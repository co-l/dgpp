#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash and -Flash-0731 independent reference (docs/
deepseek_v41_flash_plan.md G4/G5): the release's OWN layer code (inference/
model.py, engram.py in the Hub snapshot) over the real checkpoint's first
layers, against the engine's per-layer dump (dsv41_forward_check --out
states.bin --layers N).
The engine's other gates compare it to a python reference written from the
same reading of the architecture; this one is the reading DeepSeek ships
(the rule that found GLM-4.7's RoPE bug: run the reference's own layer
code on the real weights before trusting a self-written reference).

model.py drives its GEMMs and a few fused steps through tilelang kernels
(kernel.py). Those are replaced here by exact torch math on the CPU:
  * every Linear's weight is dequantized ONCE to bf16 — e4m3 x e8m0 block
    scales for the fp8 tensors, e2m1 nibbles x e8m0 for the MXFP4 experts
    (both exactly representable in bf16, so nothing is lost) — and
    `linear()` is F.linear over bf16 (fp32 accumulation, bf16 out: the
    kernels' output dtype); with --act-quant the activation is first put
    through act_quant's per-32 e4m3 / power-of-two-scale quantize-dequantize
    (the kernel's own rounding, port of kernel.py's act_quant_kernel);
  * fp4_act_quant (the indexer's q/k, the compressed KV), sparse_attn (the
    two-source attention with the sink: fp32 scores, P rounded to bf16
    before PV as the kernel does, the unrounded denominator) and
    hc_split_sinkhorn (the mHC coefficient split) are ports of the kernels'
    arithmetic;
  * the Engram table rows are read from the 101 GB shard by row slices
    (only the rows the prompt's n-gram hashes name), the experts are
    dequantized per layer on demand and freed after it.
The comparison runs twice: CHAINED (the reference's own inputs, errors
accumulate down the stack) and ISOLATED (block l fed the engine's layer
l-1 output streams AND its collapse coefficients, so each number is one
layer's own error). Relative l2 over the [T, 4, H] streams per layer.

  .venv/bin/python3 tools/dsv41_torch_reference.py --checkpoint-dir DIR \
      --engine-dump states.bin [--layers 0,1,2,3,20] [--act-quant] [--threads 20]
"""
import argparse
import importlib.util
import json
import os
import struct
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the release's quantization arithmetic in torch (kernel.py's semantics)
# ---------------------------------------------------------------------------
def e8m0_ceil_pow2(x):
    """2^ceil(log2(x)) from the fp32 bit fields (kernel.py fast_round_scale)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    k = exp + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), k)


def e4m3_round(x):
    """The nearest e4m3 value (RNE, saturating at 448, subnormals at 2^-9), fp32 in and out."""
    a = x.abs()
    s = torch.sign(x)
    sub = a < 2.0 ** -6
    r_sub = torch.round(a * 512.0) * 2.0 ** -9
    m, e = torch.frexp(a.clamp_min(2.0 ** -6))  # a = m * 2^e, m in [0.5, 1)
    q = torch.round((m * 2.0 - 1.0) * 8.0)
    r = torch.clamp((1.0 + q / 8.0) * torch.ldexp(torch.ones_like(a), e - 1), max=448.0)
    return s * torch.where(sub, r_sub, r)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_round(x):
    """The nearest e2m1 value on {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to the even code)."""
    a = x.abs()
    s = torch.sign(x)
    out = torch.full_like(a, 6.0)
    for hi, v in ((5.0, 4.0), (3.5, 3.0), (2.5, 2.0), (1.75, 1.5), (1.25, 1.0), (0.75, 0.5), (0.25, 0.0)):
        out = torch.where(a <= hi if v in (0.0, 1.0, 2.0, 4.0) else a < hi, torch.full_like(a, v), out)
    return s * out


def act_quant_dequant(x, block=32, round_scale=True):
    """act_quant(block, ue8m0) quantize-dequantize per row block: bf16 in, bf16 out."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = e8m0_ceil_pow2(amax * (1.0 / 448.0)) if round_scale else amax * (1.0 / 448.0)
    q = e4m3_round(torch.clamp(blocks / s, -448.0, 448.0)) * s
    return q.reshape(xf.shape).to(torch.bfloat16)


def fp4_quant_dequant(x, block, e4m3_scale):
    """fp4_act_quant quantize-dequantize: e8m0 scales per `block` (the indexer) or e4m3 scales
    (the compressed KV, block 16)."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block)
    if e4m3_scale:
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0 ** -9)
        s = e4m3_round(amax / 6.0)
    else:
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0 ** -126)
        s = e8m0_ceil_pow2(amax * (1.0 / 6.0))
    q = e2m1_round(torch.clamp(blocks / s, -6.0, 6.0)) * s
    return q.reshape(xf.shape).to(x.dtype)


def dequant_fp8(codes, scales, block=32):
    """e4m3 codes [N, K] x e8m0 scales [N/32, K/32] -> bf16 (exact)."""
    n, k = codes.shape
    v = codes.float().reshape(n // block, block, k // block, block)
    s = scales.float().reshape(n // block, 1, k // block, 1)
    return (v * s).reshape(n, k).to(torch.bfloat16)


def dequant_fp4(packed, scales, block=32):
    """e2m1 nibbles [N, K/2] (low nibble first) x e8m0 scales [N, K/32] -> bf16 (exact)."""
    b = packed.view(torch.uint8)
    n = b.shape[0]
    lo = (b & 0xF).to(torch.long)
    hi = (b >> 4).to(torch.long)
    nib = torch.stack([lo, hi], dim=-1).reshape(n, -1)  # [N, K]
    mag = E2M1[nib & 0x7]
    val = torch.where(nib & 0x8 != 0, -mag, mag)
    k = val.shape[1]
    s = scales.float().reshape(n, k // block, 1)
    return (val.reshape(n, k // block, block) * s).reshape(n, k).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# the kernel module stand-in (bound before model.py is imported)
# ---------------------------------------------------------------------------
ACT_QUANT = False
ATTN_FP32_P = False  # the stand-in's PV product with fp32 probabilities (a rounding-order study)


def make_kernel_module():
    m = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        y = act_quant_dequant(x, block_size, round_scale=scale_fmt is not None)
        if inplace:
            x.copy_(y)
            return x
        # the non-inplace path feeds fp8_gemm / fp4_gemm: hand over the dequantized values
        return y, None

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        y = fp4_quant_dequant(x, block_size, e4m3_scale=scale_dtype == torch.float8_e4m3fn)
        if inplace:
            x.copy_(y)
            return x
        return y, None

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
        return F.linear(a, b)

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
        return F.linear(a, b)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """The kernel's arithmetic: fp32 scores, the running max floored at -1e30, P in bf16 for
        the PV product, the unrounded denominator plus the sink term, bf16 out."""
        b, s, h, d = q.shape
        out = torch.empty_like(q)
        kvf = kv.float()
        for bi in range(b):
            idx = topk_idxs[bi].long()  # [s, topk]
            valid = idx >= 0
            rows = kvf[bi][idx.clamp_min(0)]  # [s, topk, d]
            sc = torch.einsum("shd,std->sht", q[bi].float(), rows) * softmax_scale
            sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = sc.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(sc - mx)
            denom = p.sum(dim=-1) + torch.exp(attn_sink.float().view(1, h) - mx.squeeze(-1))
            pv = torch.einsum("sht,std->shd", p if ATTN_FP32_P else p.to(torch.bfloat16).float(), rows)
            out[bi] = (pv / denom.unsqueeze(-1)).to(q.dtype)
        return out

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        b, s, _ = mixes.shape
        m = mixes.float()
        pre = torch.sigmoid(m[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
        post = 2 * torch.sigmoid(m[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
        comb = (m[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]).reshape(b, s, hc_mult, hc_mult)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    m.act_quant = act_quant
    m.fp4_act_quant = fp4_act_quant
    m.fp8_gemm = fp8_gemm
    m.fp4_gemm = fp4_gemm
    m.sparse_attn = sparse_attn
    m.hc_split_sinkhorn = hc_split_sinkhorn
    return m


def make_hadamard_module():
    """The fast_hadamard_transform stand-in: the Sylvester Walsh-Hadamard in fp32, scaled, back to
    the input dtype (the CUDA kernel's arithmetic without the SMs)."""

    def hadamard_transform(x, scale):
        n = x.shape[-1]
        assert n & (n - 1) == 0
        y = x.float()
        h = 1
        while h < n:
            y = y.view(*y.shape[:-1], n // (2 * h), 2, h)
            a = y[..., 0, :]
            b = y[..., 1, :]
            y = torch.stack((a + b, a - b), dim=-2).reshape(*y.shape[:-3], n)
            h *= 2
        return (y * scale).to(x.dtype)

    m = types.ModuleType("fast_hadamard_transform")
    m.hadamard_transform = hadamard_transform
    return m


def load_release_modules(snapshot):
    """model.py and engram.py from the snapshot's inference dir, the kernel module replaced and the
    VL modules stubbed (image_processor / vision are import-time dependencies only)."""
    inference = os.path.join(snapshot, "inference")
    sys.modules["kernel"] = make_kernel_module()
    sys.modules["fast_hadamard_transform"] = make_hadamard_module()
    ip = types.ModuleType("image_processor")
    ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
    sys.modules["image_processor"] = ip
    vis = types.ModuleType("vision")

    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    vis.Aligner = _Stub
    vis.ViT = _Stub
    sys.modules["vision"] = vis
    mods = {}
    for name in ("engram", "model"):
        path = os.path.join(inference, name + ".py")
        if name == "engram" and not os.path.exists(path):
            continue  # the 0731 release ships without the engram stage
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["model"], mods.get("engram")


# ---------------------------------------------------------------------------
# the checkpoint
# ---------------------------------------------------------------------------
class Shards:
    def __init__(self, d):
        from safetensors import safe_open

        self.d = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}
        self._open = safe_open

    def file(self, name):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = self._open(os.path.join(self.d, shard), framework="pt", device="cpu")
        return self.files[shard]

    def has(self, name):
        return name in self.index

    def get(self, name):
        return self.file(name).get_tensor(name)

    def slice(self, name):
        return self.file(name).get_slice(name)

    def linear_weight(self, prefix):
        """A Linear's weight dequantized to bf16 (fp8 x e8m0, fp4 x e8m0) or as stored (bf16/f32)."""
        w = self.get(prefix + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            s = self.get(prefix + ".scale")
            return dequant_fp8(w, s, block=w.shape[0] // s.shape[0])  # the release's weight scale grid (32, 128)
        if w.dtype == torch.int8:
            return dequant_fp4(w, self.get(prefix + ".scale"))
        return w


class TokenizerShim:
    """What engram.build_compressed_token_map reads: the raw tokenizers.Tokenizer and the size."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def read_engine_dump(path):
    routes, sels = None, None
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"DSV41ST2", b"DSV41ST3"):
            raise SystemExit(f"{path}: not a dsv41_forward_check dump")
        L, T, W = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * W * 2), dtype=np.uint16).astype(np.uint32) << 16
        states = torch.from_numpy(raw.view(np.float32).reshape(L, T, W).copy())
        pre = torch.from_numpy(np.frombuffer(f.read(L * T * 4 * 4), dtype=np.float32).reshape(L, T, 4).copy())
        ids = np.frombuffer(f.read(T * 8), dtype=np.int64).copy()
        if magic == b"DSV41ST3":
            K = struct.unpack("<i", f.read(4))[0]
            routes = np.frombuffer(f.read(L * T * K * 4), dtype=np.int32).reshape(L, T, K).copy()
            Li, ms = struct.unpack("<ii", f.read(8))
            sels = np.frombuffer(f.read(Li * T * ms * 4), dtype=np.int32).reshape(Li, T, ms).copy()
    return states, pre, ids, routes, sels


def rel_l2(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--engine-dump", required=True)
    ap.add_argument("--layers", default="0,1,2,3", help="the layers to run (chained through the first block "
                    "of each contiguous run; a lone later layer, e.g. 20, runs isolated only)")
    ap.add_argument("--act-quant", action="store_true", help="quantize every Linear's activation as the kernels do")
    ap.add_argument("--hc-bf16", action="store_true",
                    help="round the mHC coefficient matrices hc_attn_fn / hc_ffn_fn to bf16 as the engine's "
                         "loader does (the deliberate load-time rounding; the kernels are bf16-only)")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="float32: the whole reference in fp32 (the caches' quantizations kept: a near-exact "
                         "oracle to measure BOTH the engine and the release's bf16 pipeline against)")
    ap.add_argument("--save-states", default=None, help="write this run's isolated layer outputs (npz)")
    ap.add_argument("--baseline", default=None,
                    help="an earlier --save-states file (e.g. the bf16 run) to report against this run's outputs")
    ap.add_argument("--attn-fp32-p", action="store_true",
                    help="the attention stand-in keeps its probabilities in fp32 for the PV product (the "
                         "kernels round them to bf16 against a running max; a rounding-order study)")
    ap.add_argument("--fp32-down", action="store_true",
                    help="keep every expert's down projection in fp32 into the MoE sum (the engine's chain: the "
                         "release's bf16 GEMM output rounds each expert's contribution once)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.005,
                    help="the isolated per-layer relative-l2 budget over the rows routed alike, against the bf16 "
                         "reference (two bf16 pipelines rounding at other points: 0.1-0.4 %% on the real weights); "
                         "the fp32 oracle rule (--dtype float32 --baseline) is the definitive gate")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    global ACT_QUANT, ATTN_FP32_P
    ACT_QUANT = args.act_quant
    ATTN_FP32_P = args.attn_fp32_p or args.dtype == "float32"
    snap = args.checkpoint_dir
    states, pre_dump, ids, routes, sels = read_engine_dump(args.engine_dump)
    L_dump, T, W = states.shape
    print(f"engine dump: {L_dump} layers x {T} rows x {W}; act_quant {'on' if ACT_QUANT else 'off'}, "
          f"expert down projections {'fp32' if args.fp32_down else 'bf16 (the release kernels)'}, "
          f"hc_fn {'bf16 (as loaded by the engine)' if args.hc_bf16 else 'fp32 (as stored)'}")
    model, engram = load_release_modules(snap)
    has_engram = engram is not None
    pre_mix_mode = hasattr(model, "make_identity_pre_mix")
    fp8_block = getattr(model, "fp8_block_size", model.block_size)
    if hasattr(model, "rotate_activation"):
        orig_rotate = model.rotate_activation

        def rotate_activation(x):
            # the release's rotation asserts bf16 (the kernels rotate bf16 rows);
            # an fp32 pipeline carries its value, the rotation itself stays bf16.
            return orig_rotate(x.to(torch.bfloat16)).to(x.dtype)

        model.rotate_activation = rotate_activation
    cfg = json.load(open(os.path.join(snap, "inference", "config.json")))
    cfg = {k: v for k, v in cfg.items() if k in model.ModelArgs.__dataclass_fields__}
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = max(4096, ((T + 127) // 128) * 128)
    if "vision_n_layers" in model.ModelArgs.__dataclass_fields__:
        cfg["vision_n_layers"] = 0  # text only: the gate's VL bias is dead code
    margs = model.ModelArgs(**cfg)
    ref_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    torch.set_default_dtype(ref_dtype)
    model.world_size, model.rank = 1, 0
    model.default_dtype = ref_dtype  # every Linear built in the run's dtype: the dequantized weights land in it
    H = margs.dim
    hc = margs.hc_mult
    assert W == hc * H, (W, hc, H)
    shards = Shards(snap)
    if ACT_QUANT:
        real_linear = model.linear

        def quant_linear(x, weight, bias=None):
            # only the weights the checkpoint stores quantized take a quantized activation (the
            # kernels' dispatch by weight dtype); the bf16/fp32 Linears run plain
            if getattr(weight, "_dgpp_quantized", False):
                x = act_quant_dequant(x, fp8_block, True)
            return real_linear(x, weight, bias)

        model.linear = quant_linear
    # The Engram hash state (the release's own token map + multipliers) and the embedding table
    # rows read by slices from the shard.
    # Lean constructors: model.py's Linear and the Engram table allocate their storage at
    # construction (an expert layer's 27 GB of placeholders, the 98 GB table) — the weights
    # come from the shards below, so the constructors keep the shapes only.
    def lean_linear_init(self, in_features, out_features, bias=False, dtype=None):
        torch.nn.Module.__init__(self)
        self.in_features, self.out_features = in_features, out_features
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
        self.register_parameter("scale", None)
        self.register_parameter("bias", None)

    model.Linear.__init__ = lean_linear_init

    def lean_engram_init(self, num_embeddings, dim):
        torch.nn.Module.__init__(self)
        self.num_embeddings, self.dim = num_embeddings, dim
        self.part_num_embeddings = num_embeddings
        self.vocab_start_idx, self.vocab_end_idx = 0, num_embeddings
        self.block_size = fp8_block
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
        self.scale = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))

    layout, hashes = None, None
    if has_engram:
        model.ParallelEngramEmbedding.__init__ = lean_engram_init
        layout = engram.EngramLayout.from_args(margs)
        t0 = time.time()
        hash_state = engram.NgramHashState(margs, layout, TokenizerShim(os.path.join(snap, "tokenizer.json")))
        print(f"engram hash state built in {time.time() - t0:.1f} s (compressed vocab {hash_state.token_map.max().item() + 1})")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    if has_engram:
        hashes = hash_state(input_ids, 0)  # [1, T, n_engram_layers, n_hash_cols]

    def load_block(l):
        blk = model.Block(l, margs, layout) if layout is not None else model.Block(l, margs)
        with torch.no_grad():
            for name, param in list(blk.named_parameters()):
                full = f"layers.{l}.{name}"
                if "ffn.experts." in name or "engram.embed" in name:
                    continue  # the experts on demand, the table rows by slice
                if not shards.has(full):
                    raise SystemExit(f"missing tensor {full}")
                if name.endswith(".weight") and shards.has(full[:-len(".weight")] + ".scale"):
                    param.data = shards.linear_weight(full[:-len(".weight")]).to(ref_dtype)
                    param._dgpp_quantized = True
                elif param.numel() == 0:
                    # a Linear placeholder: the stored bf16/f32 tensor, in the dtype the module
                    # computes in (the ratio-2 compressor's fp32)
                    w = shards.get(full)
                    mod_dtype = torch.float32 if ".compressor." in name and margs.compress_ratios[l] > 1 else w.dtype
                    if ref_dtype == torch.float32:
                        mod_dtype = torch.float32
                    param.data = w.to(mod_dtype)
                else:
                    w = shards.get(full).to(param.dtype)
                    if args.hc_bf16 and (name.endswith("hc_attn_fn") or name.endswith("hc_ffn_fn")):
                        w = w.to(torch.bfloat16).to(param.dtype)
                    param.data = w
        # the experts: dequantized on first use, per layer
        cache = {}

        def expert_forward_fp32_down(expert, x, weights=None):
            # model.py's Expert.forward with the down projection accumulated in fp32 and left there
            # (F.linear over bf16 rounds its output to bf16; the engine's expert sum does not).
            dtype = x.dtype
            gate = expert.w1(x).float()
            up = expert.w3(x).float()
            if expert.swiglu_limit > 0:
                up = torch.clamp(up, min=-expert.swiglu_limit, max=expert.swiglu_limit)
                gate = torch.clamp(gate, max=expert.swiglu_limit)
            xx = F.silu(gate) * up
            if weights is not None:
                xx = weights * xx
            xx = xx.to(dtype)
            if ACT_QUANT and getattr(expert.w2.weight, "_dgpp_quantized", False):
                xx = act_quant_dequant(xx, fp8_block, True)
            return F.linear(xx.float(), expert.w2.weight.float())

        def make_forward(i, expert):
            def fwd(x, weights=None):
                if i not in cache:
                    ws = {}
                    for nm in ("w1", "w2", "w3"):
                        ws[nm] = shards.linear_weight(f"layers.{l}.ffn.experts.{i}.{nm}")
                    cache[i] = ws
                ws = cache[i]
                if ref_dtype == torch.float32 and ws["w1"].dtype != torch.float32:
                    ws = cache[i] = {k: v.to(torch.float32) for k, v in ws.items()}
                expert.w1.weight.data, expert.w2.weight.data, expert.w3.weight.data = ws["w1"], ws["w2"], ws["w3"]
                if args.fp32_down:
                    return expert_forward_fp32_down(expert, x, weights)
                return model.Expert.forward(expert, x, weights)
            return fwd

        for i, expert in enumerate(blk.ffn.experts):
            if expert is not None:
                expert.forward = make_forward(i, expert)
                for lin in (expert.w1, expert.w2, expert.w3):
                    lin.weight._dgpp_quantized = True
        idx = getattr(blk.attn, "indexer", None)
        if idx is not None:
            orig_idx = idx.forward

            def idx_fwd(x, qr, start_pos, offset, _orig=orig_idx):
                out = _orig(x, qr, start_pos, offset)
                sel_log[l] = out.detach().squeeze(0).cpu().numpy()
                return out

            idx.forward = idx_fwd
        if args.fp32_down:
            shared = blk.ffn.shared_experts
            shared.forward = lambda x, weights=None, e=shared: expert_forward_fp32_down(e, x, weights)
        eng = getattr(blk, "engram", None)
        if eng is not None:
            emb = eng.embed
            wname = f"layers.{l}.engram.embed.weight"
            sname = f"layers.{l}.engram.embed.scale"
            wsl, ssl = shards.slice(wname), shards.slice(sname)

            def embed_forward(indices, emb=emb, wsl=wsl, ssl=ssl):
                flat = indices.reshape(-1)
                uniq, inv = torch.unique(flat, return_inverse=True)
                rows = []
                for r in uniq.tolist():
                    v = wsl[r:r + 1].float()
                    s = ssl[r:r + 1].float()
                    rows.append((v.unflatten(-1, (-1, emb.block_size)) * s.unsqueeze(-1)).flatten(-2))
                table = torch.cat(rows, dim=0).to(torch.bfloat16).to(ref_dtype)
                return table[inv].reshape(*indices.shape, -1)

            emb.forward = embed_forward
        return blk

    layers = [int(v) for v in args.layers.split(",")]
    # The embedding: the block 0 input is embed[ids] expanded to hc copies (bf16).
    embed = shards.get("embed.weight").to(ref_dtype)
    h0 = embed[input_ids].unsqueeze(2).repeat(1, 1, hc, 1)
    pre0 = model.make_identity_pre_mix(h0, hc) if pre_mix_mode else None
    results = {}
    kept_results = {}
    # The reference's own routing per layer (the gate's picks and the near-tie margin of the
    # sixth pick over the seventh, relative to the score range), captured by a hook.
    gate_log, sel_log = {}, {}

    def hook_gate(blk, l):
        gate = blk.ffn.gate
        orig = gate.forward

        def fwd(x, image_mask=None):
            weights, indices = orig(x, image_mask)
            with torch.no_grad():
                scores = torch.nn.functional.linear(x.float(), gate.weight.float()) / getattr(gate, "gate_temp", 1.0)
                if gate.hash:
                    margin = torch.zeros(indices.size(0), dtype=torch.float32)  # the hash route is fixed per token
                else:
                    scores = torch.nn.functional.softplus(scores).sqrt()
                    if gate.bias is not None:
                        scores = scores + gate.bias
                    top = scores.topk(gate.topk + 1, dim=-1).values
                    rng = (scores.amax(dim=-1) - scores.amin(dim=-1)).clamp_min(1e-30)
                    margin = (top[:, gate.topk - 1] - top[:, gate.topk]) / rng
            gate_log[l] = (indices.sort(dim=-1).values.cpu().numpy(), margin.cpu().numpy())
            return weights, indices

        gate.forward = fwd

    if hasattr(margs, "index_source_layers"):
        index_sources = [ls for ls in margs.index_source_layers]
    else:
        index_sources = [l for l in range(margs.n_layers) if margs.compress_ratios[l] == 4]

    saved = {}
    baseline = dict(np.load(args.baseline)) if args.baseline else None

    def report(l, mode, got, blk, t0):
        """The layer's relative l2 over every row, and — with the engine's routes in the dump —
        over the rows whose routed experts agree (a row routed to a different expert is a
        different computation; it is reported with the reference's own near-tie margin). An
        index source's selections (the reference's shared topk_idxs less the window offset)
        are compared the same way."""
        want = states[l].reshape(T, hc, H)
        d = rel_l2(got, want)
        rows = np.array([rel_l2(got[t], want[t]) for t in range(T)])
        results.setdefault(l, {})[mode] = d
        line = f"layer {l:2d} {mode:8s}: rel l2 {d:.5f} (worst row {rows.max():.5f} at t{int(rows.argmax())})"
        if mode == "isolated":
            saved[f"layer{l}"] = got.float().cpu().numpy()
            if baseline is not None and f"layer{l}" in baseline:
                b = torch.from_numpy(baseline[f"layer{l}"])
                line += f"; the baseline run vs this one: rel l2 {rel_l2(b, got):.5f}"
        sel_flips = []
        if sels is not None and l in index_sources:
            si = index_sources.index(l)
            if si < sels.shape[0]:
                if getattr(model, "shared_attn", None) is not None and model.shared_attn.topk_idxs is not None:
                    ref_sel = model.shared_attn.topk_idxs[0].cpu().numpy().astype(np.int64)  # [T, topk], offset by T
                elif l in sel_log:
                    ref_sel = sel_log[l]  # the prefill indexer output, offset by T
                else:
                    ref_sel = None
                if ref_sel is not None:
                    for t in range(T):
                        r = set(int(v) - T for v in ref_sel[t] if v >= 0)
                        g = set(int(v) for v in sels[si, t] if v >= 0)
                        if r != g:
                            sel_flips.append((t, len(r ^ g)))
                line += f"; selections: {T - len(sel_flips)} rows equal"
                if sel_flips:
                    line += ", flips" + "".join(f" t{t}(±{n})" for t, n in sel_flips[:6])
        if routes is not None and l in gate_log:
            ref_ids, margin = gate_log[l]
            flips = [t for t in range(T) if not np.array_equal(ref_ids[t], np.sort(routes[l, t]))]
            sel_only = [t for t, _ in sel_flips if t not in flips]
            if sel_only:
                # A selection that differs by an entry or two at the top-k boundary (the reference
                # scores its index logits in bf16 and lets torch order the ties; the engine scores
                # in fp32): those rows' own distance says whether the swapped entries matter.
                line += f"; {len(sel_only)} rows with a selection flip only: rel l2 {rel_l2(got[sel_only], want[sel_only]):.5f}, worst {rows[sel_only].max():.5f}"
            flips = sorted(set(flips) | {t for t, _ in sel_flips})
            kept = [t for t in range(T) if t not in flips]
            if kept:
                dk = rel_l2(got[kept], want[kept])
                kept_results.setdefault(l, {})[mode] = dk
                line += f"; {len(kept)} rows routed alike: rel l2 {dk:.5f}, worst {rows[kept].max():.5f}"
            if flips:
                line += "; route flips" + "".join(f" t{t}(margin {margin[t]:.2e}, l2 {rows[t]:.3f})" for t in flips[:8])
                if len(flips) > 8:
                    line += f" +{len(flips) - 8}"
        print(line + f" in {time.time() - t0:.1f} s")

    # ---- chained: the contiguous run from layer 0 ------------------------------------
    chained = []
    l = 0
    while l in layers:
        chained.append(l)
        l += 1
    h, pre_mix = h0, pre0
    for l in chained:
        t0 = time.time()
        blk = load_block(l)
        hook_gate(blk, l)
        with torch.inference_mode():
            x = h
            eng = getattr(blk, "engram", None)
            if eng is not None:
                x = eng(x, hashes[:, :, eng.layer_hash_index, :], None)
            if pre_mix_mode:
                h, pre_mix = blk(x, 0, pre_mix, None)
            else:
                h = blk(x, 0, input_ids)
        report(l, "chained", h[0], blk, t0)
        del blk
    # ---- isolated: block l from the engine's layer l-1 output and coefficients ----------------
    if hasattr(model, "SharedAttentionRuntime"):
        model.shared_attn = model.SharedAttentionRuntime()
    for l in layers:
        if l >= L_dump:
            print(f"layer {l}: not in the engine dump ({L_dump} layers)")
            continue
        t0 = time.time()
        blk = load_block(l)
        hook_gate(blk, l)
        if l == 0:
            x_in, pre_in = h0, pre0
        else:
            x_in = states[l - 1].reshape(1, T, hc, H).to(ref_dtype)
            pre_in = pre_dump[l - 1].reshape(1, T, hc)
        with torch.inference_mode():
            x = x_in
            eng = getattr(blk, "engram", None)
            if eng is not None:
                x = eng(x, hashes[:, :, eng.layer_hash_index, :], None)
            if pre_mix_mode:
                h_out, _ = blk(x, 0, pre_in, None)
            else:
                h_out = blk(x, 0, input_ids)
        report(l, "isolated", h_out[0], blk, t0)
        del blk
    if args.save_states:
        np.savez(args.save_states, **saved)
    scored = kept_results if kept_results else results
    worst = max(v.get("isolated", 0.0) for v in scored.values())
    if baseline is not None:
        # The oracle rule (an fp32 run against a saved bf16 run): the engine is no further from
        # the near-exact evaluation than the release's own bf16 pipeline is, within 1.5x per layer.
        ratios = []
        for l, v in results.items():
            if "isolated" not in v or f"layer{l}" not in baseline or f"layer{l}" not in saved:
                continue
            b = rel_l2(torch.from_numpy(baseline[f"layer{l}"]), torch.from_numpy(saved[f"layer{l}"]))
            ratios.append((l, scored.get(l, v).get("isolated", v["isolated"]), b))
        worst_ratio = max(e / max(b, 1e-9) for _, e, b in ratios) if ratios else 0.0
        ok = worst_ratio <= 1.5
        print(f"{'OK' if ok else 'FAIL'}: the engine's distance from the fp32 evaluation over the rows routed alike "
              f"against the bf16 pipeline's: " + ", ".join(f"L{l} {e:.4f}/{b:.4f}" for l, e, b in ratios) +
              f" (worst ratio {worst_ratio:.2f}, allowed 1.5)")
    else:
        ok = worst <= args.budget
        print(f"{'OK' if ok else 'FAIL'}: worst isolated relative l2 {worst:.5f} over the rows routed alike "
              f"against the budget {args.budget} (act_quant {'on' if ACT_QUANT else 'off'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
