# DSV4-Flash-0731 port — progress

State and dated log for the [goal](DSV4-Flash-0731-port.md). Newest first.
Update the snapshot when the state changes; append log entries per
meaningful step.

## Snapshot (2026-09-29)

Serving: `d0d93c5` live (the prefill pipe, shape-routed over the
streaming GEMV; `070b661` was the GEMV-only baseline).

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| pipe **on** (d0d93c5, W4A4) | 861.5 | 16.28 | 1800 / 35 |
| GEMV (070b661, W4A4) | 748.5 | 17.0 | 1800 / 35 |
| GEMV (070b661, W4A16) | 707.8 | 30.32 | 1800 / 35 |

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| W4A4-MX prefill **on** | 748.5 | 17.0 | 1800 / 35 |
| W4A4-MX prefill **off** (W4A16) | 707.8 | 30.32 | 1800 / 35 |

- Prefill: the nsys "mma_gemv 42.6 %" is the streaming GEMV form doing
  its job — at the 0731 dense projections it beats the 16-row tile
  GEMM (2.4–4.1 TF on sm_121a) 1.3–12× for every m (bounding the gate
  to 256 rows dropped pp2000 to 450.8). The dense projections are
  **~1.17 s of the 2.67 s e2e** (bench: per layer at m=2000 wo 18.8 ms,
  wq_b 3.8 ms, wq_a 1.6 ms, wkv 1.6 ms; lm_head 62 ms). Real target: a
  fast fp8 GEMM for m > 128 (MoE-class 128-row tiles; wq_a/wkv at
  5–10 TF and wo/lm_head at 28–34 TF in the stream).
- Decode: W4A4-MX **halves** tg (15.15/17.0 vs 30.32). The MX path is a
  prefill-only design, yet something it enables costs every decode step
  (activation-quantize launch/overhead and/or the `N fallbacks` on
  sampled decode steps — see open issue 1).

## Open issues (ordered)

1. **tg regression with W4A4-MX on** (15.15 vs 30.32 t/s). Suspect the
   fused swiglu+MX-quantize kernel running on decode batches too (extra
   launch per layer per step, latency-bound), or the decode fallback
   path paying for quantized activations it doesn't use. `serve_r0.log`
   logs `slot closed: N sampled decode steps, M fallbacks` with M rising
   per slot — understand what a fallback is.
2. **pp 2.4× below parity on W4A16** (748.5 vs 1800). The dense fp8
   projections at prefill m are the target: ~1.17 s of 2.67 s e2e
   (measured per shape in `dense_gemm_path_bench`, m=2000). The
   streaming GEMV is the right form (1.3–12× over the tile kernel), so
   the lever is a fast fp8 GEMM for m > 128 — MoE-class 128-row tiles
   (the w4a4 kernel does 91.7 TF at large M; the fp8 tile kernel does
   2.9 TF). Parity math: e2e 1111 ms budget vs GPU-only
   dense 1174 + experts 248 + attn 220 + other ~480 ms — dense AND
   experts both need ~2× to even fit the budget at zero host overhead.
   (nsys burst attribution for reference: mma_gemv 722.8 ms / 42.6 %,
   w4a4_mx 247.6 / 14.6 %, attn_flash 219.9 / 13.0 %, attn_finish +
   fabric ~7 %, mhc 4.5 %.)
3. **tg on W4A16 is 87% of parity** (30.32 vs 35). Decode path:
   mma_gemv multi-problem groups are already merged; profile the rest.

## Verified facts (don't re-derive)

- Prefill profile (nsys 2025.3.2, `spark:/tmp/pp2000.nsys-rep`, W4A4 on,
  one pp2000 burst = 1696 ms GPU): mma_gemv 722.8 ms (42.6 %) — the
  streaming GEMV form, correct by design; w4a4_mx 247.6 (14.6 %, 93
  launches, small-M per expert); attn_flash 219.9 (13.0 %);
  attn_finish + fabric ~7 %; mhc 4.5 %; tile GEMM 0 ms.
- Dense projection bench (`dense_gemm_path_bench`, m=2000, e2e per
  layer ≈ 25.9 ms): wq_a 1.64 ms / 10.3 TF, wq_b 3.83 ms / 35.0 TF,
  wkv 1.58 ms / 5.3 TF, wo 18.82 ms / 28.5 TF, lm_head 61.7 ms / 34.3 TF
  (streaming GEMV); the tile kernel is 2.4–4.1 TF on the same shapes —
  GEMV wins 1.3–12×. m=300 same verdict. Total dense ≈ 1.17 s of the
  2.67 s e2e pp2000.
- Prefill pipe (`scale_gemm_pipe_kernel`: 128×128 tiles, 3-stage
  cp.async ring, ldmatrix A, in-register fp8→bf16 weight decode on the
  same dequant bridge as the tile kernel) at the 0731 shapes (bench,
  3rd path): m=2000 — wq_a 0.44 ms / 38.3 TF (3.6× over GEMV), wkv
  0.28 ms / 29.8 TF (5.6×), wo 14.60 ms / 36.8 TF (1.31×), wq_b 3.73 ms
  / 36.0 TF (1.08×), lm_head 74.8 ms / 28.3 TF (GEMV wins 69.4 / 30.5);
  m=300 — pipe wins wq_a 1.9×, wkv 1.9×, wo 1.29×; GEMV wins wq_b
  (707.9 vs 673.4) and lm_head (12483 vs 11349). Physics: the pipe
  decodes the B tile per m-tile (grid.y = m/128 amplification), so very
  wide n or small m favors the stream. Routing: pipe iff
  `m > 128 && rs ≤ 7 && cs ≥ 6 && n ≤ 32768 && (n ≤ 4096 || m ≥ 1024)`
  (pins: `scale_gemm_grid_decode_mma_prefill_shape_routes`).
- Pipe IMA root cause (m=2000, act-end 16 B read): the A row is 10
  16-byte chunks (8 data + 2 pad); the pad clamp covered only the last
  chunk — chunk 8 at the final k stage reads 16 B past the last row.
  Both pad chunks clamp to the row's data start.
- Prefill is not chunked (scheduler `prefill_chunk_limit()` default 0):
  a 2000-token prefill is one pass, m=2000. `dense_mma` default ON
  (`DGPP_DSV41_DENSE_GEMV` env disables; `dsv41/model.cpp:193`); grid
  callers: csa2 projections, LM head, MTP main proj, engram.
- Bounding the grid launchers' decode_mma gate to
  `kScaleGemmMmaMaxRows` (c9b4c93) dropped pp2000 746.9 → 450.8 t/s —
  reverted in 070b661; the hpp documents the bound as deliberate.
- `csa2_select_bench` (micro bench target) does not compile on spark:
  pre-existing signature drift vs `csa2_select_rows_prefill` (file
  identical to its committed version) — not from `c9b4c93`.

- `mma_mx4` (2X/ue8m0) fragment + scale lane mapping matches PTX ISA 9.3
  m16n8k64 u4 exactly — proven by `tools/mma_isolate_probe.cu` (standalone
  mma + host-built fragments + exact fp64 oracle, worst 5.96e-08) and by
  `tools/mx_k320_probe.cu` (real kernel, per-column, 6 shapes, all exact).
- `moe_w4a4_test` full suite green: layout vs fp64 worst 4.7e-08 (k=320
  mx), constant-data dots **exact (worst 0)** on MX + NVFP4, W4A4 vs
  W4A16 rel-L2 ~0.967, unit perf W4A4 2.01× (gate_up), 1.76× (down).
- sm_121a quirks: 4/8-byte `cp.async` issued by 128–256 threads drops
  data → the smem scale copies are plain load+store; 16-byte `cp.async`
  is fine. `cp.async` zero-fills the dst tail when src-size < cp-size
  (tail stages exact on their own).
- Unit-test gotchas that looked like kernel bugs: a view whose
  `payload` pointed at the activation codes (n-row indexing ran off the
  end into the scale arrays → per-run garbage factors), and a 1-float
  copy of the per-row global-scale buffer (rows 1+ read zero).
- Kernels: BN=128, BK=128, 256 threads, kRowBytes=80, 2-slot smem ring
  (43008 B), `DGPP_W4A4_STAGES=3` / `DGPP_W4A4_BM=64|128` /
  `DGPP_W4A4_NT` knobs exist (measured neutral at BM128/NT1).
- YaRN: the checkpoint declares it itself
  (`text_config.rope_scaling: yarn, factor 16, original 65536 → 1M,
  beta_fast 32, beta_slow 1`) — the model's 1M context **is** the YaRN
  range; native RoPE (θ 10000) covers 64K. dgpp reads that block
  (`dsv41/config.cpp`, accepts the 0731 `type` key as well as V4.1's
  `rope_type`) and applies it bit-exact vs vLLM's `yarn_scaling_rope.py`
  (`kernels/rope_scaling.{hpp,cpp}`). Not an ad-hoc extension.
- 500K context: already fits — `kv_capacity` 1,048,576 tokens,
  `admission: full` (a request is admitted only if its whole context
  fits), YaRN defined to 1M positions. `default_max_tokens` (32768) is
  the completion default when `max_tokens` is omitted, not a context
  cap. No config change needed for 500K.

## Log

### 2026-09-29 — the prefill pipe: built, unit-green, shape-routed

- `scale_gemm_pipe_kernel` (the MoE-class geometry on the scale-aware
  path): BM/BN/BK 128/128/64, 256 threads as 4m×2n warps, 3-stage
  cp.async ring (92,160 B smem), ldmatrix A, the same fp8→bf16 decode
  bridge as the tile kernel (one scalar weight scale per 128-deep k
  pair) → strict-oracle-safe. mma m16n8k16 bf16, vectorized epilogue.
- Bugs found in order: name clash (qualify `pipe::`), missing
  namespace brace, scale-index precedence (`… + (f*BK) >> 7` →
  `… + ((f*BK) >> 7)`; caught by pipe/tile ratio = s[0]/s[1]), short-GEMM
  pipeline race (stages < kStages → cp_wait no-op; runtime switch),
  pad-chunk OOB **×2** (A row = 10 chunks, 2 pad; only the last was
  clamped — chunk 8 at the final k stage read 16 B past the last row:
  the m=2000 IMA at act+4,096,000, offset +0x1000 vs the first fix's
  +0xef0 = the two cp.async sites).
- TDD: dispatch pinned by graph-capture node count (m=300/2000 → 1 pipe
  node), both-oracle correctness at m=300/2000 (strict l2_rel ~6e-5,
  semantic ~0.0024, deterministic), shape routing pinned per class
  (new test). 19/19 `scale_gemm_test`, 4/4 `csa2_layer_test`,
  `dsv41_forward_test --smoke`, 5/5 `dsv41_engine_test` green.
- Bench: pipe beats the stream 1.3–5.6× at the 0731 prefill shapes
  (above); the stream keeps lm_head-class n and wq_b @ m=300. Dense
  bench m=2000: 1.17 s → ~0.89 s (−24 %).
- Deployed `d0d93c5` (`up --replace`, 26 s warm), smoke OK. benchy A/B:
  **pp2000 748.5 → 861.5 t/s** (ttft 2323 ms, −350 ms); tg64 16.28 (no
  decode regression — the decode window ~40 t/s, prefill dominates both
  metrics). Next levers: experts (w4a4_mx 247.6 ms), attn_flash 219.9 ms,
  fabric ~190 ms.

### 2026-09-29 — prefill hunt: the GEMV form was right; the real target is a fast fp8 GEMM (070b661)

- c9b4c93 bounded the grid launchers' decode_mma gate to
  kScaleGemmMmaMaxRows (TDD: graph-capture dispatch assertion, red at 3
  nodes for m=300, green after; both-oracle correctness at m=300/2000).
  Redeploy + benchy: **pp2000 746.9 → 450.8** — the tile GEMM lost.
- `dense_gemm_path_bench` (new) over the real 0731 shapes: GEMV wins
  1.3–12× at every shape and both m (tile 2.4–4.1 TF vs stream 5–35 TF
  on sm_121a). Reverted the bound; the test now pins the streaming
  groups; benchy back to 748.5 / 17.0.
- Consequence for the plan: the prefill gap is not a wrong-path bug —
  it's that the fp8 dense GEMM is slow at prefill row counts (~1.17 s of
  2.67 s e2e, bench-measured). The lever is a MoE-class 128-row fp8
  tile kernel (the w4a4 kernel's geometry), not a dispatch change.

### 2026-09-29 — YaRN question settled; 500K context confirmed, no change

- "Why YaRN when the model allows 1M by default?" — the 1M **is** the
  YaRN: `max_position_embeddings` 1M, `rope_scaling` yarn factor 16 from
  a 64K native range, all in the checkpoint's own config. vllm/sglang
  read the same block; dgpp matches it bit for bit.
- 500K target: `kv_capacity` 1M + `admission: full` already cover it.
  Documented in the goal doc (replacing the stray "YaRN 512K" label,
  which came from the Qwen config).

### 2026-09-29 — parity bench A/B, decode regression found, docs formalized

- Deployed `18c88f8` (`up --replace`, 26 s warm). Smoke OK: exact echo +
  reasoning content.
- benchy: W4A4 on pp 746.9 / tg 15.15; W4A4 off (DGPP_MOE_W4A4=0)
  pp 707.8 / tg 30.32.
- Goal + progress docs formalized in `plans/`; benchy command recorded.
- The workspace and `~/dev/dgpp` had diverged (commit made in the cwd
  clone only); unified on `18c88f8` — workspace is ground truth,
  `origin` there points at `~/dev/dgpp`.

### 2026-09-29 — W4A4-MX twin: exact on constant data, committed 18c88f8

- The "unit-test-only constant-data failures" were **test bugs**, not
  kernel bugs: (a) `view.payload` set to the activation-codes buffer
  (B rows ≥ 128 read OOB into the scale arrays — non-deterministic
  factors per run, tile-0 exact, tiles ≥ 1 garbage; the probe passed
  because its B codes were a properly-sized buffer); (b) `dgs` copied
  with `sizeof(float)` (one row) — rows 1+ got zero globals, killing
  the NVFP4 const rows.
- After the fixes: all four const cases exact (worst 0), full suite
  green. Removed the smem-dump scaffolding (`DGPP_W4A4_DBG`) and the
  retired micro-bench targets.
- Committed `18c88f8` "w4a4: the MXFP4 (2X, ue8m0) twin for the 0731
  experts, exact on constant dots" — kernel + quantizer + launchers +
  glm moe layer wiring + tests + the two probes. Pushed fork + nicefox.

### 2026-09-29 (earlier) — sm_121a scale-copy fix + MX kernel buildout

- 4/8-byte `cp.async` drops data under 128–256-thread issue on sm_121a
  (the "exactly half" symptom): scale copies now plain load+store.
- `mma_mx4` asm, MX activation quantizer (plain + swiglu-fused), MX
  launchers, and the 0731 expert wiring landed; mma verified against
  the exact fp64 oracle before the full kernel was trusted.

### 2026-09-28 and earlier — decode + attention groundwork

- mma_gemv multi-problem decode groups (0731 dense projections in one
  launch), csa2 window form + gather bounds, dsv41 decode mHC
  fold, 0731 head collapse read — see git log (`8c081b4`, `4a04db7`,
  `fa88a7b`, `31547bf`).

## Next steps

1. Deploy the pipe (release build, `up --replace`, benchy A/B): expect
   pp2000 ~850–950 (dense 1.17 → ~0.89 s bench; nsys dense 722.8 ms →
   ~550 ms). Record the numbers.
2. Experts at prefill M (w4a4_mx 247.6 ms, small-M per expert).
3. attn_flash 219.9 ms (13 %) + fabric ~7 %.
4. Kill the decode regression: gate the MX quantize cost out of decode
   (or make the fallback free); verify with benchy tg64 ≥ 35 with
   W4A4 on.
5. Decode profile for the remaining ~5 t/s on tg (W4A16).
6. When parity holds: `serve_tools_check.sh` + `serve_agentic_streams.py`
   as the final gate
