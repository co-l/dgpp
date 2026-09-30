# DSV4-Flash-0731 port — progress

State and dated log for the [goal](DSV4-Flash-0731-port.md). Newest first.
Update the snapshot when the state changes; append log entries per
meaningful step.

## Snapshot (2026-09-30)

Live: W4A16 site default + flash DB + wo_b ws + the per-group scale
prefetch (mma_gemv fp8 decode forms). Parity 58 % on pp, ~88 % on tg.
The cp.async-staged slot kernels are merged (bitwise the register pass,
incl. the multi-pass K geometries) but OFF by default — the single-tile
form is ~39 % slower than the register pass at the live 4096/2048
geometry; the double-buffered pipeline is the next step (env:
DGPP_MOE_SLOT_CPASYNC=1).

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| + slot-kernel launch_bounds (gate 3, down 4) | 1033.0 | 30.15 (55.57 step-chunks/s fixed-bench) | 1800 / 35 |
| + per-group scale prefetch (2026-09-29 late night) | 1032.4 | ~30 (54.86 step-chunks/s fixed-bench) | 1800 / 35 |
| W4A16 site default + wo_b ws (2026-09-29 night) | 1031.9 | ~30 (54.79) | 1800 / 35 |
| + flash DB, W4A16 site default (e4f4acb) | 1033.4 | 29.1 | 1800 / 35 |

**Part memory ground truth (2026-09-29 late night, corrects the
"155 GB/s = the practical ceiling" story):** DRAM streaming peak =
**830 GB/s** (512MB–1GB plain reads), L2 = **24 MB** (not 126), L2 read
~1.9 TB/s. A 24 MB+ flush of the 16.8 MB wq_b-class slab reads at
580–600 GB/s — the honest per-slab cold number. The earlier
155 GB/s "ceiling" (experts + dense GEMVs) was 3.7–5× BELOW that
ceiling: the kernels have real headroom, the part does not. The old
273 GB/s spec peak was wrong for this part, and the 64 MB-flush
micro-benches were L2-poisoned (256 MB flushes leave a dirty-writeback
storm that starves reads to 12–24 GB/s; a 64 MB flush only half-evicts
the 24 MB L2). Benchy's tg numbers are not comparable across runs
(either: random corpus start per prompt, no seed) — the deterministic
driver (fixed 2000-token prompt, temp 0, min_tokens 512, ignore_eos)
reproduces 514 step-chunks in 9.4–9.5 s at ±0.04 %.

**Decode step fully mapped (2026-09-29 night, nsys 187 steps, wall
77.4 ms, 98.2 % GPU busy):**

- MoE experts — **revised (late night): NOT at the ceiling.** At the
  true 830 GB/s DRAM peak (and 580–600 GB/s practical slab cold) the
  experts' 155 GB/s is 3.7× off — real kernel headroom, same as the
  dense GEMVs. (Originally read "at the ceiling" against the wrong
  273 GB/s peak; the unique-traffic proof below stands: the bytes are
  the routing-set, but the streaming rate is not.)
- MoE experts — the traffic proof:
  The unique-expert trace (new env-gated `DGPP_MOE_UNIQUE_TRACE`,
  device-side, in the graph decode path): n = 30/36 ids/layer (5–6 MTP
  rows × top-6), unique = 8..36, mean **21.2** (random would be ~29 —
  the MTP rows share routing heavily). Per layer that is 94.5 MB
  (gate_up) + 47.2 MB (down) of unique-weight traffic at 611/286 us
  = **~155 GB/s — the same practical ceiling the dense GEMVs hit**
  (147–171 GB/s cold-L2 single-kernel streaming; 273 GB/s is the
  part's peak). Experts and dense GEMVs all sit at 1.6–1.7× the
  DRAM floor; the SPLITK_FILL A/B (96→192) is flat on tg — it is a
  bandwidth ceiling, not a parallelism deficit.
- **wo_b decode was missing the GEMM workspace** (every other
  `launch_scale_gemm_grid_*` call passes it) — the split-K never
  fired (64 blocks on 48 SMs). Fixed (harmless-to-small: wo_b is
  n=k=4096 = 16 MB, 1.68× floor before, streaming ceiling after).
  The earlier "wo_b 7–13× floor" micro-bench reading was a k=1024
  misread (wo_b's k is lg×o_lora = 4096).
- **The dense GEMV family (21.5 ms/step) is the whole step's
  streaming tail: every site at ~155–170 GB/s.** wq_b 16.8 MB/98 us,
  wo_b 16 MB/99 us, lm_head 265 MB/1091 us (1.12×, at the floor),
  the bf16 compressor sites 1.3×. Nothing under-filled remains
  (split-K covers every small-n site; 124 split-reduces + 92
  multi-reduces per step confirm).
- **Remaining structural levers (ranked):**
  1. Fabric fold latency: 37.2 us × 92.2/step = **3.43 ms** — the
     recorded bus node (post + peer doorbell + fold) is
     single-outstanding by the v1 contract; polling is 32 ns
     (already tight). Pipelining the next boundary's send under the
     current fold = a bus-protocol project; worth ~1.5–2.5 ms/step.
  2. mhc decode per-coefficient form: 22.2 us × 92.2 = **2.05 ms** —
     96 blocks re-read 32 KB stream rows 24× per token; the bitwise
     token-tiled form exists for prefill and is gated off for decode
     (`!decode_rows`) because the fused finish (tickets + deferred
     side-stream comb) is decode-specific. **A/B'd 2026-09-29 (night)
     and REJECTED** (`DGPP_MHC_TILE_DECODE`, unit-tested in the fused
     finish's tolerance class): tg256 24.4 tiled vs 27.6 fused — at
     decode row counts the per-coefficient form's 96-block parallelism
     beats the tiled one-block-per-4-tokens; the tiling's L2-traffic
     win only pays at prefill scale. Keep the fused form.
  3. attn_flash 1.15 ms, lm_head 0.2 ms, smalls ~1.5 ms.
  Structure ceiling: ~73.5 ms/step ≈ 33 t/s. **Parity (35 t/s ≈
  70 ms/step) needs a structural change**: the fold pipeline (1),
  a TP topology without the per-layer dense boundary (dense
  replicated / MoE split — a code project), or a higher MTP
  acceptance (2.2 tok/step now).

Serving: `c511a51` live (the prefill pipe, shape-routed over the
streaming GEMV, the bf16 mma row bound: wide-m bf16 matmuls take the
Lt tile GEMM, and the prefill attention split = 1: the flash's grid
fills the SMs at 128 blocks, so the 4-way split only bought c_main f32
traffic — 8.4 MB per (split, tile) written by the flash, read by the
finish — which the 1-split form drops 4x).

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| + flash DB, W4A16 site default (e4f4acb) | 1033.4 | 29.1 | 1800 / 35 |
| + flash DB (1c1f0c9, W4A4) | 1127.7 / 1131.2 | 15.3 | 1800 / 35 |
| pipe + Lt dot + split 1 (c511a51, W4A4) | 1117.8 | ~17 | 1800 / 35 |
| pipe + Lt dot (e0d3f14, W4A4) | 1032.1 | 16.57 | 1800 / 35 |
| pipe (d0d93c5, W4A4) | 861.5 | 16.28 | 1800 / 35 |
| GEMV (070b661, W4A4) | 748.5 | 17.0 | 1800 / 35 |
| GEMV (070b661, W4A16) | 707.8 | 30.32 | 1800 / 35 |

Cold-prefill GPU mix after both changes (nsys, ~2.28 s): w4a4 experts
~471 ms (20.7 %, ~80 TF/launch — bandwidth-bound; the BM=64 A/B was
neutral-to-worse), attn_flash ~250 ms (was 337; the c_ws write
quartered by the split), scale pipe 286 ms, fabric 165 ms,
attn_finish ~40 ms (was 135; the 1-split c_main), mhc ~90 ms, the Lt
dot GEMM (was the 343 ms mma storm).

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
- Decode: W4A4-MX **halves** tg (15.15/17.0 vs 30.32). **Solved
  2026-09-29 (evening):** the per-step decode cost is IDENTICAL in both
  modes (nsys per-step breakdown, 77.4 vs 80.0 ms/step, same kernel
  mix — decode rows < 256 take the W4A16 dequant GEMV in both worlds).
  What collapses is the **MTP draft acceptance**: the W4A4 prefill
  (block-scale fp4 experts) leaves the hidden states numerically
  shifted, and at temp 1.0 over the 129k vocab the draft's top-1 flips
  — accept p1 13–46 % (1.17–2.1 tok/step) W4A4 vs 58–92 % (2.1–4.1
  tok/step) W4A16. vllm-prod's MXFP4 path is numerically consistent
  across prefill/decode, so it never pays the split. Shipped:
  `DGPP_MOE_W4A4=0` as the live site default (site-env plumbing:
  SITE_KEYS/NODE_KEYS + the serve-side node_env allowlist + unit test,
  e4f4acb). Revisit when the W4A4 prefill and decode numerics are made
  consistent (or the acceptance is made robust).

## Open issues (ordered)

1. **Decode step cost: the experts are 53 % of a 78 ms step at 1.9× the
   DRAM floor.** Per-step nsys (W4A16, 0731): moe_slot_gate_up_swiglu_fp4
   28.2 ms (46 chains × 611 us) + moe_slot_down_fp4 13.2 ms (285 us) =
   41.4 ms/step; the floor for 7 experts × 43 layers is ~22 ms @273 GB/s.
   mma_gemv 12.9 ms, bus_allreduce 3.4 ms, mhc 2.1 ms, attention the rest.
   At the 29 t/s acceptance (2.1 tok/step) hitting 35 needs ~65 ms/step.
   Identified next step (2026-09-30): the double-buffered cp.async
   pipeline over the (now bitwise-correct, env-gated) staged slot
   kernels — stage tile p+1 while computing tile p.
2. **W4A4 prefill ↔ decode numerics split** (the tg 16-vs-29 cause, solved
   as a config for now, e4f4acb): make the block-scale fp4 prefill and the
   W4A16 dequant decode agree closely enough that the MTP acceptance
   survives (or a robust acceptance test). Until then the site default is
   W4A16 (pp 1033 vs 1131).
3. **pp 2.4× below parity on W4A16** (748.5 vs 1800; 1033.4 after the
   pipe/Lt/split-1/flash-DB work). The dense fp8 projections at prefill m
   are the target: ~1.17 s of 2.67 s e2e
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

- **The two-step (2 rows/warp) slot gate_up is bitwise-identical to the
  one-step form but perf-neutral to negative at every bench geometry**
  (2026-09-29 probe, mechanism reverted — not in the tree): gate_up
  1104 vs 1103 us @ 4096/2048, 196 vs 199 @ 2048/1024, 31 vs 35 @
  1024/512. At the live K=4096 the one-step per-row pipeline is already
  saturated (rps=1, 8 chunks/lane in flight); rows-per-warp is NOT the
  expert-kernel lever. The slot kernels sit at ~144–155 GB/s vs the
  580–600 GB/s practical slab-cold: the lever is the memory pattern
  (vllm 0731 reference at ~/dev/sparkrun-ds4/).
- **Post-split-1 cold-prefill mix (nsys `spark:/tmp/pp2000split1.nsys-rep`,
  one 2000-token burst, GPU busy 1792 / wall 1835 ms, 97.7 %):**
  moe_grouped_w4a4_mx bf16-out 301.5 (16.8 %) + f32-out 165.0 (9.2 %) =
  **466 ms (26 %)**; scale_gemm_pipe 291.8 (16.3 %, 356 launches);
  attn_flash 275.6 (15.4 %, 1401 × 196.7 us); bus_bulk_collective 173.8
  (9.7 %); mhc ~118 ms (6.5 %); attn_finish **37.6 (2.1 %, was 135 — the
  split-1 c_main cut)**; moe_grouped_mma_fp8_ldm 78; moe_accum 48;
  mma_gemv<4> 39.5.
- **The pipe's exact 0731 shapes (m=2048, via DGPP_PIPE_TRACE × nsys grid
  decode):** 1024×4096 wq_a 109.3 ms/210, 16384×1024 wq_b 78.8/42,
  4096×4096 71.3/42, 8192×1024 18.2/20, 512×4096 wkv 14.1/42 — all
  25–40 TF, none bandwidth-bound (the 1024×4096 floor is ~62 us at 273
  GB/s vs 520 us measured). wo does NOT take the pipe (the grouped
  multi-problem mma path, 369 launches).
- **Pipe kernel is latency-bound (ncu, wq_a m=2000):** occupancy 16.7 %
  (one block per SM, the 92 KB smem limit; 8/64 warps), warp cycles per
  issued instruction ~8.4, SM active ~49 %, memory ~24 %. 512 threads
  (16 warps, 4×4 warp layout) → 33 % occupancy, 3–6 % faster on every
  pipe shape (wq_a 455→426, wo 14090→13663), lm_head flat. The
  BM=64/BK=32/4-stage two-blocks-per-SM variant loses 2.2–2.5× on wide
  n (B DRAM traffic doubles: 2× the m-tile blocks re-fetch each B tile).
- **w4a4 experts hit the DRAM ceiling:** ~478 MB of weight traffic per
  launch / 1.75 ms ≈ 273 GB/s = the GB10 peak. Bandwidth-bound; no
  kernel-level win (the BM/NT knobs were neutral).

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
- Cold-prefill profile with the pipe (nsys burst, 2.28 s GPU):
  w4a4_mx gate_up 305 ms + down 166 ms (20.7 %, ~3.3 TF/launch at ~62
  rows/expert: BM=128 tiles half-empty), attn_flash 337 ms (14.8 %,
  240 us/launch × 1401), scale_gemm_pipe 286 ms (12.4 %),
  bus_bulk_collective 165 ms (7.2 %), attn_finish 130 ms, mhc ~90 ms.
  The bf16 mma storm: the indexer select dot (m = 131072 = rows × 64,
  n = 512, k = 128, f32; 20 calls) at 128-row mma groups = 21,184
  launches, 343 ms (15.1 %). mma_gemv is the GEMV form — 8 blocks on 48
  SMs, the 128-KB weight re-read per 128-row group; ~16 us/launch.
  The 16-tile (256-row) grouping measured WORSE (31.3 us/launch,
  362.6 ms: 64-accumulator pressure + one-block-per-SM smem). The Lt
  tile GEMM takes it (set_decode_mma(true, 128) on the model's bf16
  instance): pp2000 861.5 → 1032.1 t/s, ttft 2317 → 1939 ms.
  csa2 selection: 0 flips after the change (near-tie certified).
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

### 2026-09-30 — cp.async slot kernels landed (correct, env-gated); two NaN-hunt root causes; A/B says register-pass keeps the live seat

- **The cp.async-staged slot kernels (gate_up + down) are in, unit-green
  both modes, sanitizer clean.** `DGPP_MOE_SLOT_CPASYNC=1` switches the
  fp4 slot kernels to the smem-staged form (weights + scales staged with
  cp.async before the activations; the FMA chain runs off shared). The
  bitwise pin (new A/B test `moe_slot_cpasync_is_bitwise_the_register_pass`)
  holds at every shape the slot path instantiates, now including the
  3-pass 576 geometry.
- **Root cause 1 — the NaN hunt was an uninitialized accumulator.** The
  cp.async kernels declared `float acc[1];` and passed it to
  `consume_chunk`, whose first FMA READS the accumulator. The register
  path is safe because `warp_row_dots`/`row_dots` zero their accumulators
  internally; the cp.async twins broke that contract and read register
  garbage (a deterministic 0x7fffffff / FLT_MAX on the live sequence —
  the "FMA overflow" that looked data-dependent). Fixed: `= {0.f}`.
  That was the whole 512/512-NaN on the 4 decode-bitwise tests.
- **Root cause 2 — multi-pass K was half-implemented.** K in
  {576, 1152, 2304, 160, 320, 640} needs >4 chunks/lane (up to 3 passes),
  but the cp.async loop only ran `pass_chunks(0)` — a partial sum
  (256 of 576 elements at the mxfp4 576 down). The live 4096/2048
  geometry is single-pass, so A/B never caught it; the real mxfp4 576
  test did (±2 % element diffs, not NaNs — finite scales, so the missing
  terms are just missing). Fixed: both kernels now iterate
  `p in 0..passes` with `C0 = p*kMaxChunksPerLane`, the exact
  `warp_row_dots` pass structure, so the FMA order (and the bits) match.
- **The A/B test's MXFP4 pin had been VACUOUS all along** (NaN == NaN):
  its e8m0 fill `% 254` hits the 241–253 inf/NaN bytes (0 code × inf =
  NaN), and the wide bf16 x-fill (up to ~2^120) overflows the gate dots
  so `g*sigmoid(g)` NaNs the negative side. Capped the fills (e8m0 ≤ 115,
  x in 0x3C80..0x3CFF, shared e4m3 < 0x78) and added the 576 case — the
  mx pin now carries real weight.
- **Micro-bench A/B (live 4096/2048 geometry, post-fix):** register-pass
  gate 751 us (212 GB/s) / down 391 us (204 GB/s) vs cp.async single-tile
  1086 us (147) / 485 us (164) — the staged form is ~39 % SLOWER. The
  single-tile life is stage-everything → FMA: no DRAM/FMA overlap, and the
  smem round-trip costs more than the register file saved. **Decision:
  register-pass stays the live default; cp.async is env-gated and is the
  stepping stone for the double-buffered grid-stride pipeline** (stage
  tile p+1 while computing tile p — the ~2× headroom vs the 200 GB/s
  register ceiling, per the late-night-3 occupancy analysis).
- **Verification:** glm_moe_test 34/34 with CPASYNC=0 AND =1;
  compute-sanitizer memcheck 0 errors on the 16 bitwise tests (=1);
  full build links; ctest failures (mimo fixtures, dsv41 tp-bus) are
  environmental (missing .mdump staging / no live cluster), not
  MoE-related.

### 2026-09-29 (late night 3) — slot experts: occupancy is the wall; launch_bounds landed

- **ncu on the slot gate_up (isolated bench, 19 unique experts):** the
  kernel is latency-bound, not bandwidth-bound — 78 % "No Eligible",
  33 % occupancy (theoretical), IPC 0.86, 89 regs/thread → the
  register file caps the block at 2/SM (16 warps). Each warp stalls
  ~8.5 cyc on L1TEX scoreboards (weight loads) and there simply are
  not enough warps to hide it. The "144 GB/s" the bench printed was
  against the UNIQUE bytes; the real tell is the occupancy.
- **Slot reordering is already live** (the multi-token path sorts
  slots by expert id so a shared expert's 2nd read hits L2). The
  micro-bench had been using identity order — fixing that in the
  bench moved gate_up 1104→752 us and L2 hit 5 %→52 % (the bench
  now mirrors the live path). No live change there; it was the bench
  lying, not the engine.
- **launch_bounds (the real lever):** forcing the gate kernel to 3
  blocks/SM (80 regs) and down to 4 (64 regs) lifts occupancy
  (48–51 %) and each kernel ~3–4 %. Unit-green both modes. Live
  fixed-bench 54.86→55.57 step-chunks/s (+1.3 %), llama-benchy
  tg64 30.15. Small, because the experts are ~53 % of the step and
  the rest (dense GEMVs, fabric, mhc) is untouched.
- **The structural gap remains:** at ~50 % occupancy the slot kernels
  stream at ~200 GB/s against the 580–600 GB/s practical slab-cold —
  3× headroom. The fix is more in-flight weight bytes per SM without
  the register cost of the two-row form (which is occupancy-limited):
  cp.async/TMA staging of the weights into smem so the in-flight data
  lives in the async-copy queue, not registers. That is the next
  kernel project (and it is the same wall the dense GEMVs hit, so it
  is shared with the pp-1800 dense-fp8-GEMM work).

### 2026-09-29 (late night 2) — two-step slot gate_up: bitwise, perf-neutral, reverted

- **The kSteps=2 (two rows/warp) slot gate_up was probed** (pair2 path
  in fp4_gemv.cuh, kernel2 in glm_moe.cu, `DGPP_MOE_SLOT_STEPS2` gate).
  First bug found by the bitwise pins (4 tests red): kernel2's fp8
  shared branch kept the one-step warp stride (`n0 + warp*rps + i`) on
  the doubled grid stride — of a block's 2·rpb rows only 9·rps were
  covered (the first 8·rps written twice, the last 7·rps never written
  → 0x7F poison in the shared expert's act; routed slots fine). Fixed
  with the two-step warp stride (`n0 + 2*warp*rps + i`).
- **Bitwise verified two ways:** all 33 unit tests green in both modes
  (the nvfp4/mxfp4 slot-path + sliced-fold pins), and a buffer-safe
  per-slot XOR hash of (accs + written act) per launch — the CUDA
  printf buffer (~1 MB) drops per-row lines past ~768/launch, so the
  hash replaced them; every slot hashes identical 1-step vs 2-step.
- **Perf: no win anywhere** (bench with per-launch events, 19 unique
  experts × 36 slots): gate_up 1104 vs 1103 us @ H/I=4096/2048, 196 vs
  199 @ 2048/1024, 31 vs 35 @ 1024/512 (worse). The bench also now
  times gate and down separately (the old split assumed equality) and
  takes `DGPP_MOE_SLOT_BENCH_GEO=H,I`.
- **The mechanism is reverted** (no dead code in the tree); the tree
  keeps the mma `sc[]` size fix (`sc[4]` overflowed wide windows —
  `sc[W::kK/F::kQuadSpan]`) and the bench improvements.
- **Lesson:** rows-per-warp is not the expert lever — both slot
  kernels sit at ~144–155 GB/s against 580–600 GB/s practical
  slab-cold (3.7–4× headroom). Next: the memory pattern, with the
  vllm 0731 fast-support reference at ~/dev/sparkrun-ds4/.

### 2026-09-29 (late night) — the memory ground truth: 830 GB/s DRAM, 24 MB L2; scale prefetch landed

- **The part numbers everyone was quoting were wrong.** Plain-stream
  micro-benches (fixed `blockIdx` offset — an earlier version of the
  probe had every block read the whole array): DRAM 512MB–1GB reads =
  **830 GB/s**; L2 = **24 MB** (`cudaDeviceProp`), L2 reads ~1.9 TB/s.
  A slab the size of wq_b (16.8 MB) after a 24 MB+ flush: **580–600
  GB/s**. Consequences: (a) the experts' and dense GEMVs' 155 GB/s is
  3.7× below the practical ceiling — kernel headroom, not a floor;
  (b) the 273 GB/s "spec peak" in the earlier entries was wrong for
  this part; (c) flush-based "cold" benches are treacherous here — a
  256 MB flush leaves a dirty-writeback storm that starves the next
  read to 12–24 GB/s, and 64 MB only half-evicts the 24 MB L2.
- **Benchy is not A/B-grade** (two independent non-determinisms: the
  corpus start position is `np.random.randint` per request — the
  prompt differs every run, no seed flag; and at the site's temp 1.0
  default the MTP acceptance varies the token stream). The fixed
  driver (`/tmp/fixed_bench.py` pattern: fixed 2000-token prompt,
  temp 0, min_tokens 512, ignore_eos, stream) is deterministic to
  ±0.04 % (514 step-chunks / 9.4–9.5 s). All decode A/Bs below use it.
- **mma_gemv ncu (wq_b 16384×1024, m=6, ncu-flushed):** 87.2 us,
  Memory 11.2 %, Compute 29.8 %, warp CPI 13.7 — 35.8 % L1TEX
  scoreboard (the per-use `scale_row` global load inside the
  dequant→mma chain) + 32.5 % CTA barrier.
- **Scale prefetch, v1 (the whole window's 8 scale values into
  registers at window start):** isolated cold 111.3 → 100.8 us; ncu
  87.2 → 82.6 us, the barrier stall gone, CPI 13.7 → 97.5 (L1TEX now
  66 %). But **live it LOST** (54.38 vs 54.79 step-chunks/s): at
  cs=5 the window spans 8 scale blocks while each lane consumes 2 —
  an 8× overfetch of 4-byte loads on an issue-constrained inner loop.
  (Register counts identical to HEAD: ptxas reuses the 8 slots.)
- **Direct-LDG (ring off, `WarpLoads::kOn = false`) A/B:** the
  2-stage cp.async ring looked like the wall in isolation (ncu 82.6
  vs 21.6 us) but **loses live** (54.38 vs 54.79): the ring's second
  in-flight window hides the DRAM round trip better in the graph
  context than a warp's back-to-back LDG.128s. Kept the ring.
- **Landed: the per-group scale prefetch (v2).** Each lane loads only
  the scales its groups consume (`sc[kGroups]`, ≤ 4 guarded 4-byte
  loads at window start, issued before the weight smem reads; the
  ragged past-k value zeroed like the per-use guard). Bitwise the
  old arithmetic (`mma_gemv_test` 10/10, incl. the bitwise
  single-launch contracts). **Live: 54.86 vs 54.79 — best of the
  four, non-overlapping ranges; pp2000 flat (1032.4).** ~0.13 % —
  small, but the stall ncu flagged is gone from the critical path
  and nothing regressed.
- The decode kernel family is NOT "at the floor" (155 of 580+ GB/s).
  Next up: attack the mma_gemv streaming rate itself (window width /
  in-flight bytes / the per-window barrier cadence) toward the 580
  GB/s slab ceiling — the same headroom now applies to the fp4
  expert GEMVs (41.4 ms/step at 155 GB/s).

### 2026-09-29 (night) — decode step mapped to the floor; wo_b ws; expert ceiling proven

- **mhc tiled decode A/B: rejected.** `DGPP_MHC_TILE_DECODE` extends
  the prefill token-tiled dots form to the decode rows (in-block
  finish, comb included; the launcher reports finished and the
  caller's side-stream comb is skipped). Unit-green in the fused
  finish's tolerance class (the tiled dots carry a ~tens-of-fp32-ulps
  dots jitter vs the per-coefficient register form — the prefill
  tiled form's standing contract is the bf16 output's). Live tg256:
  24.4 tiled vs 27.6 fused — the per-coefficient form's 96-block
  parallelism wins at decode row counts; the tiling's L2-traffic win
  only pays at prefill scale. Default stays off; the gate + test stay
  as the evidence.

- **Unique-expert trace shipped** (env-gated, device-side, inside the
  graph decode path): `DGPP_MOE_UNIQUE_TRACE` — one 32-thread kernel
  per MoE layer per step, a 256-expert mask, `atomicOr` + one device
  printf (`[MOE] n=… unique=…`). Readings on the live world:
  n = 30/36 (5–6 MTP rows × top-6; 5031 steps at 6 rows, 402 at 5),
  unique min 8 / p50 21 / p90 27 / max 36, **mean 21.2** vs ~29 for
  random — the MTP rows route to mostly the same experts.
- **Expert DRAM floor proven**: 21.2 × 6.68 MB/rank = 142 MB/layer;
  at the ~155 GB/s practical streaming ceiling (same rate the dense
  GEMV micro-benches hit cold-L2) the floor is ~946 us/layer vs
  897 us measured — **the expert kernels are AT the ceiling**. Bytes
  are routing-set; the expert lever is exhausted. (The 273 GB/s peak
  is unreachable for this pattern; 155 GB/s is the part's practical
  GEMV streaming rate.)
- **wo_b decode missing the GEMM workspace**: every
  `launch_scale_gemm_grid_*` call in csa2 passes
  `gemm_ws_, gemm_ws_bytes_` except wo_b — split-K dead there (64
  blocks on 48 SMs). Fixed. The `DGPP_MMA_TRACE` env (new, in
  mma_gemv) gave the ground-truth per-launch shapes: the 8-warp
  bf16 class (72.3 × 98 us/step) is wq_b (n=16384 k=1024) + wo_b
  (n=k=4096 = 16 MB, now splits=2) + main_proj (n=4096 k=12288) —
  ALL at ~1.6–1.7× the DRAM floor (155–170 GB/s). Earlier "wo_b
  7–13× floor" was a k=1024 misread (its k is lg×o_lora = 4096).
- **SPLITK_FILL A/B (96 → 192): flat** (tg64 29.9 vs 30.0) — the
  dense sites are bandwidth-limited at the streaming ceiling, not
  under-filled. The fill knob stays at its default.
- **Fabric fold = 3.43 ms/step of pure RoCE latency** (37.2 us ×
  92.2 recorded bus nodes: two boundaries/layer, each a
  single-outstanding post→doorbell→fold; poll granularity 32 ns,
  already tight). Pipelining = a bus-protocol change.
- Live numbers: pp2000 1031.9, tg64 ~30 (W4A16 site default + the
  wo_b ws). Parity math: structure ceiling ~73.5 ms/step ≈ 33 t/s;
  35 t/s needs the fold pipeline, a boundary-free TP topology, or a
  higher MTP acceptance (2.2 tok/step now).

### 2026-09-29 — attn_flash double buffer + spark1 clock wedge

- attn_flash is latency-bound: per tile the random latent gather (~16 KB
  of 1024 B rows from the paged pool) is fully exposed — 196.7 us/launch
  / 8 tiles ≈ 24.6 us/tile. ncu (via the pipe, same class): no pipe >35 %,
  IPC ~1.
- Change (parity-first): the bf16 listed path now (a) keeps Q~ in
  registers (the smem copy's columns SD..SQ-1 are never read — cc max is
  6, not 14 — so reading zero there is exact), (b) double-buffers the
  latent tile with cp.async (gather of tile t+1 overlaps tile t's
  S/softmax/PV), (c) shrinks the S-exchange buffer to the listed JT's
  (8 floats/lane stride — a first cut with a 16-float stride clobbered
  the buffer and failed parity 93 %). 78.8 KB smem, one block per SM.
  csa2_layer_test 4/4, dsv41_engine_test 5/5, 126 regs, 0 spills.
- **spark1 clock wedge:** between the pipe bench (1122 t/s) and the flash
  bench, spark1's SM clock got pinned at 507 MHz (0 % boost at 96 % SM
  util, 10 W; app clock 2418). Every kernel ran 2.4–4.6× slow — the
  flash A/B (447 t/s) was contaminated, NOT a flash regression. A
  `nvidia-smi --gpu-reset` did not clear it; rebooted spark1 (~14:02
  local) — the wedge survived the reboot (still 507 MHz under load,
  dmesg clean of XIDs). Workaround in place: forced lock
  `sudo nvidia-smi -lgc 2150,2150` (min=max, no room to wedge) —
  idle clock now ~1600 MHz (was 208/507). Acceptance test pending:
  pipe micro-bench (wq_a m=2048 ≈ 426 us at healthy clocks) + dmon
  under load. Note: /tmp is tmpfs — the reboot wiped the scratch
  bench binaries; a fresh standalone `pipe_bench` (nvcc,
  scale_gemm+mma_gemv+glm_moe) is being rebuilt to verify.
- **Resolved as a spark1 hardware/firmware fault (user is fixing).**
  Evidence: the micro-bench runs 445 us on spark2 (2138 MHz under
  load) vs 1690 us on spark1 (507 MHz under load, same binary). The
  `tune-spark.sh --bench` gate (fp16 8192^3, threshold 50 TFLOPS):
  spark1 **23.8 — FAIL**, spark2 59.0 — PASS. Nothing software-side
  clears it: reboot, `--gpu-reset`, persistence-mode cycle, lock
  re-apply all fail; P0 reported, no active throttling reasons,
  temps 44-46C, no XIDs except my own OOB bench faults (since fixed).
  Bench infra on spark1 was broken independently: `~/.venv` a
  broken symlink (rebuilt: python3.12 + torch 2.14.0+cu130 at
  `~/models/.venv`), `~/benchmark.py` missing its imports (synced
  from the spark repo's scripts/), `~/comfy-ui` removed (6.1G).
  `tune-spark.sh` now locks `$MAX,$MAX` (the 200/$MAX range lock was
  the wedged mode). Until spark1 passes its bench: world stays down,
  no spark1 numbers trusted.
- ncu needs `sudo` (ERR_NVGPUCTRPERM) and `env HOME=/home/conrad` (the
  model cache is under ~conrad); a full serve boot under ncu takes >15 min
  — background it.

### 2026-09-29 — the prefill attention split: 4 → 1 (c511a51)

- attn_finish was ~130 ms (5.7 %) of the cold prefill, DRAM-bound on the
  f32 `c_main` partials ([rows, n_split, lh, 512], 8.4 MB per
  (split, 128-row tile)) — the flash writes it, the finish reads it.
- Prefill used `kPrefillSplit=4`; the flash grid is already
  `ceil(rows*lh/32) × n_split` = 128 blocks at n_split=1, plenty for 48
  SMs, so the 4-way split only bought c_main traffic, not parallelism.
  Decode is n=1 and untouched.
- A/B (benchy pp2000/tg64): split 4 = 1032.1/16.57, split 2 =
  1091.2/15.33, split 1 = 1117.8/~17. Monotonic. csa2_layer_test +
  dsv41_engine_test green (the parity gate is n_split-agnostic).

### 2026-09-29 — the bf16 mma row bound: the 343 ms launch storm dies (e0d3f14)

- Cold-prefill nsys (pipe live): w4a4 20.7 %, attn_flash 14.8 %, pipe
  12.4 %, the bf16 mma storm 15.1 % (21,184 launches of
  mma_gemv<8, bf16, f32>, gridX=8).
- The storm = the indexer select dot (m=131072 n=512 k=128 f32, 20
  calls) at 128-row mma groups (1024 launches/call). Identified via the
  DGPP_GEMM_DISPATCH trace (now prints m/n/k, env-gated) + the sqlite
  grid decode (gridX=8 = n=512) + the timeline (last 15 % = the
  index-source layers, ~1 launch/token/layer).
- First attempt: 256-row groups (16-tile form, kMmaGemvMaxRowsPerLaunch
  128→256) — measured worse: 31.3 vs 16.2 us/launch, 362.6 vs 343.5 ms
  (the form's 64 accumulators + 99 KB smem). Reverted; the dispatch is
  pinned by graph-capture node counts (128-row groups).
- Fix: `set_decode_mma(true, 128)` — wide-m bf16 matmuls fall to the Lt
  tile GEMM (deterministic per row chain; decode rows keep the mma
  chain). Tests: bf16_interface_mma_row_bound (m=128 mma / m=256 Lt,
  kernel names via graph capture), the csa2 selection test (0 flips),
  engine + forward smoke green. benchy: **pp2000 861.5 → 1032.1**
  (ttft 2317 → 1939 ms), tg64 16.57 (unchanged, prefill-dominated).

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

## Decode floor (2026-09-29, from the model config)

- 0731: 43 layers, hidden 4096, 256 routed + 1 shared, top-6, moe_inter
  2048, head_dim 512. Per decode step the experts alone stream 7 experts ×
  43 layers × 3 × 4096 × 2048 fp4 ≈ 4.7 GB ≈ 17 ms at the 273 GB/s DRAM
  ceiling → ~58 t/s theoretical. Measured 17 t/s (59 ms/step) leaves ~42 ms
  of overhead: fabric (bus_bulk on the decode critical path), MLA attention
  over the growing context, and the 43-layer launch/latency chain
  (single-token, latency-bound). Decode attack = fabric + latency, not
  expert kernels (those are the DRAM floor).

## Next steps

1. **Expert fp4 GEMV kernels: the big kernel prize.** 41.4 ms/step
   (moe_slot_gate_up 28.2 + down 13.2) at ~155 GB/s — 3.7× below the
   580–600 GB/s slab ceiling. ncu one chain (standalone driver; ncu
   attach mode needs an executable, not a running serve) for the
   stall pattern, same playbook as mma_gemv (per-use global loads on
   the dequant→mma chain, barrier cadence, in-flight bytes). A 2×
   there is ~+2 t/s.
2. **mma_gemv streaming rate: 155 → 580 GB/s.** The per-group scale
   prefetch killed the critical-path stall (54.86 live, 2026-09-29);
   the isolated cold kernel is still 82.6 us ncu-flushed vs the 29 us
   slab ceiling. Levers: window width (256→512 k, 1 block/SM smem
   cost), stage depth (the 24 KB budget is a 2026-09-14 artifact —
   re-A/B at the true ceiling), barrier cadence.
3. **Fabric fold pipeline:** 37.2 us × 92.2/step = 3.43 ms; the v1
   contract is single-outstanding. Worth ~1.5–2.5 ms/step; a
   bus-protocol project.
4. **prefill (pp 1032 vs 1800):** the dense fp8 projections at m > 128
   are the target (~1.17 s of 2.67 s e2e); a fast fp8 tile GEMM
   (MoE-class 128-row tiles) is the lever — the streaming GEMV form
   is right, its rate is not (same 3.7× gap as decode).
5. When parity holds: `serve_tools_check.sh` + `serve_agentic_streams.py`
   as the final gate; then criterion 1 (verify branch committed + pushed).
   (Branch is committed + pushed as a2fba46; the gate remains open
   until parity.)
