# DSV4-Flash-0731 port — progress

State and dated log for the [goal](DSV4-Flash-0731-port.md). Newest first.
Update the snapshot when the state changes; append log entries per
meaningful step.

## Snapshot (2026-09-29)

Serving: `c511a51` live (the prefill pipe, shape-routed over the
streaming GEMV, the bf16 mma row bound: wide-m bf16 matmuls take the
Lt tile GEMM, and the prefill attention split = 1: the flash's grid
fills the SMs at 128 blocks, so the 4-way split only bought c_main f32
traffic — 8.4 MB per (split, tile) written by the flash, read by the
finish — which the 1-split form drops 4x).

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
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

1. **Done (2026-09-29 evening):** spark1 fixed (user; 91.4 TFLOPS on the
   bench — the 59.4 vs 91.4 gap vs spark2 is the torch 2.14 vs 2.9.1
   bench-side delta, a bonus). Flash double-buffer A/B re-run:
   pp2000 1127.7 / 1131.2 t/s (vs 1122.3 baseline, +0.7 %), tg64
   16.0 / 16.2 (neutral) — KEPT. Clocks verified 2125-2132 MHz under
   load at 84 W (was 507 MHz at 10 W).
2. attn_flash (bigger lever): the double buffer only hides the gather
   latency; the b12x CuTe/TMA port
   (`~/dev/sparkrun-ds4/b12x/attention/`) is the real fix — scope as a
   project.
3. w4a4 experts ~466 ms (26 %): at the DRAM ceiling (~273 GB/s, the GB10
   peak) — no kernel-level win; the lever is traffic (weight reuse / L2).
4. Fabric: bus_bulk_collective 173.8 ms (9.7 %).
5. When parity holds: `serve_tools_check.sh` + `serve_agentic_streams.py`
   as the final gate; then criterion 1 (verify branch committed + pushed).
