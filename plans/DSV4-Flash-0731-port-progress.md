# DSV4-Flash-0731 port — progress

State and dated log for the [goal](DSV4-Flash-0731-port.md). Newest first.
Update the snapshot when the state changes; append log entries per
meaningful step.

## Snapshot (2026-09-29)

Serving: `18c88f8` live; `c9b4c93` (prefill GEMV-gate bound) built,
unit tests green, redeploy + benchy A/B pending.

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| W4A4-MX prefill **on** | 746.9 | 15.15 | 1800 / 35 |
| W4A4-MX prefill **off** (W4A16) | 707.8 | 30.32 | 1800 / 35 |

- Prefill: W4A4-MX is +5.5% over W4A16, but parity is 2.4× away even on
  W4A16 — the gap is not (only) the experts. **Root cause found**
  (nsys): the dense fp8 GEMMs (CSA2 q/k/v/o + LM head) ran the
  bandwidth GEMV form for ALL m — the grid launcher's decode_mma gate
  had no upper m bound — so a 2000-row prefill was 16 GEMV groups
  re-reading the fp8 weights 16 times (722.8 ms / 42.6 % of the
  prefill burst; tile GEMM at **0 ms**). Fixed in `c9b4c93`.
- Decode: W4A4-MX **halves** tg (15.15 vs 30.32). The MX path is a
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
2. **pp 2.4× below parity on W4A16** (707.8 vs 1800). Root cause: the
   dense fp8 GEMMs took the mma GEMV path at prefill m (grid-launcher
   gate unbounded, `c9b4c93` bounds it to `kScaleGemmMmaMaxRows`).
   Remaining prefill attribution (nsys, W4A4 on, one burst of 1696 ms):
   mma_gemv 722.8 (42.6 %, the bug), w4a4_mx experts 247.6 (14.6 %;
   small M per expert — ~47 rows vs 81920 in the 91.7 TF unit — a
   secondary target, `DGPP_W4A4_BM=64` exists), attn_flash 219.9
   (13.0 %), attn_finish/fabric ~7 %, mhc 4.5 %, router/norms the rest.
   Re-measure after `c9b4c93`.
3. **tg on W4A16 is 87% of parity** (30.32 vs 35). Decode path:
   mma_gemv multi-problem groups are already merged; profile the rest.

## Verified facts (don't re-derive)

- Prefill profile (nsys 2025.3.2, `spark:/tmp/pp2000.nsys-rep`, W4A4 on,
  one pp2000 burst = 1696 ms GPU): mma_gemv 722.8 ms (42.6 %) — the
  dense projections on the GEMV path, `c9b4c93` fixes; w4a4_mx 247.6
  (14.6 %, 93 launches, small-M per expert); attn_flash 219.9 (13.0 %);
  attn_finish + fabric ~7 %; mhc 4.5 %; tile GEMM 0 ms (it was never
  reached at prefill m). Dense FLOPs at 2.7 s → ~5.7 TF effective vs
  50–90 TF a tile GEMM gives.
- Prefill is not chunked (scheduler `prefill_chunk_limit()` default 0):
  a 2000-token prefill is one pass, m=2000.
- `dense_mma` default ON (`DGPP_DSV41_DENSE_GEMV` env disables;
  `dsv41/model.cpp:193`); the routed `launch_scale_gemm` already bounded
  the mma form to `kScaleGemmMmaMaxRows`=256 — the grid launchers were
  the only unbounded callers (csa2 projections, LM head, MTP main
  proj, engram).
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

### 2026-09-29 — prefill root cause: dense GEMMs on the GEMV path (c9b4c93)

- nsys a pp2000 burst (rank 0 under `nsys launch`, rank 1 plain): the
  prefill is dominated by `mma_gemv` (42.6 %), and the tile GEMM never
  runs. The grid launchers' decode_mma gate had no upper m bound, so
  m=2000 → 16 GEMV groups re-reading the fp8 weights (and the LM head
  at 129K vocab × 2000 rows too). TDD: new `scale_gemm_test` cases
  capture the dispatch in a CUDA graph and require the single tile
  kernel — red at 3 nodes (m=300), green after bounding the gate to
  `kScaleGemmMmaMaxRows` in both `launch_scale_gemm_grid_bf16/f32`
  (the hpp doc said "every row count" — describing the bug).
  scale_gemm_test 18/18, csa2_layer_test 4/4, dsv41_engine_test 5/5,
  dsv41_forward smoke OK.

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

1. Redeploy `c9b4c93` (`up --replace`), benchy A/B: expect pp2000 to
   jump (GEMV 42.6 % → tile GEMM); tg64 must stay ≥ 30. Then re-nsys if
   the gap remains (experts small-M, attn_flash 13 %, fabric 7 %).
2. Kill the decode regression: gate the MX quantize cost out of decode
   (or make the fallback free); verify with benchy tg64 ≥ 35 with
   W4A4 on.
3. Decode profile for the remaining ~5 t/s on tg (W4A16).
4. When parity holds: `serve_tools_check.sh` + `serve_agentic_streams.py`
   as the final gate
