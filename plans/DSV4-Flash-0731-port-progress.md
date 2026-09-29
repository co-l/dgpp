# DSV4-Flash-0731 port — progress

State and dated log for the [goal](DSV4-Flash-0731-port.md). Newest first.
Update the snapshot when the state changes; append log entries per
meaningful step.

## Snapshot (2026-09-29)

Serving: yes — `18c88f8` deployed, smoke + tools OK, benchy coherence
passes.

| path | pp2000 t/s | tg64 t/s | parity |
|------|-----------:|---------:|--------|
| W4A4-MX prefill **on** | 746.9 | 15.15 | 1800 / 35 |
| W4A4-MX prefill **off** (W4A16) | 707.8 | 30.32 | 1800 / 35 |

- Prefill: W4A4-MX is +5.5% over W4A16, but parity is 2.4× away even on
  W4A16 — the gap is not (only) the experts.
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
2. **pp 2.4× below parity on W4A16** (707.8 vs 1800). Needs a full
   prefill breakdown (nsys on serve_r0 for a pp2000 run): attention (CSA),
   KV, routing/gather, experts, epilogues. MoE GEMM alone is 2.01×
   faster in W4A4 (unit: 91.7 vs 42.9 TF), so the rest of the prefill is
   the bulk of the gap.
3. **tg on W4A16 is 87% of parity** (30.32 vs 35). Decode path:
   mma_gemv multi-problem groups are already merged; profile the rest.

## Verified facts (don't re-derive)

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

## Log

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

1. Kill the decode regression: gate the MX quantize cost out of decode
   (or make the fallback free); verify with benchy tg64 ≥ 35 with
   W4A4 on.
2. nsys a pp2000 run (W4A4 on): attribute the prefill time; chase the
   non-MoE share of the 1800 t/s gap.
3. Decode profile for the remaining ~5 t/s on tg.
4. When parity holds: `serve_tools_check.sh` + `serve_agentic_streams.py`
   as the final gate, then Good Night.
