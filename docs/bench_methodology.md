# Bench methodology

The serving goal for the DeepSeek-V4-Flash-0731 two-Spark deployment is parity with the vLLM production baseline: **prefill ≥ 2000 tok/s** and **decode ≥ 40 tok/s** at the ground-truth point below. These pages define the measurement used to claim parity and the protocol for keeping the engine trustworthy under the benchmark.

## Ground truth

The reference command runs the [llama-benchy](https://pypi.org/project/llama-benchy/) harness (v0.4.0, via `uvx`) against the live engine:

```bash
uvx llama-benchy --base-url http://192.168.1.223:8000/v1 \
  --pp 2000 --tg 64 --c 1 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731
```

- `--pp 2000` — each prompt is sliced from a cached book corpus to exactly 2000 tokens (the tokenizer is loaded from the HF model `deepseek-ai/DeepSeek-V4-Flash-0731` and must match the checkpoint's tokenizer).
- `--tg 64` — 64 output tokens per prompt.
- `--c 1` — single stream, one request in flight.
- `--model` names the tokenizer; `--served-model-name` is the `model` field sent in API calls (the engine serves `deepseek-v4-flash-0731`).
- Defaults apply: 3 runs per test, a warmup request first, an `api`-mode latency baseline (a `GET /models` round trip subtracted from every time-to-first-token), and a coherence check on the generated text.

A **pass** is defined as: warmup plus all three runs complete, the engine is still alive afterwards, and the `pp2000` and `tg64` rows exist with usable values. Any `graph walk STALLED` line in `serve_r0.log` followed by `serve: ENGINE FAILURE … exiting with status 2` is a crash, not a slow run.

## Reading the output

With concurrency 1 the table columns are `model | test | t/s (peak) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms)`.

- The `pp2000` row measures prompt processing. `est_ppt` is the pure prefill duration: end-to-end time-to-first-token minus the measured API baseline. Prefill rate = `2000 / (est_ppt / 1000)` tok/s. Parity requires `est_ppt ≤ 1000 ms`.
- The `tg64` row measures generation. The `t/s` column is peak generation throughput; at `c 1` it is the decode rate. Parity requires ≥ 40 tok/s.
- `ttfr` and `e2e_ttft` expose the split: a high `e2e_ttft` with low `est_ppt` is network/client overhead, not engine prefill.

The harness also emits `--save-result` JSON/CSV (and `--emit-progress` JSONL) with per-run values and standard deviations; use the CSV columns `est_ppt_mean` and `t_s_mean`/`peak_ts_mean` when a table row is ambiguous.

## Protocol

1. Tune the cluster first: `bash ~/dev/spark/tune-spark.sh` from conrad-mini (drops page caches, pins GPU clocks 200/2150 on both ranks).
2. Launch a fresh engine (`dgpp-cluster up --config deploy/cluster_deepseek-v4-flash-0731_mxfp4-fp8_w2.json`); a warm resident cache (`~/.cache/dgpp/resident`) makes the boot ~30 s. Confirm `GET /v1/models` before timing.
3. One benchmark at a time on the cluster; no profiler attached. The bench runs from conrad-mini over the LAN (the HTTP bind is the admin IP `192.168.1.223`, not 0.0.0.0).
4. Record the source revision (commit of the binary) next to every campaign, as `docs/benchmarks.md` does.
5. A wedge mid-run is investigated before any rate is reported: the engine exits non-zero after answering the in-flight request with `engine_failure`, so a partial result in the harness output is a failure, not a number.

## When the engine wedges

The serve watchdog reports `graph walk STALLED gen=… posted=… cell[N]@… windows=… adopted=…` every 500 ms once the replay's verdict stops publishing, then fails the engine after the pick timeout (60 s, `pick_timeout_ms` in `graph_engine.hpp`). The line's fields are the diagnostic surface:

- `gen` — global cell-generation counter of the decode walk; `windows`/`adopted` — DSpark speculative passes seen so far.
- `cell[N]` with `posted=0x0` — the host never posted work for cell N; cells before it show `r=1` (ready) / `e=1` (ended) when their kernels finished and stalled the walk downstream.
- `p1l0:d/a/s!` — the DSpark pass/layer draft/accept/select counters; the `!` flag marks the faulty node. A `509/509` form means the node's event count completed; a stuck node stays `0/0!`.

Reproduce deterministically with a plain chat completion at the failing prompt length (the 2026-09-28 wedge needed ~3.4k prompt tokens before the first decode step; short prompts never triggered it), then read `~/dgpp/log/deployments/<hash>/serve_r0.log` on spark1 for the scheduler lines around admission (`prefix cache miss`, `admitted to slot … reserve N blocks`) and the first STALLED dump.

## Regression gates

After any change to the engine that this methodology measures: unit tests on spark1 (host-only targets with `cmake -B build-test -DDGPP_ENABLE_IBV=OFF`), `scripts/serve_tools_check.sh` and `scripts/serve_agentic_streams.py` on the new binary, then this ground-truth command. The gates pass only when the engine survives them.
