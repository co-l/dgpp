# DeepSeek-V4-Flash-0731 on dgpp — goal

Serve `deepseek-ai/DeepSeek-V4-Flash-0731` on the 2× DGX Spark GB10
cluster (world 2, RoCE `rocep1s0f1`/`roceP2p1s0f1`) and reach parity with
the vllm reference implementation (the 0731 fast-support reference lives
in `~/dev/sparkrun-ds4/`).

## Parity (the measurable target)

Bench: `llama-benchy`, concurrency 1, latency mode api.

| test | target | note |
|------|--------|------|
| pp2000 | ≥ 1800 t/s | prefill 2000 tokens |
| tg64   | ≥ 35 t/s   | decode 64 tokens |

The command (record it here, it gets lost):

```bash
uvx llama-benchy --base-url http://192.168.1.223:8000/v1 \
  --pp 2000 --tg 64 --c 1 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731
```

## Correctness gates

A change lands only when, in order:

1. Host unit tests pass (spark1, `build-test-ibv`):
   `moe_w4a4_test` (layout vs fp64 + constant-data exact dots),
   `tool_parser_test`, the quantizer reference checks.
2. Serve smoke: `/v1/models` + a chat completion (low
   `reasoning_effort`) answers correctly.
3. `scripts/serve_tools_check.sh` (tools/reasoning gate).
4. `scripts/serve_agentic_streams.py` (agentic loops) when a full run is
   feasible.
5. benchy coherence test passes (it runs before the measurement).

## Scope

- Model wiring for 0731 (CSA attention, mHC, the 0731 head collapse,
  DSA/Engram tables) — done, serving.
- MoE experts in W4A4 (NVFP4 and the MXFP4 2X/ue8m0 twin for the 0731
  checkpoint) — implemented, unit-green; perf work in progress.
- Prefill and decode performance to the parity table above.

## Process — where truth lives

| location | role | rule |
|----------|------|------|
| OpenFox workspace `dsv4-flash-0731-port` | working ground truth, branch `dsv4-flash-0731-port` | edit here |
| `~/dev/dgpp` (session cwd clone) | mirror of the workspace | sync from the workspace, never diverge |
| `spark:~/dgpp` | live instance (rank 0 head + HTTP) | **never edit directly** — rsync / `git pull` only |

Git: commit in the workspace, push `fork` (github.com/co-l/dgpp) and
`nicefox` (git@nicefox.net:dgpp) on the branch `dsv4-flash-0731-port`.
`spark:~/dgpp` tracks the fork and pulls.

## Build / deploy / bench (the ritual)

```bash
# 1. tune (from conrad-mini, ~/dev/spark): drops page caches, pins clocks
./tune-spark.sh

# 2. build on spark1
ssh spark 'cd ~/dgpp && cmake --preset release && cmake --build --preset release -j 20'
ssh spark 'cd ~/dgpp && ./build-release/dgpp-serve --version'   # must show the new git hash

# 3. pre-check without booting the world (optional)
ssh spark 'cd ~/dgpp && python3 scripts/dgpp-cluster up --config deploy/cluster_deepseek-v4-flash-0731_mxfp4-fp8_w2.json --knobs=--memory-plan'

# 4. deploy (replace the running instance)
ssh spark 'cd ~/dgpp && python3 scripts/dgpp-cluster up --replace --config deploy/cluster_deepseek-v4-flash-0731_mxfp4-fp8_w2.json'

# 5. smoke + bench (conrad-mini)
curl -s http://192.168.1.223:8000/v1/models
# the benchy command from the parity section

# stop
ssh spark 'cd ~/dgpp && python3 scripts/dgpp-cluster down --config deploy/cluster_deepseek-v4-flash-0731_mxfp4-fp8_w2.json'
```

- Active config: `deploy/cluster_deepseek-v4-flash-0731_mxfp4-fp8_w2.json`
  (world 2, YaRN 512K, `model_alias: deepseek-v4-flash-0731`).
- API: `http://192.168.1.223:8000/v1` (admin IP bind, not 0.0.0.0).
- Logs: `spark:~/dgpp/log/deployments/<hash>/serve_r0.log`.
- Any `DGPP_*` env var set for the launcher reaches every rank
  (e.g. `DGPP_MOE_W4A4=0` disables the MX prefill path).
- Boot ~20 s warm (resident cache), ~4.6 min cold. NVMe low-power wobble
  shows as `allreduce STALLED` (≤ 8 s) during cold weight load — expected.

## State

Current state, open issues and the dated progress log:
[DSV4-Flash-0731-port-progress.md](DSV4-Flash-0731-port-progress.md).
