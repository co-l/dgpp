# SSE keep-alive comments for issue #49

## Change

Streaming requests previously sent headers and an initial completion chunk,
then no bytes until model output was available. Long admission waits, prefill
and pauses between visible output chunks could exceed a proxy or client's
network idle timeout.

The HTTP idle pass now sends `: keep-alive\n\n` inside a normal HTTP chunk
after the configured period of silence. It runs independently of the engine
thread. One monotonic clock belongs to each connection, so requests with
multiple choices share a timer. Queued events and successful socket writes
refresh the clock; pending output suppresses additional comments. Comments
stop on disconnect or stream termination, and do not affect generated text,
usage, finish events, scheduler state or watchdog progress.

The default interval is 30 seconds. Cluster JSON `http.sse_ping_interval`,
CLI `--sse-ping-interval`, and request `sse_ping_interval` override it in that
order. All accept integer seconds from 1 through 2147483647, or `-1` to disable.
An explicit request override requires `stream: true`. Both the native config
parser and Python deployment resolver validate the field. Rank 0 logs the
effective server setting. The CLI help, README configuration table, operations
guide and API compatibility guide document the contract.

Optional `return_progress` / `prompt_progress` events are outside this change.

## Validation

The test service uses a fake engine with real HTTP sockets. New cases cover
waiting admission, blocked synchronous prefill, gaps between tokens, active
output, both completion endpoints, multiple choices, server defaults and
request overrides, disconnect, stream termination, subsequent non-streaming
requests, unchanged usage/finish events, and invalid request values. HTTP
tests check exact SSE and HTTP chunk framing and reject multiline comments.
Config and startup tests cover default/disabled/override settings, invalid
types/ranges, launcher resolution, CLI help and JSON/CLI precedence.

The Python suite initially exposed an existing stale resolved-config fixture:
the deployment template now sets `default_max_tokens` to 32768, but the fixture
still contained 256. The fixture was updated to match the existing template;
the server's default token budget is unchanged by this change.

Completed checks:

- CI preset build with warnings as errors: HTTP/service libraries, unit,
  HTTP, service, scheduler, fabric-host, roster and server targets; the server
  binary linked successfully.
- `DGPP_TEST_FILTER=cluster_config build-ci/unit_tests`: 13 cases passed.
- `build-ci/http_server_test`: 18 cases passed.
- `DGPP_TEST_FILTER=serve_ssePing build-ci/serve_test`: all 5 new cases passed.
- `ctest --test-dir build-ci -R
  '^(unit_tests|serve_test|fabric_serve_test|scheduler_test|roster_selftest)$'
  --output-on-failure`: all 5 suites passed (267 unit, 82 service, 3 fabric-host
  and 66 scheduler cases, plus the roster self-test).
- `python3 -m unittest discover -s tests/python -p '*test.py'`: 240 tests,
  2 skipped (zsh is unavailable; startup requires a built binary). Startup
  was then run separately against the freshly built server.
- AddressSanitizer build of `http_server_test` and `serve_test`:
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 build-asan/http_server_test`
  and the same options with `DGPP_TEST_FILTER=serve_ssePing
  build-asan/serve_test`: 18 HTTP and 5 keep-alive cases passed, no sanitizer
  findings.
- Changed C++ lines formatted with clang-format 18.1.8; `git diff --check`
  passed.

- `ctest --test-dir build-ci -R '^serve_startup_test$' --output-on-failure`:
  all 10 startup tests passed, including JSON/CLI precedence and validation.
- Built `qwen_yarn_fixture_test`, `qwen_forward_test`,
  `shutdown_watchdog_test` and `engine_watchdog_test`, then ran their six host
  CTest entries (three W4A4 plan modes, YaRN fixture, shutdown watchdog and
  engine watchdog): all passed.

GPU/RDMA execution and a live model/proxy reproduction are excluded: an active
model server occupies the test machine. This change does not modify model
execution or the inter-rank protocol. The reporter is asked to test their
long-prefill client/proxy path after the merge.
