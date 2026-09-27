# Lazy SSE preambles for PR #57

The original PR delayed chat and legacy completion preambles until visible
output. This prevents a role-only event sent before prefill from distorting
clients' first-completion-event timing. Review reproduced a regression:
immediate EOS and an initial stop-string match omitted the assistant role
entirely. A legacy completion consisting only of an incomplete UTF-8 character
also skipped its preamble before the final replacement character.

The follow-up moves preamble emission into the common completion-event writer.
Each choice emits its preamble exactly once, immediately before its first
output or terminal chunk. This covers content, reasoning, tools, logprobs and
UTF-8 tails. Headers and keep-alive comments can still arrive during admission
or prefill; error-only streams retain their error and `[DONE]` sequence.

The branch includes master through `346ffaf`, including SSE keep-alives and
strict UTF-8 validation. Only rank 0's response formatting changes; model
execution and the inter-rank protocol are unaffected.

## Validation

- Built `serve_test`, `http_server_test` and `fabric_serve_test` with
  `CMAKE_BUILD_TYPE=Release` and `DGPP_WERROR=ON`.
- Three new `serve_streamPreamble` cases pass. They exercise immediate EOS,
  initial stop matches, two chat choices, both completion routes, truncated
  UTF-8 output, and preamble ordering after a blocked prefill.
- Linking the same tests against the pre-follow-up service reproduces two
  failures: empty completions and the legacy UTF-8 tail. The delay case passes
  before and after the follow-up.
- All five `serve_ssePing` cases pass, including a strengthened check that a
  queued request receives keep-alive comments without completion events.
- `ctest --test-dir build-review -R
  '^(serve_test|http_server_test|fabric_serve_test)$' --output-on-failure`
  passes all three suites: 88 serving, 18 HTTP and 3 fabric-host cases.
- Changed C++ lines were formatted with clang-format 19.1.7;
  `git diff --check` passes.

These are host tests with a fake engine and real HTTP sockets. Live-model
prefill throughput and the contributor's benchmark figures were not rerun.
