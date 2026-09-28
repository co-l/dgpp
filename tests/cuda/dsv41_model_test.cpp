// The DeepSeek-V4.1 model on the loader fixture at world 1 (plan G5's
// first gates): the cold forward runs every layer (the residual streams,
// the index sources' selections captured); a session prefill's last row
// is bitwise the forward's; decode steps and a verify batch with a
// rollback match a re-forward over the same tokens within the decode
// budget (the GEMV chain against the tile kernels).
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "kda_test_helpers.hpp"
#include "models/dsv41/loader.hpp"
#include "models/dsv41/model.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  dgpp::Dsv41TextConfig cfg;
  std::string dir;
};
Fixture make_fixture() {
  Fixture fx;
  fx.cfg = dsv41fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv41_model_fixture").string();
  dsv41fx::write_fixture(fx.cfg, fx.dir);
  (void)dsv41fx::fixture_sidecar(fx.cfg, fx.dir);
  return fx;
}
std::vector<int64_t> tokens_of(uint64_t seed, int n, int vocab) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed | 1;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(vocab));
  }
  return t;
}
// Relative l2 between two logits rows and the argmax agreement.
double rel_l2(const float* a, const float* b, size_t n) {
  double d = 0, m = 0;
  for (size_t i = 0; i < n; ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); m += double(b[i]) * b[i]; }
  return std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
}
int argmax(const float* a, size_t n) {
  int best = 0;
  for (size_t i = 1; i < n; ++i) if (a[i] > a[size_t(best)]) best = int(i);
  return best;
}
}  // namespace

// The group prefill (session_prefill_group): three cold prompts as the
// spans of one walk against each prefilled alone — the last rows' logits
// bitwise (the dense sites and the MoE tile kernel are row-invariant, the
// attention and the publication run per span, the Engram takes the
// spans), and the first decode step of each request bitwise the alone
// model's (the draft state, positions and rings per request). The exact
// prefill mode: the span limit is the walk's rows.
DGPP_TEST(dsv41_model_group_prefill_is_bitwise_the_prefills_alone) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xA1, 23, V), tokens_of(0xA2, 17, V), tokens_of(0xA3, 11, V)};
  std::vector<std::vector<float>> alone_logits, alone_step;
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    for (int r = 0; r < 3; ++r) {
      const auto o = m.session_prefill(r, prompts[static_cast<size_t>(r)]);
      alone_logits.push_back(o.logits);
      const int32_t first = argmax(o.logits.data(), size_t(V));
      const auto st = m.session_step(r, first);
      alone_step.push_back(st.logits);
    }
  }
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    require(m.prefill_group_span_limit() >= 23, "the exact prefill's span limit is the walk's rows");
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const auto outs = m.session_prefill_group({2, 0, 1}, {pp[2], pp[0], pp[1]});  // a permuted slot order
    require(outs.size() == 3, "one output per request");
    const int order[3] = {2, 0, 1};
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      require(outs[static_cast<size_t>(i)].logits.size() == size_t(V), "a last-row logits vector per request");
      require(outs[static_cast<size_t>(i)].logits == alone_logits[static_cast<size_t>(r)],
              "group prefill logits bitwise the prefill alone (request " + std::to_string(r) + ")");
    }
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      const int32_t first = argmax(alone_logits[static_cast<size_t>(r)].data(), size_t(V));
      const auto st = m.session_step(r, first);
      require(st.logits == alone_step[static_cast<size_t>(r)],
              "the first step after a group prefill bitwise the alone model's (request " + std::to_string(r) + ")");
    }
    std::printf("[ OK ] group prefill of 23 + 17 + 11 rows: last rows and first steps bitwise the prefills alone\n");
  }
}

// The same under the bounded prefill with spans wider than the window
// (16 in the fixture): each span's decoder segment is its last window
// rows, packed; the last rows' logits and the first steps bitwise the
// bounded prefills alone.
DGPP_TEST(dsv41_model_group_prefill_bounded_wide_spans_is_bitwise_the_prefills_alone) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  require(fx.cfg.sliding_window < 23, "the fixture's window is narrower than the widest prompt");
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xB1, 23, V), tokens_of(0xB2, 9, V), tokens_of(0xB3, 40, V)};
  std::vector<std::vector<float>> alone_logits, alone_step;
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    m.set_prefill_bounded(true);
    for (int r = 0; r < 3; ++r) {
      const auto o = m.session_prefill(r, prompts[static_cast<size_t>(r)]);
      alone_logits.push_back(o.logits);
      const int32_t first = argmax(o.logits.data(), size_t(V));
      alone_step.push_back(m.session_step(r, first).logits);
    }
  }
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    m.set_prefill_bounded(true);
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const auto outs = m.session_prefill_group({0, 1, 2}, pp);
    for (int r = 0; r < 3; ++r)
      require(outs[static_cast<size_t>(r)].logits == alone_logits[static_cast<size_t>(r)],
              "bounded group prefill logits bitwise the prefill alone (request " + std::to_string(r) + ")");
    for (int r = 0; r < 3; ++r) {
      const int32_t first = argmax(alone_logits[static_cast<size_t>(r)].data(), size_t(V));
      require(m.session_step(r, first).logits == alone_step[static_cast<size_t>(r)],
              "the first step after a bounded group prefill bitwise the alone model's (request " + std::to_string(r) + ")");
    }
    std::printf("[ OK ] bounded group prefill of 23 + 9 + 40 rows (window %d): bitwise the prefills alone\n", fx.cfg.sliding_window);
  }
}

DGPP_TEST(dsv41_model_forward_prefill_decode_and_rollback_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  dgpp::Dsv41Model model(fx.cfg, fx.dir, /*max_tokens=*/96, /*max_cache_tokens=*/512, dgpp::Dsv41Residency::Streaming,
                         nullptr, 0, 1, /*max_requests=*/2, /*mtp=*/false, /*decode_rows=*/8);
  // 1) the cold forward over 70 tokens (past the 16-entry select horizon at
  // ratio 2 and the candidate pool at ratio 1).
  const auto prompt = tokens_of(0xD5, 70, V);
  auto fwd = model.forward(prompt, /*capture_layers=*/true);
  const std::vector<dgpp::Dsv41Model::IndexLogits> fwd_index = model.debug_index_logits();  // before the next walk clears it
  require(fwd.logits.size() == size_t(70) * V, "forward logits rows");
  for (const float v : fwd.logits) require(std::isfinite(v), "forward logits finite");
  require(fwd.layer_states.size() == 8, "eight layer states");
  require(fwd.dsa_selections.size() == 3, "three index sources' selections (layers 2, 4, 6)");
  for (const auto& st : fwd.layer_states)
    for (const uint16_t b : st) require(std::isfinite(bf16_bits_to_float(b)), "stream finite");
  // Different rows produce different logits (the attention and the caches
  // actually feed the head).
  require(rel_l2(fwd.logits.data(), fwd.logits.data() + size_t(69) * V, size_t(V)) > 1e-3, "rows differ");

  // 2) prefill == forward bitwise on the last row.
  auto pre = model.session_prefill(0, prompt);
  require(pre.logits.size() == size_t(V), "prefill returns the last row");
  for (int i = 0; i < V; ++i)
    require(pre.logits[size_t(i)] == fwd.logits[size_t(69) * V + i], "prefill's last row bitwise the forward's");

  // The decode path against the prefill path row by row: a 69-token
  // prefill, then the prompt's own last token as a decode step, its layer
  // rows against the forward's row 69. The window-only layer and the
  // Engram layer are bitwise (the same kernels on both paths); the
  // compressed layers differ by the GEMV chain vs the tile kernels only.
  {
    dgpp::Dsv41Model m2(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const std::vector<int64_t> p69(prompt.begin(), prompt.begin() + 69);
    (void)m2.session_prefill(0, p69);
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
    const auto st = m2.session_step(0, prompt[69]);
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    const std::vector<dgpp::Dsv41Model::IndexLogits> dec_index = m2.debug_index_logits();
    const int H4 = 4 * fx.cfg.hidden_size;
    require(st.layer_states.size() == 8, "the decode walk captured every layer");
    // The index sources whose coded query (e4m3 codes + row scales) differs
    // between the decode row and the forward's row 69: from the first such
    // source's layer on, the two paths select from different logits (the
    // coding turns sub-ulp noise into code steps of 1/8 or more), so a
    // selection flip and the layer budget's excess are the coding's, not a
    // decode-path defect — the layers before it must stay within budget.
    require(fwd_index.size() == 3 && dec_index.size() == 3 && st.dsa_selections.size() == 3, "index captures on both walks");
    int first_coded_layer = 99;
    for (size_t s = 0; s < 3; ++s) {
      const auto& a = fwd_index[s]; const auto& b = dec_index[s];
      require(a.q_rows == 70 && b.q_rows == 1 && a.q_heads == b.q_heads && a.q_heads > 0, "coded index queries captured");
      const size_t w = size_t(a.q_heads) * 128;
      int diffs = 0;
      for (size_t i = 0; i < w; ++i) diffs += a.q_codes[size_t(69) * w + i] != b.q_codes[i];
      for (int h = 0; h < a.q_heads; ++h) diffs += a.q_scales[size_t(69) * a.q_heads + h] != b.q_scales[size_t(h)];
      const bool flipped = std::memcmp(st.dsa_selections[s].data(), fwd.dsa_selections[s].data() + size_t(69) * fx.cfg.index_topk,
                                       size_t(fx.cfg.index_topk) * 4) != 0;
      std::printf("[INFO] index source %zu (layer %d): coded query differs in %d of %zu codes/scales, selection %s\n", s,
                  fx.cfg.index_source_layer_ids[s], diffs, w + size_t(a.q_heads), flipped ? "FLIPPED" : "equal");
      if (flipped) require(diffs > 0, "a selection flip between the decode and the prefill row without a coded-query difference");
      if (diffs > 0 && flipped) first_coded_layer = std::min(first_coded_layer, fx.cfg.index_source_layer_ids[s]);
    }
    for (size_t l = 0; l < st.layer_states.size(); ++l) {
      double d = 0, m = 0;
      for (int i = 0; i < H4; ++i) {
        const double a = bf16_bits_to_float(st.layer_states[l][size_t(i)]);
        const double b = bf16_bits_to_float(fwd.layer_states[l][size_t(69) * H4 + i]);
        d += (a - b) * (a - b); m += b * b;
      }
      const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      std::printf("[INFO] decode vs prefill, layer %zu: rel l2 %.3e\n", l, l2);
      // Layers 0-1 are window-only (no compressed source), so their decode
      // output used to be BITWISE the forward's. Since 2026-09-14 the decode
      // window attention splits its 128 keys across the SMs (csa2_layer:
      // kWinDecodeSplit) — a single unsplit block was latency-bound and left
      // the GPU idle. The split changes the online-softmax combine's fp32
      // order, so the bf16-rounded attention output moves by about one ULP
      // (measured ~3.8e-3 relative). It is no longer bitwise, but a
      // decode-path bug would be orders of magnitude larger; the tight budget
      // still catches that. The decode-vs-re-forward audit (dsv41_decode_test,
      // certified selection flips) is the behavioural gate.
      if (l < 2) require(l2 < 8e-3, "layer " + std::to_string(l) + " (window-only) within the window-split rounding");
      else if (static_cast<int>(l) < first_coded_layer) require(l2 < 0.01, "layer " + std::to_string(l) + " within the decode budget");
      else require(l2 < 0.12, "layer " + std::to_string(l) + " past a coded selection flip within a flip's move");
    }
    // The logits: within the budget before any coded flip; past one, the
    // argmax (the generated token) is the assertion, as in the step loop below.
    if (first_coded_layer > 7)
      require(rel_l2(st.logits.data(), fwd.logits.data() + size_t(69) * V, size_t(V)) < 0.01, "decode logits within the budget");
    else
      require(argmax(st.logits.data(), size_t(V)) == argmax(fwd.logits.data() + size_t(69) * V, size_t(V)),
              "decode argmax matches the prefill's past a coded selection flip");
    // The sublayer captures of layer 0: every stage bitwise (the MoE decode
    // slot chain on the 32 x 32 shared expert grid included — the bug of
    // 2026-09-13 lived there).
    {
      dgpp::Dsv41Model m4(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
      (void)m4.forward(prompt, true);
      const auto& pf = m4.debug_sites();
      const auto& dec = m2.debug_sites();
      require(!pf.empty() && !dec.empty(), "site captures");
      const int H = fx.cfg.hidden_size;
      auto same = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t row_elems) {
        for (size_t i = 0; i < row_elems; ++i)
          if (a[i] != b[size_t(69) * row_elems + i]) return false;
        return true;
      };
      // The attention INPUT is before the window attention, so it stays
      // bitwise. From the attention output on, the decode window split (above)
      // moves the bf16 result by ~1 ULP, which propagates into the MoE input
      // and output; those three hold a tight rounding budget instead of
      // bitwise (a real slot-chain regression — the 2026-09-13 bug — is orders
      // of magnitude larger and still trips it).
      auto close = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t row_elems) {
        double d = 0, m = 0;
        for (size_t i = 0; i < row_elems; ++i) {
          const double x = bf16_bits_to_float(a[i]);
          const double y = bf16_bits_to_float(b[size_t(69) * row_elems + i]);
          d += (x - y) * (x - y);
          m += y * y;
        }
        return std::sqrt(d) / std::max(std::sqrt(m), 1e-30) < 8e-3;
      };
      require(same(dec[0].x_attn, pf[0].x_attn, size_t(H)), "layer 0 attention input bitwise");
      require(close(dec[0].attn_out, pf[0].attn_out, size_t(H)), "layer 0 attention output within the window-split rounding");
      require(close(dec[0].x_ffn, pf[0].x_ffn, size_t(H)), "layer 0 MoE input within the window-split rounding");
      require(close(dec[0].ffn_out, pf[0].ffn_out, size_t(H)), "layer 0 MoE output within the window-split rounding (the decode slot chain)");
    }
    m2.session_close(0);
  }
  // 3) decode steps: each step's logits against a re-forward over the
  // prompt plus the fed tokens (the GEMV chain vs the tile kernels).
  std::vector<int64_t> history = prompt;
  const auto feed = tokens_of(0xD6, 6, V);
  for (int k = 0; k < 3; ++k) {
    const auto step = model.session_step(0, feed[size_t(k)]);
    history.push_back(feed[size_t(k)]);
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const auto again = ref.forward(history, false);
    const float* want = again.logits.data() + (history.size() - 1) * V;
    const double l2 = rel_l2(step.logits.data(), want, size_t(V));
    std::printf("[INFO] decode step %d: rel l2 %.2e vs the re-forward, argmax %d / %d\n", k, l2,
                argmax(step.logits.data(), size_t(V)), argmax(want, size_t(V)));
    // The generated token is the assertion that matters: decode's argmax
    // equals the re-forward's. The logit-vector l2 rides the decode window
    // split's rounding (above): when it tips a near-tie selection the row's
    // logits move like a certified flip (up to ~0.1, argmax unchanged), the
    // regime dsv41_decode_test audits with its flip certification. A real
    // decode-path defect breaks the argmax or blows well past this budget.
    require(argmax(step.logits.data(), size_t(V)) == argmax(want, size_t(V)),
            "decode step " + std::to_string(k) + " argmax matches the re-forward");
    require(l2 < 0.12, "decode step " + std::to_string(k) + " within the decode budget");
  }
  // 4) a verify batch of six rows (the DSpark shape: past the GEMV's
  // four-row chunk), one accepted, then the next batch.
  const auto more = tokens_of(0xD9, 3, V);
  const std::vector<int64_t> drafts = {feed[3], feed[4], feed[5], more[0], more[1], more[2]};
  setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
  const auto ver = model.session_verify(0, drafts);
  unsetenv("DGPP_DSV41_CAPTURE_DECODE");
  const std::vector<dgpp::Dsv41Model::IndexLogits> ver_index = model.debug_index_logits();
  require(ver.logits.size() == size_t(6) * V, "verify returns every row");
  // A decode-path row against the re-forward's: the argmax is the
  // assertion; the l2 rides the near-tie rounding (dsv41_decode_test audits
  // the flips) within a wider budget — unless the row's coded index query
  // differs from the re-forward's at a source whose selection differs
  // (the coding's discontinuity: the row then reads a different attention
  // set, and its logits move like a flipped row's, ~0.35 seen). A defect
  // breaks the argmax, or the budget without that evidence.
  const auto check_row = [&](const char* what, int r, const float* got, const float* want,
                             const std::vector<dgpp::Dsv41Model::IndexLogits>& got_index,
                             const std::vector<std::vector<int32_t>>& got_sels, size_t fwd_row,
                             const std::vector<dgpp::Dsv41Model::IndexLogits>& fwd_idx,
                             const std::vector<std::vector<int32_t>>& fwd_sels) {
    const double l2 = rel_l2(got, want, size_t(V));
    int coded_flip = 0;
    for (size_t s = 0; s < fwd_idx.size() && s < got_index.size(); ++s) {
      const auto& a = fwd_idx[s]; const auto& b = got_index[s];
      if (a.q_heads == 0 || a.q_heads != b.q_heads || b.q_rows <= r) continue;
      const size_t w = size_t(a.q_heads) * 128;
      int diffs = 0;
      for (size_t i = 0; i < w; ++i) diffs += a.q_codes[fwd_row * w + i] != b.q_codes[size_t(r) * w + i];
      const size_t ms = size_t(fx.cfg.index_topk);
      const bool flipped = std::memcmp(got_sels[s].data() + size_t(r) * ms, fwd_sels[s].data() + fwd_row * ms, ms * 4) != 0;
      if (flipped && diffs > 0) ++coded_flip;
    }
    std::printf("[INFO] %s %d: rel l2 %.2e vs the re-forward%s\n", what, r, l2,
                coded_flip ? " (a coded selection flip)" : "");
    require(argmax(got, size_t(V)) == argmax(want, size_t(V)), std::string(what) + " " + std::to_string(r) + " argmax matches the re-forward");
    if (!coded_flip) require(l2 < 0.12, std::string(what) + " " + std::to_string(r) + " within the decode budget");
  };
  {
    std::vector<int64_t> h = history;
    h.insert(h.end(), drafts.begin(), drafts.end());
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const auto again = ref.forward(h, true);
    const std::vector<dgpp::Dsv41Model::IndexLogits> again_index = ref.debug_index_logits();
    for (int r = 0; r < 6; ++r)
      check_row("verify row", r, ver.logits.data() + size_t(r) * V, again.logits.data() + (history.size() + r) * V, ver_index,
                ver.dsa_selections, history.size() + r, again_index, again.dsa_selections);
  }
  model.session_rollback(0, 1);
  history.push_back(drafts[0]);
  {
    const auto tok2 = tokens_of(0xD7, 2, V);
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
    const auto ver2 = model.session_verify(0, tok2);
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    const std::vector<dgpp::Dsv41Model::IndexLogits> ver2_index = model.debug_index_logits();
    std::vector<int64_t> h = history;
    h.insert(h.end(), tok2.begin(), tok2.end());
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const auto again = ref.forward(h, true);
    const std::vector<dgpp::Dsv41Model::IndexLogits> again_index = ref.debug_index_logits();
    for (int r = 0; r < 2; ++r)
      check_row("post-rollback row", r, ver2.logits.data() + size_t(r) * V, again.logits.data() + (history.size() + r) * V,
                ver2_index, ver2.dsa_selections, history.size() + r, again_index, again.dsa_selections);
  }
  model.session_close(0);
}

// The DSpark draft through the eager session API (plan D8): after a
// prefill, the first draft call (the accepted row's main hidden into the
// draft rings, the block over [next, noise x 4], the Markov-biased row 0)
// and four chain rows; then a verify over [next, drafts], a rollback and
// the next draft — every row finite, the block's rows distinct, the
// confidence logits finite; the transcript-level gates are the engine
// test's and the reference parity the forward test's.
DGPP_TEST(dsv41_model_dspark_draft_rows_and_chain_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int B = fx.cfg.dspark_block_size;
  require(B == 5, "the fixture's block");
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident, nullptr, 0, 1, /*max_requests=*/1,
                         /*mtp=*/true, /*decode_rows=*/6);
  const auto prompt = tokens_of(0xD8, 40, V);
  const auto pre = model.session_prefill(0, prompt);
  const int next = argmax(pre.logits.data(), size_t(V));
  std::vector<int32_t> ids{static_cast<int32_t>(next)};
  std::vector<std::vector<float>> rows;
  const auto d0 = model.session_draft(0, {next});
  require(d0.logits.size() == size_t(V), "the draft returns one row");
  rows.push_back(d0.logits);
  for (int i = 0; i < B - 1; ++i) {
    const int tok = argmax(rows.back().data(), size_t(V));
    ids.push_back(tok);
    require(model.session_draft_chain_fits(0, i), "the chain fits the context");
    const auto dc = model.session_draft_chain(0, tok, i, i == 0, i == B - 2);
    require(dc.logits.size() == size_t(V), "a chain row returns one row");
    rows.push_back(dc.logits);
  }
  ids.push_back(argmax(rows.back().data(), size_t(V)));
  for (const auto& r : rows)
    for (const float v : r) require(std::isfinite(v), "draft logits finite");
  for (int i = 1; i < B; ++i) require(rel_l2(rows[size_t(i)].data(), rows[0].data(), size_t(V)) > 1e-4, "block rows differ");
  std::vector<float> conf(static_cast<size_t>(B));
  DGPP_CUDA_OK(cudaMemcpy(conf.data(), model.debug_confidence(), size_t(B) * 4, cudaMemcpyDeviceToHost));
  for (const float c : conf) require(std::isfinite(c), "confidence finite");
  std::printf("[INFO] dspark ids: next %d drafts", next);
  for (size_t i = 1; i < ids.size(); ++i) std::printf(" %d", ids[i]);
  std::printf("; confidence %.3f %.3f %.3f %.3f %.3f\n", conf[0], conf[1], conf[2], conf[3], conf[4]);
  // The verify over [next, d1 .. d5], greedy acceptance, the rollback and
  // the next draft (the accepted rows' main hidden appended to the rings).
  std::vector<int64_t> fed;
  for (const int32_t t : ids) fed.push_back(t);
  const auto ver = model.session_verify(0, fed);
  require(ver.logits.size() == size_t(B + 1) * V, "the verify's rows");
  int accepted = 1;
  while (accepted < B + 1 && argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)) == ids[size_t(accepted)]) ++accepted;
  if (accepted < B + 1) model.session_rollback(0, accepted);
  std::printf("[INFO] verify accepted %d of %d rows\n", accepted, B + 1);
  std::vector<int64_t> acc(fed.begin(), fed.begin() + accepted);
  // The draft consumes the accepted rows' tokens: row i's token is the
  // token AFTER row i (the winner); the last accepted row's is the new next.
  std::vector<int64_t> draft_tokens;
  for (int i = 1; i < accepted; ++i) draft_tokens.push_back(fed[size_t(i)]);
  draft_tokens.push_back(argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)));
  const auto d1 = model.session_draft(0, draft_tokens);
  for (const float v : d1.logits) require(std::isfinite(v), "second draft finite");
  const auto dc1 = model.session_draft_chain(0, argmax(d1.logits.data(), size_t(V)), 0, true, false);
  for (const float v : dc1.logits) require(std::isfinite(v), "second chain finite");
  model.session_close(0);
}

// engine.dense_weights = "fp8" (the DeepSeek-V4.1 family): the lm head is
// encoded to block FP8 at load (128 x 128 grid, the 0731 release's dense
// form), and the head GEMM takes the fp8 decode form. The prefill's last
// row and the first decode step stay inside the fp8 quantization budget of
// the bf16 head's logits (the weight noise, not a kernel order change).
DGPP_TEST(dsv41_model_head_fp8_logits_within_the_fp8_budget) {
  require(!dgpp::Dsv41LoaderFamily::dense_weights_fp8(), "the dense stack is the checkpoint's by default");
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const auto prompt = tokens_of(0xB1, 19, V);
  std::vector<float> ref_prefill, ref_step;
  int tok = -1;
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, /*mtp=*/false, 8);
    const auto o = m.session_prefill(0, prompt);
    ref_prefill = o.logits;
    tok = argmax(o.logits.data(), size_t(V));
    ref_step = m.session_step(0, tok).logits;
    m.session_close(0);
  }
  {
    dgpp::Dsv41LoaderFamily::set_dense_weights_fp8(true);
    struct Reset { ~Reset() { dgpp::Dsv41LoaderFamily::set_dense_weights_fp8(false); } } reset;
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, /*mtp=*/false, 8);
    require(m.head_fp8(), "the model takes the FP8 head when the family flag is on");
    const auto o = m.session_prefill(0, prompt);
    const double prefill_err = rel_l2(o.logits.data(), ref_prefill.data(), ref_prefill.size());
    require(prefill_err < 5e-2, "fp8 head prefill logits outside the fp8 budget: " + std::to_string(prefill_err));
    const auto st = m.session_step(0, tok);
    const double step_err = rel_l2(st.logits.data(), ref_step.data(), ref_step.size());
    require(step_err < 5e-2, "fp8 head decode logits outside the fp8 budget: " + std::to_string(step_err));
    std::printf("[ OK ] head FP8 (128 x 128 grid at load): prefill rel_l2 %.2e, decode rel_l2 %.2e\n", prefill_err, step_err);
  }
}

int main() { return dgpp::test::run_all(); }
