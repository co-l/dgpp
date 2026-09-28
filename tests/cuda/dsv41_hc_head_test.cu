// The V4-Flash-0731 learned head collapse (the checkpoint reference's
// Transformer.hc_head / DSparkBlock.forward_head, vllm's
// hc_head_fused_kernel_tilelang): weight-free RMSNorm of the flat 4H row,
// the fp32 mixes through the bf16 fn, the sigmoid pre with scale/base and
// hc_eps, and the fp32 weighted stream sum to bf16.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dsv41_dspark.hpp"

namespace {
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::compare_abs_rel;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct HcRef {
  int hc_mult = 4, hidden = 0;
  std::vector<uint16_t> streams;
  std::vector<uint16_t> fn;
  std::vector<float> base;
  float scale = 1.f, eps = 1e-6f, hc_eps = 1e-6f;
};

// The reference arithmetic: a faithful mirror of the kernel's exact fp32
// accumulation (per-lane fma chains, 256 lanes, the two-stage butterfly over
// the 8 warp partials) so the comparison is deterministic down to the last
// ulp; only the host's exact rsqrt/exp differ from the device's fast math,
// far below the bf16 ulp budget.
namespace mirror {
constexpr int kLanes = 256;

inline float butterfly32(const float* v0) {
  float v[32];
  for (int t = 0; t < 32; ++t) v[t] = v0[t];
  for (int off = 16; off > 0; off >>= 1) {
    float nv[32];
    for (int t = 0; t < 32; ++t) nv[t] = v[t] + v[t ^ off];
    for (int t = 0; t < 32; ++t) v[t] = nv[t];
  }
  return v[0];
}

inline float tree256(const float* lane) {
  float warp[8];
  for (int w = 0; w < 8; ++w) warp[w] = butterfly32(lane + w * 32);
  float v[32];
  for (int t = 0; t < 32; ++t) v[t] = t < 8 ? warp[t] : 0.f;
  return butterfly32(v);
}

template <typename F>
float lane_reduce(int W, F f) {
  float lane[kLanes];
  for (int t = 0; t < kLanes; ++t) {
    float acc = 0.f;
    for (int d = t; d < W; d += kLanes) acc = f(d, acc);
    lane[t] = acc;
  }
  return tree256(lane);
}
}  // namespace mirror

std::vector<uint16_t> run_ref(const HcRef& r) {
  const int rows = static_cast<int>(r.streams.size()) / (r.hc_mult * r.hidden);
  std::vector<uint16_t> out(static_cast<size_t>(rows) * r.hidden);
  for (int t = 0; t < rows; ++t) {
    const uint16_t* x = r.streams.data() + static_cast<size_t>(t) * r.hc_mult * r.hidden;
    const int W = r.hc_mult * r.hidden;
    const float ss = mirror::lane_reduce(W, [x](int d, float a) {
      const float v = bf16_bits_to_float(x[d]);
      return fmaf(v, v, a);
    });
    const float rms = rsqrtf(ss / W + r.eps);
    float pre[8];
    for (int m = 0; m < r.hc_mult; ++m) {
      const uint16_t* w = r.fn.data() + static_cast<size_t>(m) * W;
      const float acc = mirror::lane_reduce(W, [x, w](int d, float a) {
        return fmaf(bf16_bits_to_float(x[d]), bf16_bits_to_float(w[d]), a);
      });
      pre[m] = 1.f / (1.f + expf(-(acc * rms * r.scale + r.base[m]))) + r.hc_eps;
    }
    for (int d = 0; d < r.hidden; ++d) {
      float acc = 0.f;
      for (int m = 0; m < r.hc_mult; ++m) acc += pre[m] * bf16_bits_to_float(x[static_cast<size_t>(m) * r.hidden + d]);
      out[static_cast<size_t>(t) * r.hidden + d] = float_to_bf16_bits(acc);
    }
  }
  return out;
}

// The fast device rsqrt/exp drift |logit| by at most ~|acc| * 2^-21; keep the
// test data far from the sigmoid knee so that drift cannot flip a pre.
void require_logit_margin(const HcRef& r, float margin) {
  float min_logit = 1e30f;
  const int rows = static_cast<int>(r.streams.size()) / (r.hc_mult * r.hidden);
  const int W = r.hc_mult * r.hidden;
  for (int t = 0; t < rows; ++t) {
    const uint16_t* x = r.streams.data() + static_cast<size_t>(t) * W;
    const float ss = mirror::lane_reduce(W, [x](int d, float a) {
      const float v = bf16_bits_to_float(x[d]);
      return fmaf(v, v, a);
    });
    const float rms = rsqrtf(ss / W + r.eps);
    for (int m = 0; m < r.hc_mult; ++m) {
      const uint16_t* w = r.fn.data() + static_cast<size_t>(m) * W;
      const float acc = mirror::lane_reduce(W, [x, w](int d, float a) {
        return fmaf(bf16_bits_to_float(x[d]), bf16_bits_to_float(w[d]), a);
      });
      const float logit = acc * rms * r.scale + r.base[m];
      min_logit = std::min(min_logit, std::abs(logit));
    }
  }
  require(min_logit > margin, "logit margin (test data sits at the sigmoid knee)");
}

void check(const HcRef& r, const std::vector<uint16_t>& got, const std::string& what) {
  const auto want = run_ref(r);
  const long n = static_cast<long>(got.size());
  std::vector<float> gf(n), wf(n);
  for (long i = 0; i < n; ++i) {
    gf[static_cast<size_t>(i)] = bf16_bits_to_float(got[static_cast<size_t>(i)]);
    wf[static_cast<size_t>(i)] = bf16_bits_to_float(want[static_cast<size_t>(i)]);
  }
  // The 1e-6 floor is the hc_eps-weighted noise floor of near-zero (canceled)
  // outputs, where the fast-math drift of the pre's is absolute O(1e-7) and a
  // relative criterion would be meaningless; structural bugs stay O(>=1e-3).
  const auto s = compare_abs_rel(gf.data(), wf.data(), n, 32.0 * std::pow(2.0, -7.0), 1e-6);
  require_bf16(what, s, 1e-3, 0.0);
}

DevBuf upload_u16(const std::vector<uint16_t>& v) {
  DevBuf b(v.size() * sizeof(uint16_t));
  b.upload(v.data(), b.bytes);
  return b;
}

std::vector<uint16_t> download_u16(const DevBuf& b) {
  std::vector<uint16_t> v(b.bytes / sizeof(uint16_t));
  b.download(v.data(), v.size() * sizeof(uint16_t));
  return v;
}

}  // namespace

DGPP_TEST(dsv41_hc_head_small_matches_the_reference) {
  constexpr int M = 4, H = 64, R = 5;
  HcRef r;
  r.hc_mult = M;
  r.hidden = H;
  r.streams = random_bf16_bits(1, static_cast<int64_t>(M) * H * R, 0, 3);
  r.fn = random_bf16_bits(2, static_cast<int64_t>(M) * M * H, 0, 3);
  r.base = {-0.5f, 0.25f, 1.f, -2.f};
  r.scale = 1.75f;
  r.eps = 1e-6f;
  r.hc_eps = 1e-6f;
  require_logit_margin(r, 4.f);

  const auto xs = upload_u16(r.streams);
  const auto fs = upload_u16(r.fn);
  DevBuf bs(4 * sizeof(float));
  bs.upload(r.base.data(), bs.bytes);
  const float scale = r.scale;
  DevBuf os(static_cast<size_t>(R) * H * sizeof(uint16_t));
  DGPP_CUDA_OK(cudaMemset(os.p, 0xEE, os.bytes));
  dgpp::dsv41_hc_head_bf16(static_cast<const uint16_t*>(xs.p), M, H, R, static_cast<const uint16_t*>(fs.p),
                           &scale, static_cast<const float*>(bs.p), r.eps, r.hc_eps,
                           static_cast<uint16_t*>(os.p), 0);
  check(r, download_u16(os), "small");
}

DGPP_TEST(dsv41_hc_head_production_shape_matches_the_reference) {
  constexpr int M = 4, H = 5120;
  const int rows[2] = {6, 32};
  for (int i = 0; i < 2; ++i) {
    const int R = rows[i];
    HcRef r;
    r.hc_mult = M;
    r.hidden = H;
    r.streams = random_bf16_bits(10 + i, static_cast<int64_t>(M) * H * R, 0, 3);
    r.fn = random_bf16_bits(20 + i, static_cast<int64_t>(M) * M * H, 0, 3);
    r.base = {-0.5f, 0.25f, 1.f, -2.f};
    r.scale = 1.f;
    r.eps = 1e-6f;
    r.hc_eps = 1e-6f;
    require_logit_margin(r, 4.f);
    const auto xs = upload_u16(r.streams);
    const auto fs = upload_u16(r.fn);
    DevBuf bs(4 * sizeof(float));
    bs.upload(r.base.data(), bs.bytes);
    const float scale = r.scale;
    DevBuf os(static_cast<size_t>(R) * H * sizeof(uint16_t));
    dgpp::dsv41_hc_head_bf16(static_cast<const uint16_t*>(xs.p), M, H, R, static_cast<const uint16_t*>(fs.p),
                             &scale, static_cast<const float*>(bs.p), r.eps, r.hc_eps,
                             static_cast<uint16_t*>(os.p), 0);
    check(r, download_u16(os), "production shape");
  }
}

DGPP_TEST(dsv41_hc_head_is_deterministic) {
  constexpr int M = 4, H = 256, R = 3;
  HcRef r;
  r.hc_mult = M;
  r.hidden = H;
  r.streams = random_bf16_bits(7, static_cast<int64_t>(M) * H * R, 0, 3);
  r.fn = random_bf16_bits(8, static_cast<int64_t>(M) * M * H, 0, 3);
  r.base = {0.1f, -0.3f, 0.7f, 0.f};
  const auto xs = upload_u16(r.streams);
  const auto fs = upload_u16(r.fn);
  DevBuf bs(4 * sizeof(float));
  bs.upload(r.base.data(), bs.bytes);
  const float scale = 1.f;
  DevBuf o1(static_cast<size_t>(R) * H * sizeof(uint16_t)), o2(o1.bytes);
  for (int rep = 0; rep < 2; ++rep) {
    void* o = rep ? o2.p : o1.p;
    dgpp::dsv41_hc_head_bf16(static_cast<const uint16_t*>(xs.p), M, H, R, static_cast<const uint16_t*>(fs.p),
                             &scale, static_cast<const float*>(bs.p), 1e-6f, 1e-6f,
                             static_cast<uint16_t*>(o), 0);
  }
  const auto a = download_u16(o1);
  const auto b = download_u16(o2);
  for (size_t i = 0; i < a.size(); ++i) require(a[i] == b[i], "deterministic");
}

int main() { return dgpp::test::run_all(); }
