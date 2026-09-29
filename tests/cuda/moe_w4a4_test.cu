// The W4A4 NVFP4 grouped MoE GEMM (src/kernels/moe_w4a4.cu):
//  1. layouts: against an fp64 host dot of the SAME quantized activations and
//     weights (dequantized exactly) -- only fp32 summation order may differ;
//  2. quantization: against the W4A16 kernel on the bf16 activations -- the
//     error W4A4 adds, reported (the model-level gate is BFCL / tool-eval);
//  3. speed at prefill shapes against launch_moe_grouped_mma_fp4_*.
//   moe_w4a4_test [--tokens T] [--experts E] [--topk K]
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/moe_w4a4.hpp"

using namespace dgpp;

namespace {

float e2m1(uint8_t nib) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (nib & 8) ? -mag[nib & 7] : mag[nib & 7];
}
float e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}
uint16_t bf16(float f) {
  const __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
template <typename T>
T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, h.size() * sizeof(T)));
  DGPP_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <typename T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

struct Matrix {  // one expert's FP4 matrix [n, k] (NVFP4 group 16, MXFP4 group 32)
  std::vector<uint8_t> codes, scales;
  float global = 1.f;
  int group = 16;
};

struct Case {
  int tokens, experts, topk, n, k;
  std::vector<uint16_t> act;         // [tokens, k] bf16
  std::vector<Matrix> w;             // per expert
  std::vector<MoeSegment> segs;
  std::vector<int32_t> act_rows;     // [tokens * topk] -> token
  int max_rows = 0;
};

Case make_case(int tokens, int experts, int topk, int n, int k, uint32_t seed, bool mx = false) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> byte(0, 255);
  Case c{tokens, experts, topk, n, k, {}, {}, {}, {}, 0};
  c.act.resize(static_cast<size_t>(tokens) * k);
  for (size_t i = 0; i < c.act.size(); ++i) {
    float x = nd(rng);
    if (byte(rng) == 0) x *= 20.f;  // outliers, as real hidden states have
    c.act[i] = bf16(x);
  }
  std::uniform_real_distribution<float> sd(0.05f, 1.5f);
  for (int e = 0; e < experts; ++e) {
    Matrix m;
    m.group = mx ? kMxfp4Group : kFp4Group;
    m.codes.resize(static_cast<size_t>(n) * k / 2);
    for (auto& b : m.codes) b = static_cast<uint8_t>(byte(rng));
    m.scales.resize(static_cast<size_t>(n) * k / m.group);
    for (auto& s : m.scales) {
      if (mx)
        s = static_cast<uint8_t>(byte(rng) % 16 == 0 ? 0 : 118 + byte(rng) % 17);  // e8m0, no NaN
      else
        s = __nv_fp8_e4m3(sd(rng)).__x;
    }
    m.global = mx ? 1.f : 300.f + 200.f * sd(rng);
    c.w.push_back(std::move(m));
  }
  // Routing: each token picks topk distinct experts; segments by expert.
  std::vector<std::vector<int>> by_expert(experts);
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int t = 0; t < tokens; ++t) {
    std::vector<int> chosen;
    while (static_cast<int>(chosen.size()) < topk) {
      const int e = pick(rng);
      bool dup = false;
      for (int x : chosen) dup |= x == e;
      if (!dup) chosen.push_back(e);
    }
    for (int e : chosen) by_expert[e].push_back(t);
  }
  int row = 0;
  for (int e = 0; e < experts; ++e) {
    if (by_expert[e].empty()) continue;
    c.segs.push_back(MoeSegment{row, static_cast<int32_t>(by_expert[e].size()), e});
    for (int t : by_expert[e]) c.act_rows.push_back(t);
    row += static_cast<int>(by_expert[e].size());
    c.max_rows = std::max(c.max_rows, static_cast<int>(by_expert[e].size()));
  }
  return c;
}

struct Device {
  std::vector<uint8_t*> bufs;
  std::vector<float*> globals;
  MoeExpertView* views = nullptr;
};

Device upload(const Case& c) {
  Device d;
  std::vector<MoeExpertView> views(static_cast<size_t>(c.experts) * 3);
  for (int e = 0; e < c.experts; ++e) {
    uint8_t* codes = dev(c.w[e].codes);
    uint8_t* scales = dev(c.w[e].scales);
    float* g = nullptr;
    if (c.w[e].group == kFp4Group) {
      g = dev(std::vector<float>{c.w[e].global});
      d.globals.push_back(g);
    }
    d.bufs.push_back(codes);
    d.bufs.push_back(scales);
    MoeExpertView v{codes, nullptr, scales, g, 7, 7};
    v.fp4_group = c.w[e].group;
    views[static_cast<size_t>(e) * 3] = v;
  }
  d.views = dev(views);
  return d;
}

int run(int tokens, int experts, int topk, int n, int k, bool check_exact, const char* label, float static_gs = 0.f,
        bool mx = false) {
  const Case c = make_case(tokens, experts, topk, n, k, 1234u + n + k + (mx ? 7u : 0u), mx);
  const Device d = upload(c);
  const int rows = static_cast<int>(c.act_rows.size());
  uint16_t* act = dev(c.act);
  MoeSegment* segs = dev(c.segs);
  int32_t* act_rows = dev(c.act_rows);
  const size_t ss = mx ? mxfp4_act_scale_stride(k) : nvfp4_act_scale_stride(k);
  uint8_t *codes = nullptr, *scales = nullptr;
  float *gs = nullptr, *out4 = nullptr, *out16 = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&codes, static_cast<size_t>(tokens) * k / 2));
  DGPP_CUDA_OK(cudaMalloc(&scales, static_cast<size_t>(tokens) * ss));
  if (!mx) DGPP_CUDA_OK(cudaMalloc(&gs, static_cast<size_t>(tokens) * 4));
  DGPP_CUDA_OK(cudaMalloc(&out4, static_cast<size_t>(rows) * n * 4));
  DGPP_CUDA_OK(cudaMalloc(&out16, static_cast<size_t>(rows) * n * 4));
  const int ns = static_cast<int>(c.segs.size());

  auto w4a4 = [&] {
    if (mx)
      launch_quantize_rows_mx(act, k, tokens, k, codes, scales, nullptr);
    else
      launch_quantize_rows_nvfp4(act, k, tokens, k, codes, scales, gs, nullptr, static_gs);
    if (mx)
      launch_moe_grouped_w4a4_mx_f32(codes, scales, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k,
                                     nullptr);
    else
      launch_moe_grouped_w4a4_f32(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k,
                                  nullptr);
  };
  auto w4a16 = [&] {
    launch_moe_grouped_mma_fp4_f32(act, k, segs, ns, c.max_rows, 0, d.views, 0, out16, n, n, k, nullptr, act_rows,
                                   mx ? kMxfp4Group : kFp4Group);
  };
  w4a4();
  w4a16();
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> o4 = host(out4, static_cast<size_t>(rows) * n);
  const std::vector<float> o16 = host(out16, static_cast<size_t>(rows) * n);
  int fails = 0;

  // 1. layouts: fp64 dot of the quantized operands, dequantized exactly.
  if (check_exact) {
    const auto qc = host(codes, static_cast<size_t>(tokens) * k / 2);
    const auto qs = host(scales, static_cast<size_t>(tokens) * ss);
    const auto qg = mx ? std::vector<float>(static_cast<size_t>(tokens), 1.f)
                       : host(gs, static_cast<size_t>(tokens));
    double worst = 0;
    for (const MoeSegment& s : c.segs) {
      const Matrix& m = c.w[s.expert];
      for (int r = 0; r < s.rows; r += std::max(1, s.rows / 3)) {
        const int tok = c.act_rows[s.row0 + r];
        for (int col = 0; col < n; col += 37) {
          double dot = 0, mag = 0;
          for (int kk = 0; kk < k; ++kk) {
            const uint8_t ab = qc[static_cast<size_t>(tok) * k / 2 + kk / 2];
            const double as = mx ? std::ldexp(1.0, static_cast<int>(qs[tok * ss + kk / 32]) - 127)
                                 : e4m3(qs[tok * ss + kk / 16]) * qg[tok];
            const double a = e2m1(kk % 2 ? ab >> 4 : ab & 15) * as;
            const uint8_t wb = m.codes[static_cast<size_t>(col) * k / 2 + kk / 2];
            const double ws = mx
                                  ? std::ldexp(1.0, static_cast<int>(m.scales[static_cast<size_t>(col) * k / 32 + kk / 32]) - 127)
                                  : e4m3(m.scales[static_cast<size_t>(col) * k / 16 + kk / 16]) / m.global;
            const double w = e2m1(kk % 2 ? wb >> 4 : wb & 15) * ws;
            dot += a * w;
            mag += std::fabs(a * w);
          }
          const double got = o4[static_cast<size_t>(s.row0 + r) * n + col];
          const double err = std::fabs(got - dot) / (mag + 1e-30);
          if (std::getenv("DBG") && err > 1e-3 && worst < 100)
            std::printf("dbg-mx seg %d r %d col %d k %d: err %.3g got %.4g want %.4g\n", s.row0, r, col, k,
                        err, got, dot);
          worst = std::max(worst, err);
        }
      }
    }
    std::printf("[ .. ] %s layouts: worst |got - fp64| / sum|terms| = %.3g\n", label, worst);
    if (!(worst < 1e-5)) {
      std::printf("[FAIL] %s: the W4A4 kernel does not compute the quantized dot\n", label);
      ++fails;
    }
  }

  // 2. the error W4A4 adds against W4A16 (bf16 activations).
  double num = 0, den = 0;
  for (size_t i = 0; i < o4.size(); ++i) {
    num += (o4[i] - o16[i]) * (o4[i] - o16[i]);
    den += o16[i] * o16[i];
  }
  std::printf("[ .. ] %s W4A4 vs W4A16: relative l2 %.4f\n", label, std::sqrt(num / den));

  // 3. speed.
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto time = [&](auto f) {
    for (int i = 0; i < 3; ++i) f();
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) f();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    return ms / 20;
  };
  auto quant_only = [&] {
    if (mx)
      launch_quantize_rows_mx(act, k, tokens, k, codes, scales, nullptr);
    else
      launch_quantize_rows_nvfp4(act, k, tokens, k, codes, scales, gs, nullptr);
  };
  const float t4 = time(w4a4), tq = time(quant_only), t16 = time(w4a16);
  const double flop = 2.0 * rows * n * k;
  std::printf("[ .. ] %s %d rows x n %d x k %d: W4A16 %.3f ms (%.1f TF)  W4A4 %.3f ms (%.1f TF, quant %.3f)  %.2fx\n",
              label, rows, n, k, t16, flop / t16 / 1e9, t4, flop / t4 / 1e9, tq, t16 / t4);
  // The production output form (bf16 rows, the down projection's down_bf16_).
  auto gemm_bf16 = [&] {
    if (mx)
      launch_moe_grouped_w4a4_mx_bf16(codes, scales, act_rows, segs, ns, c.max_rows, d.views, 0,
                                      reinterpret_cast<uint16_t*>(out4), n, n, k, nullptr);
    else
      launch_moe_grouped_w4a4_bf16(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0,
                                   reinterpret_cast<uint16_t*>(out4), n, n, k, nullptr);
  };
  auto gemm_f32 = [&] {
    if (mx)
      launch_moe_grouped_w4a4_mx_f32(codes, scales, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k,
                                     nullptr);
    else
      launch_moe_grouped_w4a4_f32(codes, scales, gs, act_rows, segs, ns, c.max_rows, d.views, 0, out4, n, n, k,
                                  nullptr);
  };
  const float tb = time(gemm_bf16), tf = time(gemm_f32);
  std::printf("[ .. ] %s GEMM only: f32 out %.3f ms (%.1f TF), bf16 out %.3f ms (%.1f TF)\n", label, tf,
              flop / tf / 1e9, tb, flop / tb / 1e9);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  for (auto* p : d.bufs) cudaFree(p);
  for (auto* p : d.globals) cudaFree(p);
  cudaFree(d.views);
  cudaFree(act);
  cudaFree(segs);
  cudaFree(act_rows);
  cudaFree(codes);
  cudaFree(scales);
  cudaFree(gs);
  cudaFree(out4);
  cudaFree(out16);
  return fails;
}

// The MXFP4 quantizer, bitwise against the host reference (e8m0 scale of
// max(amax32, 1e-4) / 448 rounded up to a power of two; e2m1 RNE codes),
// and the fused swiglu+quantize against the two-kernel chain.
int run_mx_quant(int rows, int k, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 2.f);
  std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
  for (auto& v : x) v = bf16(nd(rng));
  const size_t ss = mxfp4_act_scale_stride(k);
  uint16_t* dx = dev(x);
  uint8_t *d1 = nullptr, *d2 = nullptr, *s1 = nullptr, *s2 = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d1, static_cast<size_t>(rows) * k / 2));
  DGPP_CUDA_OK(cudaMalloc(&d2, static_cast<size_t>(rows) * k / 2));
  DGPP_CUDA_OK(cudaMalloc(&s1, rows * ss));
  DGPP_CUDA_OK(cudaMalloc(&s2, rows * ss));
  launch_quantize_rows_mx(dx, k, rows, k, d1, s1, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto h1 = host(d1, static_cast<size_t>(rows) * k / 2);
  const auto hs1 = host(s1, rows * ss);
  int fails = 0;
  size_t bad = 0;
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < k / 32; ++g) {
      float amax = 0.f;
      float v[32];
      for (int i = 0; i < 32; ++i) {
        v[i] = std::bit_cast<float>(uint32_t(x[static_cast<size_t>(r) * k + g * 32 + i]) << 16);
        amax = std::max(amax, std::fabs(v[i]));
      }
      const uint8_t sb = e8m0_ceil_log2_byte(std::max(amax, 1e-4f) * (1.f / 448.f));
      if (hs1[static_cast<size_t>(r) * ss + g] != sb) ++bad;
      const float inv = std::ldexp(1.f, 127 - static_cast<int>(sb));
      for (int i = 0; i < 32; ++i) {
        const uint8_t nib = i % 2
                                ? static_cast<uint8_t>(h1[static_cast<size_t>(r) * k / 2 + g * 16 + i / 2] >> 4)
                                : h1[static_cast<size_t>(r) * k / 2 + g * 16 + i / 2] & 15;
        if (nib != float_to_fp4_e2m1_bits(v[i] * inv)) ++bad;
      }
    }
  }
  std::printf("[ %s ] MX quantizer vs host reference: %zu mismatches\n", bad == 0 ? "OK" : "FAIL", bad);
  if (bad) ++fails;

  // The fused swiglu + MX quantize, bitwise the two-kernel chain.
  std::vector<uint16_t> ga(x.size()), up(x.size());
  for (auto& v : ga) v = bf16(nd(rng));
  for (auto& v : up) v = bf16(nd(rng));
  uint16_t *dga = dev(ga), *dup = dev(up), *dact = dev(std::vector<uint16_t>(x.size()));
  DGPP_CUDA_OK(cudaMemset(s2, 0, rows * ss));
  launch_moe_swiglu_clamp(dga, dup, dact, int64_t(rows) * k, std::numeric_limits<float>::infinity(), 0);
  launch_quantize_rows_mx(dact, k, rows, k, d1, s1, nullptr);
  launch_swiglu_quantize_rows_mx(dga, dup, k, rows, k, std::numeric_limits<float>::infinity(), d2, s2, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto hc1 = host(d1, h1.size()), hc2 = host(d2, h1.size());
  const auto hss1 = host(s1, hs1.size()), hss2 = host(s2, hs1.size());
  const bool same = hc1 == hc2 && hss1 == hss2;
  std::printf("[ %s ] fused swiglu+MX-quantize vs the two-kernel chain: %s\n", same ? "OK" : "FAIL",
              same ? "bitwise (codes, scales)" : "DIFFER");
  if (!same) ++fails;
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dga), static_cast<void*>(dup), static_cast<void*>(dact),
                  static_cast<void*>(d1), static_cast<void*>(d2), static_cast<void*>(s1),
                  static_cast<void*>(s2)})
    cudaFree(p);
  return fails;
}

// Constant-data exact dot (2026-09-29): all codes 0x11 (e2m1 0.5), all scales
// 1.0 (e8m0 0x7F / e4m3 0x38), so every row must land on 0.25 * k exactly
// (the 0.25 terms are integers in fp32). A uniform scalar bug -- a dropped
// scale byte, a missing k half, a halved global -- shows up on every row,
// while the random-data cases above can hide factor-of-two errors inside the
// quantizer/weight noise.
int run_const(int n, int k, bool mx, const char* label) {
  const int rows = 128;
  std::vector<uint8_t> acodes(size_t(rows) * k / 2, 0x11);
  std::vector<uint8_t> ascales_mx(size_t(rows) * mxfp4_act_scale_stride(k), 0x7F);
  std::vector<uint8_t> ascales_nv(size_t(rows) * nvfp4_act_scale_stride(k), 0x38);
  std::vector<uint8_t> wcodes(size_t(n) * k / 2, 0x11);
  std::vector<uint8_t> wscales_mx(size_t(n) * k / 32, 0x7F);
  std::vector<uint8_t> wscales_nv(size_t(n) * k / 16, 0x38);
  std::vector<int32_t> actrows(rows);
  for (int i = 0; i < rows; ++i) actrows[i] = i;
  std::vector<MoeSegment> segs{{0, rows, 0}};
  MoeExpertView view;
  uint8_t *dac = nullptr, *dws = nullptr, *dwc = nullptr;
  uint8_t* das_mx = nullptr, *das_nv = nullptr, *dwsv = nullptr;
  float* dgs = nullptr;
  MoeSegment* dseg = nullptr;
  MoeExpertView* dv = nullptr;
  float* dout = nullptr;
  int32_t* dactrows = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dac, acodes.size()));
  DGPP_CUDA_OK(cudaMalloc(&dws, wscales_mx.size()));
  DGPP_CUDA_OK(cudaMalloc(&dwc, wcodes.size()));
  DGPP_CUDA_OK(cudaMalloc(&das_mx, ascales_mx.size()));
  DGPP_CUDA_OK(cudaMalloc(&das_nv, ascales_nv.size()));
  DGPP_CUDA_OK(cudaMalloc(&dwsv, wscales_nv.size()));
  DGPP_CUDA_OK(cudaMalloc(&dgs, size_t(rows) * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&dseg, segs.size() * sizeof(MoeSegment)));
  DGPP_CUDA_OK(cudaMalloc(&dv, sizeof(MoeExpertView)));
  DGPP_CUDA_OK(cudaMalloc(&dout, size_t(rows) * n * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&dactrows, actrows.size() * sizeof(int32_t)));
  cudaMemcpy(dac, acodes.data(), acodes.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dws, wscales_mx.data(), wscales_mx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dwc, wcodes.data(), wcodes.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(das_mx, ascales_mx.data(), ascales_mx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(das_nv, ascales_nv.data(), ascales_nv.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dwsv, wscales_nv.data(), wscales_nv.size(), cudaMemcpyHostToDevice);
  std::vector<float> gsv(rows, 1.f);
  cudaMemcpy(dgs, gsv.data(), gsv.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dseg, segs.data(), segs.size() * sizeof(MoeSegment), cudaMemcpyHostToDevice);
  cudaMemcpy(dactrows, actrows.data(), actrows.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  view.payload = dwc;
  view.fp4_scales = mx ? dws : dwsv;
  view.fp4_group = mx ? kMxfp4Group : kFp4Group;
  view.fp4_global = dgs;
  cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
  if (mx)
    launch_moe_grouped_w4a4_mx_f32(dac, das_mx, dactrows, dseg, 1, rows, dv, 0, dout, n, n, k, nullptr);
  else
    launch_moe_grouped_w4a4_f32(dac, das_nv, dgs, dactrows, dseg, 1, rows, dv, 0, dout, n, n, k, nullptr);
  cudaDeviceSynchronize();
  const auto o = host(dout, size_t(rows) * n);
  const float want = 0.25f * k;
  double worst = 0;
  for (const int rr : {0, 64}) {
    std::printf("  row %d factors:", rr);
    for (int c = 0; c < n; c += 64) {
      const float got = o[static_cast<size_t>(rr) * n + c];
      if (std::abs(got - want) > 1e-3f * want)
        std::printf(" c%d:%.2g", c, std::abs(got) < 1e-30f ? 0.f : got / want);
    }
    std::printf("\n");
  }
  int shown = 0;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; c += 97) {
      const double err = std::fabs(o[static_cast<size_t>(r) * n + c] - want);
      if (err > 1e-3 && shown < 8) {
        std::printf("  r %d c %d: got %.6g want %.6g\n", r, c, o[static_cast<size_t>(r) * n + c], want);
        ++shown;
      }
      worst = std::max(worst, err);
    }
  std::printf("[ .. ] %s: worst |got - 0.25k| = %.3g\n", label, worst);
  int fails = 0;
  if (!(worst < 1e-3)) {
    std::printf("[FAIL] %s: the constant-data dot is not exact\n", label);
    ++fails;
  }
  for (void* p : {static_cast<void*>(dac), static_cast<void*>(dws), static_cast<void*>(dwc),
                  static_cast<void*>(das_mx), static_cast<void*>(das_nv), static_cast<void*>(dwsv),
                  static_cast<void*>(dgs), static_cast<void*>(dseg), static_cast<void*>(dv),
                  static_cast<void*>(dout), static_cast<void*>(dactrows)})
    cudaFree(p);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("[SKIP] moe_w4a4_test: no CUDA device\n");
    return 2;
  }
  int tokens = 8192, experts = 256, topk = 10;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens") tokens = std::atoi(argv[++i]);
    else if (a == "--experts") experts = std::atoi(argv[++i]);
    else if (a == "--topk") topk = std::atoi(argv[++i]);
  }
  int fails = 0;
  const bool only_const = std::getenv("ONLY_CONST") != nullptr;
  if (!only_const) {
  // Small shapes with ragged segments and k tails for the exact layout check.
  fails += run(64, 8, 2, 256, 256, true, "small");
  fails += run(200, 16, 4, 320, 320, true, "k320");
  fails += run(100, 8, 3, 130, 2560, true, "n-tail");
  // A static global (the checkpoint's input_scale form) small enough that the
  // 20x outliers clip: the layouts must still be exact on the clipped codes.
  fails += run(200, 16, 4, 320, 320, true, "static-clip", 2.0f / (6.f * 448.f));
  // Prefill shapes (the Qwen experts per rank at TP=2: gate/up n = 320,
  // k = 2560; down n = 2560, k = 320).
  fails += run(tokens, experts, topk, 640, 2560, false, "gate_up");
  fails += run(tokens, experts, topk, 2560, 320, false, "down");
  // MXFP4 (e8m0 scales per 32, no global — the 0731 expert format).
  fails += run(64, 8, 2, 256, 256, true, "small-mx", 0.f, true);
  fails += run(200, 16, 4, 320, 320, true, "k320-mx", 0.f, true);
  fails += run(100, 8, 3, 130, 2560, true, "n-tail-mx", 0.f, true);
  // The 0731 prefill shapes (hidden 4096, inter 2048).
  fails += run(tokens, experts, topk, 2048, 4096, false, "gate_up-mx", 0.f, true);
  fails += run(tokens, experts, topk, 4096, 2048, false, "down-mx", 0.f, true);
  fails += run_mx_quant(57, 320, 9);
  }
  // Constant-data exact dots at the 0731 prefill shapes (and the k = 320
  // tail): a uniform factor-of-two bug hits every row of every tile.
  fails += run_const(320, 320, true, "const-mx-k320");
  fails += run_const(2048, 4096, true, "const-mx-gate_up");
  fails += run_const(4096, 2048, true, "const-mx-down");
  fails += run_const(320, 320, false, "const-nvfp4-k320");
  // The fused activation + quantizer against swiglu -> quantize, bitwise
  // (codes, scales, globals), at the Qwen down projection's k = 320.
  {
    const int rows = 57, k = 320, stride = 320;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> ga(size_t(rows) * stride), up(ga.size());
    for (auto& v : ga) v = dgpp::float_to_bf16_bits(nd(rng));
    for (auto& v : up) v = dgpp::float_to_bf16_bits(nd(rng));
    const size_t ss = dgpp::nvfp4_act_scale_stride(k);
    uint16_t *dg, *du, *da;
    uint8_t *c1, *c2, *s1, *s2;
    float *g1, *g2;
    DGPP_CUDA_OK(cudaMalloc(&dg, ga.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&du, up.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&da, ga.size() * 2));
    DGPP_CUDA_OK(cudaMemcpy(dg, ga.data(), ga.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(du, up.data(), up.size() * 2, cudaMemcpyHostToDevice));
    for (uint8_t** q : {&c1, &c2}) {
      DGPP_CUDA_OK(cudaMalloc(q, size_t(rows) * k / 2));
      DGPP_CUDA_OK(cudaMemset(*q, 0, size_t(rows) * k / 2));
    }
    for (uint8_t** q : {&s1, &s2}) {
      DGPP_CUDA_OK(cudaMalloc(q, rows * ss));
      DGPP_CUDA_OK(cudaMemset(*q, 0, rows * ss));
    }
    for (float** q : {&g1, &g2}) DGPP_CUDA_OK(cudaMalloc(q, rows * 4));
    const float gs = 5.0f / (6.f * 448.f), limit = std::numeric_limits<float>::infinity();
    dgpp::launch_moe_swiglu_clamp(dg, du, da, int64_t(rows) * stride, limit, 0);
    dgpp::launch_quantize_rows_nvfp4(da, stride, rows, k, c1, s1, g1, 0, gs);
    dgpp::launch_swiglu_quantize_rows_nvfp4(dg, du, stride, rows, k, limit, c2, s2, g2, 0, gs);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> hc1(size_t(rows) * k / 2), hc2(hc1.size()), hs1(rows * ss), hs2(hs1.size());
    std::vector<float> hg1(rows), hg2(rows);
    DGPP_CUDA_OK(cudaMemcpy(hc1.data(), c1, hc1.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hc2.data(), c2, hc2.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hs1.data(), s1, hs1.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hs2.data(), s2, hs2.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hg1.data(), g1, rows * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hg2.data(), g2, rows * 4, cudaMemcpyDeviceToHost));
    const bool same = hc1 == hc2 && hs1 == hs2 && hg1 == hg2;
    std::printf("[ %s ] fused swiglu+quantize vs the two-kernel chain: %s\n", same ? "OK" : "FAIL",
                same ? "bitwise (codes, scales, globals)" : "DIFFER");
    if (!same) ++fails;
    // The static global read from device memory (the layer image's copy of
    // input_scale) is bitwise the same global passed by value.
    {
      float* dgs;
      DGPP_CUDA_OK(cudaMalloc(&dgs, 4));
      DGPP_CUDA_OK(cudaMemcpy(dgs, &gs, 4, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemset(c2, 0, size_t(rows) * k / 2));
      DGPP_CUDA_OK(cudaMemset(s2, 0, rows * ss));
      dgpp::launch_quantize_rows_nvfp4(da, stride, rows, k, c2, s2, g2, 0, 0.f, dgs);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemcpy(hc2.data(), c2, hc2.size(), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hs2.data(), s2, hs2.size(), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hg2.data(), g2, rows * 4, cudaMemcpyDeviceToHost));
      const bool dev_same = hc1 == hc2 && hs1 == hs2 && hg1 == hg2;
      std::printf("[ %s ] static global from device memory vs by value: %s\n", dev_same ? "OK" : "FAIL",
                  dev_same ? "bitwise (codes, scales, globals)" : "DIFFER");
      if (!dev_same) ++fails;
      cudaFree(dgs);
    }
    for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(da), static_cast<void*>(c1),
                    static_cast<void*>(c2), static_cast<void*>(s1), static_cast<void*>(s2), static_cast<void*>(g1),
                    static_cast<void*>(g2)})
      cudaFree(p);
  }
  std::printf(fails == 0 ? "[ OK ] moe_w4a4_test\n" : "[FAIL] moe_w4a4_test: %d\n", fails);
  return fails == 0 ? 0 : 1;
}
