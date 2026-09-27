#pragma once
// Shared helpers for the KDA CUDA tests and the reference-dump parity
// runner. Header-only; every definition is inline. The dump runner lives in
// a host-only TU because its reader pulls in minijson, which nvcc's frontend
// refuses (incomplete-type vector members that g++ accepts).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "core/arena.hpp"
#include "kernels/gemm.hpp"
#include "models/kda_geometry.hpp"
#include "models/kda_layer.hpp"
#include "models/kda_reference.hpp"
#include "models/kda_state.hpp"

namespace dgpp::kda_test {

using dgpp::KdaConfig;
using dgpp::KdaGeometry;
using dgpp::KdaLayer;
using dgpp::KdaLayerWeights;
using dgpp::KdaStatePool;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;

// ---------------------------------------------------------------------------
// Device buffer RAII
// ---------------------------------------------------------------------------

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;

  DevBuf() = default;
  explicit DevBuf(size_t n) {
    if (n == 0) n = 16;  // keep non-null for pointer arithmetic
    if (cudaMalloc(&p, n) != cudaSuccess)
      throw std::runtime_error("test cudaMalloc failed");
    bytes = n;
  }
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
    o.p = nullptr;
    o.bytes = 0;
  }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      if (p) cudaFree(p);
      p = o.p;
      bytes = o.bytes;
      o.p = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() {
    if (p) cudaFree(p);
  }

  void upload(const void* host, size_t n) {
    if (n > bytes) throw std::runtime_error("upload overruns buffer");
    // 1 KiB chunks: on this platform a host-to-device copy refreshes only its
    // first 1 KiB in the kernels' loadable cache (2026-09-27), so a longer
    // single transfer would leave the tail of a reused region stale.
    size_t off = 0;
    while (off < n) {
      const size_t c = std::min(size_t(1024), n - off);
      if (cudaMemcpy(static_cast<uint8_t*>(p) + off, static_cast<const uint8_t*>(host) + off, c,
                     cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("upload failed");
      off += c;
    }
  }
  void download(void* host, size_t n) const {
    if (n > bytes) throw std::runtime_error("download overruns buffer");
    if (cudaMemcpy(host, p, n, cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("download failed");
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(p);
  }
};

// ---------------------------------------------------------------------------
// Deterministic host RNG (hash-based; identical values on every host)
// ---------------------------------------------------------------------------

inline uint64_t hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

inline uint32_t hash_u32_pair(uint64_t seed, uint64_t idx) {
  uint64_t h = hash_u64(seed * 0x9E3779B97F4A7C15ULL ^ idx);
  return static_cast<uint32_t>(h ^ (h >> 32));
}

inline float uniform_pm1(uint64_t seed, uint64_t idx, float scale) {
  const float u =
      static_cast<float>(hash_u32_pair(seed, idx) & 0xFFFFFFu) / 8388607.5f;
  return (u * 2.f - 1.f) * scale;
}

inline float uniform_01(uint64_t seed, uint64_t idx, float scale) {
  const float u =
      static_cast<float>(hash_u32_pair(seed, idx) & 0xFFFFFFu) / 8388607.0f;
  return u * scale;
}

inline float normal_f(uint64_t seed, uint64_t idx) {
  const uint32_t a = hash_u32_pair(seed, idx * 2 + 0);
  const uint32_t b = hash_u32_pair(seed ^ 0x5DEECE66DULL, idx * 2 + 1);
  const float u1 = (static_cast<float>(a & 0xFFFFFFu) + 1.0f) / 16777218.0f;
  const float u2 = static_cast<float>(b & 0xFFFFFu) / 1048576.0f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

// Round-to-nearest-even bf16 via the bit trick. Identical to
// float_to_bf16_bits for the normal-range values the generators produce
// (both are RNE there), but ~10x faster — which matters because the
// full-geometry tests generate ~700M values and sanitizers multiply that
// cost. Do not use for values that may overflow bf16 max.
inline uint16_t fast_bf16_rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}

inline std::vector<uint16_t> random_bf16_uniform(uint64_t seed, int64_t n,
                                                 float scale) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = fast_bf16_rne(uniform_pm1(seed, i, scale));
  return v;
}

inline std::vector<uint16_t> random_bf16_normal(uint64_t seed, int64_t n,
                                                float stddev) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = fast_bf16_rne(normal_f(seed, i) * stddev);
  return v;
}

inline std::vector<float> random_f32_uniform(uint64_t seed, int64_t n,
                                             float scale) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = uniform_pm1(seed, i, scale);
  return v;
}

// Test data straight from hash bits: values uniform over
// ±[2^lo_exp, 2^hi_exp) — sign, one of (hi_exp-lo_exp) exponent buckets,
// and 7 mantissa bits carved from one hash word. Parity testing needs
// distinctness and sane scale, not statistical quality (the host reference
// consumes the identical bits), and three integer ops per element keeps
// full-geometry weight sets (~700M values across the suite) cheap even
// under sanitizers. Box-Muller + a rounding encoder was pure waste here.
inline std::vector<uint16_t> random_bf16_bits(uint64_t seed, int64_t n,
                                              int lo_exp, int hi_exp) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  const uint32_t span = static_cast<uint32_t>(hi_exp - lo_exp);
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t h = hash_u32_pair(seed, i);
    const uint32_t e = static_cast<uint32_t>(lo_exp) + (h >> 8) % span;
    v[static_cast<size_t>(i)] =
        static_cast<uint16_t>((h & 0x8000u) | ((e + 127u) << 7) | (h & 0x7Fu));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

struct Stats {
  double max_abs = 0;
  double max_rel = 0;
  double l2_rel = 0;
  long n = 0;
  long mismatches = 0;  // elements exceeding the caller's ulp budget
};

// Relative comparison with an absolute floor: differences below abs_floor
// are ignored entirely. Small-magnitude elements of a subtractive recurrence
// legitimately show amplified relative error (cancellation); structural bugs
// produce O(1) absolute errors, which the floor never masks.
template <typename A, typename B>
Stats compare_abs_rel(const A* a, const B* b, long n, double rel_threshold,
                      double abs_floor = 0.0) {
  Stats s;
  s.n = n;
  double sa2 = 0, sb2 = 0, sd2 = 0;
  for (long i = 0; i < n; ++i) {
    const double x = static_cast<double>(a[i]);
    const double y = static_cast<double>(b[i]);
    const double d = std::abs(x - y);
    const double m = std::max(std::abs(x), std::abs(y));
    s.max_abs = std::max(s.max_abs, d);
    const double rel = d / std::max(m, 1e-30);
    if (d > abs_floor) {
      s.max_rel = std::max(s.max_rel, rel);
      if (rel > rel_threshold) ++s.mismatches;
    }
    sa2 += x * x;
    sb2 += y * y;
    sd2 += d * d;
  }
  s.l2_rel = std::sqrt(sd2) / std::max(std::sqrt(std::max(sa2, sb2)), 1e-30);
  return s;
}

// bf16 comparison measured in ulp budgets: a bf16 ulp is between 2^-8 and
// 2^-7 of the value, so `ulps` ulps means rel <= ulps * 2^-7. Diffs below a
// small absolute floor are ignored (near-zero outputs legitimately show
// amplified relative error; structural bugs produce O(1) diffs).
inline Stats compare_bf16(const std::vector<uint16_t>& got,
                          const std::vector<uint16_t>& want, int ulps) {
  const long n = static_cast<long>(got.size());
  std::vector<float> gf(n), wf(n);
  for (long i = 0; i < n; ++i) {
    gf[static_cast<size_t>(i)] = bf16_bits_to_float(got[static_cast<size_t>(i)]);
    wf[static_cast<size_t>(i)] = bf16_bits_to_float(want[static_cast<size_t>(i)]);
  }
  return compare_abs_rel(gf.data(), wf.data(), n, ulps * std::pow(2.0, -7.0),
                         1e-7);
}

[[noreturn]] inline void fail(const std::string& what, const Stats& s) {
  throw std::runtime_error(what + ": max_abs=" + std::to_string(s.max_abs) +
                           " max_rel=" + std::to_string(s.max_rel) +
                           " l2_rel=" + std::to_string(s.l2_rel) +
                           " mismatches=" + std::to_string(s.mismatches) + "/" +
                           std::to_string(s.n));
}

inline void require_rel(const std::string& what, const Stats& s, double rel_tol,
                        double mismatch_frac) {
  const double allow =
      mismatch_frac > 0 ? std::max(1.0, mismatch_frac * s.n) : 0.0;
  if (s.max_rel > rel_tol || s.l2_rel > rel_tol || s.mismatches > allow)
    fail(what, s);
}

// bf16 assertions: the ulp budget already permits up to ulps*2^-7 relative
// error per element (mantissa-position dependent), so a max_rel criterion
// would contradict it. What catches real bugs here: elements beyond the ulp
// budget (mismatches) and systematic drift (l2_rel).
inline void require_bf16(const std::string& what, const Stats& s,
                         double l2_tol, double mismatch_frac) {
  const double allow =
      mismatch_frac > 0 ? std::max(1.0, mismatch_frac * s.n) : 0.0;
  if (s.l2_rel > l2_tol || s.mismatches > allow) fail(what, s);
}

inline void require_bitwise(const std::string& what, const void* a, const void* b,
                            size_t bytes) {
  if (std::memcmp(a, b, bytes) != 0)
    throw std::runtime_error(what + ": bytes differ");
}

inline float ref_scale(int head_dim) {
  return static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5));
}

inline cudaStream_t test_stream() {
  static cudaStream_t s = [] {
    cudaStream_t t = nullptr;
    if (cudaStreamCreate(&t) != cudaSuccess) throw std::runtime_error("stream");
    return t;
  }();
  return s;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct TestWeights {
  std::vector<uint16_t> in_proj, f_b, g_b, conv, o_norm, o_proj;
  std::vector<float> a_log, dt_bias;

  static TestWeights random(const KdaConfig& cfg, uint64_t seed) {
    const KdaGeometry g = KdaGeometry::from_config(cfg);
    TestWeights w;
    // Bit-pattern weights at magnitudes matching the reference's scale
    // expectations; see random_bf16_bits for why this is not Box-Muller.
    w.in_proj = random_bf16_bits(seed ^ 0xA1,
                                 int64_t(g.in_proj_cols) * cfg.hidden, -6, -3);
    w.f_b = random_bf16_bits(seed ^ 0xB2,
                             int64_t(g.local_proj) * cfg.head_dim, -3, 0);
    w.g_b = random_bf16_bits(seed ^ 0xC3,
                             int64_t(g.local_proj) * cfg.head_dim, -3, 0);
    w.conv = random_bf16_bits(seed ^ 0xD4,
                              int64_t(g.conv_channels) * cfg.conv_width, -3, 0);
    w.o_norm.resize(static_cast<size_t>(cfg.head_dim));
    for (int i = 0; i < cfg.head_dim; ++i)
      w.o_norm[static_cast<size_t>(i)] =
          fast_bf16_rne(uniform_01(seed ^ 0xE5, i, 0.6f));
    w.o_proj = random_bf16_bits(seed ^ 0xF6,
                                int64_t(cfg.hidden) * g.local_proj, -6, -3);
    w.a_log = random_f32_uniform(seed ^ 0x179, g.local_heads, 0.5f);
    w.dt_bias = random_f32_uniform(seed ^ 0x28A, g.local_proj, 0.5f);
    return w;
  }
};

struct DeviceWeights {
  DevBuf in_proj, f_b, g_b, conv, a_log, dt_bias, o_norm, o_proj;

  explicit DeviceWeights(const TestWeights& tw)
      : in_proj(tw.in_proj.size() * 2),
        f_b(tw.f_b.size() * 2),
        g_b(tw.g_b.size() * 2),
        conv(tw.conv.size() * 2),
        a_log(tw.a_log.size() * 4),
        dt_bias(tw.dt_bias.size() * 4),
        o_norm(tw.o_norm.size() * 2),
        o_proj(tw.o_proj.size() * 2) {
    in_proj.upload(tw.in_proj.data(), tw.in_proj.size() * 2);
    f_b.upload(tw.f_b.data(), tw.f_b.size() * 2);
    g_b.upload(tw.g_b.data(), tw.g_b.size() * 2);
    conv.upload(tw.conv.data(), tw.conv.size() * 2);
    a_log.upload(tw.a_log.data(), tw.a_log.size() * 4);
    dt_bias.upload(tw.dt_bias.data(), tw.dt_bias.size() * 4);
    o_norm.upload(tw.o_norm.data(), tw.o_norm.size() * 2);
    o_proj.upload(tw.o_proj.data(), tw.o_proj.size() * 2);
  }

  KdaLayerWeights views() const {
    KdaLayerWeights w;
    w.in_proj = in_proj.p;
    w.f_b = f_b.p;
    w.g_b = g_b.p;
    w.conv = conv.p;
    w.a_log = a_log.as<float>();
    w.dt_bias = dt_bias.as<float>();
    w.o_norm = o_norm.p;
    w.o_proj = o_proj.p;
    return w;
  }
};

inline dgpp::kda_ref::HostWeights host_views(const TestWeights& tw) {
  dgpp::kda_ref::HostWeights w;
  w.in_proj = tw.in_proj.data();
  w.f_b = tw.f_b.data();
  w.g_b = tw.g_b.data();
  w.conv = tw.conv.data();
  w.a_log = tw.a_log.data();
  w.dt_bias = tw.dt_bias.data();
  w.o_norm = tw.o_norm.data();
  w.o_proj = tw.o_proj.data();
  return w;
}

// Arena-backed scratch for KdaLayer instances plus a shared GEMM workspace.
// `layer_count` layers may share the arena (e.g. multi-layer state tests).
struct LayerEnv {
  dgpp::Arena arena;
  dgpp::CublasLtGemm gemm;
  DevBuf ws;
  cudaStream_t stream = nullptr;

  LayerEnv(const KdaConfig& cfg, int max_tokens, cudaStream_t s,
           int layer_count = 1)
      : ws(64ull << 20) {
    dgpp::Arena::Config ac;
    ac.persistent_hot =
        KdaLayer::persistent_hot_bytes(cfg, max_tokens) * layer_count;
    arena.init(ac);
    stream = s;
  }
};

}  // namespace dgpp::kda_test
