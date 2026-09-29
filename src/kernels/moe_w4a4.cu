// W4A4 grouped MoE GEMM on GB10's native block-scaled tensor cores, in both
// FP4 scale dialects:
//  - NVFP4 (scale_vec::4X, ue4m3 scales + a per-row activation global and
//    per-matrix weight global) for the Qwen3.8 NVFP4 experts;
//  - MXFP4 (scale_vec::2X, ue8m0 scales per 32, no globals) for the 0731
//    experts (2026-09-29), whose checkpoint is authored in exactly this form.
//
// Why (2026-09-23): DGPP's moe_grouped_mma_fp4_ldm_kernel runs the NVFP4
// experts W4A16 (dequantize the e2m1 codes to bf16, HMMA m16n8k16), while
// prod SGLang runs them W4A4 through flashinfer_cutlass. sm_121a has
// mma.sync kind::mxf4nvf4.block_scale (OMMA.SF.16864): e2m1 x e2m1 with one
// ue4m3 scale per 16 elements, i.e. NVFP4 on both operands, at 4x the bf16
// instruction rate (bench/micro/mma_peak.cu: 194.6 vs 48.8 TFLOPS). The
// checkpoint is authored for W4A4 (73,728 expert input_scale tensors).
//
// Activations are quantized per row with a global gs -- the layer's static
// checkpoint input_scale (max over its experts, as SGLang's flashinfer_cutlass
// MoE does; values past 6 * 448 * gs clip, which the NVFP4 experts were
// calibrated with) or, when none is given, a dynamic amax / (6 * 448) -- and
// one e4m3 scale per 16 elements, bs = e4m3(amax16 / (6 * gs)); the codes are
// x / (bs * gs) rounded to e2m1 (cvt.rn.satfinite). x ~= e2m1 * bs * gs. The weights are DGPP's NVFP4 views unchanged: e2m1
// codes [n, k/2], e4m3 scales [n, k/16], value = e2m1 * s / g. So
//   out[row, col] = acc(row, col) * gs[row] / g.
//
// Fragment layouts (m16n8k64, PTX ISA 9.7.14.6; the same as upstream SGLang's
// validated sm120 kernel, jit/csrc/moe/nvfp4_moe_sm120.cuh): lane = 4q + t.
//   A (row-major 16 x 64 e2m1): a0 = row q bytes [4t, 4t+4), a1 = row q+8,
//      a2 = row q bytes [16 + 4t, ...), a3 = row q+8 likewise.
//   B (col-major 64 x 8): b0 = column q bytes [4t, 4t+4), b1 = +16 bytes.
//   sfa: the four ue4m3 scales (k blocks 0..3) of row q + 8 * (lane & 1);
//   sfb: the four scales of column q. Selectors zero (scale_vec::4X).
//   C: c0, c1 = row q cols 2t, 2t+1; c2, c3 = row q+8.
#include "kernels/moe_w4a4.hpp"

#include <cstdlib>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

namespace w4a4 {
constexpr int BN = 128, BK = 128;  // BK in elements: 64 code bytes per row
constexpr int kThreads = 256;
constexpr int kRowBytes = 80;  // 64 code bytes + 16 pad: conflict-free fragment reads
constexpr size_t kBCodes = size_t(BN) * kRowBytes;  // 10,240
// The row tile (2026-09-23): 64 or 128 rows. Stages: 3 at BM 64 (50,688 B)
// fit ONE block per GB10 SM; 2 fit two (DGPP_W4A4_STAGES, default 2).
// kScaleBytes: 8 (NVFP4, ue4m3 x 8 per row per stage), 4 (MXFP4, ue8m0 x 4).
template <int kBM, int kScaleBytes = 8>
struct Tile {
  static constexpr int scale_bytes = kScaleBytes;
  static constexpr size_t a_codes = size_t(kBM) * kRowBytes;
  static constexpr size_t a_scales = size_t(kBM) * kScaleBytes;
  static constexpr size_t b_scales = size_t(BN) * kScaleBytes;
  static constexpr size_t slot = a_codes + kBCodes + a_scales + b_scales;
  static constexpr int wm_warps = kBM / 32, wn_warps = 8 / wm_warps;
  static constexpr int ncols = BN / wn_warps, nt = ncols / 8;
};
template <int S, int kBM, int kScaleBytes = 8>
constexpr size_t smem_bytes_bm() { return S * Tile<kBM, kScaleBytes>::slot; }
}  // namespace w4a4

__device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes, int src_bytes) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
  else if (bytes == 8)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_nvfp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

// MXFP4 (2026-09-29, the 0731 experts): the same instruction with
// scale_vec::2X and ue8m0 scales -- one e8m0 per 32, so each scale register's
// two low bytes carry the k64 half's scales (byte-id 0; the PTX ISA's
// {byte-id, thread-id} selectors stay zero, the same thread set as 4X).
__device__ __forceinline__ void mma_mx4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}

__device__ __forceinline__ uint32_t f32x8_to_e2m1(const float* v) {
  uint32_t packed;
  asm volatile(
      "{\n.reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(packed)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return packed;
}

// kSwiglu (2026-09-23): the row is the MoE's gate/up pair, activated here --
// moe_swiglu_clamp_kernel's math and its two bf16 roundings, op for op -- so
// the down projection's NVFP4 activations come straight from gate/up without
// the bf16 act buffer's write and re-read (bitwise the two-kernel chain).
__device__ __forceinline__ float swiglu_bf16(uint16_t gb, uint16_t ub, float limit) {
  float gf = __uint_as_float(static_cast<uint32_t>(gb) << 16);
  float uf = __uint_as_float(static_cast<uint32_t>(ub) << 16);
  if (gf > limit) gf = limit;  // gate: no lower clamp (the reference's asymmetry)
  uf = fminf(fmaxf(uf, -limit), limit);
  const float silu = gf * (1.0f / (1.0f + expf(-gf)));
  const float t = __bfloat162float(__float2bfloat16_rn(silu));  // rounding 1 (silu)
  return __bfloat162float(__float2bfloat16_rn(t * uf));          // rounding 2 (product)
}

// One block per row; one thread per 16-element group (k <= 16 * blockDim).
template <bool kSwiglu>
__global__ void quantize_rows_nvfp4_kernel(const uint16_t* __restrict__ x, size_t x_stride, int rows, int k,
                                           uint8_t* __restrict__ codes, size_t code_stride,
                                           uint8_t* __restrict__ scales, size_t scale_stride,
                                           float* __restrict__ gs, float static_gs,
                                           const float* __restrict__ static_gs_dev,
                                           const uint16_t* __restrict__ up = nullptr, float limit = 0.f) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int groups = k / 16;
  const int g = threadIdx.x;
  float v[16];
  float amax = 0.f;
  if (g < groups) {
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * x_stride + g * 16);
    const uint4 w0 = src[0], w1 = src[1];
    const uint32_t w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    if constexpr (kSwiglu) {
      const uint4* us = reinterpret_cast<const uint4*>(up + static_cast<size_t>(row) * x_stride + g * 16);
      const uint4 u0 = us[0], u1 = us[1];
      const uint32_t u[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[2 * i] = swiglu_bf16(static_cast<uint16_t>(w[i] & 0xFFFFu), static_cast<uint16_t>(u[i] & 0xFFFFu), limit);
        v[2 * i + 1] = swiglu_bf16(static_cast<uint16_t>(w[i] >> 16), static_cast<uint16_t>(u[i] >> 16), limit);
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
      }
    }
  }
  // The row's amax: warp shuffles, then one word per warp.
  __shared__ float warp_max[32];
  float m = amax;
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = (blockDim.x + 31) >> 5;
    float t = threadIdx.x < nw ? warp_max[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    if (threadIdx.x == 0) warp_max[0] = t;
  }
  __syncthreads();
  const float row_amax = warp_max[0];
  // A static global (the checkpoint's input_scale) clips what exceeds
  // 6 * 448 * gs: the e4m3 block scale and the e2m1 codes both saturate.
  // The static global comes from device memory when given (the layer image's
  // copy of the checkpoint's input_scale), else from the argument.
  const float sg = static_gs_dev != nullptr ? *static_gs_dev : static_gs;
  const float global = sg > 0.f ? sg : row_amax > 0.f ? row_amax / (6.f * 448.f) : 1.f;
  if (threadIdx.x == 0) gs[row] = global;
  if (g >= groups) return;
  uint8_t* code_row = codes + static_cast<size_t>(row) * code_stride + g * 8;
  uint8_t* scale_row = scales + static_cast<size_t>(row) * scale_stride;
  const __nv_fp8_e4m3 s(amax / (6.f * global));
  const float bs = static_cast<float>(s);
  if (amax == 0.f || bs == 0.f) {
    scale_row[g] = 0;
    *reinterpret_cast<uint2*>(code_row) = make_uint2(0u, 0u);
    return;
  }
  scale_row[g] = s.__x;
  const float q = 1.f / (bs * global);
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= q;
  *reinterpret_cast<uint2*>(code_row) = make_uint2(f32x8_to_e2m1(v), f32x8_to_e2m1(v + 8));
}

// MXFP4 activation quantization (the 0731 chain, 2026-09-29): one e8m0 scale
// per 32 elements -- ceil-log2 of max(amax, 1e-4) / 448, the MX scheme --
// and e2m1(x / scale) codes, satfinite. No global scale. One block per row,
// one thread per 32-element group (k <= 32 * blockDim).
template <bool kSwiglu>
__global__ void quantize_rows_mx_kernel(const uint16_t* __restrict__ x, size_t x_stride, int rows, int k,
                                        uint8_t* __restrict__ codes, size_t code_stride,
                                        uint8_t* __restrict__ scales, size_t scale_stride,
                                        const uint16_t* __restrict__ up = nullptr, float limit = 0.f) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int groups = k / 32;
  const int g = threadIdx.x;
  float v[32];
  float amax = 0.f;
  if (g < groups) {
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * x_stride + g * 32);
    const uint4 w4[4] = {src[0], src[1], src[2], src[3]};
    const uint32_t w[16] = {w4[0].x, w4[0].y, w4[0].z, w4[0].w, w4[1].x, w4[1].y, w4[1].z, w4[1].w,
                            w4[2].x, w4[2].y, w4[2].z, w4[2].w, w4[3].x, w4[3].y, w4[3].z, w4[3].w};
    if constexpr (kSwiglu) {
      const uint4* us = reinterpret_cast<const uint4*>(up + static_cast<size_t>(row) * x_stride + g * 32);
      const uint4 u4[4] = {us[0], us[1], us[2], us[3]};
      const uint32_t u[16] = {u4[0].x, u4[0].y, u4[0].z, u4[0].w, u4[1].x, u4[1].y, u4[1].z, u4[1].w,
                              u4[2].x, u4[2].y, u4[2].z, u4[2].w, u4[3].x, u4[3].y, u4[3].z, u4[3].w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        v[2 * i] = swiglu_bf16(static_cast<uint16_t>(w[i] & 0xFFFFu), static_cast<uint16_t>(u[i] & 0xFFFFu), limit);
        v[2 * i + 1] =
            swiglu_bf16(static_cast<uint16_t>(w[i] >> 16), static_cast<uint16_t>(u[i] >> 16), limit);
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
      }
    }
  }
  if (g >= groups) return;
  const uint8_t sb = e8m0_ceil_log2_byte(std::max(amax, 1e-4f) * (1.f / 448.f));
  scales[static_cast<size_t>(row) * scale_stride + g] = sb;
  const float q = std::ldexp(1.f, 127 - static_cast<int>(sb));
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] *= q;
  uint8_t* code_row = codes + static_cast<size_t>(row) * code_stride + g * 16;
  const uint32_t p[4] = {f32x8_to_e2m1(v), f32x8_to_e2m1(v + 8), f32x8_to_e2m1(v + 16), f32x8_to_e2m1(v + 24)};
  *reinterpret_cast<uint4*>(code_row) = make_uint4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__ void store_out(uint16_t* p, float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  *p = *reinterpret_cast<const uint16_t*>(&b);
}
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_pair(uint16_t* p, float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  *reinterpret_cast<__nv_bfloat162*>(p) = v;
}
__device__ __forceinline__ void store_pair(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }

template <typename OutT, int kStages, int kBM, int kNT = 1>
__global__ __launch_bounds__(w4a4::kThreads, 2) void moe_grouped_w4a4_kernel(
    const uint8_t* __restrict__ a_codes, size_t a_code_stride, const uint8_t* __restrict__ a_scales,
    size_t a_scale_stride, const float* __restrict__ a_gs, const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views, int which,
    OutT* __restrict__ out, size_t out_stride, int n, int k, int m_tiles) {
  using namespace w4a4;
  using TL = Tile<kBM>;
  constexpr size_t kACodesT = TL::a_codes, kAScalesT = TL::a_scales, kSlotT = TL::slot;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  // kNT consecutive n-tiles per CTA: one cp.async ring runs across
  // the tiles, so a short-k GEMM (the down projection, k = 320: 2.5 stages a
  // tile) pays its pipeline prologue once per CTA, and each tile's epilogue
  // stores overlap the next tile's loads. kNT = 1 is the original kernel.
  const int n_group = static_cast<int>(blockIdx.x) / m_tiles;
  const int n_tiles_total = (n + BN - 1) / BN;
  const int t0 = n_group * kNT;
  const int tiles_here = min(kNT, n_tiles_total - t0);
  if (tiles_here <= 0) return;
  const int m0 = m_tile * kBM;
  if (m0 >= seg.rows) return;
  const int m_rows = min(kBM, seg.rows - m0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const float inv_g = 1.f / *v.fp4_global;
  const size_t w_code_stride = static_cast<size_t>(k) / 2;
  const size_t w_scale_stride = static_cast<size_t>(k) / 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int q = lane >> 2, t = lane & 3;
  const int wm = warp / TL::wn_warps, wn = warp % TL::wn_warps;  // (kBM/32) x (8 / that) warps
  const int stages = (k + BK - 1) / BK;

  // Per-thread copy assignments (fixed across stages).
  // A codes: kBM / 64 passes of (row tid/4 + 64 p, 16-byte chunk tid%4).
  constexpr int kAPasses = kBM / 64;
  const int a_chunk = tid & 3;
  int a_row[kAPasses];
  bool a_ok[kAPasses];
  const uint8_t* a_code_src[kAPasses];
#pragma unroll
  for (int p = 0; p < kAPasses; ++p) {
    a_row[p] = (tid >> 2) + 64 * p;
    a_ok[p] = a_row[p] < m_rows;
    int src_row = 0;
    if (a_ok[p]) {
      const int srow = seg.row0 + m0 + a_row[p];
      src_row = act_rows != nullptr ? act_rows[srow] : srow;
    }
    a_code_src[p] = a_codes + static_cast<size_t>(src_row) * a_code_stride;
  }
  // A scales: threads 0..63, one 8-byte copy each (row tid).
  int as_src_row = 0;
  const bool as_ok = tid < kBM && tid < m_rows;
  if (as_ok) {
    const int srow = seg.row0 + m0 + tid;
    as_src_row = act_rows != nullptr ? act_rows[srow] : srow;
  }
  const uint8_t* a_scale_src = a_scales + static_cast<size_t>(as_src_row) * a_scale_stride;
  // B codes: two 16-byte chunks per thread: rows (tid + 256 i) / 4.
  // B scales: row tid/2, 4-byte half tid%2.
  const int bs_row = tid >> 1, bs_half = tid & 1;

  auto slot_ptr = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotT; };

  // f: the flattened (tile, stage) index across this CTA's tiles.
  auto issue = [&](int f, int slot) {
    const int s = f % stages;
    const int n0 = (t0 + f / stages) * BN;
    const bool bs_ok = n0 + bs_row < n;
    const uint8_t* b_scale_src = v.fp4_scales + static_cast<size_t>(bs_ok ? n0 + bs_row : 0) * w_scale_stride;
    uint8_t* base = slot_ptr(slot);
    const int kb0 = s * (BK / 2);  // code byte offset of this stage
    const int kg0 = s * (BK / 16); // scale group offset
#pragma unroll
    for (int p = 0; p < kAPasses; ++p) {
      const int gb = kb0 + a_chunk * 16;
      const int in = a_ok[p] ? max(0, min(16, k / 2 - gb)) : 0;
      cp_async(base + a_row[p] * kRowBytes + a_chunk * 16, in > 0 ? a_code_src[p] + gb : a_codes, 16, in);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + kThreads * i;
      const int row = idx >> 2, chunk = idx & 3;
      const int gn = n0 + row;
      const int gb = kb0 + chunk * 16;
      const int in = gn < n ? max(0, min(16, k / 2 - gb)) : 0;
      const uint8_t* src = v.payload + static_cast<size_t>(gn < n ? gn : 0) * w_code_stride + gb;
      cp_async(base + kACodesT + row * kRowBytes + chunk * 16, in > 0 ? src : v.payload, 16, in);
    }
    if (tid < kBM) {
      // A plain load + store: a small (8-byte) cp.async issued by 128 threads
      // drops data on sm_121a (2026-09-29 probe) -- the MX twin copies its
      // scales the same way.
      const int in = as_ok ? max(0, min(8, k / 16 - kg0)) : 0;
      uint32_t lo = 0, hi = 0;
      const uint8_t* src = in > 0 ? a_scale_src + kg0 : a_scales;
#pragma unroll
      for (int b = 0; b < 4; ++b) lo |= static_cast<uint32_t>(b < in ? src[b] : 0) << (8 * b);
#pragma unroll
      for (int b = 4; b < 8; ++b) hi |= static_cast<uint32_t>((b - 4) < (in - 4) ? src[b] : 0) << (8 * (b - 4));
      uint32_t* dst = reinterpret_cast<uint32_t*>(base + kACodesT + kBCodes + tid * TL::scale_bytes);
      dst[0] = lo, dst[1] = hi;
    }
    {
      // A plain load + store: a 4-byte cp.async issued by all 256 threads
      // drops data on sm_121a (2026-09-29 probe) -- the MX twin's 4-byte
      // copies are plain loads for the same reason.
      const int gg = kg0 + bs_half * 4;
      const int in = bs_ok ? max(0, min(4, k / 16 - gg)) : 0;
      uint32_t word = 0;
      const uint8_t* src = in > 0 ? b_scale_src + gg : v.fp4_scales;
#pragma unroll
      for (int b = 0; b < 4; ++b) word |= static_cast<uint32_t>(b < in ? src[b] : 0) << (8 * b);
      *reinterpret_cast<uint32_t*>(base + kACodesT + kBCodes + kAScalesT + bs_row * TL::scale_bytes +
                                           bs_half * 4) = word;
    }
  };

  constexpr int NT = TL::nt, NC = TL::ncols;
  float acc[2][NT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int total = tiles_here * stages;
#pragma unroll
  for (int f = 0; f < kStages - 1; ++f) {
    if (f < total) issue(f, f);
    cp_commit();
  }
  for (int f = 0; f < total; ++f) {
    const int s = f % stages;
    cp_wait<kStages - 2>();
    __syncthreads();
    {
      const int nx = f + kStages - 1;
      if (nx < total) issue(nx, nx % kStages);
      cp_commit();
    }
    const uint8_t* base = slot_ptr(f % kStages);
    const uint8_t* ab = base;
    const uint8_t* bb = base + kACodesT;
    const uint32_t* asw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes);
    const uint32_t* bsw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes + kAScalesT);
    // ldmatrix.x4 lane addressing: lane l feeds row (l & 7) of matrix l >> 3.
    const int lm = lane >> 3, lr = lane & 7;
    // The stage's block-scale words for both k64 halves in one 8-byte load per
    // row (2026-09-23: ncu put the kernel MIO-throttled -- 20 scale LDS per
    // stage per warp beside 12 ldmatrix for 32 MMAs; this halves the scale LDS).
    uint2 sfa2[2], sfb2[NT];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int sr = wm * 32 + i * 16 + q + ((lane & 1) << 3);
      sfa2[i] = *reinterpret_cast<const uint2*>(asw + sr * 2);
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) sfb2[j] = *reinterpret_cast<const uint2*>(bsw + (wn * NC + j * 8 + q) * 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {  // two k64 steps per stage
      uint32_t af[2][4], sfa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        // matrices: rows +0/+8 (lm & 1) x bytes +0/+16 (lm >> 1) -> a0, a1, a2, a3.
        const int row = wm * 32 + i * 16 + (lm & 1) * 8 + lr;
        ldmatrix_x4(af[i], ab + row * kRowBytes + kh * 32 + (lm >> 1) * 16);
        sfa[i] = kh == 0 ? sfa2[i].x : sfa2[i].y;
      }
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        // matrices: bytes +0/+16 (lm & 1) x n-tiles j/j+1 (lm >> 1) -> b0, b1 of each.
        uint32_t bf[4];
        const int col = wn * NC + jp * 16 + (lm >> 1) * 8 + lr;
        ldmatrix_x4(bf, bb + col * kRowBytes + kh * 32 + (lm & 1) * 16);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = jp * 2 + h;
          const uint32_t sfb = kh == 0 ? sfb2[j].x : sfb2[j].y;
#pragma unroll
          for (int i = 0; i < 2; ++i) mma_nvfp4(acc[i][j], af[i], bf[2 * h], bf[2 * h + 1], sfa[i], sfb);
        }
      }
    }
    if (s == stages - 1) {  // this tile's last stage: store it, clear for the next
      const int n0 = (t0 + f / stages) * BN;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = wm * 32 + i * 16 + q + h * 8;
          if (mm >= m_rows) continue;
          const int srow = seg.row0 + m0 + mm;
          const float scale = a_gs[act_rows != nullptr ? act_rows[srow] : srow] * inv_g;
          OutT* orow = out + static_cast<size_t>(srow) * out_stride;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * NC + j * 8 + 2 * t;
            // Pairs as one 4-byte (bf16x2) / 8-byte (float2) store: col is even
            // and rows are even-strided (the element-at-a-time stores made the
            // output-heavy down projection store-instruction bound).
            if (col + 1 < n) {
              store_pair(orow + col, acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale);
            } else if (col < n) {
              store_out(orow + col, acc[i][j][2 * h] * scale);
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    }
  }
  cp_wait<0>();
}

}  // namespace

// The MXFP4 twin (2026-09-29): the same tile machinery, e8m0 scales (4 bytes
// per row per stage, no global scale on either operand, the epilogue stores
// the raw accumulator), mma_mx4.
template <typename OutT, int kStages, int kBM, int kNT = 1>
__global__ __launch_bounds__(w4a4::kThreads, 2) void moe_grouped_w4a4_mx_kernel(
    const uint8_t* __restrict__ a_codes, size_t a_code_stride, const uint8_t* __restrict__ a_scales,
    size_t a_scale_stride, const int32_t* __restrict__ act_rows, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int m_tiles) {
  using namespace w4a4;
  using TL = Tile<kBM, 4>;
  constexpr size_t kACodesT = TL::a_codes, kAScalesT = TL::a_scales, kSlotT = TL::slot;
  extern __shared__ __align__(16) uint8_t smem[];
  const MoeSegment seg = segs[blockIdx.y];
  const int m_tile = static_cast<int>(blockIdx.x) % m_tiles;
  const int n_group = static_cast<int>(blockIdx.x) / m_tiles;
  const int n_tiles_total = (n + BN - 1) / BN;
  const int t0 = n_group * kNT;
  const int tiles_here = min(kNT, n_tiles_total - t0);
  if (tiles_here <= 0) return;
  const int m0 = m_tile * kBM;
  if (m0 >= seg.rows) return;
  const int m_rows = min(kBM, seg.rows - m0);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const size_t w_code_stride = static_cast<size_t>(k) / 2;
  const size_t w_scale_stride = static_cast<size_t>(k) / 32;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int q = lane >> 2, t = lane & 3;
  const int wm = warp / TL::wn_warps, wn = warp % TL::wn_warps;  // (kBM/32) x (8 / that) warps
  const int stages = (k + BK - 1) / BK;

  constexpr int kAPasses = kBM / 64;
  const int a_chunk = tid & 3;
  int a_row[kAPasses];
  bool a_ok[kAPasses];
  const uint8_t* a_code_src[kAPasses];
#pragma unroll
  for (int p = 0; p < kAPasses; ++p) {
    a_row[p] = (tid >> 2) + 64 * p;
    a_ok[p] = a_row[p] < m_rows;
    int src_row = 0;
    if (a_ok[p]) {
      const int srow = seg.row0 + m0 + a_row[p];
      src_row = act_rows != nullptr ? act_rows[srow] : srow;
    }
    a_code_src[p] = a_codes + static_cast<size_t>(src_row) * a_code_stride;
  }
  // A scales: one 4-byte copy per row (ue8m0 x 4 per stage).
  int as_src_row = 0;
  const bool as_ok = tid < kBM && tid < m_rows;
  if (as_ok) {
    const int srow = seg.row0 + m0 + tid;
    as_src_row = act_rows != nullptr ? act_rows[srow] : srow;
  }
  const uint8_t* a_scale_src = a_scales + static_cast<size_t>(as_src_row) * a_scale_stride;

  auto slot_ptr = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotT; };

  auto issue = [&](int f, int slot) {
    const int s = f % stages;
    const int n0 = (t0 + f / stages) * BN;
    const uint8_t* b_scale_src = v.fp4_scales + static_cast<size_t>(min(n0 + tid, n - 1)) * w_scale_stride;
    uint8_t* base = slot_ptr(slot);
    const int kb0 = s * (BK / 2);  // code byte offset of this stage
    const int kg0 = s * (BK / 32); // scale group offset
#pragma unroll
    for (int p = 0; p < kAPasses; ++p) {
      const int gb = kb0 + a_chunk * 16;
      const int in = a_ok[p] ? max(0, min(16, k / 2 - gb)) : 0;
      cp_async(base + a_row[p] * kRowBytes + a_chunk * 16, in > 0 ? a_code_src[p] + gb : a_codes, 16, in);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + kThreads * i;
      const int row = idx >> 2, chunk = idx & 3;
      const int gn = n0 + row;
      const int gb = kb0 + chunk * 16;
      const int in = gn < n ? max(0, min(16, k / 2 - gb)) : 0;
      const uint8_t* src = v.payload + static_cast<size_t>(gn < n ? gn : 0) * w_code_stride + gb;
      cp_async(base + kACodesT + row * kRowBytes + chunk * 16, in > 0 ? src : v.payload, 16, in);
    }
    if (as_ok) {
      // A plain load + store: a 4-byte cp.async issued by all 256 threads
      // drops data on sm_121a (2026-09-29: the high half of the word arrived
      // zero, exactly halving every row on constant activations).
      const int in = max(0, min(4, k / 32 - kg0));
      uint32_t word = 0;
      const uint8_t* src = in > 0 ? a_scale_src + kg0 : a_scales;
#pragma unroll
      for (int b = 0; b < 4; ++b) word |= static_cast<uint32_t>(b < in ? src[b] : 0) << (8 * b);
      *reinterpret_cast<uint32_t*>(base + kACodesT + kBCodes + tid * 4) = word;
    }
    if (tid < BN) {
      const int gn = n0 + tid;
      const int in = gn < n ? max(0, min(4, k / 32 - kg0)) : 0;
      // A plain load + store: the checkpoint's scale row is k/32 bytes wide
      // (a multiple of 2 only -- k = 320 leaves the odd columns 2 bytes off,
      // and cp.async needs a 4-byte-aligned source), and the 4-byte cp.async
      // form is the one sm_121a drops under 256-thread issue.
      uint32_t word = 0;
      const uint8_t* src = in > 0 ? b_scale_src + kg0 : v.fp4_scales;
#pragma unroll
      for (int b = 0; b < 4; ++b) word |= static_cast<uint32_t>(b < in ? src[b] : 0) << (8 * b);
      *reinterpret_cast<uint32_t*>(base + kACodesT + kBCodes + kAScalesT + tid * 4) = word;
    }
  };

  constexpr int NT = TL::nt, NC = TL::ncols;
  float acc[2][NT][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int total = tiles_here * stages;
#pragma unroll
  for (int f = 0; f < kStages - 1; ++f) {
    if (f < total) issue(f, f);
    cp_commit();
  }
  for (int f = 0; f < total; ++f) {
    const int s = f % stages;
    cp_wait<kStages - 2>();
    __syncthreads();
    {
      const int nx = f + kStages - 1;
      if (nx < total) issue(nx, nx % kStages);
      cp_commit();
    }
    const uint8_t* base = slot_ptr(f % kStages);
    const uint8_t* ab = base;
    const uint8_t* bb = base + kACodesT;
    const uint32_t* asw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes);
    const uint32_t* bsw = reinterpret_cast<const uint32_t*>(base + kACodesT + kBCodes + kAScalesT);
    // The stage's e8m0 scale words: 4 bytes per row (two k64 halves), the low
    // word's two bytes for the first half, the high word's for the second.
    uint32_t sfa2[2], sfb2[NT];
#pragma unroll
    for (int i = 0; i < 2; ++i)
      sfa2[i] = asw[wm * 32 + i * 16 + q + ((lane & 1) << 3)];
#pragma unroll
    for (int j = 0; j < NT; ++j) sfb2[j] = bsw[wn * NC + j * 8 + q];
    // ldmatrix.x4 lane addressing: lane l feeds row (l & 7) of matrix l >> 3.
    const int lm = lane >> 3, lr = lane & 7;
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {  // two k64 steps per stage
      uint32_t af[2][4], sfa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int row = wm * 32 + i * 16 + (lm & 1) * 8 + lr;
        ldmatrix_x4(af[i], ab + row * kRowBytes + kh * 32 + (lm >> 1) * 16);
        sfa[i] = kh == 0 ? sfa2[i] & 0xFFFFu : sfa2[i] >> 16;
      }
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        uint32_t bf[4];
        const int col = wn * NC + jp * 16 + (lm >> 1) * 8 + lr;
        ldmatrix_x4(bf, bb + col * kRowBytes + kh * 32 + (lm & 1) * 16);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int j = jp * 2 + h;
          const uint32_t sfb = kh == 0 ? sfb2[j] & 0xFFFFu : sfb2[j] >> 16;
#pragma unroll
          for (int i = 0; i < 2; ++i) mma_mx4(acc[i][j], af[i], bf[2 * h], bf[2 * h + 1], sfa[i], sfb);
        }
      }
    }
    if (s == stages - 1) {  // this tile's last stage: store it, clear for the next
      const int n0 = (t0 + f / stages) * BN;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = wm * 32 + i * 16 + q + h * 8;
          if (mm >= m_rows) continue;
          const int srow = seg.row0 + m0 + mm;
          OutT* orow = out + static_cast<size_t>(srow) * out_stride;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * NC + j * 8 + 2 * t;
            if (col + 1 < n)
              store_pair(orow + col, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            else if (col < n)
              store_out(orow + col, acc[i][j][2 * h]);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    }
  }
  cp_wait<0>();
}

size_t nvfp4_act_scale_stride(int k) { return static_cast<size_t>(((k / 16) + 7) / 8 * 8); }
size_t mxfp4_act_scale_stride(int k) { return static_cast<size_t>(((k / 32) + 3) / 4 * 4); }

void launch_quantize_rows_nvfp4(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, float* gs, cudaStream_t stream, float static_gs,
                                const float* static_gs_dev) {
  if (rows <= 0) return;
  if (k % 64 != 0 || k / 16 > 1024) throw std::invalid_argument("quantize nvfp4: k must be a multiple of 64, <= 16384");
  if (x_stride % 8 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
    throw std::invalid_argument("quantize nvfp4: rows must be 16-byte aligned");
  const int threads = ((k / 16) + 31) / 32 * 32;
  quantize_rows_nvfp4_kernel<false><<<rows, threads, 0, stream>>>(x, x_stride, rows, k, codes,
                                                                   static_cast<size_t>(k) / 2, scales,
                                                                   nvfp4_act_scale_stride(k), gs, static_gs, static_gs_dev);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_swiglu_quantize_rows_nvfp4(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                       float limit, uint8_t* codes, uint8_t* scales, float* gs,
                                       cudaStream_t stream, float static_gs, const float* static_gs_dev) {
  if (rows <= 0) return;
  if (k % 64 != 0 || k / 16 > 1024) throw std::invalid_argument("swiglu quantize nvfp4: k % 64, <= 16384");
  if (stride % 8 != 0 || reinterpret_cast<uintptr_t>(gate) % 16 != 0 || reinterpret_cast<uintptr_t>(up) % 16 != 0)
    throw std::invalid_argument("swiglu quantize nvfp4: rows must be 16-byte aligned");
  const int threads = ((k / 16) + 31) / 32 * 32;
  quantize_rows_nvfp4_kernel<true><<<rows, threads, 0, stream>>>(gate, stride, rows, k, codes,
                                                                  static_cast<size_t>(k) / 2, scales,
                                                                  nvfp4_act_scale_stride(k), gs, static_gs, static_gs_dev, up, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
static void launch_w4a4(const uint8_t* codes, const uint8_t* scales, const float* gs, const int32_t* act_rows,
                        const MoeSegment* segs, int n_segs, int max_rows, const MoeExpertView* views, int which,
                        OutT* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  using namespace w4a4;
  if (n_segs <= 0 || n <= 0) return;
  if (k % 64 != 0) throw std::invalid_argument("moe w4a4: k must be a multiple of 64");
  if (max_rows <= 0) throw std::invalid_argument("moe w4a4: max_rows");
  // The epilogue stores column pairs (bf16x2 / float2): rows must be even-strided and aligned.
  if (out_stride % 2 != 0 || (reinterpret_cast<uintptr_t>(out) % (2 * sizeof(OutT))) != 0)
    throw std::invalid_argument("moe w4a4: out rows must be even-strided and pair-aligned");
  static const int stages = [] {
    const char* e = std::getenv("DGPP_W4A4_STAGES");
    return e != nullptr && std::atoi(e) == 3 ? 3 : 2;
  }();
  // DGPP_W4A4_BM=64|128 (default 128; idle GPU, moe_w4a4_test at 8,192
  // tokens x top-10: gate/up 3.40 -> 3.27 ms, down 5.27 -> 4.89 ms).
  static const int bm = [] {
    const char* e = std::getenv("DGPP_W4A4_BM");
    return e != nullptr && std::atoi(e) == 64 ? 64 : 128;
  }();
  static const bool opted = [] {  // once per process and OutT, thread-safe
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 64>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<2, 64>())));
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 3, 64>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<3, 64>())));
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes_bm<2, 128>())));
    return true;
  }();
  (void)opted;
  const int m_tiles = (max_rows + bm - 1) / bm;
  const unsigned n_tiles = static_cast<unsigned>((n + BN - 1) / BN);
  const size_t cs = static_cast<size_t>(k) / 2, ss = nvfp4_act_scale_stride(k);
  // n-tiles per CTA (DGPP_W4A4_NT=1|2|4; measured neutral, default 1).
  static const int nt_env = [] {
    const char* e = std::getenv("DGPP_W4A4_NT");
    const int v = e != nullptr ? std::atoi(e) : 1;
    return v == 2 || v == 4 ? v : 1;
  }();
  if (bm == 128 && nt_env > 1) {
    static const bool opted_nt = [] {
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128, 2>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes_bm<2, 128>())));
      DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_kernel<OutT, 2, 128, 4>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes_bm<2, 128>())));
      return true;
    }();
    (void)opted_nt;
    const unsigned groups = (n_tiles + nt_env - 1) / nt_env;
    const dim3 g2(groups * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
    if (nt_env == 2)
      moe_grouped_w4a4_kernel<OutT, 2, 128, 2><<<g2, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
          codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
    else
      moe_grouped_w4a4_kernel<OutT, 2, 128, 4><<<g2, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
          codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const dim3 grid(n_tiles * static_cast<unsigned>(m_tiles), static_cast<unsigned>(n_segs), 1u);
  if (bm == 128)
    moe_grouped_w4a4_kernel<OutT, 2, 128><<<grid, kThreads, smem_bytes_bm<2, 128>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  else if (stages == 3)
    moe_grouped_w4a4_kernel<OutT, 3, 64><<<grid, kThreads, smem_bytes_bm<3, 64>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  else
    moe_grouped_w4a4_kernel<OutT, 2, 64><<<grid, kThreads, smem_bytes_bm<2, 64>(), stream>>>(
        codes, cs, scales, ss, gs, act_rows, segs, views, which, out, out_stride, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_grouped_w4a4_bf16(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                  const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                  const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                  int k, cudaStream_t stream) {
  launch_w4a4<uint16_t>(codes, scales, gs, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                        stream);
}
void launch_moe_grouped_w4a4_f32(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                 const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                 const MoeExpertView* views, int which, float* out, size_t out_stride, int n, int k,
                                 cudaStream_t stream) {
  launch_w4a4<float>(codes, scales, gs, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                     stream);
}

void launch_quantize_rows_mx(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                             uint8_t* scales, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k % 32 != 0 || k / 32 > 1024) throw std::invalid_argument("quantize mx: k must be a multiple of 32, <= 32768");
  if (x_stride % 32 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0)
    throw std::invalid_argument("quantize mx: rows must be 16-byte aligned");
  const int threads = ((k / 32) + 31) / 32 * 32;
  quantize_rows_mx_kernel<false><<<rows, threads, 0, stream>>>(x, x_stride, rows, k, codes,
                                                               static_cast<size_t>(k) / 2, scales,
                                                               mxfp4_act_scale_stride(k));
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_swiglu_quantize_rows_mx(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                    float limit, uint8_t* codes, uint8_t* scales, cudaStream_t stream) {
  if (rows <= 0) return;
  if (k % 32 != 0 || k / 32 > 1024) throw std::invalid_argument("swiglu quantize mx: k % 32, <= 32768");
  if (stride % 32 != 0 || reinterpret_cast<uintptr_t>(gate) % 16 != 0 || reinterpret_cast<uintptr_t>(up) % 16 != 0)
    throw std::invalid_argument("swiglu quantize mx: rows must be 16-byte aligned");
  const int threads = ((k / 32) + 31) / 32 * 32;
  quantize_rows_mx_kernel<true><<<rows, threads, 0, stream>>>(gate, stride, rows, k, codes,
                                                              static_cast<size_t>(k) / 2, scales,
                                                              mxfp4_act_scale_stride(k), up, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
static void launch_w4a4_mx(const uint8_t* codes, const uint8_t* scales, const int32_t* act_rows,
                           const MoeSegment* segs, int n_segs, int max_rows, const MoeExpertView* views, int which,
                           OutT* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  using namespace w4a4;
  if (n_segs <= 0 || n <= 0) return;
  if (k % 64 != 0) throw std::invalid_argument("moe w4a4 mx: k must be a multiple of 64");
  if (max_rows <= 0) throw std::invalid_argument("moe w4a4 mx: max_rows");
  if (out_stride % 2 != 0 || (reinterpret_cast<uintptr_t>(out) % (2 * sizeof(OutT))) != 0)
    throw std::invalid_argument("moe w4a4 mx: out rows must be even-strided and pair-aligned");
  static const int bm = [] {
    const char* e = std::getenv("DGPP_W4A4_BM");
    return e != nullptr && std::atoi(e) == 64 ? 64 : 128;
  }();
  static const bool opted = [] {  // once per process and OutT, thread-safe
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_mx_kernel<OutT, 2, 64, 1>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_bytes_bm<2, 64, 4>())));
    DGPP_CUDA_OK(cudaFuncSetAttribute(moe_grouped_w4a4_mx_kernel<OutT, 2, 128, 1>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_bytes_bm<2, 128, 4>())));
    return true;
  }();
  (void)opted;
  const int m_tiles = (max_rows + bm - 1) / bm;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN) * static_cast<unsigned>(m_tiles),
                  static_cast<unsigned>(n_segs), 1u);
  if (bm == 128)
    moe_grouped_w4a4_mx_kernel<OutT, 2, 128, 1><<<grid, kThreads, smem_bytes_bm<2, 128, 4>(), stream>>>(
        codes, static_cast<size_t>(k) / 2, scales, mxfp4_act_scale_stride(k), act_rows, segs, views, which, out,
        out_stride, n, k, m_tiles);
  else
    moe_grouped_w4a4_mx_kernel<OutT, 2, 64, 1><<<grid, kThreads, smem_bytes_bm<2, 64, 4>(), stream>>>(
        codes, static_cast<size_t>(k) / 2, scales, mxfp4_act_scale_stride(k), act_rows, segs, views, which, out,
        out_stride, n, k, m_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_grouped_w4a4_mx_bf16(const uint8_t* codes, const uint8_t* scales, const int32_t* act_rows,
                                     const MoeSegment* segs, int n_segs, int max_rows,
                                     const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream) {
  launch_w4a4_mx<uint16_t>(codes, scales, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                           stream);
}
void launch_moe_grouped_w4a4_mx_f32(const uint8_t* codes, const uint8_t* scales, const int32_t* act_rows,
                                    const MoeSegment* segs, int n_segs, int max_rows,
                                    const MoeExpertView* views, int which, float* out, size_t out_stride, int n,
                                    int k, cudaStream_t stream) {
  launch_w4a4_mx<float>(codes, scales, act_rows, segs, n_segs, max_rows, views, which, out, out_stride, n, k,
                        stream);
}

}  // namespace dgpp
