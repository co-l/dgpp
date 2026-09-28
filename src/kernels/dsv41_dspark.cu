#include "kernels/dsv41_dspark.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

__global__ void stream_mean_kernel(const uint16_t* __restrict__ streams, int hc_mult, int hidden, int rows,
                                   uint16_t* __restrict__ out, int64_t out_stride) {
  const int t = blockIdx.x;
  if (t >= rows) return;
  const uint16_t* x = streams + static_cast<size_t>(t) * static_cast<size_t>(hc_mult) * hidden;
  const float inv = 1.0f / static_cast<float>(hc_mult);
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j < hc_mult; ++j) s += bf16_bits_to_float(x[static_cast<size_t>(j) * hidden + d]);
    out[static_cast<size_t>(t) * out_stride + d] = float_to_bf16_bits(s * inv);
  }
}

__global__ void block_rows_kernel(const int64_t* __restrict__ step_pos, const int64_t* __restrict__ tokens,
                                  const int32_t* __restrict__ req_ids, int groups, int rows_per_group, int block,
                                  int64_t noise_id, int64_t* __restrict__ pos_out, int64_t* __restrict__ tok_out,
                                  int32_t* __restrict__ req_out, int32_t* __restrict__ spans_out) {
  const int g = blockIdx.x;
  if (g >= groups) return;
  // One block per group; thread 0 scans the group's rows (rows_per_group <= 32).
  __shared__ int64_t s_p;
  __shared__ int64_t s_next;
  if (threadIdx.x == 0) {
    int64_t best = -1;
    int64_t next = noise_id;
    for (int r = 0; r < rows_per_group; ++r) {
      const int64_t p = step_pos[g * rows_per_group + r];
      if (p > best) {
        best = p;
        next = tokens[g * rows_per_group + r];
      }
    }
    s_p = best < 0 ? -1 : best + 1;
    s_next = next;
    spans_out[g] = g * block;
    if (g == 0) spans_out[groups] = groups * block;
  }
  __syncthreads();
  for (int k = threadIdx.x; k < block; k += blockDim.x) {
    const int row = g * block + k;
    pos_out[row] = s_p < 0 ? -1 : s_p + k;
    tok_out[row] = k == 0 ? s_next : noise_id;
    req_out[row] = req_ids[g * rows_per_group];
  }
}

constexpr int kMaxRank = 512;

// Warp-per-row GEMV: lane `lig` owns a 16-byte (8-element) slice of the
// row's rank, the dot is its FMAs plus the fixed xor tree across the row's
// lane group (rank a multiple of 8, <= 512: one slice per lane at
// rank <= 256, two at 512). The e-vector is the shared form's.
__global__ void markov_bias_kernel(const float* __restrict__ base, int64_t base_group_stride, int block_row,
                                   const uint16_t* __restrict__ markov_embed, const uint16_t* __restrict__ markov_head,
                                   int rank, int vocab_begin, int count, const int64_t* __restrict__ tok,
                                   int tok_stride, float* __restrict__ out, int64_t out_group_stride, int rows_out) {
  const int g = blockIdx.y;
  __shared__ float e[kMaxRank];
  // A padding group's token (-1: a closed slot) biases nothing.
  const int64_t t = tok[static_cast<size_t>(g) * tok_stride];
  for (int r = threadIdx.x; r < rank; r += blockDim.x)
    e[r] = t < 0 ? 0.f : bf16_bits_to_float(markov_embed[static_cast<size_t>(t) * rank + r]);
  __syncthreads();
  const int lpr = rank < 256 ? rank / 8 : 32;  // lanes per row (a power of two)
  const int chunks = rank / (lpr * 8);         // 8-element slices per lane (1 or 2)
  const int rows_per_warp = 32 / lpr;
  const int row = blockIdx.x * (8 * rows_per_warp) + (threadIdx.x / 32) * rows_per_warp +
                  (threadIdx.x % 32) / lpr;
  if (row >= count) return;
  const int lig = threadIdx.x % lpr;
  const uint16_t* h = markov_head + static_cast<size_t>(vocab_begin + row) * rank + lig * 8;
  float acc = 0.f;
  if (t >= 0) {
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      if (ch >= chunks) break;
      const uint4 w = *reinterpret_cast<const uint4*>(h + static_cast<size_t>(ch) * lpr * 8);
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint16_t p0 = static_cast<uint16_t>(ws[q] & 0xFFFFu);
        const uint16_t p1 = static_cast<uint16_t>(ws[q] >> 16);
        const int r0 = ch * lpr * 8 + lig * 8 + 2 * q;
        acc = fmaf(e[r0], bf16_bits_to_float(p0), acc);
        acc = fmaf(e[r0 + 1], bf16_bits_to_float(p1), acc);
      }
    }
    for (int off = lpr / 2; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  }
  const float b = base[static_cast<size_t>(g) * base_group_stride + static_cast<size_t>(block_row) * count + row];
  const float val = b + acc;
  float* o = out + static_cast<size_t>(g) * out_group_stride + row;
  if (lig == 0)
    for (int j = 0; j < rows_out; ++j) o[static_cast<size_t>(j) * count] = val;
}

__global__ void confidence_kernel(const uint16_t* __restrict__ x, int64_t x_group_stride, int block_row, int hidden,
                                  const uint16_t* __restrict__ markov_embed, int rank, const int64_t* __restrict__ tok,
                                  int tok_stride, const float* __restrict__ w, float* __restrict__ conf_out,
                                  int conf_stride) {
  const int g = blockIdx.x;
  const uint16_t* xr = x + static_cast<size_t>(g) * x_group_stride + static_cast<size_t>(block_row) * hidden;
  const int64_t t = tok[static_cast<size_t>(g) * tok_stride];
  const uint16_t* e = markov_embed + static_cast<size_t>(t < 0 ? 0 : t) * rank;
  float acc = 0.f;
  if (t >= 0) {
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) acc = fmaf(w[d], bf16_bits_to_float(xr[d]), acc);
    for (int r = threadIdx.x; r < rank; r += blockDim.x) acc = fmaf(w[hidden + r], bf16_bits_to_float(e[r]), acc);
  }
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) conf_out[static_cast<size_t>(g) * conf_stride] = red[0];
}

}  // namespace

void dsv41_stream_mean_bf16(const uint16_t* streams, int hc_mult, int hidden, int rows, uint16_t* out,
                            int64_t out_stride, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!streams || !out || hc_mult <= 0 || hidden <= 0 || out_stride < hidden)
    throw std::invalid_argument("dsv41_stream_mean_bf16: arguments");
  stream_mean_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(streams, hc_mult, hidden, rows, out, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                             int rows_per_group, int block, int64_t noise_id, int64_t* pos_out, int64_t* tok_out,
                             int32_t* req_out, int32_t* spans_out, cudaStream_t stream) {
  if (!step_pos || !tokens || !req_ids || !pos_out || !tok_out || !req_out || !spans_out)
    throw std::invalid_argument("dsv41_dspark_block_rows: null buffer");
  if (groups <= 0 || rows_per_group <= 0 || rows_per_group > 32 || block <= 0 || block > 32)
    throw std::invalid_argument("dsv41_dspark_block_rows: shape");
  block_rows_kernel<<<static_cast<unsigned>(groups), 32, 0, stream>>>(step_pos, tokens, req_ids, groups, rows_per_group,
                                                                       block, noise_id, pos_out, tok_out, req_out,
                                                                       spans_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_markov_bias(const float* base, int64_t base_group_stride, int block_row, const uint16_t* markov_embed,
                              const uint16_t* markov_head, int rank, int vocab_begin, int count, const int64_t* tok,
                              int tok_stride, int groups, float* out, int64_t out_group_stride, int rows_out,
                              cudaStream_t stream) {
  if (!base || !markov_embed || !markov_head || !tok || !out) throw std::invalid_argument("dsv41_dspark_markov_bias: null buffer");
  if (rank <= 0 || rank > kMaxRank || count <= 0 || groups <= 0 || rows_out <= 0 || block_row < 0 || tok_stride <= 0)
    throw std::invalid_argument("dsv41_dspark_markov_bias: shape");
  if (rank % 8 != 0 || (rank < 256 && (rank / 8) & (rank / 8 - 1)) != 0)
    throw std::invalid_argument("dsv41_dspark_markov_bias: rank must be 8 x a power of two");
  const int lpr = rank < 256 ? rank / 8 : 32;
  const int rows_per_block = 8 * (32 / lpr);
  const dim3 grid(static_cast<unsigned>((count + rows_per_block - 1) / rows_per_block), static_cast<unsigned>(groups));
  markov_bias_kernel<<<grid, 256, 0, stream>>>(base, base_group_stride, block_row, markov_embed, markov_head, rank,
                                               vocab_begin, count, tok, tok_stride, out, out_group_stride, rows_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_confidence(const uint16_t* x, int64_t x_group_stride, int block_row, int hidden,
                             const uint16_t* markov_embed, int rank, const int64_t* tok, int tok_stride,
                             const float* w, int groups, float* conf_out, int conf_stride, cudaStream_t stream) {
  if (!x || !markov_embed || !tok || !w || !conf_out) throw std::invalid_argument("dsv41_dspark_confidence: null buffer");
  if (hidden <= 0 || rank <= 0 || groups <= 0 || block_row < 0 || tok_stride <= 0 || conf_stride <= 0)
    throw std::invalid_argument("dsv41_dspark_confidence: shape");
  confidence_kernel<<<static_cast<unsigned>(groups), 256, 0, stream>>>(x, x_group_stride, block_row, hidden, markov_embed,
                                                                        rank, tok, tok_stride, w, conf_out, conf_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int M>
__global__ void hc_head_kernel(const uint16_t* __restrict__ x, int hidden, int rows, const uint16_t* __restrict__ fn,
                               const float* __restrict__ scale, const float* __restrict__ base, float eps, float hc_eps,
                               uint16_t* __restrict__ out) {
  const int t = blockIdx.x;
  if (t >= rows) return;
  const int W = M * hidden;
  const uint16_t* xr = x + static_cast<size_t>(t) * static_cast<size_t>(W);
  __shared__ float red[32];
  {
    float ss = 0.f;
    for (int d = threadIdx.x; d < W; d += blockDim.x) {
      const float v = bf16_bits_to_float(xr[d]);
      ss = fmaf(v, v, ss);
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = (static_cast<int>(blockDim.x) / 32 > threadIdx.x) ? red[threadIdx.x] : 0.f;
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / W + eps);
    __syncthreads();
    float pre[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint16_t* w = fn + static_cast<size_t>(m) * static_cast<size_t>(W);
      float acc = 0.f;
      for (int d = threadIdx.x; d < W; d += blockDim.x)
        acc = fmaf(bf16_bits_to_float(xr[d]), bf16_bits_to_float(w[d]), acc);
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
      if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
      __syncthreads();
      if (threadIdx.x < 32) {
        float v = (static_cast<int>(blockDim.x) / 32 > threadIdx.x) ? red[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) red[0] = v;
      }
      __syncthreads();
      pre[m] = 1.f / (1.f + expf(-(red[0] * rms * scale[0] + base[m]))) + hc_eps;
      // Barrier before the next m clobbers red[]: pre[m] just read red[0],
      // and the leaders' next-m store must not race a lagging read.
      __syncthreads();
    }
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
      float acc = 0.f;
      #pragma unroll
      for (int m = 0; m < M; ++m) acc = fmaf(pre[m], bf16_bits_to_float(xr[static_cast<size_t>(m) * hidden + d]), acc);
      out[static_cast<size_t>(t) * hidden + d] = float_to_bf16_bits(acc);
    }
  }
}

void dsv41_hc_head_bf16(const uint16_t* x, int hc_mult, int hidden, int rows, const uint16_t* fn, const float* scale,
                        const float* base, float eps, float hc_eps, uint16_t* out, cudaStream_t stream) {
  if (!x || !fn || !scale || !base || !out) throw std::invalid_argument("dsv41_hc_head_bf16: null buffer");
  if (hc_mult <= 0 || hc_mult > 8 || hidden <= 0 || rows <= 0) throw std::invalid_argument("dsv41_hc_head_bf16: shape");
  switch (hc_mult) {
    case 1: hc_head_kernel<1><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, hidden, rows, fn, scale, base, eps, hc_eps, out); break;
    case 2: hc_head_kernel<2><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, hidden, rows, fn, scale, base, eps, hc_eps, out); break;
    case 4: hc_head_kernel<4><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, hidden, rows, fn, scale, base, eps, hc_eps, out); break;
    case 8: hc_head_kernel<8><<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, hidden, rows, fn, scale, base, eps, hc_eps, out); break;
    default: throw std::invalid_argument("dsv41_hc_head_bf16: hc_mult must be 1, 2, 4 or 8");
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
