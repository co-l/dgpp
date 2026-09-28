#pragma once
// The streaming tensor-core decode GEMM (2026-09-14, the batched decode's
// dense projections and the lm head): out[m, n] = act[m, k] x W[n, k]^T for
// m <= 32 activation rows, the weights read ONCE whatever m.
//
// WHY: the decode GEMV cores (fp8_gemv.cuh, bf16_gemv.cu) chunk the rows by
// four and pay kRows FMAs per weight element on the CUDA cores, so past a
// few rows a projection is issue-bound (the six-slot DeepSeek-V4.1 batch:
// 30 rows = eight chunks, 56 ms of a 249 ms step in dense fp8 projections
// and 18 ms in the 331 MB head). This kernel keeps the GEMV's memory shape
// — one warp per eight weight rows, a quad of lanes per row, 16-byte chunks,
// eight in flight per lane, no barrier in the k loop — and applies every
// chunk to all m rows with mma.sync m16n8k16 (bf16 in, fp32 accumulate):
// the per-weight cost is a dequant, a fixed in-quad shuffle and a quarter
// of an mma, whatever m.
//
// NUMERICS: the weight VALUES are the dequant bridge's (bf16(e4m3 x scale),
// exact for the e8m0 scales); the activations are the bf16 rows as given;
// the fp32 accumulation is the mma's over the k16 slices in ascending k
// order — deterministic, and without split-K the SAME chain whatever m
// (a padded row never touches another row's accumulators), so a batched row is
// bitwise the row alone at m = 1 through this kernel. It is not bitwise
// the GEMV cores' chain (a different fp32 order, inside the oracle budgets).
// With workspace, groups of at most 32 rows may use split-K. Their chain
// is independent of the group's row count, but differs from wider groups'
// unsplit chain. A caller changing batch size must preserve this boundary.
//
// CONTRACT: m >= 1 (rows 1..128 in one launch — 1/2/4/8 sixteen-row tiles
// by the count — and wider m in 128-row groups, each group its own launch
// re-reading the weights); k % 64 == 0 (a quad of lanes loads 4 x 16 k; a chunk lies inside one scale block:
// cs >= 4); 16-byte-aligned weight rows; act rows 16-byte aligned with an
// even-16-byte stride; out row stride >= n. fp8: w [n, k] e4m3 with block
// scales f32 [ceil(n / 2^rs), ceil(k / 2^cs)]. bf16: w [n, k] bf16.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kMmaGemvMaxRows = 32;
// The split-K's block cap: a problem's partials region is sized for this many
// splits, so the multi form's workspace is the sum over its problems of
// kMmaGemvMaxSplit * kMmaGemvMaxRows * n * sizeof(float).
constexpr int kMmaGemvMaxSplit = 16;
// Rows per launch: the widest single form (8 tiles); m above it runs in
// groups of this many rows, the weights read once per group.
constexpr int kMmaGemvMaxRowsPerLaunch = 128;

// fp8 weights with block scales; out bf16 or f32 (the epilogue store is the
// only difference: bf16(out_f32) == out_bf16 bit for bit).
// ws / ws_bytes (2026-09-21): a device workspace lets the decode forms (m <=
// 32) split the k range across blocks when a small n leaves the grid
// under-filled (fp32 partials in ws, one reduce launch; the split count a
// function of the shape only, so a row's chain is still the same whatever
// m rides in the launch). nullptr: the unsplit form, as before.
void launch_mma_gemv_fp8_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                              const float* scales, uint16_t* out, int m, int n, int k,
                              size_t out_stride, int rs, int cs, cudaStream_t stream,
                              void* ws = nullptr, size_t ws_bytes = 0);
void launch_mma_gemv_fp8_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                             const float* scales, float* out, int m, int n, int k,
                             size_t out_stride, int rs, int cs, cudaStream_t stream,
                             void* ws = nullptr, size_t ws_bytes = 0);
// bf16 weights (the lm head, the small dense sites); out bf16 or f32. The
// ws pair rides the fp8 pair's split-K contract (2026-09-28): a small n
// under-fills the grid, and a workspace splits the k range across blocks.
void launch_mma_gemv_bf16_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w,
                               uint16_t* out, int m, int n, int k, size_t out_stride,
                               cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);
void launch_mma_gemv_bf16_f32(const uint16_t* act, size_t act_stride, const uint16_t* w,
                              float* out, int m, int n, int k, size_t out_stride,
                              cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);
// The shape the kernel takes (k a multiple of 16, aligned pointers, m in range).
bool mma_gemv_shape_ok(const void* w, const void* act, size_t act_stride, int m, int k);

// The multi-problem decode form (2026-09-28, the 0731 dense groups): up to
// four problems in one launch, each with its own act / weights / output,
// all at the same m and k (the per-layer projections of one input: the
// hidden's wq_a + wkv, the lora's wq_b + index q, the attention out's
// output-group folds). Each problem runs the decode forms' block body over
// its own (n, k) — the same width and split count the single launch would
// choose for its n — so a problem's chain is the single launch's, and the
// launches it replaces only differ by the width rule's max-n choice
// (tolerance-equal, the decode forms' contract). The partials share one
// workspace, region per problem.
struct MmaGemvMultiProblem {
  const uint16_t* act;  // [m, act_stride]
  size_t act_stride;
  const void* w;  // fp8: e4m3 [n, k]; bf16: bf16 [n, k]
  const float* scales;  // fp8 only (nullptr for bf16)
  void* out;  // f32 or bf16 per the form
  int n;
  size_t out_stride;
};
constexpr int kMmaGemvMaxProblems = 4;
void launch_mma_gemv_multi_fp8_f32(const MmaGemvMultiProblem* probs, int np, int m, int k, int rs,
                                   int cs, cudaStream_t stream, void* ws, size_t ws_bytes);
void launch_mma_gemv_multi_fp8_bf16(const MmaGemvMultiProblem* probs, int np, int m, int k, int rs,
                                    int cs, cudaStream_t stream, void* ws, size_t ws_bytes);
void launch_mma_gemv_multi_bf16_f32(const MmaGemvMultiProblem* probs, int np, int m, int k, int rs,
                                    int cs, cudaStream_t stream, void* ws, size_t ws_bytes);
void launch_mma_gemv_multi_bf16_bf16(const MmaGemvMultiProblem* probs, int np, int m, int k, int rs,
                                     int cs, cudaStream_t stream, void* ws, size_t ws_bytes);

// The decode forms' block width override (warps of 8 weight rows: 1, 2, 4,
// 8; 0 restores the rule by n) — the timing sweep's knob, not a serving
// setting: the rows' chains do not depend on it.
void mma_gemv_set_decode_width(int warps);

}  // namespace dgpp
