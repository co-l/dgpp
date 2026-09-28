#pragma once
// The DSpark draft's glue kernels (docs/deepseek_v41_flash_plan.md §1.7,
// D8; the reference `DSparkBlock`): the target layers' stream mean, the
// block rows off the accepted verify rows, the Markov-biased head row and
// the confidence logit. Every launch is graph-capturable (kernels only).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// out[t, :] (row stride out_stride elements) = bf16(mean over the hc_mult
// streams of streams[t, :, :]) — the fp32 sum in stream order, one
// rounding (torch's bf16 mean: `h.mean(dim=2)` of the attention input at
// a DSpark target layer).
void dsv41_stream_mean_bf16(const uint16_t* streams, int hc_mult, int hidden, int rows, uint16_t* out,
                            int64_t out_stride, cudaStream_t stream);

// The block rows off the draft rows the core staged (glm_spec_draft_rows'
// contract: step_pos[r] the accepted rows' positions, -1 padding, tokens[r]
// their winners; the eager path stages every row real). For each group g
// of `rows_per_group` rows: P = 1 + max(step_pos) over the group (-1 when
// no row is real: the group's block is padding) and `next` = the token of
// that row; the block's rows g * block + k get pos P + k (or -1), the
// tokens [next, noise, noise, ...], the group's request id (row
// g * rows_per_group's) and one span per group (spans[g] = g * block,
// spans[groups] = groups * block).
void dsv41_dspark_block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                             int rows_per_group, int block, int64_t noise_id, int64_t* pos_out, int64_t* tok_out,
                             int32_t* req_out, int32_t* spans_out, cudaStream_t stream);

// The Markov-biased head row: for each group g and each j in [0, rows_out),
//   out[g * out_group_stride + j * count + v] = base[g * base_group_stride + block_row * count + v]
//                                             + sum_r embed[tok_g][r] * head[vocab_begin + v][r]
// over this rank's vocab slice [vocab_begin, vocab_begin + count) — the
// reference's `logits[:, i].add_(markov_head(output_ids[:, i]))` with the
// bf16 embedding row and the bf16 head rows, fp32 accumulation (the head's
// linear over one row: one fp32 dot per vocab entry, in rank order).
// tok_g = tok[g * tok_stride].
void dsv41_dspark_markov_bias(const float* base, int64_t base_group_stride, int block_row, const uint16_t* markov_embed,
                              const uint16_t* markov_head, int rank, int vocab_begin, int count, const int64_t* tok,
                              int tok_stride, int groups, float* out, int64_t out_group_stride, int rows_out,
                              cudaStream_t stream);

// The confidence logit of block row `block_row` per group (the reference's
// `confidence_head(x, markov_embed)`: the fp32 projection of
// [x_k | embed(tok_{k-1})], x the hc_pre'd hidden BEFORE the norm):
//   conf[g * conf_stride] = sum_d w[d] * x[g * x_group_stride + block_row * hidden + d]
//                         + sum_r w[hidden + r] * embed[tok_g][r]
void dsv41_dspark_confidence(const uint16_t* x, int64_t x_group_stride, int block_row, int hidden,
                             const uint16_t* markov_embed, int rank, const int64_t* tok, int tok_stride,
                             const float* w, int groups, float* conf_out, int conf_stride, cudaStream_t stream);

// The V4-Flash-0731 learned head collapse (the checkpoint reference's
// `hc_head`, vllm's `hc_head_fused_kernel_tilelang`), over the hc_mult
// residual streams BEFORE the final norm:
//   rms   = rsqrt(mean over the flat (hc_mult * hidden) row of x^2 + eps)
//   mix_m = dot(x[t, :], fn[m, :]) * rms      (the whole flat row, vllm's F.linear)
//   pre_m = sigmoid(mix_m * scale + base[m]) + hc_eps
//   out[t, d] = bf16(sum_m pre_m * x[t, m, d])
// x is bf16 [rows, hc_mult * hidden], fn bf16 [hc_mult, hc_mult * hidden],
// base f32 [hc_mult], scale f32 [1]; fp32 accumulation, one final rounding.
// One block per row; graph-capturable.
void dsv41_hc_head_bf16(const uint16_t* x, int hc_mult, int hidden, int rows, const uint16_t* fn,
                        const float* scale, const float* base, float eps, float hc_eps, uint16_t* out,
                        cudaStream_t stream);

}  // namespace dgpp
