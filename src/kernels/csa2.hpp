#pragma once
// CSA2 (Compressed Sparse Attention 2, DeepSeek-V4.1-Flash) kernels
// (2026-09-13, docs/deepseek_v41_flash_plan.md §1.3, D6/D7): the pieces
// the DSA kernels (kernels/dsa.hpp) do not already provide. The layer
// object (models/dsv41/csa2_layer.hpp) composes them with the DSA
// select / attention / append kernels:
//   * the window rows live in per-request rings in the fp8_block format
//     (kernels/latent_format.hpp) and are attended by dsa_attn_partial /
//     dsa_attn_listed over a slot list — the ring IS a one-block cache;
//   * the compressed main KV lives in the paged pool in the fp4_block
//     format (dsa_latent_append / the attention kernels), the index keys
//     in the planar e4m3 + row-scale index cache the select kernels
//     stream (an fp4 e8m0/32 value is exact there — csa2_index_*);
//   * the plain selection is dsa_select_decode / dsa_select_prefill at
//     kpool 1 with relu over `pos_sel` = (pos + 1) / ratio - 1 (the
//     compressed entries visible to a query as a "position"); the
//     candidate stage (the top-2048 blocks of 8 entries, the newest
//     block pinned) and the restricted stage (the top-512 among a
//     candidate list) are the kernels below.
// Numerics mirror inference/model.py + kernel.py: the one-rounding
// RMSNorm (fp32 interior, bf16 out), the complex rotation over adjacent
// pairs with fp32 angles p * inv_freq[i] and cosf/sinf (no table), the
// release's quantizers, the sink in the softmax denominator only.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/latent_format.hpp"

namespace dgpp {

constexpr int kCsa2Latent = 512;    // head_dim: the latent width, K == V
constexpr int kCsa2Rope = 64;       // the rotated tail (qk_rope_head_dim)
constexpr int kCsa2IndexDim = 128;  // index_head_dim
constexpr int kCsa2CandidateMaxBlocks = 2048;  // candidate_topk_blocks (the select bound)

// Opts the candidate-select kernels into large dynamic shared memory
// (idempotent; must run outside graph capture).
void csa2_prepare_kernel_smem();

// ---- norms and rotations ---------------------------------------------------
// y = bf16(w * (x * rsqrt(mean(x^2) + eps))): the reference's RMSNorm —
// fp32 interior, the weight applied in fp32, ONE bf16 rounding (unlike
// GLM's two-rounding glm_rmsnorm_bf16). bf16 x [rows, dim] at x_stride
// elements per row; y likewise (y may alias x with equal strides).
void csa2_rmsnorm_bf16(const void* x, int64_t x_stride, const void* w, void* y,
                       int64_t y_stride, int rows, int dim, float eps,
                       cudaStream_t stream);

// The rotary inverse frequencies as the reference's precompute_freqs_cis
// builds them (fp32): 1 / theta^(2i/dim), and with original_seq_len > 0
// the YaRN mix — dims whose wavelength fits the training context keep
// their frequency, those far beyond it are divided by `factor`, the
// beta_fast..beta_slow band in between fades linearly. out: [rope_dim/2].
// The one builder lives in kernels/rope_scaling.cpp (yarn_rope_inv_freq_host),
// which the Qwen QSA rope shares; this name and these arguments are the
// CSA2 callers' (original_seq_len is that builder's correction max position).
void csa2_rope_inv_freq_host(int rope_dim, double theta, int64_t original_seq_len,
                             double factor, double beta_fast, double beta_slow,
                             float* out);

// Rotates adjacent pairs (x[2i], x[2i+1]) of `rope_dim` elements at every
// (row, head) segment in place: angle = fp32(pos) * inv_freq[i] (the
// reference's fp32 outer product), (c, s) = (cosf, sinf), the complex
// product in fp32, ONE bf16 rounding; `inverse` conjugates (the output's
// rotation removed). Rows with pos < 0 are left untouched.
//   x: bf16 at x + r * row_stride + h * head_stride (elements): the FIRST
//   element of the segment to rotate (the caller offsets to a tail).
void csa2_rope_apply(void* x, int64_t row_stride, int64_t head_stride, int heads,
                     int rope_dim, const int64_t* pos, const float* inv_freq,
                     bool inverse, int64_t rows, cudaStream_t stream);

// ---- positions ------------------------------------------------------------------
// out[r] = pos[r] < 0 ? -1 : (pos[r] + 1) / ratio - 1: the compressed
// entries a query at pos can see, minus one — the "position" the DSA
// select and dense-attention kernels take (visible = out + 1). A query
// that sees no entry gets -1 (a padding row to those kernels).
void csa2_entry_positions(const int64_t* pos, int64_t* out, int rows, int ratio,
                          cudaStream_t stream);
// out[r] = src[r] < 0 ? -1 : src[r] * mul + add (an entry's rotation
// position j * ratio; a chunk row's absolute position).
void csa2_scaled_positions(const int64_t* src, int64_t* out, int rows, int64_t mul,
                           int64_t add, cudaStream_t stream);
// out[r] = pos[r] < 0 ? -1 : pos[r] % ring_slots (a row's ring slot, the
// "position" the ring-as-cache append takes).
void csa2_ring_slot_positions(const int64_t* pos, int64_t* out, int rows, int ring_slots,
                              cudaStream_t stream);

// ---- the window ring ---------------------------------------------------------------
// A layer's ring: [max_requests][ring_slots] rows of row_bytes (the
// fp8_block form of the roped kv). Presented to the DSA kernels as a cache
// with one block per request: block_tables = the identity [max_requests]
// (csa2_ring_table), blocks_per_request 1, block_tokens = ring_slots, a
// row's "position" = its slot.
void csa2_ring_table(int32_t* table, int max_requests, cudaStream_t stream);
// Decode: the slot list of the positions [max(0, pos - window + 1), pos]
// (ascending) per row into list [rows, window] (-1 padded) and counts.
// The DSpark block's lists (the reference `get_dspark_topk_idxs`): the
// rows come in groups of `block` (row i is row i % block of its block, at
// position pos[i] = P + i % block with P the block's first position; a
// padding block has pos -1). Every row of a block lists the ring slots of
// positions [max(0, P - window) .. P - 1] (the real rows the ring holds
// up to the block: the window relative to P - 1, the last accepted
// position) then the block's own slots (positions P .. P + block - 1) —
// bidirectional inside the block; the list stride is window + block, the
// count min(window, P) + block (0 for a padding block).
void csa2_dspark_window_slots(const int64_t* pos, int rows, int block, int window, int ring_slots, int32_t* list,
                              int32_t* counts, cudaStream_t stream);
void csa2_window_slots_decode(const int64_t* pos, int rows, int window, int ring_slots,
                              int32_t* list, int32_t* counts, cudaStream_t stream);
// Prefill over a chunk of T rows at positions pos0 + i, attended from a
// scratch of [window - 1 + T] rows where row j holds position pos0 -
// (window - 1) + j: the list of chunk row i is the scratch rows of the
// positions [max(0, pos0 + i - window + 1), pos0 + i].
// `floor`: the earliest position a row may attend (the bounded decoder's
// replay start; 0 for the whole context).
void csa2_window_slots_prefill(int64_t pos0, int T, int window, int32_t* list,
                               int32_t* counts, cudaStream_t stream, int64_t floor = 0);
// The scratch's first window - 1 rows from the request's ring (positions
// pos0 - window + 1 .. pos0 - 1; a negative position is a zero row).
void csa2_window_scratch_prologue(const void* ring, int ring_slots, int64_t pos0,
                                  int window, size_t row_bytes, void* scratch,
                                  cudaStream_t stream, int64_t floor = 0);
// The chunk's last min(T, ring_slots) rows (scratch rows window - 1 + i)
// into the request's ring at slot (pos0 + i) % ring_slots.
void csa2_window_ring_writeback(const void* scratch, int window, int64_t pos0, int T,
                                int ring_slots, size_t row_bytes, void* ring,
                                cudaStream_t stream);

// ---- the indexer's fp4 e8m0/32 forms in the planar index cache ---------------------
// The reference quantizes the index q and k to e2m1 with an e8m0 scale per
// 32 (fp4_act_quant) and dequantizes in place. Stored here as e4m3 codes
// with one power-of-two row scale S = 2^(k_max - 6) (k_max the row's
// largest block exponent): every block within 14 binades of the largest
// is exact in e4m3; a farther block loses codes and is counted in
// *violations (may be null; one count per row).
//   q: bf16 [rows, heads, 128] (the tail already rotated) -> q_fp8 [rows *
//   heads, 128] e4m3, q_scale [rows * heads] fp32.
void csa2_index_q_quant(const void* q, int rows, int heads, void* q_fp8, float* q_scale,
                        unsigned* violations, cudaStream_t stream);
//   k: bf16 [n, 128] (normed, the tail rotated) -> the index cache slots of
//   entries `entries` (int64, -1 skipped) through the block table
//   (`entries_per_block` entries per physical block): index_k [slots,
//   128] e4m3, index_scale [slots] fp32.
void csa2_index_k_append(const void* k, const int32_t* req_ids, const int64_t* entries,
                         int n, const int32_t* block_tables, int blocks_per_request,
                         int entries_per_block, void* index_k, float* index_scale,
                         unsigned* violations, cudaStream_t stream);
// w_folded[i] = (fp32(bf16 w[i]) * fold) * q_scale[i]: the reference's
// weights_proj output (bf16) times the softmax scale (v4.1: 128^-0.5 *
// 32^-0.5 = 2^-6; 0731: 128^-0.5 * 64^-0.5 = 8192^-0.5) with the q scale
// folded in (the bf16 path passes a nullptr scale).
void csa2_fold_weights(const void* w_bf16, const float* q_scale, float fold, float* out, int64_t n,
                       cudaStream_t stream);

// ---- the compressor at ratio 2 --------------------------------------------------------
// The pooled latent of a pair: per channel, softmax over the two gate
// scores (fp32), the weighted kv sum (fp32) rounded to bf16, then the
// one-rounding RMSNorm over the 512 with `norm_w`.
// Prefill: kv / score fp32 [T, 512] for chunk rows at even-aligned
// positions; pairs (2j, 2j+1) inside the chunk -> latent_out bf16 [T / 2,
// 512]; an odd trailing row is stashed in `tail` (fp32 [2, 512]: kv then
// score) when tail is non-null.
void csa2_compress_pairs_prefill(const float* kv, const float* score, int T,
                                 const void* norm_w, float eps, void* latent_out,
                                 float* tail, cudaStream_t stream);
// Decode: rows in request spans (start, len) in position order. An even
// position stashes its (kv, score) into the request's tail; an odd
// position pools the tail with itself into latent_out[t] and reports
// entries_out[t] = pos / 2 (-1 on every other row, whose latent row is
// zeroed). tail_snapshots (optional, speculative rows): fp32 [tokens, 2,
// 512] — after every row that is not its request's last, the tail as it
// stands; rolling back to `a` accepted rows = copying row start + a - 1
// over the tail. Padding rows (pos < 0) touch nothing.
void csa2_compress_decode_update(const float* kv, const float* score, const int32_t* req_ids,
                                 const int64_t* pos, const int32_t* req_spans,
                                 int num_requests, const void* norm_w, float eps,
                                 float* tails, void* latent_out, int64_t* entries_out,
                                 int tokens, float* tail_snapshots, cudaStream_t stream);

// ---- the 0731 activation roundings (the release's in-place quantizers) ------
// The 0731 reference applies its activation quantizers IN PLACE: the caches
// store the dequantized bf16 values (inference/kernel.py act_quant with
// scale_fmt None / fp4_act_quant, the indexer's Hadamard rotation). One
// thread per row; y may alias x.
// fp8 over the non-rope 448 (seven blocks of 64): s = fp32(max(absmax,
// 1e-4) * (1 / 448)) (no power-of-two rounding), code = e4m3(x / s),
// y = bf16(dequant(code) * s); the last 64 (the rope tail) pass through.
void csa2_actquant8_dequant_bf16(const uint16_t* x, uint16_t* y, int rows, cudaStream_t stream);
// The Hadamard rotation of the release's rotate_activation (the indexer's
// 128 dims): y = bf16(H128(x) * (1 / sqrt(128))).
void csa2_hadamard128_bf16(const uint16_t* x, uint16_t* y, int rows, cudaStream_t stream);
// fp4 over the full 128 (four blocks of 32): s = 2^ceil(log2(max(absmax,
// 6 * 2^-126) / 6)) (e8m0), code = e2m1(x / s), y = bf16(dequant(code) * s).
void csa2_fp4_dequant_bf16(const uint16_t* x, uint16_t* y, int rows, cudaStream_t stream);
// The 0731 per-head q renormalization (before the rotation): the reference's
// q *= rsqrt(mean(q^2) + eps) per head — the mean in fp32 (bf16 result),
// the + eps and the rsqrt in bf16, the element product one bf16 rounding.
// q: bf16 [rows, heads, 512].
void csa2_q_renorm_bf16(uint16_t* q, int rows, int heads, float eps, cudaStream_t stream);

// p[0..n) = -inf (fp32): the compressor states' score-half init.
void csa2_fill_inf(float* p, int n, cudaStream_t stream);

// The 0731 selection runs on the dequantized bf16 values (no fp8 codes, no
// scales). The visible entries' 128-dim bf16 rows gathered contiguous
// (block table, `entries_per_block` per block).
void csa2_gather_index_bf16(const int32_t* block_table, int entries_per_block, const void* index_k, int n,
                            void* out, cudaStream_t stream);
// The 0731 index rows into the cache slots (the bf16 planar form, 128 bf16
// per row; entries -1 skipped).
void csa2_index_k_append_bf16(const void* k, const int32_t* req_ids, const int64_t* entries, int n,
                              const int32_t* block_tables, int blocks_per_request, int entries_per_block,
                              void* index_k, cudaStream_t stream);
// Decode: the dequantized-dot topk over every visible entry (no candidate
// pool): q_bf16 [rows, heads, 128] (the dequantized, rotated tail included),
// w_folded [rows, heads] (the weights with the softmax scale folded in),
// index_k the bf16 index rows. The topk of entry ids (ascending) and counts.
void csa2_select_bf16_decode(const void* q_bf16, const float* w_folded, const int32_t* req_ids,
                             const int64_t* pos_sel, int rows, const int32_t* block_tables,
                             int blocks_per_request, const void* index_k, int entries_per_block,
                             int heads, int select_k, uint64_t* keys_ws, int64_t max_entries,
                             int32_t* topk_out, int32_t* counts, cudaStream_t stream);
// The C128A sequential selection (the reference's get_compress_topk_idxs):
// topk_out[r][k] = k for k < min(pos_sel[r], col), counts_out[r] = that
// count. The visible entries' prefix, no scoring.
void csa2_sequential_topk(const int64_t* pos_sel, int32_t* topk_out, int32_t* counts_out, int rows, int col,
                          cudaStream_t stream);

// ---- the compressors at ratios 4 and 128 (the 0731 overlap windows) ----------
// The 0731 reference's Compressor: per channel the fp32 softmax over the
// window's items (max-subtracted, p_j = e_j / den), the fp32 weighted kv
// sum, one bf16 rounding, then the one-rounding RMSNorm. The ratio-4 window
// overlaps: it pools the previous window's first kv half against its own
// second half (eight items of head_dim). The request state (fp32 per
// request, `states`): the reference's kv_state then score_state — ratio 4,
// 16 slots of 2*head_dim (the last four tokens' raw kv in slots 0..3, the
// current window's kv and score + ape in 4..7, 0..3 taking 4..7 over after
// a completed window); ratio 128, 256 slots of head_dim. The scores are
// stored raw; the score half of a fresh state is the reference's -inf (an
// unwritten slot's softmax weight is exactly zero), which the pool's reset
// fills in (Csa2PoolShape::tail_inf).
// Prefill: kv / score fp32 [T, 2D] (ratio 4) at chunk rows, ape fp32
// [4, 2D] -> latent_out bf16 [T / 4, D] (the normed windows, not yet
// rotated); the chunk's boundary state (the reference's kv_state /
// score_state after the chunk) into `tail` (fp32 [16, 2D]) when non-null.
// `D` is 512 (the main latent) or 128 (the indexer compressor's).
void csa2_compress4_prefill(const float* kv, const float* score, int T, int D, const float* ape,
                            const void* norm_w, float eps, void* latent_out, float* tail,
                            cudaStream_t stream);
// Decode: rows in request spans (start, len) in position order, the state
// of request r at `states` + r * (16 * 2D). A row at position p stashes its
// (kv, score + ape[p % 4]); p with (p + 1) % 4 == 0 pools the eight items
// into latent_out[t], reports entries_out[t] = p / 4 and shifts the state
// (a non-entry row zeroes its latent, entry -1). snapshots (optional,
// speculative rows): fp32 [tokens, 16, 2D] — after every row that is not its
// request's last, the state as it stands; rolling back to `a` accepted rows
// = copying row start + a - 1 over the request's state. Padding rows (pos <
// 0) touch nothing.
void csa2_compress4_decode(const float* kv, const float* score, const int32_t* req_ids,
                           const int64_t* pos, const int32_t* req_spans, int num_requests, int D,
                           const float* ape, const void* norm_w, float eps, float* states,
                           void* latent_out, int64_t* entries_out, int tokens, float* snapshots,
                           cudaStream_t stream);
// The ratio-128 compressor (the full window, no overlap): kv / score fp32
// [T, 512], ape fp32 [128, 512] -> latent_out bf16 [T / 128, 512]; when
// `tail` is non-null, the chunk's boundary state (256 slots: the remainder
// rows' kv and score + ape in slots 0..T % 128 - 1, the rest zero) is
// written to it (fp32 [256, 512]) for the decode continuation.
void csa2_compress128_prefill(const float* kv, const float* score, int T, const float* ape,
                              const void* norm_w, float eps, void* latent_out, float* tail,
                              cudaStream_t stream);
// Decode as the ratio-4 machine with 128 slots: the state of request r at
// `states` + r * (256 * 512); p with (p + 1) % 128 == 0 pools, entry p / 128;
// snapshots fp32 [tokens, 256, 512].
void csa2_compress128_decode(const float* kv, const float* score, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans, int num_requests,
                             const float* ape, const void* norm_w, float eps, float* states,
                             void* latent_out, int64_t* entries_out, int tokens, float* snapshots,
                             cudaStream_t stream);

// ---- the candidate and restricted selections --------------------------------------
// The candidate pool of a row (the reference's select_candidate_blocks):
// the complete blocks of `block_size` entries among [0, pos_sel + 1)
// scored by the max of their entries, the last complete block pinned
// (+inf) when it is the newest, the `topk_blocks` best kept — cand_out
// [rows, topk_blocks] ascending block ids (-1 padded), cand_counts — plus,
// implicitly, the newest PARTIAL block's entries, which every consumer
// appends (a query at pos_sel with visible = pos_sel + 1 entries: the
// entries [visible / block_size * block_size, visible)).
// Scratch for the two decode selectors: uint64 keys [max_rows, max_entries].
// max_entries bounds each row's visible compressed entries, not token positions.
// The same row stride is used at every replay size; no counters or initialization
// are needed. A DSA selection workspace with these bounds is also sufficient.
size_t csa2_select_workspace_bytes(int max_rows, int64_t max_entries);
// Decode: parallel stripes score the index cache (the unchanged fp8 dot,
// relu, folded weight and entry scale); exact radix selection keeps the
// block maxima. When all complete blocks fit, emit their IDs without scoring.
// keys_ws has csa2_select_workspace_bytes(rows, max_entries) bytes; max_entries
// must bound the actual visible entry count. No allocation occurs at launch.
void csa2_select_candidates_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids,
                                   const int64_t* pos_sel, int rows, const int32_t* block_tables,
                                   int blocks_per_request, const void* index_k,
                                   const float* index_scale, int entries_per_block, int heads,
                                   int block_size, int topk_blocks, uint64_t* keys_ws,
                                   int64_t max_entries, int32_t* cand_out, int32_t* cand_counts,
                                   cudaStream_t stream);
// Decode, a restricted (or short) selection: the `select_k` best entries
// among a row's candidate pool (cand: the block ids [rows, cand_stride],
// cand_counts, block_size — the partial newest block appended) — or,
// with cand null, among every visible entry (the plain selection for the
// tests and the short-context case) — ascending in topk_out [rows,
// select_k] (-1 padded) with counts. Candidate block IDs must be distinct and
// ascending, as produced above. Uses the same scratch contract; the parallel
// scores retain original entry IDs in their tie keys. If every entry fits,
// emit the IDs directly. Both launches are graph-capturable kernel nodes.
void csa2_select_listed_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids,
                               const int64_t* pos_sel, int rows, const int32_t* block_tables,
                               int blocks_per_request, const void* index_k,
                               const float* index_scale, int entries_per_block, int heads,
                               const int32_t* cand, int cand_stride, const int32_t* cand_counts,
                               int block_size, int select_k, uint64_t* keys_ws, int64_t max_entries,
                               int32_t* topk_out, int32_t* counts, cudaStream_t stream);
// Prefill: the per-row logits over entries [0, n) from the dot buffer
// (fp32 [rows * heads, dot_stride], row r head h at r * heads + h):
// sum_h w_folded[r, h] * relu(dot) * k_scale[j]; entries at or past the
// row's visible count are -inf. logits: fp32 [rows, logits_stride].
void csa2_logits_prefill(const float* dot, int64_t dot_stride, const float* w_folded,
                         const float* k_scale, const int64_t* pos_sel, int rows, int64_t n,
                         int heads, float* logits, int64_t logits_stride,
                         cudaStream_t stream);
// Prefill selections over the logits rows: the plain / restricted top-k
// (cand null: every visible entry; else the block ids as above) and the
// candidate stage.
void csa2_select_rows_prefill(const float* logits, int64_t logits_stride,
                              const int64_t* pos_sel, int rows, int heads, int select_k,
                              const int32_t* cand, int cand_stride, const int32_t* cand_counts,
                              int block_size, int32_t* topk_out, int32_t* counts,
                              cudaStream_t stream);
void csa2_select_candidates_prefill(const float* logits, int64_t logits_stride,
                                    const int64_t* pos_sel, int rows, int heads, int block_size,
                                    int topk_blocks, int32_t* cand_out, int32_t* cand_counts,
                                    cudaStream_t stream);

// ---- the attention finish -----------------------------------------------------------------
// Merges the main-KV partials (n_main splits in dsa_attn_partial's layout;
// null / 0 on a window-only layer) with the window partials (n_win
// splits) and the sink (exp(sink_h - m) in the denominator only, the
// reference's), normalizes, rounds to bf16, then removes the query's
// rotation from the last 64 of every head (the reference's inverse
// apply_rotary_emb on the bf16 output). out: bf16 [rows, local_heads *
// 512]; padding rows (pos < 0) are zero.
void csa2_attn_finish(const float* m_main, const float* l_main, const float* c_main,
                      int n_main, const float* m_win, const float* l_win, const float* c_win,
                      int n_win, const float* sink, int rows, int local_heads,
                      const int64_t* pos, const float* inv_freq, void* out,
                      cudaStream_t stream);

}  // namespace dgpp
