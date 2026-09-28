#pragma once
// One CSA2 attention layer's forward (2026-09-13, docs/deepseek_v41_flash_plan.md
// §1.3, D6): the low-rank q (wq_a -> RMSNorm -> wq_b per local head, the
// tail rotated), the window latent (wkv -> RMSNorm -> rotated -> the
// fp8_block ring row), the compressor (a kv source: ratio 1 a projection +
// norm, ratio 2 the gated pair pooling with the per-request tail), the
// index keys (wk -> RMSNorm -> rotated at the entry's position -> the
// planar index cache) and the compressed main KV (rotated -> fp4_block
// rows), the indexer (an index source: wq_b -> rotated -> fp4 e8m0/32 ->
// the planar q with folded weights), the selection (plain, the candidate
// stage, or restricted to the candidate source's pool), the two-source
// attention (the window ring and the selected main rows, the sink in the
// denominator), the inverse rotation, the grouped wo_a and wo_b.
//
// One object serves every layer of a model: rebind() points it at a
// layer's weights and role; the shared scratch keeps the last index
// source's selection (topk / counts) and the candidate source's pool
// (cand / cand_counts) for the layers that reuse them — the full
// GLM-5.3's "select once, attend N times" contract restated for eight
// index sources and one candidate source. A reusing enqueue must see the
// same call shape (rows, chunk, request) the producing enqueue saw.
//
// The decode path is graph-capturable (call prepare() outside capture;
// the batch's req_ids / pos / req_spans are device buffers re-uploaded
// between replays); prefill is a control-path operation (one request per
// call, pool-aligned chunks, host staging of positions).
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/csa2.hpp"
#include "kernels/gemm.hpp"
#include "models/dsv41/csa2_state.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

struct Csa2Config {
  // The dense fp8 projections' decode form: the streaming tensor-core GEMM
  // (rows 1..32 in one launch, the weights read once) or the 4-row GEMV
  // chunks. Tolerance-equal forms; the model sets every site alike.
  bool dense_mma = true;
  int hidden = 5120;
  int q_lora = 1280;
  int o_lora = 1024;
  int num_heads = 64;
  int o_groups = 8;
  int index_heads = 32;
  int index_topk = 512;
  int candidate_block = 8;
  int candidate_blocks = 2048;
  int window = 128;
  int ring_slots = 160;
  int block_tokens = 128;
  int fp8_block = 32;  // the dense fp8 scale grid (32: V4.1, 128: 0731)
  float eps = 1e-20f;
  int tp = 1;
  // The 0731 release's cache forms (v4.1 defaults): the dequantized bf16
  // window / main rows and the dequantized bf16 index rows.
  LatentFormat ring_format = LatentFormat::kFp8Block;
  LatentFormat main_format = LatentFormat::kFp4Block;
  bool index_bf16 = false;
  // The 0731 per-head q renormalization (before the rotation).
  bool q_renorm = false;
  // The weights' softmax-scale fold (v4.1: 2^-6; 0731: 8192^-0.5).
  float fold_scale = 0.015625f;
  int local_heads() const { return num_heads / tp; }
  // log2 of the fp8 scale block (5: 32, 7: 128) — the scale gemm's rs / cs.
  int scale_shift() const { return fp8_block == 128 ? 7 : 5; }
  int local_groups() const { return o_groups / tp; }
  int heads_per_group() const { return num_heads / o_groups; }
  static void validate(const Csa2Config& c);
};

struct Csa2LayerWeights {
  GlmQuantMatrix wq_a, wkv, wq_b, wo_a, wo_b;  // fp8, 32 x 32 grids
  const uint16_t* q_norm = nullptr;
  const uint16_t* kv_norm = nullptr;
  const float* attn_sink = nullptr;    // f32 [local_heads]
  GlmQuantMatrix idx_wq_b;             // index sources
  const uint16_t* idx_wp = nullptr;
  const uint16_t* idx_wk = nullptr;    // kv sources
  const uint16_t* idx_k_norm = nullptr;
  const uint16_t* comp_wkv = nullptr;  // kv sources (bf16 [width, hidden])
  const uint16_t* comp_wgate = nullptr;  // ratio 2 (v4.1), ratio 4 (0731)
  const uint16_t* comp_norm = nullptr;
  const float* comp_ape = nullptr;     // 0731: the compressor's APE [ratio, 2*width]
  const uint16_t* idx_comp_wkv = nullptr;  // 0731: the indexer's compressor [256, hidden]
  const uint16_t* idx_comp_wgate = nullptr;
  const float* idx_comp_ape = nullptr;
  const uint16_t* idx_comp_norm = nullptr;
  const float* inv_freq = nullptr;     // device [32]: the layer's rotary table
  int ratio = 0;                       // 0: window only (1 / 2: v4.1; 4 / 128: 0731)
  int cache_ord = -1;                  // the main / index cache read (ratio > 0)
  int tail_ord = -1;                   // the compressor tail (ratio 2; 0731: main + indexer)
  int idx_tail_ord = -1;               // 0731: the indexer compressor's tail
  bool kv_source = false;
  bool index_source = false;
  bool candidate_source = false;
  bool uses_candidates = false;
};

class Csa2Layer {
 public:
  Csa2Layer(IGemm& gemm, const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, void* scratch,
            size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes, int max_decode_rows = 16,
            int decode_n_split = 32, size_t dot_budget = 64ull << 20, int max_requests = 4);
  static size_t scratch_bytes(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                              int max_decode_rows = 16, int decode_n_split = 32,
                              size_t dot_budget = 64ull << 20, int max_requests = 4);

  void rebind(const Csa2LayerWeights& w, int layer);
  // GEMM plans for `tokens` rows and the shared-memory opt-ins; outside capture.
  bool prepare(int tokens);
  // One request's chunk [pos0, pos0 + tokens) (pos0 a multiple of the block
  // size when the layer publishes; the request's blocks must already cover
  // pos0 + tokens). `floor`: the earliest position the window attends (the
  // bounded decoder's replay start; 0 for the context); `publish` false: a
  // kv source's entries were published before (publish_prefill) — the
  // rows attend and select only (the replay segment).
  // `row_base`: the chunk's first row within the walk's rows when several
  // requests' chunks share one walk (the group prefill): the selection
  // state the following layers reuse (topk_, counts_, the candidate pool)
  // and its shape record live at that offset, so every span keeps its own.
  void enqueue_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                       void* out, cudaStream_t stream, int64_t floor = 0, bool publish = true, int row_base = 0);
  // A kv source's publish alone over the chunk's rows: the compressor and
  // the index keys into the pool (the bounded prefill publishes the
  // decoder's global KV for every prompt row, then runs the segment).
  void publish_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                       cudaStream_t stream);
  // A decode batch of `tokens` rows over `num_requests` spans (the DSA
  // contract: contiguous rows per request in position order, pos -1
  // padding rows, all-padding spans allowed). tail_snapshots (optional):
  // fp32 [tokens, 2, 512] per-row compressor-tail snapshots on the
  // ratio-2 kv sources.
  void enqueue_decode(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                      const int32_t* req_spans, int num_requests, int tokens, void* out, cudaStream_t stream,
                      float* tail_snapshots = nullptr);

  // ---- the DSpark draft (plan D8) on a window-only draft layer ----------------
  // append_window_rows: the ring rows of `n` real tokens — x [n, hidden] the
  // projected main hidden (main_x), the latent rotated at pos, the fp8 row
  // at pos % ring_slots; pos -1 rows write nothing; n <= min(max_tokens,
  // ring_slots) (one slot per row: the caller passes a chunk's tail).
  void append_window_rows(const void* x, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos, int n,
                          cudaStream_t stream);
  // enqueue_draft_block: the block's rows (`rows` = groups x block; req_ids /
  // pos device per row, pos = the block's first position + the row's index
  // in its block, -1 for a padding block): q and the rows' own window
  // latents from `hidden_in` (rotated at pos, into the ring at the block's
  // slots — past the window of every later real query, so never read
  // again), the attention over the ring's real rows up to the block and
  // the block itself (bidirectional inside it), wo_a / wo_b into `out`.
  void enqueue_draft_block(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                           int rows, int block, void* out, cudaStream_t stream);
  static constexpr int kMaxDraftBlock = 8;

  const Csa2Config& config() const { return cfg_; }
  int layer() const { return layer_; }
  // Index-key / q exactness violations so far (a host read; synchronizes).
  unsigned index_violations() const;
  // Probes (valid until the next enqueue on the shared scratch).
  const int32_t* debug_topk() const { return topk_; }
  // The topk rows' column count (the indexer's topk, widened for the C128A
  // sequential selection's full visible lists).
  int topk_stride() const { return sel_col_; }
  const int32_t* debug_counts() const { return counts_; }
  const int32_t* debug_cand() const { return cand_; }
  const int32_t* debug_cand_counts() const { return cand_counts_; }
  const uint16_t* debug_attn_out() const { return o_; }
  const uint16_t* debug_q() const { return q_; }
  const uint8_t* debug_index_q_codes() const { return q_fp8_; }   // [rows * heads, 128] e4m3 (csa2_index_q_quant)
  const float* debug_index_q_scales() const { return q_scale_; }  // [rows * heads]
  const uint16_t* debug_kv() const { return kv_; }
  const uint16_t* debug_latent() const { return latent_; }
  const int64_t* debug_entries() const { return entries_; }
  // The last prefill select tile's logits [rows, stride] (fp32, -inf past
  // the row's visible entries; the parity tests' certification source).
  const float* debug_logits() const { return logits_; }
  int debug_logits_rows() const { return dbg_logits_rows_; }
  int64_t debug_logits_stride() const { return dbg_logits_stride_; }
  int64_t debug_logits_entries() const { return dbg_logits_entries_; }

  static constexpr int kPrefillAttnRows = 128;
  static constexpr int kPrefillSplit = 4;
  // The decode window attention splits its 128 keys this many ways so a
  // one-row window is not a single latency-bound block (the 2026-09-14
  // profile). A FIXED count, independent of the row count, so a token's
  // window is computed identically whether it is a one-row decode step or
  // row 0 of a multi-row MTP verify (the speculative transcript must equal
  // the plain greedy one). Eight keeps the split's fp32-combine drift from
  // the forward inside the decode audit's certified-flip budget while giving
  // ~8x the occupancy of a single block; prefill stays unsplit.
  static constexpr int kWinDecodeSplit = 8;
  static constexpr int kEntryPad = 256;

 private:
  struct Layout {
    size_t total = 0;
    size_t qr, kv, q, o, oa, idx_q, q_fp8, q_scale, w, w_folded, comp_kv, comp_score, idx_comp_kv,
         idx_comp_score, latent, ik;
    size_t pos, req_ids, req_zero, slots, pos_sel, entries, ent_pos, scratch_pos, iota, one_block;
    size_t wlist, wcounts, wscratch, dlist, dcounts, topk, counts, cand, cand_counts;
    size_t m_main, l_main, c_main, m_win, l_win, c_win;
    size_t gather_k, gather_scale, dot, logits, select_ws, counter, violations, snap_scratch;
    int tile_cap = 0;
    int sel_col = 0;
    int64_t max_entries = 0;
    int ws_slots = 0;
    int ws_win_rows = 0;
  };
  static Layout layout(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                       int decode_n_split, size_t dot_budget, int max_requests = 4);

  enum class SelKind { kNone, kPrefill, kDecode };
  struct SelShape {
    SelKind kind = SelKind::kNone;
    int rows = 0;
    int64_t start = -1;
    int req = -1;
    const int64_t* pos = nullptr;
  };
  void require_shape(const SelShape& have, const SelShape& want, const char* what) const;
  // The decode rows the multi-problem dense form groups (1..32, the mma's
  // decode rows; wider rows keep the per-site launches).
  bool dense_mma_grouped(int tokens) const { return cfg_.dense_mma && tokens >= 1 && tokens <= 32; }

  // The projections and the window row of `tokens` rows at `pos` (device).
  void project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream);
  void project_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                  bool gemm_done = false);
  void indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                     bool idx_q_done = false);
  // The index keys and the main rows of `n` entries (entries_ / ent_pos_
  // set; latent_ holds the normed, unrotated latents).
  void publish_entries(Csa2StatePool& pool, const int32_t* req_ids, int n, cudaStream_t stream);
  // A kv source's chunk: the compressor over `tokens` rows (pos_ / req_ids_
  // staged), the entries published.
  void publish_rows(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens, cudaStream_t stream);
  void stage_prefill_rows(int req, int64_t pos0, int tokens, cudaStream_t stream);
  // The attention of rows [row0, row0 + rows) of the current call: the
  // window partials over `win_cache` (a ring or the prefill scratch; the
  // per-row slot lists `list` of stride `list_stride` with `counts`), the
  // main partials over the selection, the finish into o_.
  void attend_rows(Csa2StatePool& pool, const int32_t* req_ids_win, const uint8_t* win_cache, int win_block_tokens,
                   const int32_t* win_table, const int32_t* req_ids_main, const int64_t* pos, int row0, int rows,
                   int n_split_main, cudaStream_t stream, const int32_t* list, int list_stride,
                   const int32_t* counts, bool split_window = false, int sel_base = 0);
  void project_out(int tokens, void* out, cudaStream_t stream);
  void validate_pool(const Csa2StatePool& pool) const;

  IGemm& gemm_;
  Csa2Config cfg_;
  Csa2LayerWeights w_;
  int layer_ = -1;
  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int max_decode_rows_ = 16;
  int decode_n_split_ = 32;
  int tile_cap_ = 0;
  int sel_col_ = 0;
  int64_t max_entries_ = 0;
  int64_t gather_zeroed_ = 0;
  int ws_slots_ = 0;
  int ws_win_rows_ = 0;
  float attn_scale_ = 0.f;
  // The selection and candidate-pool shape records, per row base (0: a
  // single-request walk; a group prefill's spans at their first rows).
  std::unordered_map<int, SelShape> sel_by_base_, cand_by_base_;
  SelShape& sel_at(int base) { return sel_by_base_[base]; }
  SelShape& cand_at(int base) { return cand_by_base_[base]; }
  int dbg_logits_rows_ = 0;
  int64_t dbg_logits_stride_ = 0, dbg_logits_entries_ = 0;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  std::vector<int64_t> host_i64_;
  std::vector<int32_t> host_i32_;

  uint8_t* scratch_ = nullptr;
  uint16_t* qr_ = nullptr;        // [T, q_lora]
  uint16_t* kv_ = nullptr;        // [T, 512]
  uint16_t* q_ = nullptr;         // [T, lh * 512]
  uint16_t* o_ = nullptr;         // [T, lh * 512]
  uint16_t* oa_ = nullptr;        // [T, lg * o_lora]
  uint16_t* idx_q_ = nullptr;     // [T, 32 * 128]
  uint8_t* q_fp8_ = nullptr;      // [T * 32, 128]
  float* q_scale_ = nullptr;      // [T * 32]
  uint16_t* iw_ = nullptr;        // [T, 32] the indexer weights
  float* w_folded_ = nullptr;     // [T * 32]
  float* comp_kv_ = nullptr;      // [T, 1024] (the 0731 compressor's 2*width rows)
  float* comp_score_ = nullptr;   // [T, 1024]
  float* idx_comp_kv_ = nullptr;  // [T, 256] (the 0731 indexer's compressor)
  float* idx_comp_score_ = nullptr;  // [T, 256]
  uint16_t* latent_ = nullptr;    // [T, 512]
  uint16_t* ik_ = nullptr;        // [T, 128]
  int64_t* pos_ = nullptr;        // [T] (prefill staging)
  int32_t* req_ids_ = nullptr;    // [T] (prefill staging)
  int32_t* req_zero_ = nullptr;   // [T] zeros
  int64_t* slots_ = nullptr;      // [T] ring slots
  int64_t* pos_sel_ = nullptr;    // [T]
  int64_t* entries_ = nullptr;    // [T]
  int64_t* ent_pos_ = nullptr;    // [T]
  int64_t* scratch_pos_ = nullptr;  // [T] window-scratch rows
  int64_t* iota_ = nullptr;       // [T]
  int32_t* one_block_ = nullptr;  // [1] = 0
  int32_t* wlist_ = nullptr;      // [T, window]
  int32_t* wcounts_ = nullptr;    // [T]
  int32_t* dlist_ = nullptr;      // [decode rows, window + kMaxDraftBlock] the draft block's lists
  int32_t* dcounts_ = nullptr;    // [decode rows]
  uint8_t* wscratch_ = nullptr;   // [(window - 1 + T) rows]
  int32_t* topk_ = nullptr;       // [T, index_topk]
  int32_t* counts_ = nullptr;     // [T]
  int32_t* cand_ = nullptr;       // [T, candidate_blocks]
  int32_t* cand_counts_ = nullptr;  // [T]
  float* m_main_ = nullptr;       // [ws_slots, lh]
  float* l_main_ = nullptr;
  float* c_main_ = nullptr;       // [ws_slots, lh, 512]
  float* m_win_ = nullptr;        // [ws_win_rows, lh]
  float* l_win_ = nullptr;
  float* c_win_ = nullptr;        // [ws_win_rows, lh, 512]
  uint8_t* gather_k_ = nullptr;   // [max_entries, 128]
  float* gather_scale_ = nullptr; // [max_entries]
  float* dot_ = nullptr;          // [tile_cap * 32, max_entries]
  float* logits_ = nullptr;       // [tile_cap, max_entries]
  void* select_ws_ = nullptr;
  int32_t* counter_ws_ = nullptr;
  unsigned* violations_ = nullptr;
  float* snap_scratch_ = nullptr;  // [max_requests, 256, 512] the ratio-128 snapshots' base
};

}  // namespace dgpp
