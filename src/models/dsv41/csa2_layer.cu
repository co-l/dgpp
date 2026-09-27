#include "models/dsv41/csa2_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/dsa.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
int64_t round_up_to(int64_t v, int64_t g) { return (v + g - 1) / g * g; }
}  // namespace

void Csa2Config::validate(const Csa2Config& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("csa2 layer: ") + what); };
  if (c.hidden <= 0 || c.hidden % 32 != 0) fail("hidden must be a positive multiple of 32");
  if (c.q_lora <= 0 || c.q_lora % 32 != 0) fail("q_lora must be a positive multiple of 32");
  if (c.o_lora <= 0 || c.o_lora % 32 != 0) fail("o_lora must be a positive multiple of 32");
  if (c.tp <= 0 || c.num_heads % c.tp != 0 || c.o_groups % c.tp != 0 || c.num_heads % c.o_groups != 0)
    fail("heads and groups must divide by tp, heads by groups");
  if (c.local_heads() < 4 || (c.local_heads() & (c.local_heads() - 1)) != 0)
    fail("local heads must be a power of two >= 4 (the attention head-group tiling)");
  if (c.index_heads != 32 && c.index_heads != 64) fail("the selection kernels take 32 or 64 index heads");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > 1024)
    fail("index_topk must be a power of two <= 1024");
  if (c.candidate_block <= 0 || c.candidate_blocks <= 0 || c.candidate_blocks > kCsa2CandidateMaxBlocks ||
      (c.candidate_blocks & (c.candidate_blocks - 1)) != 0)
    fail("candidate_blocks must be a power of two <= 2048");
  if (c.window <= 0 || c.ring_slots < c.window + 16) fail("the ring must hold the window plus the spec rows");
  if (c.block_tokens <= 0 || c.block_tokens % 2 != 0) fail("block_tokens must be a positive even number");
  if (!(c.eps >= 0.f)) fail("eps");
}

Csa2Layer::Layout Csa2Layer::layout(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                                    int max_decode_rows, int decode_n_split, size_t dot_budget) {
  Csa2Config::validate(cfg);
  if (max_tokens <= 0 || max_cache_tokens <= 0) throw std::invalid_argument("csa2 layer: max_tokens / max_cache_tokens");
  if (max_decode_rows <= 0 || max_decode_rows > 32 || max_decode_rows > max_tokens)
    throw std::invalid_argument("csa2 layer: max_decode_rows must be in [1, min(32, max_tokens)]");
  if (decode_n_split <= 0 || dot_budget == 0) throw std::invalid_argument("csa2 layer: decode_n_split / dot_budget");
  const int lh = cfg.local_heads(), lg = cfg.local_groups();
  const size_t T = static_cast<size_t>(max_tokens);
  Layout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += std::max<size_t>(bytes, 16);
    return at;
  };
  L.max_entries = round_up_to(max_cache_tokens, kEntryPad);
  int tile_cap = static_cast<int>(dot_budget / (size_t(cfg.index_heads) * size_t(L.max_entries) * 4));
  tile_cap = std::max(1, std::min(tile_cap, max_tokens));
  L.tile_cap = tile_cap;
  L.ws_slots = std::max(std::max(max_decode_rows, 8) * std::max(decode_n_split, 8), kPrefillAttnRows * kPrefillSplit);
  L.ws_win_rows = std::max(std::max(max_decode_rows, 8), kPrefillAttnRows);
  L.qr = alloc(T * cfg.q_lora * 2);
  L.kv = alloc(T * kCsa2Latent * 2);
  L.q = alloc(T * lh * kCsa2Latent * 2);
  L.o = alloc(T * lh * kCsa2Latent * 2);
  L.oa = alloc(T * lg * cfg.o_lora * 2);
  L.idx_q = alloc(T * cfg.index_heads * kCsa2IndexDim * 2);
  L.q_fp8 = alloc(T * cfg.index_heads * kCsa2IndexDim);
  L.q_scale = alloc(T * cfg.index_heads * 4);
  L.w = alloc(T * cfg.index_heads * 2);
  L.w_folded = alloc(T * cfg.index_heads * 4);
  L.comp_kv = alloc(T * 2 * kCsa2Latent * 4);
  L.comp_score = alloc(T * 2 * kCsa2Latent * 4);
  L.idx_comp_kv = alloc(T * 2 * kCsa2IndexDim * 4);
  L.idx_comp_score = alloc(T * 2 * kCsa2IndexDim * 4);
  L.latent = alloc(T * kCsa2Latent * 2);
  L.ik = alloc(T * kCsa2IndexDim * 2);
  L.pos = alloc(T * 8);
  L.req_ids = alloc(T * 4);
  L.req_zero = alloc(T * 4);
  L.slots = alloc(T * 8);
  L.pos_sel = alloc(T * 8);
  L.entries = alloc(T * 8);
  L.ent_pos = alloc(T * 8);
  L.scratch_pos = alloc(T * 8);
  L.iota = alloc(T * 8);
  L.one_block = alloc(16);
  L.wlist = alloc(T * cfg.window * 4);
  L.wcounts = alloc(T * 4);
  L.dlist = alloc(size_t(max_decode_rows) * size_t(cfg.window + kMaxDraftBlock) * 4);
  L.dcounts = alloc(size_t(max_decode_rows) * 4);
  L.wscratch = alloc((size_t(cfg.window) - 1 + T) * latent_row_bytes(cfg.ring_format, kCsa2Latent));
  // The C128A sequential selection lists every visible entry (up to
    // max_cache_tokens / 128), so the selection column is wider than the
    // indexer's topk when the context outruns it.
    L.sel_col = std::max<int>(cfg.index_topk, int((max_cache_tokens + 127) / 128));
  L.topk = alloc(size_t(T) * size_t(L.sel_col) * 4);
  L.counts = alloc(T * 4);
  L.cand = alloc(T * cfg.candidate_blocks * 4);
  L.cand_counts = alloc(T * 4);
  L.m_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_main = alloc(size_t(L.ws_slots) * lh * kCsa2Latent * 4);
  // The window partials carry a split dimension too (the decode fix below):
  // sized like the main source so a small-row window can fill the GPU.
  L.m_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_win = alloc(size_t(L.ws_slots) * lh * kCsa2Latent * 4);
  L.gather_k = alloc(size_t(L.max_entries) * kCsa2IndexDim * 2);  // bf16 rows (the fp8 form fits)
  L.gather_scale = alloc(size_t(L.max_entries) * 4);
  L.dot = alloc(size_t(tile_cap) * cfg.index_heads * size_t(L.max_entries) * 4);
  L.logits = alloc(size_t(tile_cap) * size_t(L.max_entries) * 4);
  L.select_ws = alloc(dsa_select_workspace_bytes(max_decode_rows, L.max_entries));
  L.counter = alloc(16);
  L.violations = alloc(16);
  L.total = align256(off);
  return L;
}

size_t Csa2Layer::scratch_bytes(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                                int decode_n_split, size_t dot_budget) {
  return layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget).total;
}

Csa2Layer::Csa2Layer(IGemm& gemm, const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, void* scratch,
                     size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes, int max_decode_rows,
                     int decode_n_split, size_t dot_budget)
    : gemm_(gemm), cfg_(cfg), max_tokens_(max_tokens), max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows), decode_n_split_(decode_n_split), gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget);
  if (scratch == nullptr || scratch_capacity < L.total) throw std::invalid_argument("csa2 layer: scratch too small");
  scratch_ = static_cast<uint8_t*>(scratch);
  tile_cap_ = L.tile_cap;
  sel_col_ = L.sel_col;
  max_entries_ = L.max_entries;
  ws_slots_ = L.ws_slots;
  ws_win_rows_ = L.ws_win_rows;
  attn_scale_ = static_cast<float>(std::pow(static_cast<double>(kCsa2Latent), -0.5));
  const auto at = [&](size_t off) { return scratch_ + off; };
  qr_ = reinterpret_cast<uint16_t*>(at(L.qr));
  kv_ = reinterpret_cast<uint16_t*>(at(L.kv));
  q_ = reinterpret_cast<uint16_t*>(at(L.q));
  o_ = reinterpret_cast<uint16_t*>(at(L.o));
  oa_ = reinterpret_cast<uint16_t*>(at(L.oa));
  idx_q_ = reinterpret_cast<uint16_t*>(at(L.idx_q));
  q_fp8_ = at(L.q_fp8);
  q_scale_ = reinterpret_cast<float*>(at(L.q_scale));
  iw_ = reinterpret_cast<uint16_t*>(at(L.w));
  w_folded_ = reinterpret_cast<float*>(at(L.w_folded));
  comp_kv_ = reinterpret_cast<float*>(at(L.comp_kv));
  comp_score_ = reinterpret_cast<float*>(at(L.comp_score));
  idx_comp_kv_ = reinterpret_cast<float*>(at(L.idx_comp_kv));
  idx_comp_score_ = reinterpret_cast<float*>(at(L.idx_comp_score));
  latent_ = reinterpret_cast<uint16_t*>(at(L.latent));
  ik_ = reinterpret_cast<uint16_t*>(at(L.ik));
  pos_ = reinterpret_cast<int64_t*>(at(L.pos));
  req_ids_ = reinterpret_cast<int32_t*>(at(L.req_ids));
  req_zero_ = reinterpret_cast<int32_t*>(at(L.req_zero));
  slots_ = reinterpret_cast<int64_t*>(at(L.slots));
  pos_sel_ = reinterpret_cast<int64_t*>(at(L.pos_sel));
  entries_ = reinterpret_cast<int64_t*>(at(L.entries));
  ent_pos_ = reinterpret_cast<int64_t*>(at(L.ent_pos));
  scratch_pos_ = reinterpret_cast<int64_t*>(at(L.scratch_pos));
  iota_ = reinterpret_cast<int64_t*>(at(L.iota));
  one_block_ = reinterpret_cast<int32_t*>(at(L.one_block));
  wlist_ = reinterpret_cast<int32_t*>(at(L.wlist));
  wcounts_ = reinterpret_cast<int32_t*>(at(L.wcounts));
  dlist_ = reinterpret_cast<int32_t*>(at(L.dlist));
  dcounts_ = reinterpret_cast<int32_t*>(at(L.dcounts));
  wscratch_ = at(L.wscratch);
  topk_ = reinterpret_cast<int32_t*>(at(L.topk));
  counts_ = reinterpret_cast<int32_t*>(at(L.counts));
  cand_ = reinterpret_cast<int32_t*>(at(L.cand));
  cand_counts_ = reinterpret_cast<int32_t*>(at(L.cand_counts));
  m_main_ = reinterpret_cast<float*>(at(L.m_main));
  l_main_ = reinterpret_cast<float*>(at(L.l_main));
  c_main_ = reinterpret_cast<float*>(at(L.c_main));
  m_win_ = reinterpret_cast<float*>(at(L.m_win));
  l_win_ = reinterpret_cast<float*>(at(L.l_win));
  c_win_ = reinterpret_cast<float*>(at(L.c_win));
  gather_k_ = at(L.gather_k);
  gather_scale_ = reinterpret_cast<float*>(at(L.gather_scale));
  dot_ = reinterpret_cast<float*>(at(L.dot));
  logits_ = reinterpret_cast<float*>(at(L.logits));
  select_ws_ = at(L.select_ws);
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.counter));
  violations_ = reinterpret_cast<unsigned*>(at(L.violations));
  // The constants: the iota, the zero request ids, the one-block table,
  // the select workspace and counters (zeroed once).
  std::vector<int64_t> iota(static_cast<size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) iota[static_cast<size_t>(i)] = i;
  DGPP_CUDA_OK(cudaMemcpy(iota_, iota.data(), iota.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(req_zero_, 0, static_cast<size_t>(max_tokens) * 4));
  DGPP_CUDA_OK(cudaMemset(one_block_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(select_ws_, 0, dsa_select_workspace_bytes(max_decode_rows, max_entries_)));
  DGPP_CUDA_OK(cudaMemset(counter_ws_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(violations_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(wscratch_, 0, (size_t(cfg.window) - 1 + size_t(max_tokens)) * latent_row_bytes(cfg.ring_format, kCsa2Latent)));
}

void Csa2Layer::rebind(const Csa2LayerWeights& w, int layer) {
  if (w.wq_a.payload == nullptr || w.wkv.payload == nullptr || w.wq_b.payload == nullptr || w.wo_a.payload == nullptr ||
      w.wo_b.payload == nullptr || w.q_norm == nullptr || w.kv_norm == nullptr || w.attn_sink == nullptr ||
      w.inv_freq == nullptr)
    throw std::invalid_argument("csa2 layer: a null projection, norm, sink or rotary table");
  if (w.ratio != 0 && w.ratio != 1 && w.ratio != 2 && w.ratio != 4 && w.ratio != 128)
    throw std::invalid_argument("csa2 layer: ratio must be 0, 1, 2, 4 or 128");
  if (w.ratio > 0 && w.cache_ord < 0) throw std::invalid_argument("csa2 layer: a compressing layer needs its cache");
  if (w.ratio == 0 && (w.kv_source || w.index_source || w.candidate_source || w.uses_candidates))
    throw std::invalid_argument("csa2 layer: a window-only layer owns no compressor, indexer or candidates");
  if (w.kv_source) {
    if (w.ratio == 1 && (w.comp_wkv == nullptr || w.comp_norm == nullptr ||
                         (!cfg_.index_bf16 && (w.idx_wk == nullptr || w.idx_k_norm == nullptr))))
      throw std::invalid_argument("csa2 layer: a kv source needs its compressor and index-key weights");
    if (w.ratio == 2 && (w.comp_wkv == nullptr || w.comp_norm == nullptr || w.comp_wgate == nullptr || w.tail_ord < 0 ||
                         w.idx_wk == nullptr || w.idx_k_norm == nullptr))
      throw std::invalid_argument("csa2 layer: a kv source needs its compressor and index-key weights");
    if (w.ratio == 4 && (w.comp_wkv == nullptr || w.comp_norm == nullptr || w.comp_wgate == nullptr ||
                         w.comp_ape == nullptr || w.tail_ord < 0 || w.idx_comp_wkv == nullptr ||
                         w.idx_comp_wgate == nullptr || w.idx_comp_norm == nullptr || w.idx_comp_ape == nullptr ||
                         w.idx_tail_ord < 0))
      throw std::invalid_argument("csa2 layer: a C4A source needs both compressors and the index keys");
    if (w.ratio == 128 && (w.comp_wkv == nullptr || w.comp_norm == nullptr || w.comp_wgate == nullptr ||
                           w.comp_ape == nullptr || w.tail_ord < 0))
      throw std::invalid_argument("csa2 layer: a C128A source needs its block compressor");
  }
  if (w.index_source && (w.idx_wq_b.payload == nullptr || w.idx_wp == nullptr))
    throw std::invalid_argument("csa2 layer: an index source needs its indexer weights");
  if ((w.candidate_source || w.uses_candidates) && !w.index_source)
    throw std::invalid_argument("csa2 layer: the candidate stages belong to index sources");
  if (w.candidate_source && w.uses_candidates) throw std::invalid_argument("csa2 layer: a candidate source uses no pool");
  if (w.wq_a.rows != cfg_.q_lora || w.wq_a.cols != cfg_.hidden || w.wkv.rows != kCsa2Latent || w.wkv.cols != cfg_.hidden ||
      w.wq_b.rows != int64_t(cfg_.local_heads()) * kCsa2Latent || w.wq_b.cols != cfg_.q_lora ||
      w.wo_a.rows != int64_t(cfg_.local_groups()) * cfg_.o_lora || w.wo_a.cols != int64_t(cfg_.heads_per_group()) * kCsa2Latent ||
      w.wo_b.rows != cfg_.hidden || w.wo_b.cols != int64_t(cfg_.local_groups()) * cfg_.o_lora)
    throw std::invalid_argument("csa2 layer: projection geometry disagrees with the config");
  if (w.index_source && (w.idx_wq_b.rows != int64_t(cfg_.index_heads) * kCsa2IndexDim || w.idx_wq_b.cols != cfg_.q_lora))
    throw std::invalid_argument("csa2 layer: indexer geometry disagrees with the config");
  w_ = w;
  layer_ = layer;
}

bool Csa2Layer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prepare rows out of range");
  dsa_prepare_kernel_smem();
  csa2_prepare_kernel_smem();
  bool ok = true;
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, 2 * kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, 2 * kCsa2IndexDim, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16, size_t(kCsa2Latent));
  ok &= gemm_.ensure_plan(tokens, cfg_.index_heads, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  return ok;
}

unsigned Csa2Layer::index_violations() const {
  unsigned v = 0;
  DGPP_CUDA_OK(cudaMemcpy(&v, violations_, 4, cudaMemcpyDeviceToHost));
  return v;
}

void Csa2Layer::require_shape(const SelShape& have, const SelShape& want, const char* what) const {
  if (have.kind != want.kind || have.rows != want.rows || have.start != want.start || have.req != want.req ||
      have.pos != want.pos)
    throw std::logic_error(std::string("csa2 layer: ") + what +
                           " reused by a call of another shape (the producing layer's enqueue must precede it "
                           "with the same rows, chunk and request)");
}

void Csa2Layer::validate_pool(const Csa2StatePool& pool) const {
  if (!pool.initialized()) throw std::invalid_argument("csa2 layer: the pool is not initialized");
  if (pool.shape().ring_slots != cfg_.ring_slots || pool.shape().block_tokens != cfg_.block_tokens)
    throw std::invalid_argument("csa2 layer: the pool's ring / block geometry disagrees with the config");
  if (layer_ < 0 || layer_ >= pool.shape().layers) throw std::invalid_argument("csa2 layer: rebind a layer first");
  if (w_.ratio > 0 && (w_.cache_ord >= pool.caches() || pool.cache_ratio(w_.cache_ord) != w_.ratio))
    throw std::invalid_argument("csa2 layer: the layer's cache ordinal / ratio disagrees with the pool");
  if (pool.shape().main_format != cfg_.main_format || pool.shape().ring_format != cfg_.ring_format ||
      pool.shape().index_bf16 != cfg_.index_bf16)
    throw std::invalid_argument("csa2 layer: the pool's cache forms disagree with the config");
  if (w_.kv_source && w_.ratio > 0 && w_.ratio != 1 && w_.tail_ord >= pool.shape().tail_ordinals)
    throw std::invalid_argument("csa2 layer: tail ordinal out of the pool's range");
  if (w_.kv_source && w_.ratio == 4 && w_.idx_tail_ord >= pool.shape().tail_ordinals)
    throw std::invalid_argument("csa2 layer: the indexer tail ordinal out of the pool's range");
}

// ---- the projections -------------------------------------------------------------------------

void Csa2Layer::project_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wkv.payload, w_.wkv.scales, kv_, tokens, kCsa2Latent,
                              cfg_.hidden, stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
  csa2_rmsnorm_bf16(kv_, kCsa2Latent, w_.kv_norm, kv_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
  csa2_rope_apply(kv_ + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, pos, w_.inv_freq, false,
                  tokens, stream);
  if (cfg_.ring_format == LatentFormat::kBf16) csa2_actquant8_dequant_bf16(kv_, kv_, tokens, stream);
}

void Csa2Layer::project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  const int lh = cfg_.local_heads();
  launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wq_a.payload, w_.wq_a.scales, qr_, tokens, cfg_.q_lora,
                              cfg_.hidden, stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
  csa2_rmsnorm_bf16(qr_, cfg_.q_lora, w_.q_norm, qr_, cfg_.q_lora, tokens, cfg_.q_lora, cfg_.eps, stream);
  project_kv(hidden_in, tokens, pos, stream);
  launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.wq_b.payload, w_.wq_b.scales, q_, tokens,
                              lh * kCsa2Latent, cfg_.q_lora, stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
  if (cfg_.q_renorm) csa2_q_renorm_bf16(q_, tokens, lh, cfg_.eps, stream);
  csa2_rope_apply(q_ + (kCsa2Latent - kCsa2Rope), int64_t(lh) * kCsa2Latent, kCsa2Latent, lh, kCsa2Rope, pos,
                  w_.inv_freq, false, tokens, stream);
}

void Csa2Layer::indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const int heads = cfg_.index_heads;
  launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.idx_wq_b.payload, w_.idx_wq_b.scales, idx_q_, tokens,
                              heads * kCsa2IndexDim, cfg_.q_lora, stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
  csa2_rope_apply(idx_q_ + (kCsa2IndexDim - kCsa2Rope), int64_t(heads) * kCsa2IndexDim, kCsa2IndexDim, heads,
                  kCsa2Rope, pos, w_.inv_freq, false, tokens, stream);
  if (cfg_.index_bf16) {
    csa2_hadamard128_bf16(idx_q_, idx_q_, int64_t(tokens) * heads, stream);
    csa2_fp4_dequant_bf16(idx_q_, idx_q_, int64_t(tokens) * heads, stream);
  } else {
    csa2_index_q_quant(idx_q_, tokens, heads, q_fp8_, q_scale_, violations_, stream);
  }
  gemm_.matmul(hidden_in, w_.idx_wp, iw_, tokens, heads, cfg_.hidden, DType::BF16, GemmOut::BF16,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
  csa2_fold_weights(iw_, cfg_.index_bf16 ? nullptr : q_scale_, cfg_.fold_scale, w_folded_, int64_t(tokens) * heads, stream);
}

void Csa2Layer::publish_entries(Csa2StatePool& pool, const int32_t* req_ids, int n, cudaStream_t stream) {
  if (n <= 0) return;
  const int ord = w_.cache_ord;
  const int epb = pool.entries_per_block(ord);
  if (!cfg_.index_bf16) {
    // The index keys from the unrotated latents: wk -> RMSNorm -> the tail
    // rotated at the entry's position -> the planar form.
    gemm_.matmul(latent_, w_.idx_wk, ik_, n, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16,
                 size_t(kCsa2Latent), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_rmsnorm_bf16(ik_, kCsa2IndexDim, w_.idx_k_norm, ik_, kCsa2IndexDim, n, kCsa2IndexDim, cfg_.eps, stream);
    csa2_rope_apply(ik_ + (kCsa2IndexDim - kCsa2Rope), kCsa2IndexDim, kCsa2IndexDim, 1, kCsa2Rope, ent_pos_, w_.inv_freq,
                    false, n, stream);
    csa2_index_k_append(ik_, req_ids, entries_, n, pool.block_tables(), int(pool.total_blocks()), epb, pool.index_k(ord),
                        pool.index_scale(ord), violations_, stream);
  }
  // The main rows: the latent's tail rotated, then the append (the pool's
  // format: fp4_block v4.1, the dequantized bf16 0731).
  csa2_rope_apply(latent_ + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, ent_pos_, w_.inv_freq,
                  false, n, stream);
  if (w_.ratio == 4 || w_.ratio == 128)
    csa2_actquant8_dequant_bf16(latent_, latent_, n, stream);
  dsa_latent_append(latent_, req_ids, entries_, n, pool.block_tables(), int(pool.total_blocks()), epb, pool.main(ord),
                    kCsa2Latent, stream, pool.main_format());
}

void Csa2Layer::attend_rows(Csa2StatePool& pool, const int32_t* req_ids_win, const uint8_t* win_cache,
                            int win_block_tokens, const int32_t* win_table, const int32_t* req_ids_main,
                            const int64_t* pos, int row0, int rows, int n_split_main, cudaStream_t stream,
                            const int32_t* list, int list_stride, const int32_t* counts, bool split_window,
                            int sel_base) {
  const int lh = cfg_.local_heads();
  if (rows > ws_win_rows_ || rows * n_split_main > ws_slots_) throw std::logic_error("csa2 layer: attention tile too wide");
  const uint16_t* q = q_ + size_t(row0) * lh * kCsa2Latent;
  // The window source. At decode (one or a few rows) an unsplit window is a
  // single thread block per head group — latency-bound at ~4 warps while the
  // rest of the GPU idles (the 2026-09-14 profile: 140 us for 128 keys). The
  // decode caller (`split_window`) splits it across the key dimension the way
  // the compressed source is, sized to fill the SMs. Prefill's row tile
  // already fills the GPU, so it stays one split and is byte-identical to
  // before (the reference-parity gates unchanged); decode's window is then no
  // longer bit-identical to the forward's single block — a different fp32
  // combine order, the rounding class the decode audit certifies.
  // n_split_win <= n_split_main, so it fits ws_slots.
  const int n_split_win = split_window ? std::min(n_split_main, kWinDecodeSplit) : 1;
  dsa_attn_partial(q, win_cache, req_ids_win, list + size_t(row0) * list_stride, list_stride, counts + row0, rows,
                   n_split_win, lh, kCsa2Latent, win_block_tokens, win_table, 1, attn_scale_, m_win_, l_win_, c_win_,
                   stream, pool.ring_format(), nullptr, 0);
  int n_main = 0;
  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    const int32_t* topk = topk_ + size_t(sel_base + row0) * cfg_.index_topk;
    const int32_t* counts = counts_ + sel_base + row0;
    n_main = n_split_main;
    if (!(lh >= 16 && lh % 16 == 0 &&
          dsa_attn_listed(q, pool.main(ord), req_ids_main, topk, sel_col_, counts, rows, n_main, lh, kCsa2Latent,
                          epb, pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_,
                          stream, pool.main_format(), nullptr, 0)))
      dsa_attn_partial(q, pool.main(ord), req_ids_main, topk, sel_col_, counts, rows, n_main, lh, kCsa2Latent,
                       epb, pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_,
                       stream, pool.main_format(), nullptr, 0);
  }
  csa2_attn_finish(n_main ? m_main_ : nullptr, n_main ? l_main_ : nullptr, n_main ? c_main_ : nullptr, n_main, m_win_,
                   l_win_, c_win_, n_split_win, w_.attn_sink, rows, lh, pos, w_.inv_freq,
                   o_ + size_t(row0) * lh * kCsa2Latent, stream);
}

void Csa2Layer::project_out(int tokens, void* out, cudaStream_t stream) {
  const int lh = cfg_.local_heads(), lg = cfg_.local_groups(), hpg = cfg_.heads_per_group();
  const int K = hpg * kCsa2Latent;
  // wo_a is block-diagonal over the groups: group g projects its own heads.
  for (int g = 0; g < lg; ++g) {
    const size_t row_off = size_t(g) * cfg_.o_lora;
    launch_scale_gemm_grid_bf16(o_ + size_t(g) * K, size_t(lh) * kCsa2Latent, w_.wo_a.payload + row_off * K,
                                w_.wo_a.scales + (row_off / cfg_.fp8_block) * (size_t(K) / cfg_.fp8_block), oa_ + row_off, tokens, cfg_.o_lora, K,
                                stream, size_t(lg) * cfg_.o_lora, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
  }
  launch_scale_gemm_grid_bf16(oa_, size_t(lg) * cfg_.o_lora, w_.wo_b.payload, w_.wo_b.scales,
                              static_cast<uint16_t*>(out), tokens, cfg_.hidden, lg * cfg_.o_lora, stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), cfg_.dense_mma);
}

// ---- decode ---------------------------------------------------------------------------------------

void Csa2Layer::enqueue_decode(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                               const int32_t* req_spans, int num_requests, int tokens, void* out, cudaStream_t stream,
                               float* tail_snapshots) {
  validate_pool(pool);
  if (tokens <= 0 || tokens > max_decode_rows_) throw std::invalid_argument("csa2 layer: decode rows out of range");
  if (num_requests <= 0 || num_requests > pool.shape().max_requests)
    throw std::invalid_argument("csa2 layer: request count out of range");
  if (!hidden_in || !req_ids || !pos || !req_spans || !out) throw std::invalid_argument("csa2 layer: null buffer");
  const SelShape shape{SelKind::kDecode, tokens, -1, -1, pos};

  project_q_kv(hidden_in, tokens, pos, stream);
  // The window row into the ring (this batch's rows are visible to its
  // own queries: the ring is read after the append).
  csa2_ring_slot_positions(pos, slots_, tokens, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, tokens, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kCsa2Latent,
                    stream, pool.ring_format());
  csa2_window_slots_decode(pos, tokens, cfg_.window, cfg_.ring_slots, wlist_, wcounts_, stream);

  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    if (w_.kv_source) {
      if (w_.ratio == 4) {
        // The C4A main latent and the indexer's own compressor, per request
        // span; the completed windows publish into the pool.
        gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, 2 * kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, 2 * kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        csa2_compress4_decode(comp_kv_, comp_score_, req_ids, pos, req_spans, num_requests, kCsa2Latent, w_.comp_ape,
                              w_.comp_norm, cfg_.eps, pool.tails(w_.tail_ord), latent_, entries_, tokens,
                              tail_snapshots, stream);
        csa2_scaled_positions(entries_, ent_pos_, tokens, 4, 0, stream);
        gemm_.matmul(hidden_in, w_.idx_comp_wkv, idx_comp_kv_, tokens, 2 * kCsa2IndexDim, cfg_.hidden, DType::BF16,
                     GemmOut::F32, size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        gemm_.matmul(hidden_in, w_.idx_comp_wgate, idx_comp_score_, tokens, 2 * kCsa2IndexDim, cfg_.hidden, DType::BF16,
                     GemmOut::F32, size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        csa2_compress4_decode(idx_comp_kv_, idx_comp_score_, req_ids, pos, req_spans, num_requests, kCsa2IndexDim,
                              w_.idx_comp_ape, w_.idx_comp_norm, cfg_.eps, pool.tails(w_.idx_tail_ord), ik_, nullptr,
                              tokens, nullptr, stream);
        csa2_rope_apply(ik_ + (kCsa2IndexDim - kCsa2Rope), kCsa2IndexDim, kCsa2IndexDim, 1, kCsa2Rope, ent_pos_,
                        w_.inv_freq, false, tokens, stream);
        csa2_hadamard128_bf16(ik_, ik_, tokens, stream);
        csa2_fp4_dequant_bf16(ik_, ik_, tokens, stream);
        csa2_index_k_append_bf16(ik_, req_ids, entries_, tokens, pool.block_tables(), int(pool.total_blocks()),
                                 pool.entries_per_block(ord), pool.index_k(ord), stream);
      } else if (w_.ratio == 128) {
        gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        csa2_compress128_decode(comp_kv_, comp_score_, req_ids, pos, req_spans, num_requests, w_.comp_ape, w_.comp_norm,
                                cfg_.eps, pool.tails(w_.tail_ord), latent_, entries_, tokens, tail_snapshots, stream);
        csa2_scaled_positions(entries_, ent_pos_, tokens, 128, 0, stream);
      } else if (w_.ratio == 2) {
        gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        csa2_compress_decode_update(comp_kv_, comp_score_, req_ids, pos, req_spans, num_requests, w_.comp_norm, cfg_.eps,
                                    pool.tails(w_.tail_ord), latent_, entries_, tokens, tail_snapshots, stream);
        csa2_scaled_positions(entries_, ent_pos_, tokens, 2, 0, stream);
      } else {
        gemm_.matmul(hidden_in, w_.comp_wkv, latent_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16,
                     size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
        csa2_rmsnorm_bf16(latent_, kCsa2Latent, w_.comp_norm, latent_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
        csa2_scaled_positions(pos, entries_, tokens, 1, 0, stream);
        csa2_scaled_positions(pos, ent_pos_, tokens, 1, 0, stream);
      }
      publish_entries(pool, req_ids, tokens, stream);
    }
    csa2_entry_positions(pos, pos_sel_, tokens, w_.ratio, stream);
    if (w_.index_source) {
      indexer_query(hidden_in, tokens, pos, stream);
      const int epb = pool.entries_per_block(ord);
      if (cfg_.index_bf16) {
        csa2_select_bf16_decode(idx_q_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
                                int(pool.total_blocks()), pool.index_k(ord), epb, cfg_.index_heads, cfg_.index_topk,
                                static_cast<uint64_t*>(select_ws_), max_entries_, topk_, counts_, stream);
      } else if (w_.candidate_source) {
        csa2_select_candidates_decode(
            q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
            int(pool.total_blocks()), pool.index_k(ord), pool.index_scale(ord), epb,
            cfg_.index_heads, cfg_.candidate_block, cfg_.candidate_blocks,
            static_cast<uint64_t*>(select_ws_), max_entries_, cand_, cand_counts_, stream);
        cand_at(0) = shape;
        csa2_select_listed_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
                                  int(pool.total_blocks()), pool.index_k(ord),
                                  pool.index_scale(ord), epb, cfg_.index_heads, cand_,
                                  cfg_.candidate_blocks, cand_counts_, cfg_.candidate_block,
                                  cfg_.index_topk, static_cast<uint64_t*>(select_ws_), max_entries_,
                                  topk_, counts_, stream);
      } else if (w_.uses_candidates) {
        require_shape(cand_at(0), shape, "the candidate pool");
        csa2_select_listed_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
                                  int(pool.total_blocks()), pool.index_k(ord),
                                  pool.index_scale(ord), epb, cfg_.index_heads, cand_,
                                  cfg_.candidate_blocks, cand_counts_, cfg_.candidate_block,
                                  cfg_.index_topk, static_cast<uint64_t*>(select_ws_), max_entries_,
                                  topk_, counts_, stream);
      } else {
        dsa_select_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(), int(pool.total_blocks()),
                          pool.index_k(ord), pool.index_scale(ord), epb, cfg_.index_heads, kCsa2IndexDim,
                          cfg_.index_topk, 1, cfg_.index_topk, topk_, counts_, select_ws_, max_entries_, counter_ws_, 0,
                          stream, true);
      }
      sel_at(0) = shape;
    } else if (w_.ratio == 128) {
      // The C128A sequential selection: every visible entry.
      csa2_sequential_topk(pos_sel_, topk_, counts_, tokens, sel_col_, stream);
      sel_at(0) = shape;
    } else {
      require_shape(sel_at(0), shape, "the selection");
    }
  }
  attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table(), req_ids, pos, 0, tokens,
              decode_n_split_, stream, wlist_, cfg_.window, wcounts_, /*split_window=*/true);
  project_out(tokens, out, stream);
}

// ---- the DSpark draft --------------------------------------------------------------------------------

void Csa2Layer::append_window_rows(const void* x, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos, int n,
                                   cudaStream_t stream) {
  validate_pool(pool);
  if (n <= 0 || n > max_tokens_ || n > cfg_.ring_slots) throw std::invalid_argument("csa2 layer: window rows out of range");
  if (!x || !req_ids || !pos) throw std::invalid_argument("csa2 layer: null buffer");
  project_kv(x, n, pos, stream);
  csa2_ring_slot_positions(pos, slots_, n, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, n, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kCsa2Latent,
                    stream, pool.ring_format());
}

void Csa2Layer::enqueue_draft_block(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                                    int rows, int block, void* out, cudaStream_t stream) {
  validate_pool(pool);
  if (w_.ratio != 0) throw std::logic_error("csa2 layer: the draft block runs on a window-only layer");
  if (block <= 0 || block > kMaxDraftBlock || rows <= 0 || rows % block != 0 || rows > max_decode_rows_)
    throw std::invalid_argument("csa2 layer: draft block rows out of range");
  if (cfg_.ring_slots < cfg_.window + block) throw std::invalid_argument("csa2 layer: the ring must hold window + block");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("csa2 layer: null buffer");
  project_q_kv(hidden_in, rows, pos, stream);
  csa2_ring_slot_positions(pos, slots_, rows, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, rows, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kCsa2Latent,
                    stream, pool.ring_format());
  csa2_dspark_window_slots(pos, rows, block, cfg_.window, cfg_.ring_slots, dlist_, dcounts_, stream);
  attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table(), req_ids, pos, 0, rows, 1, stream,
              dlist_, cfg_.window + block, dcounts_);  // the builder's stride
  project_out(rows, out, stream);
}

// ---- prefill --------------------------------------------------------------------------------------

void Csa2Layer::stage_prefill_rows(int req, int64_t pos0, int tokens, cudaStream_t stream) {
  // Host staging (prefill is never captured): positions and request ids.
  host_i64_.assign(static_cast<size_t>(tokens), 0);
  host_i32_.assign(static_cast<size_t>(tokens), req);
  for (int i = 0; i < tokens; ++i) host_i64_[static_cast<size_t>(i)] = pos0 + i;
  DGPP_CUDA_OK(cudaMemcpyAsync(pos_, host_i64_.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(req_ids_, host_i32_.data(), static_cast<size_t>(tokens) * 4, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // the staging vectors are reused by the callers
}

void Csa2Layer::publish_rows(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                             cudaStream_t stream) {
  int n_entries = 0;
  if (w_.ratio == 4) {
    // The C4A main latent (the overlap window) and the indexer's own
    // compressor (the bf16 index keys at the index dimension).
    gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, 2 * kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, 2 * kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_compress4_prefill(comp_kv_, comp_score_, tokens, kCsa2Latent, w_.comp_ape, w_.comp_norm, cfg_.eps, latent_,
                           pool.tails(w_.tail_ord) + size_t(req) * pool.tail_bytes_per_request(w_.tail_ord) / sizeof(float), stream);
    n_entries = tokens / 4;
    csa2_scaled_positions(iota_, entries_, n_entries, 1, pos0 / 4, stream);
    csa2_scaled_positions(entries_, ent_pos_, n_entries, 4, 0, stream);
    gemm_.matmul(hidden_in, w_.idx_comp_wkv, idx_comp_kv_, tokens, 2 * kCsa2IndexDim, cfg_.hidden, DType::BF16,
                 GemmOut::F32, size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden_in, w_.idx_comp_wgate, idx_comp_score_, tokens, 2 * kCsa2IndexDim, cfg_.hidden, DType::BF16,
                 GemmOut::F32, size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_compress4_prefill(idx_comp_kv_, idx_comp_score_, tokens, kCsa2IndexDim, w_.idx_comp_ape, w_.idx_comp_norm,
                           cfg_.eps, ik_,
                           pool.tails(w_.idx_tail_ord) + size_t(req) * pool.tail_bytes_per_request(w_.idx_tail_ord) / sizeof(float), stream);
    csa2_rope_apply(ik_ + (kCsa2IndexDim - kCsa2Rope), kCsa2IndexDim, kCsa2IndexDim, 1, kCsa2Rope, ent_pos_,
                    w_.inv_freq, false, n_entries, stream);
    csa2_hadamard128_bf16(ik_, ik_, n_entries, stream);
    csa2_fp4_dequant_bf16(ik_, ik_, n_entries, stream);
    csa2_index_k_append_bf16(ik_, req_ids_, entries_, n_entries, pool.block_tables(), int(pool.total_blocks()),
                             pool.entries_per_block(w_.cache_ord), pool.index_k(w_.cache_ord), stream);
  } else if (w_.ratio == 128) {
    // The C128A block latent (the full window; the chunk's boundary state
    // into the tail when the chunk is not block-aligned).
    gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_compress128_prefill(comp_kv_, comp_score_, tokens, w_.comp_ape, w_.comp_norm, cfg_.eps, latent_,
                             tokens % 128 != 0
                                 ? pool.tails(w_.tail_ord) + size_t(req) * pool.tail_bytes_per_request(w_.tail_ord) / sizeof(float)
                                 : nullptr,
                             stream);
    n_entries = tokens / 128;
    csa2_scaled_positions(iota_, entries_, n_entries, 1, pos0 / 128, stream);
    csa2_scaled_positions(entries_, ent_pos_, n_entries, 128, 0, stream);
  } else if (w_.ratio == 2) {
    gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_compress_pairs_prefill(comp_kv_, comp_score_, tokens, w_.comp_norm, cfg_.eps, latent_,
                                pool.tails(w_.tail_ord) + size_t(req) * 2 * kCsa2Latent, stream);
    n_entries = tokens / 2;
    csa2_scaled_positions(iota_, entries_, n_entries, 1, pos0 / 2, stream);
    csa2_scaled_positions(entries_, ent_pos_, n_entries, 2, 0, stream);
  } else {
    gemm_.matmul(hidden_in, w_.comp_wkv, latent_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_rmsnorm_bf16(latent_, kCsa2Latent, w_.comp_norm, latent_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
    n_entries = tokens;
    csa2_scaled_positions(pos_, entries_, tokens, 1, 0, stream);
    csa2_scaled_positions(pos_, ent_pos_, tokens, 1, 0, stream);
  }
  publish_entries(pool, req_ids_, n_entries, stream);
}

void Csa2Layer::publish_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                                cudaStream_t stream) {
  validate_pool(pool);
  if (!w_.kv_source || w_.ratio <= 0) throw std::logic_error("csa2 layer: publish_prefill on a layer that is not a kv source");
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prefill rows out of range");
  if (req < 0 || req >= pool.shape().max_requests) throw std::invalid_argument("csa2 layer: request out of range");
  if (pos0 < 0 || pos0 % cfg_.block_tokens != 0)
    throw std::invalid_argument("csa2 layer: a prefill chunk starts on a block boundary");
  if (pool.request_blocks(req) * cfg_.block_tokens < pos0 + tokens)
    throw std::invalid_argument("csa2 layer: the request's blocks do not cover the chunk");
  if (!hidden_in) throw std::invalid_argument("csa2 layer: null buffer");
  stage_prefill_rows(req, pos0, tokens, stream);
  publish_rows(hidden_in, pool, req, pos0, tokens, stream);
}

void Csa2Layer::enqueue_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                                void* out, cudaStream_t stream, int64_t floor, bool publish, int row_base) {
  validate_pool(pool);
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prefill rows out of range");
  if (row_base < 0 || row_base + tokens > max_tokens_)
    throw std::invalid_argument("csa2 layer: the chunk's row base and rows exceed max_tokens");
  if (req < 0 || req >= pool.shape().max_requests) throw std::invalid_argument("csa2 layer: request out of range");
  if (pos0 < 0 || (publish && pos0 % cfg_.block_tokens != 0))
    throw std::invalid_argument("csa2 layer: a prefill chunk starts on a block boundary");
  if (floor < 0 || floor > pos0) throw std::invalid_argument("csa2 layer: the replay floor lies at or before the chunk");
  if (pool.request_blocks(req) * cfg_.block_tokens < pos0 + tokens)
    throw std::invalid_argument("csa2 layer: the request's blocks do not cover the chunk");
  if (!hidden_in || !out) throw std::invalid_argument("csa2 layer: null buffer");
  const SelShape shape{SelKind::kPrefill, tokens, pos0, req, nullptr};
  stage_prefill_rows(req, pos0, tokens, stream);

  project_q_kv(hidden_in, tokens, pos_, stream);
  // The window scratch: the ring's last window - 1 rows (those at or past
  // the floor), then the chunk's rows as a one-block cache (scratch row
  // window - 1 + i).
  uint8_t* ring = pool.ring(layer_) + size_t(req) * pool.ring_bytes_per_request();
  const size_t rb = pool.ring_row_bytes();
  const int win_rows = cfg_.window - 1 + tokens;
  csa2_window_scratch_prologue(ring, cfg_.ring_slots, pos0, cfg_.window, rb, wscratch_, stream, floor);
  csa2_scaled_positions(iota_, scratch_pos_, tokens, 1, cfg_.window - 1, stream);
  dsa_latent_append(kv_, req_zero_, scratch_pos_, tokens, one_block_, 1, win_rows, wscratch_, kCsa2Latent, stream,
                    pool.ring_format());
  csa2_window_slots_prefill(pos0, tokens, cfg_.window, wlist_, wcounts_, stream, floor);

  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    if (w_.kv_source && publish) publish_rows(hidden_in, pool, req, pos0, tokens, stream);
    csa2_entry_positions(pos_, pos_sel_, tokens, w_.ratio, stream);
    if (w_.index_source) {
      indexer_query(hidden_in, tokens, pos_, stream);
      // The visible entries of the chunk's last row, gathered contiguous
      // for the dot GEMM (the padded tail zeroed once for the sanitizer).
      const int64_t n_gather = (pos0 + tokens) / w_.ratio;
      const int64_t padded_n = n_gather > 0 ? round_up_to(n_gather, kEntryPad) : 0;
      if (padded_n > max_entries_) throw std::invalid_argument("csa2 layer: the context exceeds max_cache_tokens");
      if (padded_n > gather_zeroed_) {
        if (cfg_.index_bf16)
          DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * kCsa2IndexDim * 2, 0,
                                       size_t(padded_n - gather_zeroed_) * kCsa2IndexDim * 2, stream));
        else {
          DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * kCsa2IndexDim, 0,
                                       size_t(padded_n - gather_zeroed_) * kCsa2IndexDim, stream));
          DGPP_CUDA_OK(cudaMemsetAsync(gather_scale_ + gather_zeroed_, 0, size_t(padded_n - gather_zeroed_) * 4, stream));
        }
        gather_zeroed_ = padded_n;
      }
      if (n_gather > 0) {
        if (cfg_.index_bf16)
          csa2_gather_index_bf16(pool.block_tables() + size_t(req) * pool.total_blocks(), epb, pool.index_k(ord), n_gather,
                                 gather_k_, stream);
        else
          dsa_gather_index_pools(pool.block_tables() + size_t(req) * pool.total_blocks(), epb, pool.index_k(ord),
                                 pool.index_scale(ord), n_gather, gather_k_, gather_scale_, kCsa2IndexDim, stream);
      }
      for (int row0 = 0; row0 < tokens; row0 += tile_cap_) {
        const int rows = std::min(tile_cap_, tokens - row0);
        if (padded_n > 0) {
          if (cfg_.index_bf16) {
            gemm_.matmul(idx_q_ + size_t(row0) * cfg_.index_heads * kCsa2IndexDim, gather_k_, dot_, rows * cfg_.index_heads,
                         int(padded_n), kCsa2IndexDim, DType::BF16, GemmOut::F32, size_t(kCsa2IndexDim), gemm_ws_,
                         gemm_ws_bytes_, stream);
            csa2_logits_prefill(dot_, padded_n, w_folded_ + size_t(row0) * cfg_.index_heads, nullptr, pos_sel_ + row0,
                                rows, n_gather, cfg_.index_heads, logits_, padded_n, stream);
          } else {
            gemm_.matmul(q_fp8_ + size_t(row0) * cfg_.index_heads * kCsa2IndexDim, gather_k_, dot_, rows * cfg_.index_heads,
                         int(padded_n), kCsa2IndexDim, DType::F8_E4M3, GemmOut::F32, size_t(kCsa2IndexDim), gemm_ws_,
                         gemm_ws_bytes_, stream);
            csa2_logits_prefill(dot_, padded_n, w_folded_ + size_t(row0) * cfg_.index_heads, gather_scale_, pos_sel_ + row0,
                                rows, n_gather, cfg_.index_heads, logits_, padded_n, stream);
          }
        }
        const int64_t stride = std::max<int64_t>(padded_n, 1);
        dbg_logits_rows_ = padded_n > 0 ? rows : 0;
        dbg_logits_stride_ = stride;
        dbg_logits_entries_ = n_gather;
        const size_t srow = size_t(row_base + row0);  // the selection state's row
        if (w_.candidate_source)
          csa2_select_candidates_prefill(logits_, stride, pos_sel_ + row0, rows, cfg_.index_heads, cfg_.candidate_block,
                                         cfg_.candidate_blocks,
                                         cand_ + srow * cfg_.candidate_blocks, cand_counts_ + srow, stream);
        const bool restricted = w_.candidate_source || w_.uses_candidates;
        csa2_select_rows_prefill(logits_, stride, pos_sel_ + row0, rows, cfg_.index_heads, cfg_.index_topk,
                                 restricted ? cand_ + srow * cfg_.candidate_blocks : nullptr, cfg_.candidate_blocks,
                                 restricted ? cand_counts_ + srow : nullptr, cfg_.candidate_block,
                                 topk_ + srow * cfg_.index_topk, counts_ + srow, stream);
      }
      if (w_.candidate_source) cand_at(row_base) = shape;
      else if (w_.uses_candidates) require_shape(cand_at(row_base), shape, "the candidate pool");
      sel_at(row_base) = shape;
    } else if (w_.ratio == 128) {
      // The C128A sequential selection: every visible entry.
      csa2_sequential_topk(pos_sel_, topk_, counts_, tokens, sel_col_, stream);
      sel_at(row_base) = shape;
    } else {
      require_shape(sel_at(row_base), shape, "the selection");
    }
  }
  for (int row0 = 0; row0 < tokens; row0 += kPrefillAttnRows) {
    const int rows = std::min(kPrefillAttnRows, tokens - row0);
    attend_rows(pool, req_zero_, wscratch_, win_rows, one_block_, req_ids_, pos_ + row0, row0, rows, kPrefillSplit,
                stream, wlist_, cfg_.window, wcounts_, false, row_base);
  }
  csa2_window_ring_writeback(wscratch_, cfg_.window, pos0, tokens, cfg_.ring_slots, rb, ring, stream);
  project_out(tokens, out, stream);
}

}  // namespace dgpp
