#include "models/dsv41/engram_layer.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {

struct Dsv41EngramLayer::StageArgs {
  const Dsv41EngramTables* tables = nullptr;
  const int32_t* ids = nullptr;
  uint8_t* dst = nullptr;
  size_t layer_bytes = 0;
  int n = 0, entries = 0;
  std::atomic<int> error{0};
  std::string what;
};

void Dsv41EngramLayer::stage_callback(void* user) {
  StageArgs* a = static_cast<StageArgs*>(user);
  try {
    const int L = static_cast<int>(a->tables->tables.size());
    for (int l = 0; l < L; ++l)
      a->tables->tables[static_cast<size_t>(l)]->gather(a->ids + static_cast<size_t>(l) * a->entries,
                                                        static_cast<int64_t>(L) * a->entries, a->tables->sel.data(),
                                                        a->tables->rows_local(), a->n,
                                                        a->dst + static_cast<size_t>(l) * a->layer_bytes);
  } catch (const std::exception& e) {
    a->what = e.what();
    a->error.store(1, std::memory_order_release);
    DGPP_LOG_ERROR("Dsv41EngramLayer: the Engram staging failed: {}", e.what());
  }
}

size_t Dsv41EngramLayer::staging_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens) {
  const size_t L = cfg.engram_layer_ids.size();
  const size_t entries = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
  const size_t rows_local = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * (cfg.engram_n_heads / world);
  const size_t row_bytes = static_cast<size_t>(cfg.engram_head_dim) + cfg.engram_head_dim / 32;
  return static_cast<size_t>(max_tokens) * L * entries * 4 + L * static_cast<size_t>(max_tokens) * rows_local * row_bytes;
}
size_t Dsv41EngramLayer::device_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens) {
  const size_t rows_local = static_cast<size_t>(cfg.engram_max_ngram_size - 1) * (cfg.engram_n_heads / world);
  return static_cast<size_t>(max_tokens) * rows_local * cfg.engram_head_dim * 2 +
         static_cast<size_t>(max_tokens) * (cfg.hc_mult + 1) * cfg.hidden_size * 2 +
         static_cast<size_t>(cfg.vocab_size) * 4 + 4096;
}

Dsv41EngramLayer::Dsv41EngramLayer(const Dsv41TextConfig& cfg, const Dsv41EngramTables& tables,
                                   const Dsv41EngramSidecar& sidecar, int max_tokens, int max_requests)
    : cfg_(cfg), tables_(tables), max_tokens_(max_tokens) {
  layers_ = static_cast<int>(cfg.engram_layer_ids.size());
  if (layers_ <= 0 || static_cast<int>(tables.tables.size()) != layers_)
    throw std::invalid_argument("Dsv41EngramLayer: the tables must cover every Engram layer");
  if (max_tokens <= 0 || max_requests <= 0) throw std::invalid_argument("Dsv41EngramLayer: shape");
  if (tables.head_dim != cfg.engram_head_dim || tables.heads != cfg.engram_n_heads ||
      tables.ngrams != cfg.engram_max_ngram_size - 1)
    throw std::invalid_argument("Dsv41EngramLayer: the tables' geometry disagrees with the config");
  entries_ = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
  // The hash constants.
  const size_t L = static_cast<size_t>(layers_);
  if (sidecar.token_map.size() != static_cast<size_t>(cfg.vocab_size) ||
      sidecar.multipliers.size() != L * cfg.engram_max_ngram_size ||
      sidecar.primes.size() != L * entries_ || sidecar.offsets.size() != L * entries_)
    throw std::invalid_argument("Dsv41EngramLayer: the sidecar's shape disagrees with the config");
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_token_map_), sidecar.token_map.size() * 4));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_mult_), sidecar.multipliers.size() * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_primes_), sidecar.primes.size() * 8));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_offsets_), sidecar.offsets.size() * 8));
  DGPP_CUDA_OK(cudaMemcpy(d_token_map_, sidecar.token_map.data(), sidecar.token_map.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_mult_, sidecar.multipliers.data(), sidecar.multipliers.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_primes_, sidecar.primes.data(), sidecar.primes.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_offsets_, sidecar.offsets.data(), sidecar.offsets.size() * 8, cudaMemcpyHostToDevice));
  hash_.token_map = d_token_map_;
  hash_.multipliers = d_mult_;
  hash_.primes = d_primes_;
  hash_.offsets = d_offsets_;
  hash_.pad_class = sidecar.pad_class;
  hash_.layers = layers_;
  hash_.max_ngram = cfg.engram_max_ngram_size;
  hash_.heads = cfg.engram_n_heads;
  hash_.vocab = cfg.vocab_size;
  // The pinned ids (device-mapped: the hash kernel writes them, the host
  // callback reads them) and the staged rows.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_), static_cast<size_t>(max_tokens) * L * entries_ * 4,
                             cudaHostAllocMapped));
  staged_layer_bytes_ = static_cast<size_t>(max_tokens) * tables.rows_local() * tables.tables[0]->row_bytes();
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&staged_), L * staged_layer_bytes_, cudaHostAllocMapped));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&e_), static_cast<size_t>(max_tokens) * width_local() * 2));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&kv_),
                          static_cast<size_t>(max_tokens) * (cfg.hc_mult + 1) * cfg.hidden_size * 2));
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
  stage_args_.push_back(std::make_unique<StageArgs>());
  stage_args_.push_back(std::make_unique<StageArgs>());
}

Dsv41EngramLayer::~Dsv41EngramLayer() {
  if (side_) cudaStreamSynchronize(side_);
  if (fork_) cudaEventDestroy(fork_);
  if (join_) cudaEventDestroy(join_);
  if (side_) cudaStreamDestroy(side_);
  cudaFreeHost(h_ids_);
  cudaFreeHost(staged_);
  cudaFree(e_);
  cudaFree(kv_);
  cudaFree(d_token_map_);
  cudaFree(d_mult_);
  cudaFree(d_primes_);
  cudaFree(d_offsets_);
}

void Dsv41EngramLayer::check_staged() const {
  for (const auto& a : stage_args_)
    if (a->error.load(std::memory_order_acquire))
      throw std::runtime_error("Dsv41EngramLayer: an Engram staging failed on the host: " + a->what);
}

void Dsv41EngramLayer::stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                             const int32_t* req_spans, int num_requests, const int32_t* ctx, cudaStream_t stream) {
  if (rows <= 0) return;
  if (rows > max_tokens_) throw std::invalid_argument("Dsv41EngramLayer: rows exceed max_tokens");
  // A new staging may follow a consumed or an abandoned one: the side
  // stream serializes the host gathers, so the pinned buffers are free
  // once the previous callback has run, which the fork below waits for.
  check_staged();
  int32_t* d_ids = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ids), h_ids_, 0));
  dsv41_engram_hash_ids_rows(tokens, rows, req_ids, pos, req_spans, num_requests, ctx, hash_, d_ids, stream);
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  DGPP_CUDA_OK(cudaStreamIsCapturing(stream, &cs));
  StageArgs* a = nullptr;
  if (cs != cudaStreamCaptureStatusNone) {
    stage_args_.push_back(std::make_unique<StageArgs>());
    a = stage_args_.back().get();
  } else {
    a = stage_args_[static_cast<size_t>(eager_slot_)].get();
    eager_slot_ ^= 1;
  }
  a->tables = &tables_;
  a->ids = h_ids_;
  a->dst = staged_;
  a->layer_bytes = staged_layer_bytes_;
  a->n = rows;
  a->entries = entries_;
  DGPP_CUDA_OK(cudaEventRecord(fork_, stream));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, fork_, 0));
  DGPP_CUDA_OK(cudaLaunchHostFunc(side_, &Dsv41EngramLayer::stage_callback, a));
  DGPP_CUDA_OK(cudaEventRecord(join_, side_));
  staged_rows_ = rows;
}

void Dsv41EngramLayer::context_rows(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                                    const int32_t* req_spans, int num_requests, int32_t* ctx, int32_t* ctx_rows,
                                    cudaStream_t stream) {
  dsv41_engram_context_rows(tokens, rows, req_ids, pos, req_spans, num_requests, d_token_map_, cfg_.engram_max_ngram_size,
                            ctx, ctx_rows, stream);
}

void Dsv41EngramLayer::embed(int table_index, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (table_index < 0 || table_index >= layers_) throw std::invalid_argument("Dsv41EngramLayer: table index");
  if (staged_rows_ != rows) throw std::logic_error("Dsv41EngramLayer: embed() without a matching stage()");
  DGPP_CUDA_OK(cudaStreamWaitEvent(stream, join_, 0));
  uint8_t* d_staged = nullptr;
  DGPP_CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_staged), staged_, 0));
  dsv41_engram_gather_staged_bf16(d_staged + static_cast<size_t>(table_index) * staged_layer_bytes_, rows,
                                  tables_.rows_local(), cfg_.engram_head_dim, e_, stream);
}

void Dsv41EngramLayer::project(const Dsv41EngramLayerWeights& w, int rows, uint16_t* kv_dst, cudaStream_t stream) {
  if (rows <= 0) return;
  const int K = width_local();
  const int N = (cfg_.hc_mult + 1) * cfg_.hidden_size;
  if (w.wkv.payload == nullptr || w.wkv.rows != N || w.wkv.cols != K)
    throw std::invalid_argument("Dsv41EngramLayer: wkv geometry disagrees with the rank's rows");
  launch_scale_gemm_grid_bf16(e_, static_cast<size_t>(K), w.wkv.payload, w.wkv.scales, kv_dst ? kv_dst : kv_, rows, N, K,
                              stream, 0, cfg_.scale_shift(), cfg_.scale_shift(), dense_mma_);
}

void Dsv41EngramLayer::gate(const Dsv41EngramLayerWeights& w, uint16_t* x, const uint16_t* kv, int rows,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (w.q_weight == nullptr || w.k_weight == nullptr) throw std::invalid_argument("Dsv41EngramLayer: null gate weights");
  dsv41_engram_gate_rows(x, kv, w.q_weight, w.k_weight, rows, cfg_.hc_mult, cfg_.hidden_size, cfg_.rms_norm_eps, stream);
}

}  // namespace dgpp
