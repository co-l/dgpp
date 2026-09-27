#include "models/dsv41/csa2_state.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {
size_t padded(size_t b) { return (b + 255) / 256 * 256; }
void check_shape(const Csa2PoolShape& s) {
  if (s.layers <= 0 || s.max_requests <= 0 || s.token_slots <= 0 || s.block_tokens <= 0 ||
      s.token_slots % s.block_tokens != 0 || s.ring_slots <= 0 || s.tail_ordinals < 0)
    throw std::invalid_argument("csa2 state pool: invalid shape");
  for (const int r : s.cache_ratio)
    if ((r != 1 && r != 2 && r != 4 && r != 128) || s.block_tokens % r != 0)
      throw std::invalid_argument("csa2 state pool: a cache ratio must be 1, 2, 4 or 128 and divide the block");
  if (s.tail_floats.size() != static_cast<size_t>(s.tail_ordinals) ||
      s.tail_inf.size() != static_cast<size_t>(s.tail_ordinals))
    throw std::invalid_argument("csa2 state pool: a tail size per tail ordinal");
  for (size_t t = 0; t < s.tail_inf.size(); ++t)
    if (s.tail_inf[t] < 0 || s.tail_inf[t] > s.tail_floats[t])
      throw std::invalid_argument("csa2 state pool: the -inf count must fit the tail");
  if (s.token_slots / s.block_tokens * s.block_tokens >= (int64_t(1) << 21))
    throw std::invalid_argument("csa2 state pool: entry id space exceeds 2^21 (the select keys)");
}
}  // namespace

Csa2StatePool::~Csa2StatePool() {
  for (uint8_t* p : main_) cudaFree(p);
  for (uint8_t* p : index_k_) cudaFree(p);
  for (float* p : index_scale_) cudaFree(p);
  cudaFree(ring_base_);
  cudaFree(ring_table_);
  cudaFree(tail_base_);
}

void Csa2StatePool::init(const Csa2PoolShape& shape) {
  if (initialized_) throw std::logic_error("csa2 state pool: init twice");
  check_shape(shape);
  shape_ = shape;
  table_.init(shape.max_requests, shape.block_tokens, shape.token_slots / shape.block_tokens);
  const int n = caches();
  main_.assign(static_cast<size_t>(n), nullptr);
  index_k_.assign(static_cast<size_t>(n), nullptr);
  index_scale_.assign(static_cast<size_t>(n), nullptr);
  const size_t index_row = shape_.index_bf16 ? 2 * kCsa2IndexDim : kCsa2IndexDim;
  for (int o = 0; o < n; ++o) {
    const size_t slots = static_cast<size_t>(entry_slots(o));
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&main_[static_cast<size_t>(o)]), slots * main_row_bytes()));
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&index_k_[static_cast<size_t>(o)]), slots * index_row));
    if (!shape_.index_bf16)
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&index_scale_[static_cast<size_t>(o)]),
                              slots * sizeof(float)));
  }
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&ring_base_),
                          static_cast<size_t>(shape.layers) * shape.max_requests * ring_bytes_per_request()));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&ring_table_), static_cast<size_t>(shape.max_requests) * 4));
  csa2_ring_table(ring_table_, shape.max_requests, nullptr);
  size_t tail_bytes = 0;
  for (const int f : shape.tail_floats) tail_bytes += static_cast<size_t>(f) * sizeof(float);
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&tail_base_),
                          std::max<size_t>(16, tail_bytes * shape.max_requests)));
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

size_t Csa2StatePool::cache_bytes(const Csa2PoolShape& s) {
  check_shape(s);
  const int64_t blocks = s.token_slots / s.block_tokens;
  size_t total = 0;
  for (const int r : s.cache_ratio) {
    const size_t slots = static_cast<size_t>(blocks) * static_cast<size_t>(s.block_tokens / r);
    total += padded(slots * latent_row_bytes(s.main_format, kCsa2Latent));
    total += padded(slots * (s.index_bf16 ? 2 * kCsa2IndexDim : kCsa2IndexDim));
    if (!s.index_bf16) total += padded(slots * sizeof(float));
  }
  total += padded(static_cast<size_t>(s.layers) * s.max_requests * s.ring_slots *
                  latent_row_bytes(s.ring_format, kCsa2Latent));
  total += padded(static_cast<size_t>(s.max_requests) * 4);
  size_t tail_bytes = 0;
  for (const int f : s.tail_floats) tail_bytes += static_cast<size_t>(f) * sizeof(float);
  total += padded(tail_bytes * s.max_requests);
  total += padded(PagedBlockTable::table_bytes(s.max_requests, blocks));
  return total;
}

void Csa2StatePool::check_ord(int ord, const char* what) const {
  if (!initialized_) throw std::logic_error(std::string("csa2 state pool: ") + what + " before init");
  if (ord < 0 || ord >= caches())
    throw std::out_of_range(std::string("csa2 state pool: ") + what + " cache ordinal " + std::to_string(ord));
}
uint8_t* Csa2StatePool::main(int ord) const {
  check_ord(ord, "main");
  return main_[static_cast<size_t>(ord)];
}
uint8_t* Csa2StatePool::index_k(int ord) const {
  check_ord(ord, "index_k");
  return index_k_[static_cast<size_t>(ord)];
}
float* Csa2StatePool::index_scale(int ord) const {
  check_ord(ord, "index_scale");
  return index_scale_[static_cast<size_t>(ord)];
}
uint8_t* Csa2StatePool::ring(int layer) const {
  if (!initialized_ || layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("csa2 state pool: ring layer " + std::to_string(layer));
  return ring_base_ + static_cast<size_t>(layer) * shape_.max_requests * ring_bytes_per_request();
}
float* Csa2StatePool::tails(int tail_ord) const {
  if (!initialized_ || tail_ord < 0 || tail_ord >= shape_.tail_ordinals)
    throw std::out_of_range("csa2 state pool: tail ordinal " + std::to_string(tail_ord));
  size_t stride = 0;
  for (int o = 0; o < tail_ord; ++o) stride += shape_.tail_floats[static_cast<size_t>(o)];
  return tail_base_ + stride * shape_.max_requests;
}

void Csa2StatePool::reset_request(int req, cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("csa2 state pool: reset_request before init");
  if (req < 0 || req >= shape_.max_requests) throw std::out_of_range("csa2 state pool: request");
  for (int l = 0; l < shape_.layers; ++l)
    DGPP_CUDA_OK(cudaMemsetAsync(ring(l) + static_cast<size_t>(req) * ring_bytes_per_request(), 0,
                                 ring_bytes_per_request(), stream));
  for (int t = 0; t < shape_.tail_ordinals; ++t) {
    const int tf = shape_.tail_floats[static_cast<size_t>(t)];
    DGPP_CUDA_OK(cudaMemsetAsync(tails(t) + static_cast<size_t>(req) * tf, 0, static_cast<size_t>(tf) * 4, stream));
    const int inf = shape_.tail_inf[static_cast<size_t>(t)];
    if (inf > 0)
      csa2_fill_inf(tails(t) + static_cast<size_t>(req) * shape_.tail_floats[static_cast<size_t>(t)] +
                        static_cast<size_t>(shape_.tail_floats[static_cast<size_t>(t)] - inf),
                    inf, stream);
  }
  table_.release_request_blocks(req, stream);
}

void Csa2StatePool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("csa2 state pool: reset_all before init");
  for (int o = 0; o < caches(); ++o) {
    const size_t slots = static_cast<size_t>(entry_slots(o));
    DGPP_CUDA_OK(cudaMemsetAsync(main_[static_cast<size_t>(o)], 0, slots * main_row_bytes(), stream));
    DGPP_CUDA_OK(cudaMemsetAsync(index_k_[static_cast<size_t>(o)], 0, slots * index_row_bytes(), stream));
    if (!shape_.index_bf16)
      DGPP_CUDA_OK(cudaMemsetAsync(index_scale_[static_cast<size_t>(o)], 0, slots * sizeof(float), stream));
  }
  DGPP_CUDA_OK(cudaMemsetAsync(ring_base_, 0,
                               static_cast<size_t>(shape_.layers) * shape_.max_requests * ring_bytes_per_request(), stream));
  if (shape_.tail_ordinals > 0) {
    size_t tail_bytes = 0;
    for (const int f : shape_.tail_floats) tail_bytes += static_cast<size_t>(f) * sizeof(float);
    DGPP_CUDA_OK(cudaMemsetAsync(tail_base_, 0, tail_bytes * shape_.max_requests, stream));
    for (int t = 0; t < shape_.tail_ordinals; ++t) {
      const int tf = shape_.tail_floats[static_cast<size_t>(t)], inf = shape_.tail_inf[static_cast<size_t>(t)];
      if (inf <= 0) continue;
      float* p = tails(t) + static_cast<size_t>(tf - inf);
      for (int r = 0; r < shape_.max_requests; ++r) csa2_fill_inf(p + static_cast<size_t>(r) * tf, inf, stream);
    }
  }
  table_.reset_all(stream);
}

void Csa2StatePool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  for (int o = 0; o < caches(); ++o) {
    const size_t epb = static_cast<size_t>(entries_per_block(o));
    const size_t mb = epb * main_row_bytes();
    DGPP_CUDA_OK(cudaMemcpyAsync(main_[static_cast<size_t>(o)] + static_cast<size_t>(dst) * mb,
                                 main_[static_cast<size_t>(o)] + static_cast<size_t>(src) * mb, mb,
                                 cudaMemcpyDeviceToDevice, stream));
    const size_t kb = epb * index_row_bytes();
    DGPP_CUDA_OK(cudaMemcpyAsync(index_k_[static_cast<size_t>(o)] + static_cast<size_t>(dst) * kb,
                                 index_k_[static_cast<size_t>(o)] + static_cast<size_t>(src) * kb, kb,
                                 cudaMemcpyDeviceToDevice, stream));
    if (!shape_.index_bf16)
      DGPP_CUDA_OK(cudaMemcpyAsync(index_scale_[static_cast<size_t>(o)] + static_cast<size_t>(dst) * epb,
                                   index_scale_[static_cast<size_t>(o)] + static_cast<size_t>(src) * epb,
                                   epb * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
