// micro_dense_gemm_path_bench: the 0731 dense projections at prefill row
// counts — dgpp's tile GEMM vs the streaming GEMV groups (the decode_mma
// path) vs the prefill pipe kernel, machine-parseable:
//   RES tag path m n k us tflops
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/log.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {

struct Shape {
  const char* tag;
  int n, k;
};

struct Case {
  const char* tag;
  int m, n, k;
};

const std::vector<Shape>& projections() {
  static const std::vector<Shape> kShapes{
      {"wq_a", 1024, 4096},   // q lora
      {"wq_b", 32768, 1024},  // 64 heads x 512 latent
      {"wkv", 512, 4096},     // single kv head
      {"wo", 4096, 32768},    // latent -> hidden
      {"lm_head", 129280, 4096},
  };
  return kShapes;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    DGPP_LOG_ERROR("{} failed: {}", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct Buf {
  uint16_t* act{};
  uint8_t* w{};
  float* s{};
  uint16_t* out{};
};

Buf alloc(int m, int n, int k) {
  Buf b;
  check_cuda(cudaMalloc(&b.act, size_t(m) * k * 2), "malloc act");
  check_cuda(cudaMalloc(&b.w, size_t(n) * k), "malloc w");
  check_cuda(cudaMalloc(&b.s, size_t((n + 127) / 128) * ((k + 127) / 128) * 4),
             "malloc s");
  check_cuda(cudaMalloc(&b.out, size_t(m) * n * 2), "malloc out");
  check_cuda(cudaMemset(b.act, 0x3C, size_t(m) * k * 2), "memset act");
  check_cuda(cudaMemset(b.w, 0x3C, size_t(n) * k), "memset w");
  check_cuda(cudaMemset(b.s, 0x3C, size_t((n + 127) / 128) * ((k + 127) / 128) * 4),
             "memset s");
  return b;
}

// Paths: 0 gemv (streaming groups), 1 tile (legacy GEMM), 2 pipe
// (the prefill pipe kernel via the grid launcher, decode_mma on).
double bench_path(int path, const Case& c, Buf& b, cudaStream_t stream) {
  cudaEvent_t beg{}, end{};
  check_cuda(cudaEventCreate(&beg), "event begin");
  check_cuda(cudaEventCreate(&end), "event end");
  auto launch = [&] {
    if (path == 0)
      dgpp::launch_mma_gemv_fp8_bf16(b.act, c.k, b.w, b.s, b.out, c.m, c.n, c.k, c.n, 7,
                                     7, stream, nullptr, 0);
    else if (path == 2)
      dgpp::launch_scale_gemm_grid_bf16(b.act, c.k, b.w, b.s, b.out, c.m, c.n, c.k,
                                        stream, c.n, 7, 7, /*decode_mma=*/true);
    else
      dgpp::launch_scale_gemm_grid_bf16(b.act, c.k, b.w, b.s, b.out, c.m, c.n, c.k,
                                        stream, c.n, 7, 7, /*decode_mma=*/false);
  };
  const int iters = c.m >= 1024 ? 3 : 10;
  for (int w = 0; w < 2; ++w) launch();
  check_cuda(cudaStreamSynchronize(stream), "warmup sync");
  std::array<double, 3> samples{};
  for (int smp = 0; smp < 3; ++smp) {
    check_cuda(cudaEventRecord(beg, stream), "record begin");
    for (int i = 0; i < iters; ++i) launch();
    check_cuda(cudaEventRecord(end, stream), "record end");
    check_cuda(cudaEventSynchronize(end), "sync end");
    float ms = 0.f;
    check_cuda(cudaEventElapsedTime(&ms, beg, end), "elapsed");
    samples[smp] = double(ms) / iters;
  }
  std::sort(samples.begin(), samples.end());
  const double us = samples[1] * 1000.0;
  const double tflops = double(c.m) * c.n * c.k * 2.0 / (us * 1e-6) / 1e12;
  check_cuda(cudaEventDestroy(beg), "destroy begin");
  check_cuda(cudaEventDestroy(end), "destroy end");
  const char* name = path == 0 ? "gemv" : (path == 2 ? "pipe" : "tile");
  std::printf("RES %s %s m=%d n=%d k=%d %.1f us %.1f TFLOPS\n", c.tag, name, c.m, c.n,
              c.k, us, tflops);
  return us;
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  cudaStream_t stream{};
  check_cuda(cudaStreamCreate(&stream), "stream create");
  int failed = 0;
  const int paths[] = {1, 0, 2};  // tile, gemv, pipe
  for (const int m : {300, 2000}) {
    for (const auto& sh : projections()) {
      const Case c{sh.tag, m, sh.n, sh.k};
      Buf b = alloc(m, sh.n, sh.k);
      double best = 1e30;
      const char* best_name = "?";
      for (const int p : paths) {
        const double us = bench_path(p, c, b, stream);
        if (us < best) {
          best = us;
          best_name = p == 0 ? "gemv" : (p == 2 ? "pipe" : "tile");
        }
      }
      std::printf("RES %s verdict m=%d: best=%s %.1f us\n", sh.tag, m, best_name,
                  best);
      cudaFree(b.act);
      cudaFree(b.w);
      cudaFree(b.s);
      cudaFree(b.out);
      ++failed;  // every shape prints one verdict; count for the exit code
    }
  }
  check_cuda(cudaStreamDestroy(stream), "stream destroy");
  DGPP_LOG_INFO("done");
  return failed == int(projections().size()) * 2 ? 0 : 1;
}
