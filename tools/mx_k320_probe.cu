// Bisect the constant-data halving: vary one operand's value at a time.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"
#include "kernels/moe_w4a4.hpp"

using namespace dgpp;

int main(int argc, char** argv) {
  // probe rows n k a_code b_code a_scale b_scale
  const int rows = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const uint8_t ac = (uint8_t)std::atoi(argv[4]), bc = (uint8_t)std::atoi(argv[5]);
  const uint8_t as = (uint8_t)std::atoi(argv[6]), bs = (uint8_t)std::atoi(argv[7]);
  std::vector<uint8_t> acodes(size_t(rows) * k / 2),
      ascales(size_t(rows) * mxfp4_act_scale_stride(k), as);
  for (int r = 0; r < rows; ++r) {
    const uint8_t v = static_cast<uint8_t>((r % 6) + 1);  // 0.5, 1, 1.5, 2, 3, 4
    for (int c = 0; c < k / 2; ++c) acodes[static_cast<size_t>(r) * (k / 2) + c] = static_cast<uint8_t>(v * 17);
  }
  (void)ac;
  std::vector<uint8_t> wcodes(size_t(n) * k / 2, 0x11), wscales(size_t(n) * k / 32, bs);
  std::vector<uint8_t> wscales_nv(size_t(n) * ((k / 16 + 7) / 8 * 8), 0x38);
  std::vector<uint8_t> ascales_nv(size_t(rows) * ((k / 16 + 7) / 8 * 8), 0x38);
  std::vector<float> host_gs(rows, 1.0f);  // the NVFP4 kernel reads one per row
  std::vector<MoeSegment> segs{{0, rows, 0}};
  MoeExpertView view;
  uint8_t *dac, *das, *dw, *dws;
  uint8_t *dwsv, *dasv;
  float* dgs;
  MoeSegment* dseg;
  MoeExpertView* dv;
  float* out;
  cudaMalloc(&dac, acodes.size());
  cudaMalloc(&das, ascales.size());
  cudaMalloc(&dw, wcodes.size());
  cudaMalloc(&dws, wscales.size());
  cudaMalloc(&dwsv, wscales_nv.size());
  cudaMalloc(&dasv, ascales_nv.size());
  cudaMalloc(&dgs, size_t(rows) * sizeof(float));
  cudaMalloc(&dseg, sizeof(MoeSegment));
  cudaMalloc(&dv, sizeof(MoeExpertView));
  cudaMalloc(&out, size_t(rows) * n * 4);
  cudaMemcpy(dac, acodes.data(), acodes.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(das, ascales.data(), ascales.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, wcodes.data(), wcodes.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dws, wscales.data(), wscales.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dwsv, wscales_nv.data(), wscales_nv.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dasv, ascales_nv.data(), ascales_nv.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dgs, host_gs.data(), host_gs.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dseg, segs.data(), sizeof(MoeSegment), cudaMemcpyHostToDevice);
  view.payload = dw;
  view.fp4_scales = dws;
  view.fp4_group = 32;
  view.fp4_global = dgs;
  cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
  std::vector<int32_t> actrows(rows);
  for (int i = 0; i < rows; ++i) actrows[i] = i;
  int32_t* dactrows;
  cudaMalloc(&dactrows, actrows.size() * 4);
  cudaMemcpy(dactrows, actrows.data(), actrows.size() * 4, cudaMemcpyHostToDevice);
  if (argc > 9 && std::strcmp(argv[9], "warm") == 0) {  // a random-data MX GEMM first
    std::mt19937 rng(7);
    std::vector<uint8_t> rac(size_t(rows) * k / 2), rws(size_t(n) * k / 2);
    std::vector<uint8_t> ras(size_t(rows) * mxfp4_act_scale_stride(k)), rws2(size_t(n) * k / 32);
    for (auto& v : rac) v = (uint8_t)rng();
    for (auto& v : rws) v = (uint8_t)rng();
    for (auto& v : ras) v = (uint8_t)(rng() % 16 == 0 ? 0 : 118 + rng() % 17);
    for (auto& v : rws2) v = (uint8_t)(rng() % 16 == 0 ? 0 : 118 + rng() % 17);
    uint8_t *dr1 = nullptr, *dr2 = nullptr, *dr3 = nullptr, *dr4 = nullptr;
    float* drout = nullptr;
    cudaMalloc(&dr1, rac.size());
    cudaMalloc(&dr2, ras.size());
    cudaMalloc(&dr3, rws.size());
    cudaMalloc(&dr4, rws2.size());
    cudaMalloc(&drout, size_t(rows) * n * 4);
    cudaMemcpy(dr1, rac.data(), rac.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dr2, ras.data(), ras.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dr3, rws.data(), rws.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dr4, rws2.data(), rws2.size(), cudaMemcpyHostToDevice);
    view.fp4_scales = dr4;
    view.fp4_group = 32;
    cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
    launch_moe_grouped_w4a4_mx_f32(dr1, dr2, dactrows, dseg, 1, rows, dv, 0, drout, n, n, k, nullptr);
    cudaDeviceSynchronize();
    view.fp4_scales = dws;
    cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
    cudaFree(dr1);
    cudaFree(dr2);
    cudaFree(dr3);
    cudaFree(dr4);
    cudaFree(drout);
    std::printf("warm done\n");
  }
  launch_moe_grouped_w4a4_mx_f32(dac, das, dactrows, dseg, 1, rows, dv, 0, out, n, n, k, nullptr);
  cudaDeviceSynchronize();
  std::vector<float> h(size_t(rows) * n);
  cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
  if (argc > 8 && std::strcmp(argv[8], "dump") == 0) {
    const uint8_t* d = reinterpret_cast<const uint8_t*>(h.data());
    int bad = 0;
    std::printf("dump: A smem (stage 0) rows 0..31 x 64B vs expected\n");
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 64; ++c)
        if (d[r * 64 + c] != acodes[static_cast<size_t>(r) * (k / 2) + c]) {
          std::printf("  A row %d byte %d: %02x want %02x\n", r, c, d[r * 64 + c],
                      acodes[static_cast<size_t>(r) * (k / 2) + c]);
          if (++bad > 16) break;
        }
    std::printf("  %s\n", bad == 0 ? "A smem exact" : "A smem BROKEN");
    const uint32_t* af = reinterpret_cast<const uint32_t*>(d + 2048);
    bad = 0;
    for (int wm = 0; wm < 4 && bad < 16; ++wm)
      for (int i = 0; i < 2; ++i)
        for (int kh = 0; kh < 2; ++kh)
          for (int lane = 0; lane < 32; ++lane) {
            const int q = lane >> 2, t = lane & 3;
            const int base = wm * 32 + i * 16;
            const uint32_t want[4] = {
                *reinterpret_cast<const uint32_t*>(&acodes[(base + q) * (k / 2) + kh * 32 + 4 * t]),
                *reinterpret_cast<const uint32_t*>(&acodes[(base + 8 + q) * (k / 2) + kh * 32 + 4 * t]),
                *reinterpret_cast<const uint32_t*>(&acodes[(base + q) * (k / 2) + kh * 32 + 16 + 4 * t]),
                *reinterpret_cast<const uint32_t*>(&acodes[(base + 8 + q) * (k / 2) + kh * 32 + 16 + 4 * t]),
            };
            const uint32_t* got = af + ((wm * 2 + i) * 2 + kh) * 128 + lane * 4;
            for (int j = 0; j < 4; ++j)
              if (got[j] != want[j]) {
                std::printf("  af wm %d i %d kh %d lane %d reg %d: %08x want %08x\n", wm, i, kh, lane, j, got[j],
                            want[j]);
                if (++bad > 16) break;
              }
          }
    std::printf("  %s\n", bad == 0 ? "af exact" : "af BROKEN");
    const uint32_t* sfa = reinterpret_cast<const uint32_t*>(d + 10240);
    bad = 0;
    for (int wm = 0; wm < 4 && bad < 16; ++wm)
      for (int i = 0; i < 2; ++i)
        for (int lane = 0; lane < 32; ++lane)
          if (sfa[(wm * 2 + i) * 32 + lane] != 0x00007F7Fu) {
            std::printf("  sfa wm %d i %d lane %d: %08x\n", wm, i, lane, sfa[(wm * 2 + i) * 32 + lane]);
            ++bad;
          }
    std::printf("  %s\n", bad == 0 ? "sfa exact" : "sfa BROKEN");
    // B smem rows 0..31 x 64B at [12288, 14336).
    const uint8_t* bsm = d + 12288;
    bad = 0;
    for (int r = 0; r < 32 && bad < 16; ++r)
      for (int c = 0; c < 64; ++c)
        if (bsm[r * 64 + c] != wcodes[static_cast<size_t>(r) * (k / 2) + c]) {
          std::printf("  B row %d byte %d: %02x want %02x\n", r, c, bsm[r * 64 + c],
                      wcodes[static_cast<size_t>(r) * (k / 2) + c]);
          ++bad;
        }
    std::printf("  %s\n", bad == 0 ? "B smem exact" : "B smem BROKEN");
    // B fragments: bf[(jp*2+kh)*32 + lane][4]; for n-tile j=jp*2+h, k-offset
    // kh*64, thread (r=lane/4, cc=lane%4): b0 = row j*8+r bytes kh*32+cc*4,
    // b1 = row j*8+r bytes kh*32+16+cc*4.
    const uint32_t* bfr = reinterpret_cast<const uint32_t*>(d + 14336);
    bad = 0;
    for (int j = 0; j < 8 && bad < 16; ++j)
      for (int kh = 0; kh < 2; ++kh)
        for (int lane = 0; lane < 32; ++lane) {
          const int r = lane >> 2, cc = lane & 3;
          const int jp = j >> 1, h = j & 1;
          const uint32_t eb0 = *reinterpret_cast<const uint32_t*>(&bsm[(j * 8 + r) * 64 + kh * 32 + cc * 4]);
          const uint32_t eb1 =
              *reinterpret_cast<const uint32_t*>(&bsm[(j * 8 + r) * 64 + kh * 32 + 16 + cc * 4]);
          const uint32_t* got = bfr + ((jp * 2 + kh) * 32 + lane) * 4;
          if (got[2 * h] != eb0 || got[2 * h + 1] != eb1) {
            std::printf("  bf j %d kh %d lane %d: b0 %08x want %08x | b1 %08x want %08x\n", j, kh, lane,
                        got[2 * h], eb0, got[2 * h + 1], eb1);
            ++bad;
          }
        }
    std::printf("  %s\n", bad == 0 ? "B frags exact" : "B frags BROKEN");
    const uint32_t* sfb = reinterpret_cast<const uint32_t*>(d + 11264);
    bad = 0;
    for (int j = 0; j < 8; ++j)
      for (int lane = 0; lane < 32; ++lane)
        if (sfb[j * 32 + lane] != 0x7F7F7F7Fu) {
          std::printf("  sfb j %d lane %d: %08x\n", j, lane, sfb[j * 32 + lane]);
          ++bad;
        }
    std::printf("  %s\n", bad == 0 ? "sfb exact" : "sfb BROKEN");
    return 0;
  }
  const int iters = argc > 8 ? std::atoi(argv[8]) : 1;
  for (int it = 0; it < iters; ++it) {
    int bad = 0;
    for (int c = 0; c < n; c += 64)
      for (int r = 0; r < rows; ++r) {
        static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        const float want = static_cast<float>(k * mag[(r % 6) + 1] * 0.5);
        if (std::abs(h[static_cast<size_t>(r) * n + c] - want) > 1e-3f * want) ++bad;
      }
    if (bad) {
      std::printf("iter %d: %d bad cells\n", it, bad);
      for (int c = 0; c < n; c += 64) {
        int badrows = 0, firstbad = -1;
        float badval = 0;
        for (int r = 0; r < rows; ++r) {
          static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
          const float want = static_cast<float>(k * mag[(r % 6) + 1] * 0.5);
          if (std::abs(h[static_cast<size_t>(r) * n + c] - want) > 1e-3f * want) {
            if (firstbad < 0) {
              firstbad = r;
              badval = h[static_cast<size_t>(r) * n + c];
            }
            ++badrows;
          }
        }
        if (badrows) std::printf("  col %d: %d bad rows (row %d: %.6g)\n", c, badrows, firstbad, badval);
      }
    }
    launch_moe_grouped_w4a4_mx_f32(dac, das, dactrows, dseg, 1, rows, dv, 0, out, n, n, k, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
  }
  if (iters > 1) {
    std::printf("  (loop done)\n");
    return 0;
  }
  std::printf("rows %d n %d k %d a %d b %d as %d bs %d: mx\n", rows, n, k, ac, bc, as, bs);
  for (int c = 0; c < n; c += 64) {
    int badrows = 0, firstbad = -1;
    float badval = 0;
    for (int r = 0; r < rows; ++r) {
      static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
      const float want = static_cast<float>(k * mag[(r % 6) + 1] * 0.5);
      if (std::abs(h[static_cast<size_t>(r) * n + c] - want) > 1e-3f * want) {
        if (firstbad < 0) {
          firstbad = r;
          badval = h[static_cast<size_t>(r) * n + c];
        }
        ++badrows;
      }
    }
    if (badrows) std::printf("  col %d: %d bad rows (row %d: %.6g)\n", c, badrows, firstbad, badval);
  }
  for (int r = 0; r < rows; ++r) {
    static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const float want = static_cast<float>(k * mag[(r % 6) + 1] * 0.5);
    if (std::abs(h[static_cast<size_t>(r) * n] - want) > 1e-3f * want)
      std::printf("  row %d: %.6g (want %.6g)\n", r, h[static_cast<size_t>(r) * n], want);
  }
  // The NVFP4 twin on the same codes: e4m3 scales (0x38 = 1.0) + the global.
  view.fp4_scales = dwsv;
  view.fp4_group = 16;
  cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
  launch_moe_grouped_w4a4_f32(dac, dasv, dgs, dactrows, dseg, 1, rows, dv, 0, out, n, n, k, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
  std::printf("nvfp4\n");
  for (int r = 0; r < rows; ++r) {
    static const double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const float want = static_cast<float>(k * mag[(r % 6) + 1] * 0.5);
    if (std::abs(h[static_cast<size_t>(r) * n] - want) > 1e-3f * want)
      std::printf("  row %d: %.6g (want %.6g)\n", r, h[static_cast<size_t>(r) * n], want);
  }
  // The W4A16 twin on bf16(0.5) activations: the same quantized weights.
  // (The NVFP4 section re-pointed the view at the e4m3 scales: restore the
  // MX view, whose 0x38 bytes the e8m0 path would read as 2^-71.)
  view.fp4_scales = dws;
  view.fp4_group = 32;
  cudaMemcpy(dv, &view, sizeof(view), cudaMemcpyHostToDevice);
  std::vector<uint16_t> act(size_t(rows) * k);
  for (auto& v : act) v = 0x3F00;  // bf16 0.5
  uint16_t* dact;
  cudaMalloc(&dact, act.size() * 2);
  cudaMemcpy(dact, act.data(), act.size() * 2, cudaMemcpyHostToDevice);
  launch_moe_grouped_mma_fp4_f32(dact, k, dseg, 1, rows, 0, dv, 0, out, n, n, k, nullptr, nullptr, 32);
  cudaDeviceSynchronize();
  cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost);
  std::printf("w4a16\n");
  for (int r = 0; r < rows; ++r) {
    const float want = 0.25f * k;  // act 0.5 (all rows) x B 0.5 x scale 1.0
    if (std::abs(h[static_cast<size_t>(r) * n] - want) > 1e-3f * want)
      std::printf("  row %d: %.6g (want %.6g)\n", r, h[static_cast<size_t>(r) * n], want);
  }
  cudaFree(dact);
  cudaFree(dac);
  cudaFree(das);
  cudaFree(dw);
  cudaFree(dws);
  cudaFree(dseg);
  cudaFree(dv);
  cudaFree(out);
  cudaFree(dactrows);
  return 0;
}
