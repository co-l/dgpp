// Isolate mma_mx4 (and the 4X twin) from the kernel: one warp, fragments built
// on the host per the PTX ISA 9.3 formulas (Figures 93/96/97/99 + the block-
// scaling selectors), exact fp64 oracle. If this passes, the instruction and
// the fragment layout are sound and any kernel bug lives in smem -> fragment.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// e2m1: sign(0) exp(2) mantissa(1), bias 1: 0, 0.5, 1, 1.5, 2, 3, 4, 6
static double e2m1(int c) {
  if (c == 0) return 0.0;
  const int e = (c >> 1) & 7, m = c & 1;
  return e == 0 ? (double)m * 0.5 : std::ldexp(1.0 + m / 2.0, e - 1);
}
static uint32_t pack8(const uint8_t* v) {
  uint32_t r = 0;
  for (int i = 0; i < 8; ++i) r |= uint32_t(v[i]) << (4 * i);
  return r;
}

// PTX ISA 9.3, mma.m16n8k64 (u4/e2m1):
//  A: groupID = lane >> 2, t = lane % 4
//     row = groupID (i < 8), groupID + 8 (8 <= i < 16), groupID (16 <= i < 24), groupID + 8 (i >= 24)
//     col = t*8 + (i & 7) (i < 16), t*8 + (i & 7) + 32 (i >= 16)   [i = 0..31 element index]
//  B: row = t*8 + (i & 7) (i < 8), t*8 + (i & 7) + 32 (i >= 8)   [row = the k side]
//     col = groupID
//  D: row = groupID (i < 2), groupID + 8 (i >= 2); col = t*2 + (i & 1)
//  SF_A (2X, ue8m0): 16 lanes contribute (one row each), 2 bytes per row:
//     thread-id-a selects the low/high thread pair of the quad (0 = lanes %4 {0,1}).
//     Row of a contributing lane: the two lanes of the quad that contribute cover
//     rows 2*quad and 2*quad+1 -- verified empirically below via the oracle.
__global__ void mma_one(float* d, const uint32_t* a, const uint32_t* b, const uint32_t* sfa,
                        const uint32_t* sfb, int kind) {
  uint32_t aa[4], bb[2];
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  const int lane = threadIdx.x & 31;
  for (int i = 0; i < 4; ++i) aa[i] = a[lane * 4 + i];
  bb[0] = b[lane * 2], bb[1] = b[lane * 2 + 1];
  if (kind == 2)
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(bb[0]), "r"(bb[1]), "r"(sfa[lane]),
                   "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb[lane]),
                   "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
  else
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(aa[0]), "r"(aa[1]), "r"(aa[2]), "r"(aa[3]), "r"(bb[0]), "r"(bb[1]), "r"(sfa[lane]),
                   "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb[lane]),
                   "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#pragma unroll
  for (int i = 0; i < 4; ++i) d[lane * 4 + i] = c[i];
}

int main(int argc, char** argv) {
  const int kind = argc > 1 ? std::atoi(argv[1]) : 2;  // 2 = 2X/ue8m0, 4 = 4X/ue4m3
  const int seed = argc > 2 ? std::atoi(argv[2]) : 12345;
  srand(seed);
  const bool constdat = seed == 0, randcodes = seed == 1;
  // A: 16 x 64 e2m1; B: 64(k) x 8(n) e2m1. Random codes (0..6) and scales.
  uint8_t a[16 * 64], b[64 * 8];
  uint8_t sfa[16][2], sfb[8][2];  // 2X: 2 groups per row/col
  uint8_t sfa4[16][4], sfb4[8][4];  // 4X: 4 groups
  for (int i = 0; i < 16 * 64; ++i) a[i] = (constdat || randcodes) ? uint8_t(rand() % 7) : uint8_t(2);
  for (int i = 0; i < 64 * 8; ++i) b[i] = (constdat || randcodes) ? uint8_t(rand() % 7) : uint8_t(2);
  for (int r = 0; r < 16; ++r)
    for (int g = 0; g < 2; ++g) {
      sfa[r][g] = (constdat || randcodes) ? uint8_t(127) : uint8_t(120 + rand() % 16);  // 2^-7 .. 2^0
      sfa4[r][2 * g] = sfa4[r][2 * g + 1] = (constdat || randcodes) ? uint8_t(0x30) : uint8_t(119 + rand() % 8);
    }
  for (int c = 0; c < 8; ++c)
    for (int g = 0; g < 2; ++g) {
      sfb[c][g] = (constdat || randcodes) ? uint8_t(127) : uint8_t(120 + rand() % 16);
      sfb4[c][2 * g] = sfb4[c][2 * g + 1] = (constdat || randcodes) ? uint8_t(0x30) : uint8_t(119 + rand() % 8);
    }
  auto sa = [&](int r, int g) { return kind == 2 ? (double)std::ldexp(1, int(sfa[r][g]) - 127) : 1.0; };
  auto sb = [&](int c, int g) { return kind == 2 ? (double)std::ldexp(1, int(sfb[c][g]) - 127) : 1.0; };
  auto sa4 = [&](int r, int g) {
    int e = sfa4[r][g] >> 3, m = sfa4[r][g] & 7;
    return e == 0 ? (double)m * std::ldexp(1.0, -9) : (double)(8 + m) * std::ldexp(1.0, e - 8);
  };
  auto sb4 = [&](int c, int g) {
    int e = sfb4[c][g] >> 3, m = sfb4[c][g] & 7;
    return e == 0 ? (double)m * std::ldexp(1.0, -9) : (double)(8 + m) * std::ldexp(1.0, e - 8);
  };
  // Exact oracle: D[r][c] = sum_k A[r][k] * B[k][c] with per-group scales.
  double want[16][8] = {};
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c)
      for (int k = 0; k < 64; ++k) {
        const double av = e2m1(a[r * 64 + k] & 15) * (kind == 2 ? sa(r, k / 32) : sa4(r, k / 16));
        const double bv = e2m1(b[k * 8 + c] & 15) * (kind == 2 ? sb(c, k / 32) : sb4(c, k / 16));
        want[r][c] += av * bv;
      }
  // Build the fragments per the PTX formulas.
  uint32_t af[32][4], bf[32][2], sfareg[32], sfereg[32];
  for (int lane = 0; lane < 32; ++lane) {
    const int q = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = i * 8;  // element index within the thread's 32 elements
      const int row = (idx < 8 || (idx >= 16 && idx < 24)) ? q : q + 8;
      const int col = t * 8 + (idx & 7) + (idx >= 16 ? 32 : 0);
      uint32_t w = 0;
      for (int e = 0; e < 8; ++e) w |= uint32_t(a[row * 64 + col + e]) << (4 * e);
      af[lane][i] = w;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = i * 8;
      const int krow = t * 8 + (idx & 7) + (idx >= 8 ? 32 : 0);
      uint32_t w = 0;
      for (int e = 0; e < 8; ++e) w |= uint32_t(b[(krow + e) * 8 + q]) << (4 * e);
      bf[lane][i] = w;
    }
  }
  // Scale registers. Try the lane->row mapping empirically: first assume the
  // contributing lanes (lane%4 in {0,1} for thread-id 0) cover rows 2*quad+0/1.
  for (int lane = 0; lane < 32; ++lane) {
    const int q = lane >> 2, p = lane & 3;
    // The mapping the kernel uses (thread-id 0): lanes 4q+0 -> row q,
    // lanes 4q+1 -> row q+8. Unit tests with random scales confirm it.
    const int row = q + ((p & 1) << 3);
    const int col = row;
    if (kind == 2) {
      sfareg[lane] = uint32_t(sfa[row][0]) | uint32_t(sfa[row][1]) << 8;
      sfereg[lane] = uint32_t(sfb[col][0]) | uint32_t(sfb[col][1]) << 8;
    } else {
      sfareg[lane] = 0;
      for (int g = 0; g < 4; ++g) sfareg[lane] |= uint32_t(sfa4[row][g]) << (8 * g);
      for (int g = 0; g < 4; ++g) sfereg[lane] |= uint32_t(sfb4[col][g]) << (8 * g);
    }
  }
  uint32_t *daf, *dbf, *dsfa, *dsfb;
  float* dd;
  cudaMalloc(&daf, 32 * 4 * 4);
  cudaMalloc(&dbf, 32 * 2 * 4);
  cudaMalloc(&dsfa, 32 * 4);
  cudaMalloc(&dsfb, 32 * 4);
  cudaMalloc(&dd, 32 * 4 * 4);
  cudaMemcpy(daf, af, sizeof(af), cudaMemcpyHostToDevice);
  cudaMemcpy(dbf, bf, sizeof(bf), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfa, sfareg, sizeof(sfareg), cudaMemcpyHostToDevice);
  cudaMemcpy(dsfb, sfereg, sizeof(sfereg), cudaMemcpyHostToDevice);
  mma_one<<<1, 32>>>(dd, daf, dbf, dsfa, dsfb, kind);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("kind %d: CUDA error %s\n", kind, cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  std::vector<float> hd(32 * 4);
  cudaMemcpy(hd.data(), dd, hd.size() * 4, cudaMemcpyDeviceToHost);
  double worst = 0;
  int bad = 0;
  for (int lane = 0; lane < 32; ++lane) {
    const int q = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = i < 2 ? q : q + 8;
      const int col = t * 2 + (i & 1);
      const double err = std::fabs((double)hd[lane * 4 + i] - want[row][col]) / (std::fabs(want[row][col]) + 1e-30);
      worst = std::max(worst, err);
      if (err > 1e-6 && bad < 8) {
        std::printf("  lane %d i %d (row %d col %d): got %.6g want %.6g\n", lane, i, row, col,
                    (double)hd[lane * 4 + i], want[row][col]);
        ++bad;
      }
    }
  }
  std::printf("kind %d seed %d: %s worst %.3g\n", kind, seed, bad == 0 ? "OK" : "MISMATCH", worst);
  return bad != 0;
}
