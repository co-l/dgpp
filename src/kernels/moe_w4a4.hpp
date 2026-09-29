#pragma once
// W4A4 NVFP4 grouped MoE GEMM on sm_121a's block-scaled tensor cores
// (mma kind::mxf4nvf4, OMMA.SF.16864). See moe_w4a4.cu.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"

namespace dgpp {

// The activation scale buffer's row stride in bytes: k/16 rounded up to 8.
size_t nvfp4_act_scale_stride(int k);

// bf16 rows [rows, k] (stride x_stride elements) -> NVFP4: codes [rows, k/2]
// (e2m1 pairs, low nibble first), scales [rows, nvfp4_act_scale_stride(k)]
// (e4m3, one per 16), gs[rows] (fp32 per-row global). x ~= e2m1 * s * gs.
// A static global (the checkpoint's input_scale; over-range values clip) is
// *static_gs_dev when that pointer is given, else static_gs; a global <= 0
// means a dynamic amax / (6 * 448) per row.
void launch_quantize_rows_nvfp4(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, float* gs, cudaStream_t stream, float static_gs = 0.f,
                                const float* static_gs_dev = nullptr);
// The MoE activation fused in (moe_swiglu_clamp's math and roundings): gate/up rows
// [rows, stride] -> NVFP4 codes of silu(gate) * up, bitwise the swiglu -> quantize chain.
void launch_swiglu_quantize_rows_nvfp4(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                       float limit, uint8_t* codes, uint8_t* scales, float* gs,
                                       cudaStream_t stream, float static_gs,
                                       const float* static_gs_dev = nullptr);

// The MXFP4 (e8m0 scales per 32, no global) activation scale buffer's row
// stride in bytes: k/32 rounded up to 4.
size_t mxfp4_act_scale_stride(int k);

// bf16 rows [rows, k] (stride x_stride elements) -> MXFP4: codes [rows, k/2]
// (e2m1 pairs, low nibble first), scales [rows, mxfp4_act_scale_stride(k)]
// (e8m0, one per 32, 2^(b - 127); x ~= e2m1 * scale). No global scale.
void launch_quantize_rows_mx(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                             uint8_t* scales, cudaStream_t stream);
// The MoE activation fused in (moe_swiglu_clamp's math and roundings): gate/up rows
// [rows, stride] -> MXFP4 codes of silu(gate) * up, bitwise the swiglu -> quantize chain.
void launch_swiglu_quantize_rows_mx(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                    float limit, uint8_t* codes, uint8_t* scales, cudaStream_t stream);

// out[seg.row0 + m, :n] = act(row) . W(expert, which)^T for every segment,
// act(row) the quantized row act_rows[seg.row0 + m] (or seg.row0 + m when
// act_rows is null). Same segment/view/out conventions as
// launch_moe_grouped_mma_fp4_{bf16,f32}; k % 64 == 0.
void launch_moe_grouped_w4a4_bf16(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                  const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                  const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                  int k, cudaStream_t stream);
void launch_moe_grouped_w4a4_f32(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                 const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                 const MoeExpertView* views, int which, float* out, size_t out_stride, int n, int k,
                                 cudaStream_t stream);

// The MXFP4 (e8m0 scales, no global) twins of the NVFP4 launchers above: the
// MX experts' fp4_group (32) weights unchanged, the activations in the MX
// quantization above, mma kind::mxf4nvf4 scale_vec::2X.
void launch_moe_grouped_w4a4_mx_bf16(const uint8_t* codes, const uint8_t* scales,
                                     const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                     const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                     int k, cudaStream_t stream);
void launch_moe_grouped_w4a4_mx_f32(const uint8_t* codes, const uint8_t* scales,
                                    const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                    const MoeExpertView* views, int which, float* out, size_t out_stride, int n,
                                    int k, cudaStream_t stream);

}  // namespace dgpp
