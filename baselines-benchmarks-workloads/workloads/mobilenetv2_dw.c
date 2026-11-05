#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "benchmark_common.h"

// MobileNetV2 3x3 Depthwise Convolution layer dimensions:
// Input feature map: H_IN=10, W_IN=10 (8x8 valid output patches -> M=64 spatial patches)
// Kernel: 3x3 spatial window (9 weights padded to K=16 to align with 4x4, 8x8, 16x16 systolic grids)
// Channels: N=32 depthwise output channels
#define H_IN        10
#define W_IN        10
#define K_H         3
#define K_W         3
#define H_OUT       (H_IN - K_H + 1) // 8
#define W_OUT       (W_IN - K_W + 1) // 8

#define DW_M        (H_OUT * W_OUT)  // 64 spatial output patches
#define DW_K        16               // 3x3 = 9 taps padded to 16
#define DW_N        32               // 32 channels

// Static aligned buffers for deterministic physical/virtual memory mapping in gem5 SE mode
static int8_t  raw_input[H_IN * W_IN * DW_N] __attribute__((aligned(64)));
static int8_t  im2col_act[DW_M * DW_K]       __attribute__((aligned(64)));
static int8_t  dw_weights[DW_K * DW_N]       __attribute__((aligned(64)));
static int32_t dw_output[DW_M * DW_N]        __attribute__((aligned(64)));

// Lower 3x3 spatial neighborhoods into contiguous row patches for systolic DMA streaming
static void mobilenetv2_im2col(const int8_t *in_tensor, int8_t *patch_matrix) {
    memset(patch_matrix, 0, sizeof(int8_t) * DW_M * DW_K);
    uint32_t row = 0;
    for (uint32_t r = 0; r < H_OUT; ++r) {
        for (uint32_t c = 0; c < W_OUT; ++c) {
            uint32_t k_idx = 0;
            for (uint32_t kr = 0; kr < K_H; ++kr) {
                for (uint32_t kc = 0; kc < K_W; ++kc) {
                    // Representative channel slice lowered into patch row
                    uint32_t in_idx = ((r + kr) * W_IN + (c + kc)) * DW_N;
                    patch_matrix[row * DW_K + k_idx] = in_tensor[in_idx];
                    k_idx++;
                }
            }
            row++;
        }
    }
}

int main(void) {
    init_int8_tensor(raw_input, H_IN * W_IN * DW_N, 0x1001U);
    init_int8_tensor(dw_weights, DW_K * DW_N, 0x2002U);

    // Zero out padding rows in the weight matrix (indices k = 9..15)
    for (uint32_t k = 9; k < DW_K; ++k) {
        for (uint32_t n = 0; n < DW_N; ++n) {
            dw_weights[k * DW_N + n] = 0;
        }
    }

    // Perform im2col lowering to prepare activation tiles
    mobilenetv2_im2col(raw_input, im2col_act);

    // Execute kernel (O3CPU Baseline, Mock NPU, or Live Hardware MMIO)
    uint64_t cycles = execute_gemm_kernel(
        im2col_act, dw_weights, dw_output,
        DW_M, DW_N, DW_K, DEFAULT_DATAFLOW
    );

    uint64_t total_macs = (uint64_t)DW_M * DW_N * DW_K;

#if defined(MOCK_NPU)
    int status = verify_against_golden(
        im2col_act, dw_weights, dw_output,
        DW_M, DW_N, DW_K, "MobileNetV2_3x3_DW"
    );
    printf("[MOCK_NPU] MobileNetV2_3x3_DW | Out[0]=%d Out[255]=%d | Total MACs=%lu\n",
           dw_output[0], dw_output[255], (unsigned long)total_macs);
    return status;
#elif defined(RUN_CPU_BASELINE)
    printf("[O3CPU_BASELINE] MobileNetV2_3x3_DW | M=%u N=%u K=%u | CPU_Cycles=%lu | Total_MACs=%lu | Out[0]=%d\n",
           DW_M, DW_N, DW_K, (unsigned long)cycles, (unsigned long)total_macs, dw_output[0]);
    return 0;
#else
    printf("[SBR_NPU_HW] MobileNetV2_3x3_DW | Mode=%s | M=%u N=%u K=%u | NPU_Cycles=%lu | Total_MACs=%lu | Out[0]=%d\n",
           (DEFAULT_DATAFLOW == DATAFLOW_WS) ? "WS" : "OS",
           DW_M, DW_N, DW_K, (unsigned long)cycles, (unsigned long)total_macs, dw_output[0]);
    return 0;
#endif
}