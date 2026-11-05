#include <stdio.h>
#include <stdint.h>
#include "benchmark_common.h"

// Transformer Attention Q, K, V Projection dimensions (64x64 matrices)
#define ATTN_M      64   // Sequence length (tokens)
#define ATTN_K      64   // Input embedding dimension (d_model)
#define ATTN_N      64   // Projection head dimension (d_k / d_v)

// 64-byte aligned static tensors in user space for clean DMA streaming
static int8_t  X_in[ATTN_M * ATTN_K]   __attribute__((aligned(64)));
static int8_t  W_Q[ATTN_K * ATTN_N]    __attribute__((aligned(64)));
static int8_t  W_K[ATTN_K * ATTN_N]    __attribute__((aligned(64)));
static int8_t  W_V[ATTN_K * ATTN_N]    __attribute__((aligned(64)));

static int32_t Out_Q[ATTN_M * ATTN_N]  __attribute__((aligned(64)));
static int32_t Out_K[ATTN_M * ATTN_N]  __attribute__((aligned(64)));
static int32_t Out_V[ATTN_M * ATTN_N]  __attribute__((aligned(64)));

int main(void) {
    init_int8_tensor(X_in, ATTN_M * ATTN_K, 0xA001U);
    init_int8_tensor(W_Q,  ATTN_K * ATTN_N, 0xB002U);
    init_int8_tensor(W_K,  ATTN_K * ATTN_N, 0xC003U);
    init_int8_tensor(W_V,  ATTN_K * ATTN_N, 0xD004U);

    uint64_t total_cycles = 0;

    // 1. Query Projection: Q = X * W_Q (64x64x64)
    total_cycles += execute_gemm_kernel(
        X_in, W_Q, Out_Q, ATTN_M, ATTN_N, ATTN_K, DEFAULT_DATAFLOW
    );

    // 2. Key Projection: K = X * W_K (64x64x64)
    total_cycles += execute_gemm_kernel(
        X_in, W_K, Out_K, ATTN_M, ATTN_N, ATTN_K, DEFAULT_DATAFLOW
    );

    // 3. Value Projection: V = X * W_V (64x64x64)
    total_cycles += execute_gemm_kernel(
        X_in, W_V, Out_V, ATTN_M, ATTN_N, ATTN_K, DEFAULT_DATAFLOW
    );

    uint64_t total_macs = 3ULL * ATTN_M * ATTN_N * ATTN_K; // 786,432 MACs

#if defined(MOCK_NPU)
    int err = 0;
    err |= verify_against_golden(X_in, W_Q, Out_Q, ATTN_M, ATTN_N, ATTN_K, "Transformer_Proj_Q");
    err |= verify_against_golden(X_in, W_K, Out_K, ATTN_M, ATTN_N, ATTN_K, "Transformer_Proj_K");
    err |= verify_against_golden(X_in, W_V, Out_V, ATTN_M, ATTN_N, ATTN_K, "Transformer_Proj_V");
    printf("[MOCK_NPU] Transformer_QKV_64x64 | Q[0]=%d K[0]=%d V[0]=%d | Total MACs=%lu\n",
           Out_Q[0], Out_K[0], Out_V[0], (unsigned long)total_macs);
    return err;
#elif defined(RUN_CPU_BASELINE)
    printf("[O3CPU_BASELINE] Transformer_QKV_64x64 | 3x(64x64x64) | CPU_Cycles=%lu | Total_MACs=%lu | Q[0]=%d K[0]=%d V[0]=%d\n",
           (unsigned long)total_cycles, (unsigned long)total_macs, Out_Q[0], Out_K[0], Out_V[0]);
    return 0;
#else
    printf("[SBR_NPU_HW] Transformer_QKV_64x64 | Mode=%s | 3x(64x64x64) | NPU_Cycles=%lu | Total_MACs=%lu | Q[0]=%d K[0]=%d V[0]=%d\n",
           (DEFAULT_DATAFLOW == DATAFLOW_WS) ? "WS" : "OS",
           (unsigned long)total_cycles, (unsigned long)total_macs, Out_Q[0], Out_K[0], Out_V[0]);
    return 0;
#endif
}