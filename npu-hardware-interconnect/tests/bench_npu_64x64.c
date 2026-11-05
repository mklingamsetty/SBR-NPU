#include <stdio.h>
#include <stdint.h>
#include "../../sbr_npu_mmio.h"

#define DIM 64

int8_t A[DIM * DIM] __attribute__((aligned(64)));
int8_t B[DIM * DIM] __attribute__((aligned(64)));
int32_t C_ws[DIM * DIM] __attribute__((aligned(64)));
int32_t C_os[DIM * DIM] __attribute__((aligned(64)));

int main() {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            A[i * DIM + j] = (int8_t)((i + j) % 7 + 1);
            B[i * DIM + j] = (i == j) ? 2 : 1;
        }
    }

    // 1. Weight-Stationary (WS) 64x64 INT8 GEMM
    sbr_npu_gemm(A, B, C_ws, DIM, DIM, DIM, DATAFLOW_WS);
    uint64_t ws_cycles = REG_NPU_CYCLES;

    // 2. Output-Stationary (OS) 64x64 INT8 GEMM
    sbr_npu_gemm(A, B, C_os, DIM, DIM, DIM, DATAFLOW_OS);
    uint64_t os_cycles = REG_NPU_CYCLES;

    printf("[SWEEP_RESULT] WS_CYCLES=%lu OS_CYCLES=%lu C0=%d C_LAST=%d\n",
           (unsigned long)ws_cycles, (unsigned long)os_cycles,
           C_ws[0], C_os[DIM * DIM - 1]);
    return 0;
}
