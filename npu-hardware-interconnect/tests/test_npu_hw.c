#include <stdio.h>
#include <stdint.h>
#include "../../sbr_npu_mmio.h"

int8_t A[16] __attribute__((aligned(64))) = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    1, 1, 1, 1,
    2, 2, 2, 2
};

int8_t B[16] __attribute__((aligned(64))) = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

int32_t C_ws[16] __attribute__((aligned(64))) = {0};
int32_t C_os[16] __attribute__((aligned(64))) = {0};

int main() {
    printf("--- Launching SBR-NPU Hardware Verification ---\n");

    sbr_npu_gemm(A, B, C_ws, 4, 4, 4, DATAFLOW_WS);
    uint64_t ws_cycles = REG_NPU_CYCLES;
    printf("[WS Mode] C[0]=%d, C[5]=%d, C[10]=%d, C[15]=%d | NPU Cycles=%lu\n",
           C_ws[0], C_ws[5], C_ws[10], C_ws[15], (unsigned long)ws_cycles);

    sbr_npu_gemm(A, B, C_os, 4, 4, 4, DATAFLOW_OS);
    uint64_t os_cycles = REG_NPU_CYCLES;
    printf("[OS Mode] C[0]=%d, C[5]=%d, C[10]=%d, C[15]=%d | NPU Cycles=%lu\n",
           C_os[0], C_os[5], C_os[10], C_os[15], (unsigned long)os_cycles);

    return 0;
}
