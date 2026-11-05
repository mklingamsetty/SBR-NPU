#ifndef SBR_NPU_MMIO_H
#define SBR_NPU_MMIO_H

#include <stdint.h>

// Base address mapped in gem5 SE mode Python config
#define SBR_NPU_BASE       0x40000000UL

// Register offsets
#define REG_CTRL_STATUS    (*(volatile uint32_t *)(SBR_NPU_BASE + 0x00))
#define REG_DATAFLOW_MODE  (*(volatile uint32_t *)(SBR_NPU_BASE + 0x04))
#define REG_DIM_M          (*(volatile uint32_t *)(SBR_NPU_BASE + 0x08))
#define REG_DIM_N          (*(volatile uint32_t *)(SBR_NPU_BASE + 0x0C))
#define REG_DIM_K          (*(volatile uint32_t *)(SBR_NPU_BASE + 0x10))
#define REG_WEIGHT_ADDR    (*(volatile uint64_t *)(SBR_NPU_BASE + 0x18))
#define REG_ACT_ADDR       (*(volatile uint64_t *)(SBR_NPU_BASE + 0x20))
#define REG_OUT_ADDR       (*(volatile uint64_t *)(SBR_NPU_BASE + 0x28))
#define REG_NPU_CYCLES     (*(volatile uint64_t *)(SBR_NPU_BASE + 0x30))

// Control bits and Dataflow modes
#define NPU_START_BIT      0x01
#define NPU_BUSY_BIT       0x02
#define NPU_DONE_BIT       0x04

#define DATAFLOW_WS        0
#define DATAFLOW_OS        1

// Shared driver function called by both synthetic tests and AI workloads
static inline void sbr_npu_gemm(const int8_t *act, const int8_t *weight, int32_t *out,
                                uint32_t M, uint32_t N, uint32_t K, uint32_t mode) {
#ifdef MOCK_NPU
    // Chase uses this software golden model to verify benchmark math independently
    (void)mode;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                acc += (int32_t)act[m * K + k] * (int32_t)weight[k * N + n];
            }
            out[m * N + n] = acc;
        }
    }
#else
    // Murali & final integration use real MMIO writes to the gem5 NPU object
    REG_DATAFLOW_MODE = mode;
    REG_DIM_M         = M;
    REG_DIM_N         = N;
    REG_DIM_K         = K;
    REG_ACT_ADDR      = (uint64_t)(uintptr_t)act;
    REG_WEIGHT_ADDR   = (uint64_t)(uintptr_t)weight;
    REG_OUT_ADDR      = (uint64_t)(uintptr_t)out;

    // Trigger DMA + Systolic execution and poll until completion
    REG_CTRL_STATUS   = NPU_START_BIT;
    while ((REG_CTRL_STATUS & NPU_DONE_BIT) == 0) {
        // Busy-wait for NPU completion
    }
#endif
}

#endif // SBR_NPU_MMIO_H