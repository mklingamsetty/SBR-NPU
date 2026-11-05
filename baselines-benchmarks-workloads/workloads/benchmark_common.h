#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/sbr_npu_mmio.h"

#ifndef DEFAULT_DATAFLOW
#define DEFAULT_DATAFLOW DATAFLOW_WS
#endif

// Read RISC-V hardware cycle CSR (rdcycle) when running on gem5 O3CPU
static inline uint64_t read_cpu_cycles(void) {
#if defined(__riscv)
    uint64_t cycles = 0;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return 0;
#endif
}

// Deterministic pseudo-random INT8 tensor initialization (avoids external file I/O in gem5 SE mode)
static inline void init_int8_tensor(int8_t *buf, uint32_t size, uint32_t seed) {
    uint32_t state = seed ? seed : 0x13579BDFU;
    for (uint32_t i = 0; i < size; ++i) {
        state = state * 1664525U + 1013904223U;
        // Bound values to [-16, 15] to prevent artificial overflow while exercising signed INT8 multipliers
        buf[i] = (int8_t)(((int32_t)(state >> 24) % 32) - 16);
    }
}

// Software INT8 GEMM baseline executed on the gem5 O3CPU core (Tiled for cache locality comparison)
static inline void cpu_baseline_gemm(const int8_t *act, const int8_t *weight, int32_t *out,
                                     uint32_t M, uint32_t N, uint32_t K) {
    const uint32_t TILE = 16;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            out[m * N + n] = 0;
        }
    }

    for (uint32_t m0 = 0; m0 < M; m0 += TILE) {
        uint32_t m_lim = (m0 + TILE < M) ? (m0 + TILE) : M;
        for (uint32_t k0 = 0; k0 < K; k0 += TILE) {
            uint32_t k_lim = (k0 + TILE < K) ? (k0 + TILE) : K;
            for (uint32_t n0 = 0; n0 < N; n0 += TILE) {
                uint32_t n_lim = (n0 + TILE < N) ? (n0 + TILE) : N;
                for (uint32_t m = m0; m < m_lim; ++m) {
                    for (uint32_t k = k0; k < k_lim; ++k) {
                        int32_t a_val = (int32_t)act[m * K + k];
                        for (uint32_t n = n0; n < n_lim; ++n) {
                            out[m * N + n] += a_val * (int32_t)weight[k * N + n];
                        }
                    }
                }
            }
        }
    }
}

// Unified dispatcher that selects between O3CPU Baseline, Mock NPU verification, and Live NPU MMIO
static inline uint64_t execute_gemm_kernel(const int8_t *act, const int8_t *weight, int32_t *out,
                                           uint32_t M, uint32_t N, uint32_t K, uint32_t mode) {
#if defined(RUN_CPU_BASELINE)
    (void)mode;
    uint64_t start = read_cpu_cycles();
    cpu_baseline_gemm(act, weight, out, M, N, K);
    uint64_t end = read_cpu_cycles();
    return (end - start);
#elif defined(MOCK_NPU)
    sbr_npu_gemm(act, weight, out, M, N, K, mode);
    return 0;
#else
    sbr_npu_gemm(act, weight, out, M, N, K, mode);
    return REG_NPU_CYCLES;
#endif
}

// Verification helper used when compiled with -DMOCK_NPU
static inline int verify_against_golden(const int8_t *act, const int8_t *weight, const int32_t *npu_out,
                                        uint32_t M, uint32_t N, uint32_t K, const char *layer_name) {
    int32_t *ref = (int32_t *)malloc(sizeof(int32_t) * M * N);
    if (!ref) {
        fprintf(stderr, "[ERROR] Memory allocation failed during verification.\n");
        return -1;
    }
    cpu_baseline_gemm(act, weight, ref, M, N, K);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
        if (ref[i] != npu_out[i]) {
            if (mismatches < 5) {
                fprintf(stderr, "[MISMATCH] %s idx %u: expected %d, got %d\n",
                        layer_name, i, ref[i], npu_out[i]);
            }
            mismatches++;
        }
    }
    free(ref);

    if (mismatches == 0) {
        printf("[VERIFY PASS] %s (%ux%ux%u): 0 mismatches against golden reference.\n",
               layer_name, M, N, K);
        return 0;
    } else {
        printf("[VERIFY FAIL] %s (%ux%ux%u): %u mismatches detected!\n",
               layer_name, M, N, K, mismatches);
        return 1;
    }
}

#endif // BENCHMARK_COMMON_H