#!/usr/bin/env bash
set -euo pipefail

# Path to your gem5 RISC-V binary (override via: GEM5_BIN=/path/to/gem5.opt ./scripts/run_simulations.sh)
GEM5_BIN="${GEM5_BIN:-$HOME/gem5/build/RISCV/gem5.opt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

BIN_DIR="$ROOT_DIR/bin"
BASE_OUT="$ROOT_DIR/results/m5out_baseline"
SWEEP_OUT="$ROOT_DIR/results/m5out_npu_sweep"
CONFIG_SCRIPT="$SCRIPT_DIR/run_o3_baseline.py"

MODE="${1:-baseline}" # Pass "baseline" (Phase 1) or "npu_sweep" (Phase 2)

if [[ ! -x "$GEM5_BIN" ]]; then
    echo "[ERROR] gem5 binary not found at: $GEM5_BIN"
    echo "Set GEM5_BIN=/path/to/gem5/build/RISCV/gem5.opt and re-run."
    exit 1
fi

echo "=== Step 1: Compiling RISC-V Benchmark Binaries ==="
make -C "$ROOT_DIR" riscv

if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
    echo "=== Step 2: Running Phase 1 gem5 O3CPU Baselines ==="
    for WORKLOAD in mobilenet transformer; do
        for BUS_BITS in 64 128 256; do
            OUT_DIR="$BASE_OUT/${WORKLOAD}_o3_bus${BUS_BITS}"
            mkdir -p "$OUT_DIR"
            echo " -> Simulating O3CPU Baseline: $WORKLOAD (Bus=${BUS_BITS}-bit)"
            "$GEM5_BIN" --outdir="$OUT_DIR" "$CONFIG_SCRIPT" \
                --cmd="$BIN_DIR/${WORKLOAD}_cpu_base.riscv" \
                --bus-width-bits="$BUS_BITS" \
                > "$OUT_DIR/sim_stdout.log" 2>&1
        done
    done
    echo "[DONE] O3CPU baseline stats saved under $BASE_OUT/"
fi

if [[ "$MODE" == "npu_sweep" || "$MODE" == "all" ]]; then
    echo "=== Step 3: Running Phase 2 SBR-NPU Design Space Sweep ==="
    for WORKLOAD in mobilenet transformer; do
        for DF in ws os; do
            for DIM in 4 8 16; do
                for SPAD in 16 32 64; do
                    for BUS_BITS in 64 128 256; do
                        RUN_ID="${WORKLOAD}_${DF}_N${DIM}_spad${SPAD}KB_bus${BUS_BITS}"
                        OUT_DIR="$SWEEP_OUT/$RUN_ID"
                        mkdir -p "$OUT_DIR"
                        echo " -> Simulating SBR-NPU: $RUN_ID"
                        "$GEM5_BIN" --outdir="$OUT_DIR" "$CONFIG_SCRIPT" \
                            --cmd="$BIN_DIR/${WORKLOAD}_npu_${DF}.riscv" \
                            --enable-npu \
                            --npu-dim="$DIM" \
                            --spad-kb="$SPAD" \
                            --bus-width-bits="$BUS_BITS" \
                            > "$OUT_DIR/sim_stdout.log" 2>&1
                    done
                done
            done
        done
    done
    echo "[DONE] NPU sweep stats saved under $SWEEP_OUT/"
fi