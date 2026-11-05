#!/usr/bin/env bash
set -euo pipefail

GEM5_BIN="${GEM5_BIN:-$HOME/Github_Repositories/EEL5764-Computer-Architecture/gem5/build/RISCV/gem5.opt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

BIN_DIR="$ROOT_DIR/bin"
BASE_OUT="$ROOT_DIR/results/m5out_baseline"
SWEEP_OUT="$ROOT_DIR/results/m5out_npu_sweep"
CONFIG_SCRIPT="$SCRIPT_DIR/run_o3_baseline.py"
MODE="${1:-all}"
MAX_JOBS=$(nproc)

make -C "$ROOT_DIR" riscv

if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
    echo "=== Running 6 gem5 O3CPU Baselines in Parallel ==="
    for WORKLOAD in mobilenet transformer; do
        for BUS_BITS in 64 128 256; do
            OUT_DIR="$BASE_OUT/${WORKLOAD}_o3_bus${BUS_BITS}"
            mkdir -p "$OUT_DIR"
            "$GEM5_BIN" --outdir="$OUT_DIR" "$CONFIG_SCRIPT" \
                --cmd="$BIN_DIR/${WORKLOAD}_cpu_base.riscv" \
                --bus-width-bits="$BUS_BITS" > "$OUT_DIR/sim_stdout.log" 2>&1 &
        done
    done
    wait
    echo "[DONE] All 6 O3CPU baselines finished."
fi

if [[ "$MODE" == "npu_sweep" || "$MODE" == "all" ]]; then
    echo "=== Running 108 Live SBR-NPU Sweep Simulations in Parallel ==="
    job_count=0
    for WORKLOAD in mobilenet transformer; do
        for DF in ws os; do
            for DIM in 4 8 16; do
                for SPAD in 16 32 64; do
                    for BUS_BITS in 64 128 256; do
                        RUN_ID="${WORKLOAD}_${DF}_N${DIM}_spad${SPAD}KB_bus${BUS_BITS}"
                        OUT_DIR="$SWEEP_OUT/$RUN_ID"
                        mkdir -p "$OUT_DIR"
                        "$GEM5_BIN" --outdir="$OUT_DIR" "$CONFIG_SCRIPT" \
                            --cmd="$BIN_DIR/${WORKLOAD}_npu_${DF}.riscv" \
                            --enable-npu --npu-dim="$DIM" --spad-kb="$SPAD" \
                            --bus-width-bits="$BUS_BITS" > "$OUT_DIR/sim_stdout.log" 2>&1 &
                        job_count=$((job_count + 1))
                        if (( job_count % MAX_JOBS == 0 )); then wait; fi
                    done
                done
            done
        done
    done
    wait
    echo "[DONE] All 108 SBR-NPU sweep simulations finished."
fi
