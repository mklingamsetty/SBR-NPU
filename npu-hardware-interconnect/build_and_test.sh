#!/usr/bin/env bash
set -e

GEM5_DIR="$HOME/Github_Repositories/EEL5764-Computer-Architecture/gem5"
TRACK_A_DIR="$HOME/Github_Repositories/SBR-NPU/npu-hardware-interconnect"

echo "[1/4] Syncing SbrNpu SimObject into gem5 source tree..."
mkdir -p "$GEM5_DIR/src/dev/sbr_npu"
cp "$TRACK_A_DIR/gem5_sbr_npu/"* "$GEM5_DIR/src/dev/sbr_npu/"

echo "[2/4] Building gem5.opt (incremental build)..."
cd "$GEM5_DIR"
scons --ignore-style build/RISCV/gem5.opt -j$(nproc)

echo "[3/4] Cross-compiling standalone RISC-V NPU test binary..."
cd "$TRACK_A_DIR"
riscv64-linux-gnu-gcc -O2 -static tests/test_npu_hw.c -o tests/test_npu_hw.riscv

echo "[4/4] Running SBR-NPU verification in gem5 SE mode..."
"$GEM5_DIR/build/RISCV/gem5.opt" \
    --outdir="$TRACK_A_DIR/m5out_test" \
    "$TRACK_A_DIR/configs/run_sbr_npu.py" \
    --cpu-type=O3CPU \
    --array-dim=8 \
    --scratchpad-kb=32 \
    --bus-width-bits=128 \
    -c "$TRACK_A_DIR/tests/test_npu_hw.riscv"

echo "--- Extracted SBR-NPU Hardware Telemetry ---"
grep "system.sbr_npu." "$TRACK_A_DIR/m5out_test/stats.txt"
