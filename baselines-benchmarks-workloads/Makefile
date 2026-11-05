# Compiler configuration
HOST_CC   := gcc
RISCV_CC  := riscv64-linux-gnu-gcc

CFLAGS_COMMON := -O3 -Wall -Wextra -I./include -I./workloads
RISCV_FLAGS   := $(CFLAGS_COMMON) -static -march=rv64gc -mabi=lp64d

BIN_DIR := bin
SRC_MN  := workloads/mobilenetv2_dw.c
SRC_TR  := workloads/transformer_qkv.c

# Target binary lists
HOST_MOCK_BINS := $(BIN_DIR)/mobilenet_mock.host \
                  $(BIN_DIR)/transformer_mock.host

RISCV_BASE_BINS := $(BIN_DIR)/mobilenet_cpu_base.riscv \
                   $(BIN_DIR)/transformer_cpu_base.riscv

RISCV_NPU_BINS := $(BIN_DIR)/mobilenet_npu_ws.riscv \
                  $(BIN_DIR)/mobilenet_npu_os.riscv \
                  $(BIN_DIR)/transformer_npu_ws.riscv \
                  $(BIN_DIR)/transformer_npu_os.riscv

.PHONY: all host riscv test clean dirs

all: dirs host riscv

dirs:
	@mkdir -p $(BIN_DIR)

# 1. Host Mock Binaries (for immediate standalone verification without waiting on Student 1)
host: dirs $(HOST_MOCK_BINS)

$(BIN_DIR)/mobilenet_mock.host: $(SRC_MN) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(HOST_CC) $(CFLAGS_COMMON) -DMOCK_NPU $(SRC_MN) -o $@

$(BIN_DIR)/transformer_mock.host: $(SRC_TR) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(HOST_CC) $(CFLAGS_COMMON) -DMOCK_NPU $(SRC_TR) -o $@

# 2. RISC-V Binaries for gem5 SE Mode (O3CPU Baseline + NPU WS/OS Hardware Binaries)
riscv: dirs $(RISCV_BASE_BINS) $(RISCV_NPU_BINS)

$(BIN_DIR)/mobilenet_cpu_base.riscv: $(SRC_MN) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DRUN_CPU_BASELINE $(SRC_MN) -o $@

$(BIN_DIR)/transformer_cpu_base.riscv: $(SRC_TR) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DRUN_CPU_BASELINE $(SRC_TR) -o $@

$(BIN_DIR)/mobilenet_npu_ws.riscv: $(SRC_MN) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DDEFAULT_DATAFLOW=DATAFLOW_WS $(SRC_MN) -o $@

$(BIN_DIR)/mobilenet_npu_os.riscv: $(SRC_MN) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DDEFAULT_DATAFLOW=DATAFLOW_OS $(SRC_MN) -o $@

$(BIN_DIR)/transformer_npu_ws.riscv: $(SRC_TR) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DDEFAULT_DATAFLOW=DATAFLOW_WS $(SRC_TR) -o $@

$(BIN_DIR)/transformer_npu_os.riscv: $(SRC_TR) include/sbr_npu_mmio.h workloads/benchmark_common.h
	$(RISCV_CC) $(RISCV_FLAGS) -DDEFAULT_DATAFLOW=DATAFLOW_OS $(SRC_TR) -o $@

# Run standalone software verification
test: host
	@echo "=== Running Standalone Mock NPU Verification ==="
	./$(BIN_DIR)/mobilenet_mock.host
	./$(BIN_DIR)/transformer_mock.host

clean:
	rm -rf $(BIN_DIR)/*.host $(BIN_DIR)/*.riscv