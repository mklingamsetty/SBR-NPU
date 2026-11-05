#!/usr/bin/env python3
"""
Automated gem5 + McPAT/CACTI Power & Evaluation Metric Pipeline (Student 2)
Computes:
  1. Execution Latency (T_CPU vs T_NPU cycles) & Speedup (T_CPU / T_NPU)
  2. Main Memory (DRAM) Access Reduction (%) & L1/L2 Cache Evictions
  3. System Interconnect Bus Saturation Rate (%)
  4. Dynamic Energy per MAC (pJ/MAC) & Energy-Delay Product (EDP)
"""

import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Calibrated 28nm CACTI 7.0 & McPAT Component Energy Parameters
# Used when compiling energy totals from gem5 + CACTI/McPAT activity factors
# ---------------------------------------------------------------------------
CLOCK_FREQ_HZ = 2.0e9          # 2 GHz clock (0.5 ns period)
CLOCK_PERIOD_S = 1.0 / CLOCK_FREQ_HZ

# O3CPU Core & Cache Energy per event (28nm HP node from McPAT/CACTI characterization)
E_O3_INSTRUCTION_PJ = 42.5     # Fetch + Decode + Rename + Issue + ROB control overhead per inst
E_O3_ALU_OP_PJ      = 4.8      # Integer ALU / scalar multiplier operation
E_L1_ACCESS_PJ      = 14.2     # 32 KB 4-way L1 cache read/write access (pJ)
E_L2_ACCESS_PJ      = 58.0     # 256 KB 8-way L2 cache access (pJ)
E_DRAM_BYTE_PJ      = 160.0    # Off-chip DDR4 access energy per byte transferred (20 pJ/bit)
P_O3_STATIC_MW      = 185.0    # O3CPU + L1/L2 static leakage power (mW)

# SBR-NPU CACTI Scratchpad SRAM per-byte access energy & leakage by capacity (16KB, 32KB, 64KB)
CACTI_SPAD_PARAMS = {
    16: {"e_byte_pj": 0.92, "leakage_mw": 4.2,  "tile_reuse_factor": 0.72},
    32: {"e_byte_pj": 1.18, "leakage_mw": 7.8,  "tile_reuse_factor": 0.88},
    64: {"e_byte_pj": 1.54, "leakage_mw": 14.5, "tile_reuse_factor": 0.95},
}

# SBR-NPU Systolic PE Array (8-bit Multiplier + 32-bit Accumulator + Local Register File)
E_PE_MAC_WS_PJ = 0.38          # Weight-stationary MAC dynamic energy (weights held in local PE reg)
E_PE_MAC_OS_PJ = 0.34          # Output-stationary MAC dynamic energy (32-bit psum held in local PE reg)
P_PE_LEAKAGE_MW = 0.18         # Leakage power per PE (mW)
E_DMA_BUS_BYTE_PJ = 2.4        # On-chip SystemXBar DMA transfer energy per byte (pJ)

WORKLOAD_MACS = {
    "mobilenet":   64 * 32 * 16,       # 32,768 MACs
    "transformer": 3 * 64 * 64 * 64,   # 786,432 MACs
}

# Match flexible stat counter names in gem5 stats.txt
def parse_stats_file(stats_path: Path) -> dict:
    stats = {}
    if not stats_path.exists():
        return stats
    pattern = re.compile(r"^([a-zA-Z0-9_\.\-:]+)\s+([0-9\.eE\+\-]+)")
    with open(stats_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            m = pattern.match(line)
            if m:
                key, val_str = m.group(1), m.group(2)
                try:
                    stats[key] = float(val_str)
                except ValueError:
                    pass
    return stats

def get_stat(stats: dict, candidates: list, default: float = 0.0) -> float:
    for key in candidates:
        if key in stats:
            return stats[key]
        # Substring fallback for varying gem5 hierarchy prefixes
        for k, v in stats.items():
            if k.endswith(key):
                return v
    return default

def generate_mock_gem5_stats(base_dir: Path, sweep_dir: Path):
    """
    Generates realistic mock gem5 stats.txt files adhering to the Day-1 Standardized
    Statistics Naming contract so Student 2 can verify the full pipeline independently.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mock O3CPU Baseline Runs
    baseline_profiles = {
        "mobilenet":   {"cycles": 194500,  "insts": 285000,  "dram_bytes": 118400, "l1_repl": 1420, "l2_repl": 310},
        "transformer": {"cycles": 4620000, "insts": 6840000, "dram_bytes": 2480000, "l1_repl": 38400, "l2_repl": 8920},
    }

    for wl, prof in baseline_profiles.items():
        for bus_bits in [64, 128, 256]:
            bus_scale = {64: 1.14, 128: 1.00, 256: 0.95}[bus_bits]
            cycles = int(prof["cycles"] * bus_scale)
            out_sub = base_dir / f"{wl}_o3_bus{bus_bits}"
            out_sub.mkdir(parents=True, exist_ok=True)
            with open(out_sub / "stats.txt", "w") as f:
                f.write("---------- Begin Simulation Statistics ----------\n")
                f.write(f"simTicks                                 {cycles * 500}\n")
                f.write(f"system.cpu.numCycles                     {cycles}\n")
                f.write(f"simInsts                                 {prof['insts']}\n")
                f.write(f"system.cpu.dcache.replacements           {prof['l1_repl']}\n")
                f.write(f"system.l2cache.replacements              {prof['l2_repl']}\n")
                f.write(f"system.mem_ctrl.bytesRead                {int(prof['dram_bytes'] * 0.72)}\n")
                f.write(f"system.mem_ctrl.bytesWritten             {int(prof['dram_bytes'] * 0.28)}\n")
                f.write(f"system.membus.pktSize                    {prof['dram_bytes']}\n")
                f.write("---------- End Simulation Statistics ----------\n")

    # 2. Mock SBR-NPU Multidimensional Sweep Runs
    for wl, total_macs in WORKLOAD_MACS.items():
        base_dram = baseline_profiles[wl]["dram_bytes"]
        # Minimum theoretical tensor footprint (Weights + Activations + Outputs)
        compulsory_bytes = 9216 if wl == "mobilenet" else 65536
        for df in ["ws", "os"]:
            for dim in [4, 8, 16]:
                for spad in [16, 32, 64]:
                    for bus_bits in [64, 128, 256]:
                        run_id = f"{wl}_{df}_N{dim}_spad{spad}KB_bus{bus_bits}"
                        out_sub = sweep_dir / run_id
                        out_sub.mkdir(parents=True, exist_ok=True)

                        # Compute analytical systolic + DMA cycle latency
                        pe_count = dim * dim
                        compute_cycles = math.ceil(total_macs / pe_count) + (3 * dim)
                        # OS slightly outperforms on deep inner dimension K=64; WS slightly better on K=16
                        if wl == "mobilenet" and df == "ws":
                            compute_cycles = int(compute_cycles * 0.93)
                        elif wl == "transformer" and df == "os":
                            compute_cycles = int(compute_cycles * 0.91)

                        reuse = CACTI_SPAD_PARAMS[spad]["tile_reuse_factor"]
                        dma_bytes = int(compulsory_bytes / reuse + (1.0 - reuse) * 0.22 * base_dram)

                        bus_bytes_per_cycle = bus_bits // 8
                        dma_cycles = math.ceil(dma_bytes / (bus_bytes_per_cycle * 0.75))
                        # Double-buffering overlaps 80% of DMA prefetching with systolic compute
                        npu_cycles = max(compute_cycles, dma_cycles) + int(0.20 * min(compute_cycles, dma_cycles))
                        host_total_cycles = npu_cycles + 1850  # MMIO setup overhead

                        with open(out_sub / "stats.txt", "w") as f:
                            f.write("---------- Begin Simulation Statistics ----------\n")
                            f.write(f"simTicks                                 {host_total_cycles * 500}\n")
                            f.write(f"system.cpu.numCycles                     {host_total_cycles}\n")
                            f.write(f"simInsts                                 4200\n")
                            f.write(f"system.sbr_npu.npu_cycles                {npu_cycles}\n")
                            f.write(f"system.sbr_npu.dma_bytes_transferred     {dma_bytes}\n")
                            f.write(f"system.sbr_npu.mac_count                 {total_macs}\n")
                            f.write(f"system.cpu.dcache.replacements           18\n")
                            f.write(f"system.l2cache.replacements              4\n")
                            f.write(f"system.mem_ctrl.bytesRead                {int(dma_bytes * 0.68)}\n")
                            f.write(f"system.mem_ctrl.bytesWritten             {int(dma_bytes * 0.32)}\n")
                            f.write(f"system.membus.pktSize                    {dma_bytes}\n")
                            f.write("---------- End Simulation Statistics ----------\n")

def evaluate_all_runs(base_dir: Path, sweep_dir: Path, output_csv: Path):
    rows = []
    baseline_Lookup = {}

    # 1. Process O3CPU Baseline Runs
    for wl in ["mobilenet", "transformer"]:
        total_macs = WORKLOAD_MACS[wl]
        for bus_bits in [64, 128, 256]:
            stats_path = base_dir / f"{wl}_o3_bus{bus_bits}" / "stats.txt"
            stats = parse_stats_file(stats_path)
            if not stats:
                continue

            cpu_cycles = get_stat(stats, ["system.cpu.numCycles", "numCycles"], 1.0)
            sim_insts = get_stat(stats, ["simInsts", "system.cpu.committedInsts"], cpu_cycles * 1.4)
            l1_repl = get_stat(stats, ["system.cpu.dcache.replacements::total", "system.cpu.dcache.replacements", "dcache.replacements"], 0.0)
            l2_repl = get_stat(stats, ["system.l2cache.replacements::total", "system.l2cache.replacements", "l2cache.replacements"], 0.0)
            dram_read = get_stat(stats, ["system.mem_ctrl.dram.bytesRead::total", "system.mem_ctrl.bytesRead", "bytesRead::total", "bytesRead"], 0.0)
            dram_write = get_stat(stats, ["system.mem_ctrl.dram.bytesWritten::total", "system.mem_ctrl.bytesWritten", "bytesWritten::total", "bytesWritten"], 0.0)
            dram_bytes = dram_read + dram_write

            # Peak bus bandwidth bytes over elapsed cycles
            peak_bus_bytes = cpu_cycles * (bus_bits // 8)
            bus_saturation_pct = min(100.0, (dram_bytes / max(1.0, peak_bus_bytes)) * 100.0)

            # McPAT + CACTI O3CPU Dynamic & Static Energy Calculation (in picojoules)
            dyn_core_pj = sim_insts * (E_O3_INSTRUCTION_PJ + E_O3_ALU_OP_PJ)
            dyn_cache_pj = (sim_insts * 0.45 * E_L1_ACCESS_PJ) + ((l1_repl + l2_repl) * E_L2_ACCESS_PJ)
            dyn_dram_pj = dram_bytes * E_DRAM_BYTE_PJ
            total_dyn_pj = dyn_core_pj + dyn_cache_pj + dyn_dram_pj

            exec_time_s = cpu_cycles * CLOCK_PERIOD_S
            static_pj = (P_O3_STATIC_MW * 1e-3) * exec_time_s * 1e12
            total_energy_nj = (total_dyn_pj + static_pj) * 1e-3
            pj_per_mac = total_dyn_pj / total_macs
            edp_nj_s = total_energy_nj * exec_time_s

            baseline_Lookup[(wl, bus_bits)] = {
                "cycles": cpu_cycles,
                "dram_bytes": dram_bytes,
                "pj_per_mac": pj_per_mac,
                "edp_nj_s": edp_nj_s,
            }

            rows.append({
                "workload": wl,
                "config_type": "O3CPU_Baseline",
                "dataflow": "Scalar_OoO",
                "array_dim": 0,
                "spad_kb": 0,
                "bus_bits": bus_bits,
                "cycles": int(cpu_cycles),
                "speedup": 1.00,
                "dram_bytes": int(dram_bytes),
                "dram_reduction_pct": 0.00,
                "l1_l2_evictions": int(l1_repl + l2_repl),
                "bus_saturation_pct": round(bus_saturation_pct, 2),
                "pj_per_mac": round(pj_per_mac, 3),
                "total_energy_nj": round(total_energy_nj, 3),
                "edp_nj_s": float(f"{edp_nj_s:.4e}"),
            })

    # 2. Process SBR-NPU Sweep Runs
    run_Pattern = re.compile(r"^(mobilenet|transformer)_(ws|os)_N(\d+)_spad(\d+)KB_bus(\d+)$")
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = run_Pattern.match(sub.name)
        if not m:
            continue
        wl, df, dim_s, spad_s, bus_s = m.groups()
        dim, spad_kb, bus_bits = int(dim_s), int(spad_s), int(bus_s)
        total_macs = WORKLOAD_MACS[wl]

        stats = parse_stats_file(sub / "stats.txt")
        if not stats:
            continue

        npu_cycles = get_stat(stats, ["system.sbr_npu.npuCycles", "system.sbr_npu.npu_cycles", "npuCycles", "npu_cycles"], 1.0)
        dma_bytes = (
            get_stat(stats, ["system.sbr_npu.dmaBytesRead", "dmaBytesRead"], 0.0)
            + get_stat(stats, ["system.sbr_npu.dmaBytesWritten", "dmaBytesWritten"], 0.0)
        )
        if dma_bytes == 0.0:
            dma_bytes = get_stat(stats, ["system.sbr_npu.dma_bytes_transferred", "dma_bytes_transferred"], 0.0)

        l1_repl = get_stat(stats, ["system.cpu.dcache.replacements"], 0.0)
        l2_repl = get_stat(stats, ["system.l2cache.replacements"], 0.0)

        # Retrieve matching O3CPU baseline (defaulting to 128-bit bus baseline if specific width missing)
        base_ref = baseline_Lookup.get((wl, bus_bits)) or baseline_Lookup.get((wl, 128))
        base_cycles = base_ref["cycles"] if base_ref else npu_cycles
        base_dram = base_ref["dram_bytes"] if base_ref else dma_bytes

        speedup = base_cycles / max(1.0, npu_cycles)
        dram_reduction_pct = max(0.0, (1.0 - (dma_bytes / max(1.0, base_dram))) * 100.0)

        peak_bus_bytes = npu_cycles * (bus_bits // 8)
        bus_saturation_pct = min(100.0, (dma_bytes / max(1.0, peak_bus_bytes)) * 100.0)

        # CACTI Scratchpad + Systolic PE Array + DMA Energy Model
        spad_cfg = CACTI_SPAD_PARAMS[spad_kb]
        e_mac_pj = E_PE_MAC_WS_PJ if df == "ws" else E_PE_MAC_OS_PJ
        pe_dyn_pj = total_macs * e_mac_pj
        # Double-buffered scratchpad reads weights + activations and writes partial/final outputs
        spad_access_bytes = total_macs * 2.25 / dim
        spad_dyn_pj = spad_access_bytes * spad_cfg["e_byte_pj"]
        dram_and_dma_dyn_pj = dma_bytes * (E_DRAM_BYTE_PJ + E_DMA_BUS_BYTE_PJ)

        total_dyn_pj = pe_dyn_pj + spad_dyn_pj + dram_and_dma_dyn_pj
        exec_time_s = npu_cycles * CLOCK_PERIOD_S
        total_leakage_mw = spad_cfg["leakage_mw"] + (dim * dim * P_PE_LEAKAGE_MW)
        static_pj = (total_leakage_mw * 1e-3) * exec_time_s * 1e12

        total_energy_nj = (total_dyn_pj + static_pj) * 1e-3
        pj_per_mac = total_dyn_pj / total_macs
        edp_nj_s = total_energy_nj * exec_time_s

        rows.append({
            "workload": wl,
            "config_type": "SBR_NPU",
            "dataflow": df.upper(),
            "array_dim": dim,
            "spad_kb": spad_kb,
            "bus_bits": bus_bits,
            "cycles": int(npu_cycles),
            "speedup": round(speedup, 2),
            "dram_bytes": int(dma_bytes),
            "dram_reduction_pct": round(dram_reduction_pct, 2),
            "l1_l2_evictions": int(l1_repl + l2_repl),
            "bus_saturation_pct": round(bus_saturation_pct, 2),
            "pj_per_mac": round(pj_per_mac, 3),
            "total_energy_nj": round(total_energy_nj, 3),
            "edp_nj_s": float(f"{edp_nj_s:.4e}"),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload", "config_type", "dataflow", "array_dim", "spad_kb", "bus_bits",
        "cycles", "speedup", "dram_bytes", "dram_reduction_pct", "l1_l2_evictions",
        "bus_saturation_pct", "pj_per_mac", "total_energy_nj", "edp_nj_s"
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SUCCESS] Evaluated {len(rows)} configurations -> {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="SBR-NPU McPAT/CACTI Power & Metrics Pipeline")
    parser.add_argument("--base-dir", default="results/m5out_baseline", help="O3CPU stats root directory")
    parser.add_argument("--sweep-dir", default="results/m5out_npu_sweep", help="SBR-NPU sweep stats root directory")
    parser.add_argument("--out-csv", default="results/evaluation_metrics.csv", help="Output CSV path")
    parser.add_argument("--generate-mock-stats", action="store_true",
                        help="Populate realistic mock gem5 stats.txt files to test the pipeline before integration")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    sweep_dir = Path(args.sweep_dir)
    out_csv = Path(args.out_csv)

    if args.generate_mock_stats:
        print("[MOCK] Generating Day-1 contract gem5 stats.txt files for standalone pipeline testing...")
        generate_mock_gem5_stats(base_dir, sweep_dir)

    evaluate_all_runs(base_dir, sweep_dir, out_csv)

if __name__ == "__main__":
    main()