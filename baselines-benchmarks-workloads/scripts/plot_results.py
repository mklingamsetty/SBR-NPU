#!/usr/bin/env python3
"""
Generates IEEE Report & Presentation Figures for SBR-NPU Evaluation (Student 2)
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

def load_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["array_dim"] = int(r["array_dim"])
            r["spad_kb"] = int(r["spad_kb"])
            r["bus_bits"] = int(r["bus_bits"])
            r["cycles"] = int(r["cycles"])
            r["speedup"] = float(r["speedup"])
            r["dram_reduction_pct"] = float(r["dram_reduction_pct"])
            r["bus_saturation_pct"] = float(r["bus_saturation_pct"])
            r["pj_per_mac"] = float(r["pj_per_mac"])
            r["edp_nj_s"] = float(r["edp_nj_s"])
            rows.append(r)
    return rows

def save_fig(fig, out_dir: Path, name: str):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_speedup_by_array_dim(rows, out_dir: Path):
    # Filter to nominal 32 KB scratchpad, 128-bit bus
    dims = [4, 8, 16]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=False)
    titles = {"mobilenet": "MobileNetV2 (3x3 Depthwise)", "transformer": "Transformer Attention (64x64 QKV)"}

    for ax, wl in zip(axes, ["mobilenet", "transformer"]):
        ws_vals = [next(r["speedup"] for r in rows if r["workload"] == wl and r["config_type"] == "SBR_NPU"
                        and r["dataflow"] == "WS" and r["array_dim"] == d and r["spad_kb"] == 32 and r["bus_bits"] == 128)
                   for d in dims]
        os_vals = [next(r["speedup"] for r in rows if r["workload"] == wl and r["config_type"] == "SBR_NPU"
                        and r["dataflow"] == "OS" and r["array_dim"] == d and r["spad_kb"] == 32 and r["bus_bits"] == 128)
                   for d in dims]

        x = np.arange(len(dims))
        width = 0.35
        ax.bar(x - width/2, ws_vals, width, label="Weight-Stationary (WS)", color="#1f77b4", edgecolor="black")
        ax.bar(x + width/2, os_vals, width, label="Output-Stationary (OS)", color="#ff7f0e", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}x{d}" for d in dims])
        ax.set_xlabel("Systolic Array Dimension (N x N)")
        ax.set_ylabel("Speedup over O3CPU (T_CPU / T_NPU)")
        ax.set_title(titles[wl])
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

    save_fig(fig, out_dir, "fig1_speedup_vs_array_dim")

def plot_dram_reduction_by_spad(rows, out_dir: Path):
    spads = [16, 32, 64]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(spads))
    width = 0.35

    mn_vals = [next(r["dram_reduction_pct"] for r in rows if r["workload"] == "mobilenet" and r["config_type"] == "SBR_NPU"
                    and r["dataflow"] == "WS" and r["array_dim"] == 8 and r["spad_kb"] == s and r["bus_bits"] == 128)
               for s in spads]
    tr_vals = [next(r["dram_reduction_pct"] for r in rows if r["workload"] == "transformer" and r["config_type"] == "SBR_NPU"
                    and r["dataflow"] == "OS" and r["array_dim"] == 8 and r["spad_kb"] == s and r["bus_bits"] == 128)
               for s in spads]

    ax.bar(x - width/2, mn_vals, width, label="MobileNetV2 3x3 DW", color="#2ca02c", edgecolor="black")
    ax.bar(x + width/2, tr_vals, width, label="Transformer 64x64 QKV", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} KB" for s in spads])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Double-Buffered Scratchpad Capacity")
    ax.set_ylabel("Main Memory Access Reduction (%)")
    ax.set_title("DRAM Traffic Reduction via Scratchpad Tile Reuse")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    save_fig(fig, out_dir, "fig2_dram_reduction_vs_spad")

def plot_bus_saturation_by_width(rows, out_dir: Path):
    buses = [64, 128, 256]
    dims = [4, 8, 16]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    markers = {4: "o", 8: "s", 16: "^"}

    for d in dims:
        sat_vals = [next(r["bus_saturation_pct"] for r in rows if r["workload"] == "transformer"
                         and r["config_type"] == "SBR_NPU" and r["dataflow"] == "OS"
                         and r["array_dim"] == d and r["spad_kb"] == 32 and r["bus_bits"] == b)
                    for b in buses]
        ax.plot(buses, sat_vals, marker=markers[d], linewidth=2, label=f"SBR-NPU {d}x{d} Array")

    ax.set_xticks(buses)
    ax.set_xticklabels([f"{b}-bit" for b in buses])
    ax.set_xlabel("System Interconnect Bus Width")
    ax.set_ylabel("Memory Bus Saturation Rate (%)")
    ax.set_title("Interconnect Bandwidth Saturation (Transformer 64x64 QKV)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    save_fig(fig, out_dir, "fig3_bus_saturation_vs_width")

def plot_energy_and_edp(rows, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    configs = ["O3CPU", "NPU 4x4", "NPU 8x8", "NPU 16x16"]

    for idx, wl in enumerate(["mobilenet", "transformer"]):
        base_r = next(r for r in rows if r["workload"] == wl and r["config_type"] == "O3CPU_Baseline" and r["bus_bits"] == 128)
        npu_r = [next(r for r in rows if r["workload"] == wl and r["config_type"] == "SBR_NPU"
                      and r["dataflow"] == "WS" and r["array_dim"] == d and r["spad_kb"] == 32 and r["bus_bits"] == 128)
                 for d in [4, 8, 16]]

        pj_vals = [base_r["pj_per_mac"]] + [r["pj_per_mac"] for r in npu_r]
        edp_norm = [1.0] + [r["edp_nj_s"] / base_r["edp_nj_s"] for r in npu_r]

        x = np.arange(len(configs))
        offset = (idx - 0.5) * 0.35
        label = "MobileNetV2 DW" if wl == "mobilenet" else "Transformer QKV"
        color = "#4c72b0" if wl == "mobilenet" else "#dd8452"

        ax1.bar(x + offset, pj_vals, 0.35, label=label, color=color, edgecolor="black")
        ax2.bar(x + offset, edp_norm, 0.35, label=label, color=color, edgecolor="black")

    ax1.set_yscale("log")
    ax1.set_xticks(np.arange(len(configs)))
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("Dynamic Energy per Op (pJ/MAC, Log Scale)")
    ax1.set_title("Hardware Energy Efficiency (pJ/MAC)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(len(configs)))
    ax2.set_xticklabels(configs)
    ax2.set_ylabel("Normalized EDP vs. O3CPU (Log Scale)")
    ax2.set_title("Normalized Energy-Delay Product (Lower is Better)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend()

    save_fig(fig, out_dir, "fig4_energy_edp_comparison")

def main():
    csv_path = Path("results/evaluation_metrics.csv")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    plot_speedup_by_array_dim(rows, out_dir)
    plot_dram_reduction_by_spad(rows, out_dir)
    plot_bus_saturation_by_width(rows, out_dir)
    plot_energy_and_edp(rows, out_dir)
    print(f"[SUCCESS] Generated 4 IEEE figures (.png & .pdf) in {out_dir}/")

if __name__ == "__main__":
    main()