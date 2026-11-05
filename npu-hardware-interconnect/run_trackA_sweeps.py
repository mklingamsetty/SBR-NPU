#!/usr/bin/env python3
import csv, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
GEM5 = f"{HOME}/Github_Repositories/EEL5764-Computer-Architecture/gem5/build/RISCV/gem5.opt"
CFG  = "configs/run_sbr_npu.py"
BIN  = "tests/bench_npu_64x64.riscv"
OUT  = "results"
os.makedirs(OUT, exist_ok=True)

ARRAY_DIMS = [4, 8, 16]
SRAM_KBS   = [16, 32, 64]
BUS_BITS   = [64, 128, 256]
NAIVE_MEM_BYTES = 2 * (64 * 64 * 64) * (1 + 1 + 4)

def run_one(params):
    arr, sram, bus = params
    tag = f"N{arr}_S{sram}KB_B{bus}b"
    outdir = f"{OUT}/m5out_{tag}"
    cmd = [
        GEM5, f"--outdir={outdir}", CFG,
        "--cpu-type=O3CPU", f"--array-dim={arr}",
        f"--scratchpad-kb={sram}", f"--bus-width-bits={bus}",
        "-c", BIN
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = re.search(r"WS_CYCLES=(\d+)\s+OS_CYCLES=(\d+)", proc.stdout)
    ws_cyc = int(m.group(1)) if m else 0
    os_cyc = int(m.group(2)) if m else 0

    stats = {}
    with open(f"{outdir}/stats.txt") as f:
        for line in f:
            if line.startswith("system.sbr_npu."):
                parts = line.split()
                stats[parts[0].replace("system.sbr_npu.", "")] = float(parts[1])

    npu_cyc  = int(stats.get("npuCycles", 1))
    comp_cyc = int(stats.get("computeCycles", 0))
    dma_cyc  = int(stats.get("dmaCycles", 0))
    hid_cyc  = int(stats.get("dmaHiddenCycles", 0))
    rd_bytes = int(stats.get("dmaBytesRead", 0))
    wr_bytes = int(stats.get("dmaBytesWritten", 0))
    tot_bytes = rd_bytes + wr_bytes
    tiles = int(stats.get("tileCount", 0))

    mem_red = (1.0 - (tot_bytes / NAIVE_MEM_BYTES)) * 100.0
    bus_sat = min(100.0, (dma_cyc / max(1, npu_cyc)) * 100.0)

    print(f"Done {tag:15s} | WS={ws_cyc:5d} | OS={os_cyc:5d} | "
          f"HiddenDMA={hid_cyc:5d} | MemRed={mem_red:5.1f}% | BusSat={bus_sat:5.1f}%")
    return {
        "array_dim": arr, "scratchpad_kb": sram, "bus_width_bits": bus,
        "ws_cycles": ws_cyc, "os_cycles": os_cyc, "total_npu_cycles": npu_cyc,
        "compute_cycles": comp_cyc, "dma_cycles": dma_cyc, "dma_hidden_cycles": hid_cyc,
        "dma_bytes_read": rd_bytes, "dma_bytes_written": wr_bytes,
        "total_dma_bytes": tot_bytes, "total_tiles": tiles,
        "mem_reduction_pct": round(mem_red, 2),
        "bus_saturation_pct": round(bus_sat, 2)
    }

configs = [(a, s, b) for a in ARRAY_DIMS for s in SRAM_KBS for b in BUS_BITS]
print("Launching 27 gem5 SBR-NPU design-space simulations...")
with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
    rows = list(ex.map(run_one, configs))

rows.sort(key=lambda r: (r["array_dim"], r["scratchpad_kb"], r["bus_width_bits"]))
csv_path = f"{OUT}/trackA_27_sweeps.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

plt.figure(figsize=(7, 4.5))
for bus in BUS_BITS:
    sub = [r for r in rows if r["scratchpad_kb"] == 32 and r["bus_width_bits"] == bus]
    plt.plot([f"{r['array_dim']}x{r['array_dim']}" for r in sub],
             [r["total_npu_cycles"] for r in sub], marker="o", linewidth=2, label=f"{bus}-bit Bus")
plt.title("SBR-NPU Execution Cycles vs. Array Size & Bus Width (32KB SRAM)")
plt.xlabel("Systolic Array Dimension (NxN)")
plt.ylabel("Total NPU Execution Cycles (WS + OS)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_array_bus_scaling.png", dpi=300)

sub2 = [r for r in rows if r["array_dim"] == 8 and r["bus_width_bits"] == 128]
fig, ax1 = plt.subplots(figsize=(7, 4.5))
x_labels = [f"{r['scratchpad_kb']} KB" for r in sub2]
ax1.bar(x_labels, [r["total_dma_bytes"] / 1024.0 for r in sub2], width=0.4, label="Off-Chip DMA Traffic (KB)")
ax1.set_xlabel("Double-Buffered Scratchpad Capacity")
ax1.set_ylabel("Total Off-Chip DMA Traffic (KB)")
ax2 = ax1.twinx()
ax2.plot(x_labels, [r["total_npu_cycles"] for r in sub2], color="darkred", marker="s", linewidth=2, label="Total NPU Cycles")
ax2.set_ylabel("Total NPU Execution Cycles")
plt.title("Scratchpad Tile Reuse: Off-Chip DMA Traffic & Latency (8x8, 128-bit)")
fig.tight_layout()
plt.savefig(f"{OUT}/fig2_scratchpad_reuse.png", dpi=300)

sub3 = [r for r in rows if r["scratchpad_kb"] == 32 and r["bus_width_bits"] == 128]
plt.figure(figsize=(7, 4.5))
x = range(len(sub3))
plt.bar([i - 0.18 for i in x], [r["ws_cycles"] for r in sub3], width=0.35, label="Weight-Stationary (WS)")
plt.bar([i + 0.18 for i in x], [r["os_cycles"] for r in sub3], width=0.35, label="Output-Stationary (OS)")
plt.xticks(list(x), [f"{r['array_dim']}x{r['array_dim']}" for r in sub3])
plt.xlabel("Systolic Array Dimension (NxN)")
plt.ylabel("Execution Cycles (64x64 INT8 Layer)")
plt.title("Reconfigurable Dataflow Comparison: WS vs. OS (32KB SRAM, 128-bit Bus)")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_ws_vs_os_dataflow.png", dpi=300)

print(f"\nSaved all 27 sweep results to {csv_path} and 3 plots in {OUT}/")
