#!/usr/bin/env python3
"""
gem5 SE Mode Configuration Script for SBR-NPU Project (EEL 5764)
Supports:
  1. Baseline RISC-V O3CPU execution with L1/L2 cache hierarchy
  2. Configurable system interconnect width (64-bit, 128-bit, 256-bit)
  3. Plug-and-play hook for Student 1's memory-mapped SBR-NPU at 0x40000000
"""

import argparse
import os
import sys

import m5
from m5.objects import *

def parse_args():
    parser = argparse.ArgumentParser(description="gem5 O3CPU Baseline & SBR-NPU Runner")
    parser.add_argument("--cmd", required=True, help="Path to RISC-V static binary")
    parser.add_argument("--bus-width-bits", type=int, default=128, choices=[64, 128, 256],
                        help="System interconnect bus width in bits (64, 128, or 256)")
    parser.add_argument("--l1d-size", default="32kB", help="L1 Data Cache size")
    parser.add_argument("--l1i-size", default="32kB", help="L1 Instruction Cache size")
    parser.add_argument("--l2-size", default="256kB", help="L2 Unified Cache size")
    parser.add_argument("--enable-npu", action="store_true",
                        help="Map 0x40000000 MMIO region and attach SBR-NPU if present")
    parser.add_argument("--npu-dim", type=int, default=8, choices=[4, 8, 16],
                        help="Systolic array dimension N (4, 8, or 16)")
    parser.add_argument("--spad-kb", type=int, default=32, choices=[16, 32, 64],
                        help="Total double-buffered scratchpad SRAM capacity in KB (16, 32, 64)")
    return parser.parse_args()

# Standard L1/L2 Cache definitions for gem5 classic memory model
class L1Cache(Cache):
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 20

class L2Cache(Cache):
    assoc = 8
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 32
    tgts_per_mshr = 12

def build_system(args):
    system = System()
    system.clk_domain = SrcClockDomain(clock="2GHz", voltage_domain=VoltageDomain())
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange("512MB")]

    # Out-of-Order RISC-V Baseline Core
    system.cpu = RiscvO3CPU()
    system.cpu.createInterruptController()

    # Parameterized System Interconnect Bus (64-bit = 8B, 128-bit = 16B, 256-bit = 32B)
    bus_width_bytes = args.bus_width_bits // 8
    system.membus = SystemXBar(width=bus_width_bytes)
    system.l2bus = L2XBar(width=bus_width_bytes)

    # L1 I/D Caches
    system.cpu.icache = L1Cache(size=args.l1i_size)
    system.cpu.dcache = L1Cache(size=args.l1d_size)
    system.cpu.icache.cpu_side = system.cpu.icache_port
    system.cpu.dcache.cpu_side = system.cpu.dcache_port
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    # Unified L2 Cache
    system.l2cache = L2Cache(size=args.l2_size)
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports

    # System port & Main Memory Controller
    system.system_port = system.membus.cpu_side_ports
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR4_2400_8x8(range=system.mem_ranges[0])
    system.mem_ctrl.port = system.membus.mem_side_ports

    # Optional hook for Student 1's SBR-NPU SimObject (Base Address: 0x40000000)
    if args.enable_npu and hasattr(m5.objects, "SbrNpu"):
        system.sbr_npu = getattr(m5.objects, "SbrNpu")(
            pio_addr=0x40000000,
            array_dim=args.npu_dim,
            scratchpad_kb=args.spad_kb,
        )
        # Connect PIO responder port and bypassed DMA requestor port directly to membus
        system.sbr_npu.pio = system.membus.mem_side_ports
        system.sbr_npu.dma = system.membus.cpu_side_ports

    # Configure SE Workload
    system.workload = SEWorkload.init_compatible(args.cmd)
    process = Process(cmd=[args.cmd])
    system.cpu.workload = process
    system.cpu.createThreads()

    return system, process

def main():
    args = parse_args()
    system, process = build_system(args)

    root = Root(full_system=False, system=system)
    m5.instantiate()

    # Map virtual address 0x40000000 -> physical address 0x40000000 for user-space MMIO
    if args.enable_npu:
        process.map(0x40000000, 0x40000000, 4096, cacheable=False)

    print(f"[gem5] Starting simulation for {args.cmd} (Bus={args.bus_width_bits}-bit)")
    exit_event = m5.simulate()
    print(f"[gem5] Exited @ tick {m5.curTick()} because {exit_event.getCause()}")

if __name__ == "__main__":
    main()