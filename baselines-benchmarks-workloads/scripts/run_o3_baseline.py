#!/usr/bin/env python3
import argparse
import m5
from m5.objects import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--bus-width-bits", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--l1d-size", default="32kB")
    parser.add_argument("--l1i-size", default="32kB")
    parser.add_argument("--l2-size", default="256kB")
    parser.add_argument("--enable-npu", action="store_true")
    parser.add_argument("--npu-dim", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--spad-kb", type=int, default=32, choices=[16, 32, 64])
    return parser.parse_args()

class L1Cache(Cache):
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 8

class L2Cache(Cache):
    assoc = 8
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 32
    tgts_per_mshr = 8

def main():
    args = parse_args()
    system = System()
    system.clk_domain = SrcClockDomain(clock="2GHz", voltage_domain=VoltageDomain())
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange("512MB")]

    system.cpu = RiscvO3CPU()
    system.cpu.createInterruptController()

    bus_bytes = args.bus_width_bits // 8
    system.l1xbar = SystemXBar(forward_latency=0, frontend_latency=0, response_latency=0, width=32)
    system.l2bus = L2XBar(width=bus_bytes)
    system.membus = SystemXBar(width=bus_bytes)

    system.cpu.icache = L1Cache(size=args.l1i_size)
    system.cpu.dcache = L1Cache(size=args.l1d_size, addr_ranges=[AddrRange("512MB")])

    system.cpu.icache.cpu_side = system.cpu.icache_port
    system.cpu.dcache_port = system.l1xbar.cpu_side_ports
    system.cpu.dcache.cpu_side = system.l1xbar.mem_side_ports

    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

    system.l2cache = L2Cache(size=args.l2_size, addr_ranges=[AddrRange("512MB")])
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports

    system.system_port = system.membus.cpu_side_ports
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR4_2400_8x8(range=system.mem_ranges[0])
    system.mem_ctrl.port = system.membus.mem_side_ports

    if args.enable_npu:
        system.sbr_npu = SbrNpu(
            pio_addr=0x40000000,
            pio_latency="1ns",
            array_dim=args.npu_dim,
            scratchpad_size=args.spad_kb * 1024,
            bus_width_bits=args.bus_width_bits,
        )
        system.sbr_npu.pio = system.l1xbar.mem_side_ports

    system.workload = RiscvSEWorkload.init_compatible(args.cmd)
    process = Process(cmd=[args.cmd])
    system.cpu.workload = process
    system.cpu.createThreads()

    root = Root(full_system=False, system=system)
    m5.instantiate()

    if args.enable_npu:
        process.map(0x40000000, 0x40000000, 0x1000, False)

    print(f"[gem5] Starting simulation for {args.cmd} (Bus={args.bus_width_bits}-bit)")
    exit_event = m5.simulate()
    print(f"[gem5] Exited @ tick {m5.curTick()} because {exit_event.getCause()}")

if __name__ in ("__main__", "__m5_main__"):
    main()
