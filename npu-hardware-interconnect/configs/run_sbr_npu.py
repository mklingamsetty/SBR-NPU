import argparse
import m5
from m5.objects import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cmd", required=True)
parser.add_argument("--cpu-type", default="O3CPU")
parser.add_argument("--array-dim", type=int, default=8)
parser.add_argument("--scratchpad-kb", type=int, default=32)
parser.add_argument("--bus-width-bits", type=int, default=128)
args = parser.parse_args()

system = System()
system.clk_domain = SrcClockDomain(clock="2GHz", voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

system.cpu = RiscvO3CPU() if args.cpu_type == "O3CPU" else RiscvTimingSimpleCPU()

system.cpu.icache = Cache(size="32kB", assoc=4, tag_latency=2, data_latency=2, response_latency=2, mshrs=4, tgts_per_mshr=8)
system.cpu.dcache = Cache(size="32kB", assoc=4, tag_latency=2, data_latency=2, response_latency=2, mshrs=16, tgts_per_mshr=8, addr_ranges=[AddrRange("512MB")])

# l1xbar routes 0x0-0x1FFFFFFF to L1 dcache and 0x40000000 directly to SBR-NPU MMIO
system.l1xbar = SystemXBar(forward_latency=0, frontend_latency=0, response_latency=0, width=32)
system.membus = SystemXBar(width=args.bus_width_bits // 8)

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache_port = system.l1xbar.cpu_side_ports
system.cpu.dcache.cpu_side = system.l1xbar.mem_side_ports

system.cpu.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache.mem_side = system.membus.cpu_side_ports
system.cpu.createInterruptController()

system.sbr_npu = SbrNpu(
    pio_addr=0x40000000,
    pio_latency="1ns",
    array_dim=args.array_dim,
    scratchpad_size=args.scratchpad_kb * 1024,
    bus_width_bits=args.bus_width_bits,
)
system.sbr_npu.pio = system.l1xbar.mem_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8(range=system.mem_ranges[0])
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

system.workload = RiscvSEWorkload.init_compatible(args.cmd)
process = Process(cmd=[args.cmd])
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()
process.map(0x40000000, 0x40000000, 0x1000, False)

print(f"**** REAL SIMULATION (SBR-NPU: {args.array_dim}x{args.array_dim}, {args.scratchpad_kb}KB SRAM, {args.bus_width_bits}-bit bus) ****")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
