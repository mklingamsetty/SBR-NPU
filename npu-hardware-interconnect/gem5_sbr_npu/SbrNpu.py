from m5.params import *
from m5.objects.Device import BasicPioDevice

class SbrNpu(BasicPioDevice):
    type = 'SbrNpu'
    cxx_header = "dev/sbr_npu/sbr_npu.hh"
    cxx_class = "gem5::SbrNpu"

    array_dim = Param.Unsigned(8, "Systolic array dimension N (NxN)")
    scratchpad_size = Param.Unsigned(32768, "Total double-buffered SRAM capacity in bytes")
    bus_width_bits = Param.Unsigned(128, "System interconnect DMA bus width in bits")
