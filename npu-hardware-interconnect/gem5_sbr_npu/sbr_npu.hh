#ifndef __DEV_SBR_NPU_HH__
#define __DEV_SBR_NPU_HH__

#include <cstdint>
#include <vector>

#include "base/statistics.hh"
#include "dev/io_device.hh"
#include "params/SbrNpu.hh"
#include "sim/eventq.hh"

namespace gem5
{

class SbrNpu : public BasicPioDevice
{
  private:
    // MMIO Register Offsets matching sbr_npu_mmio.h
    static constexpr Addr REG_CTRL_STATUS   = 0x00;
    static constexpr Addr REG_DATAFLOW_MODE = 0x04;
    static constexpr Addr REG_DIM_M         = 0x08;
    static constexpr Addr REG_DIM_N         = 0x0C;
    static constexpr Addr REG_DIM_K         = 0x10;
    static constexpr Addr REG_WEIGHT_ADDR   = 0x18;
    static constexpr Addr REG_ACT_ADDR      = 0x20;
    static constexpr Addr REG_OUT_ADDR      = 0x28;
    static constexpr Addr REG_NPU_CYCLES    = 0x30;

    // Control Bits
    static constexpr uint32_t NPU_START_BIT = 0x01;
    static constexpr uint32_t NPU_BUSY_BIT  = 0x02;
    static constexpr uint32_t NPU_DONE_BIT  = 0x04;

    // Hardware Configuration Parameters (for multidimensional sweeps)
    const uint32_t arrayDim;
    const uint32_t scratchpadSize;
    const uint32_t busWidthBits;

    // Internal MMIO Registers
    uint32_t ctrlStatus   = 0;
    uint32_t dataflowMode = 0;
    uint32_t dimM         = 0;
    uint32_t dimN         = 0;
    uint32_t dimK         = 0;
    uint64_t weightAddr   = 0;
    uint64_t actAddr      = 0;
    uint64_t outAddr      = 0;
    uint64_t lastCycles   = 0;

    // Buffered output computed during NPU execution
    std::vector<int32_t> pendingOutput;

    // Event to model non-blocking hardware execution delay in gem5 ticks
    EventFunctionWrapper finishEvent;

    // Helper functions for cache-bypassing DMA reads/writes in SE mode
    bool dmaReadVirt(Addr vaddr, uint8_t *dst, size_t bytes);
    bool dmaWriteVirt(Addr vaddr, const uint8_t *src, size_t bytes);

    void startNpuOperation();
    void completeNpuOperation();

    struct SbrNpuStats : public statistics::Group
    {
        SbrNpuStats(statistics::Group *parent);
        statistics::Scalar numInvocations;
        statistics::Scalar npuCycles;
        statistics::Scalar computeCycles;
        statistics::Scalar dmaCycles;
        statistics::Scalar dmaHiddenCycles;
        statistics::Scalar dmaBytesRead;
        statistics::Scalar dmaBytesWritten;
        statistics::Scalar totalMacs;
        statistics::Scalar tileCount;
    } stats;

  public:
    SbrNpu(const SbrNpuParams &p);

    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;
};

} // namespace gem5

#endif // __DEV_SBR_NPU_HH__
