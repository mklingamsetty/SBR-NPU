#include "dev/sbr_npu/sbr_npu.hh"
#include <algorithm>
#include <cmath>
#include "base/logging.hh"
#include "cpu/thread_context.hh"
#include "mem/packet_access.hh"
#include "mem/page_table.hh"
#include "mem/port_proxy.hh"
#include "sim/process.hh"
#include "sim/system.hh"

namespace gem5
{

SbrNpu::SbrNpuStats::SbrNpuStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(numInvocations, statistics::units::Count::get(), "NPU invocations"),
      ADD_STAT(npuCycles, statistics::units::Cycle::get(), "Total NPU cycles"),
      ADD_STAT(computeCycles, statistics::units::Cycle::get(), "Systolic compute cycles"),
      ADD_STAT(dmaCycles, statistics::units::Cycle::get(), "Raw DMA cycles"),
      ADD_STAT(dmaHiddenCycles, statistics::units::Cycle::get(), "DMA cycles hidden by double buffer"),
      ADD_STAT(dmaBytesRead, statistics::units::Byte::get(), "DMA bytes read"),
      ADD_STAT(dmaBytesWritten, statistics::units::Byte::get(), "DMA bytes written"),
      ADD_STAT(totalMacs, statistics::units::Count::get(), "Total INT8 MACs"),
      ADD_STAT(tileCount, statistics::units::Count::get(), "Total SRAM tiles")
{
}

SbrNpu::SbrNpu(const SbrNpuParams &p)
    : BasicPioDevice(p, 0x1000),
      arrayDim(p.array_dim),
      scratchpadSize(p.scratchpad_size),
      busWidthBits(p.bus_width_bits),
      finishEvent([this]{ completeNpuOperation(); }, name() + ".finishEvent"),
      stats(this)
{
}

bool SbrNpu::dmaReadVirt(Addr vaddr, uint8_t *dst, size_t bytes)
{
    if (sys->threads.empty()) return false;
    auto *proc = sys->threads[0]->getProcessPtr();
    if (!proc || !proc->pTable) return false;
    for (size_t i = 0; i < bytes; ++i) {
        Addr paddr = 0;
        if (!proc->pTable->translate(vaddr + i, paddr)) return false;
        sys->physProxy.readBlob(paddr, dst + i, 1);
    }
    return true;
}

bool SbrNpu::dmaWriteVirt(Addr vaddr, const uint8_t *src, size_t bytes)
{
    if (sys->threads.empty()) return false;
    auto *proc = sys->threads[0]->getProcessPtr();
    if (!proc || !proc->pTable) return false;
    for (size_t i = 0; i < bytes; ++i) {
        Addr paddr = 0;
        if (!proc->pTable->translate(vaddr + i, paddr)) return false;
        sys->physProxy.writeBlob(paddr, src + i, 1);
    }
    return true;
}

void SbrNpu::startNpuOperation()
{
    if (dimM == 0 || dimN == 0 || dimK == 0) {
        ctrlStatus = NPU_DONE_BIT;
        return;
    }
    ctrlStatus = NPU_BUSY_BIT;

    const size_t actBytes = static_cast<size_t>(dimM) * dimK;
    const size_t weightBytes = static_cast<size_t>(dimK) * dimN;
    const size_t outElements = static_cast<size_t>(dimM) * dimN;

    std::vector<int8_t> actBuf(actBytes, 0);
    std::vector<int8_t> weightBuf(weightBytes, 0);
    pendingOutput.assign(outElements, 0);

    dmaReadVirt(actAddr, reinterpret_cast<uint8_t *>(actBuf.data()), actBytes);
    dmaReadVirt(weightAddr, reinterpret_cast<uint8_t *>(weightBuf.data()), weightBytes);

    for (uint32_t m = 0; m < dimM; ++m) {
        for (uint32_t n = 0; n < dimN; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < dimK; ++k) {
                acc += static_cast<int32_t>(actBuf[m * dimK + k]) *
                       static_cast<int32_t>(weightBuf[k * dimN + n]);
            }
            pendingOutput[m * dimN + n] = acc;
        }
    }

    uint32_t maxTileDim = (scratchpadSize <= 16384) ? 16 : (scratchpadSize <= 32768 ? 24 : 32);
    

    const uint32_t tilesM = (dimM + maxTileDim - 1) / maxTileDim;
    const uint32_t tilesN = (dimN + maxTileDim - 1) / maxTileDim;
    const uint32_t tilesK = (dimK + maxTileDim - 1) / maxTileDim;
    const uint64_t totalTiles = static_cast<uint64_t>(tilesM) * tilesN * tilesK;

    const uint32_t sysM = (dimM + arrayDim - 1) / arrayDim;
    const uint32_t sysN = (dimN + arrayDim - 1) / arrayDim;
    const uint32_t sysK = (dimK + arrayDim - 1) / arrayDim;

    uint64_t totalComp = (dataflowMode == 0)
        ? (static_cast<uint64_t>(sysN) * sysK * arrayDim +
           static_cast<uint64_t>(sysM) * sysN * sysK * (2 * arrayDim - 1))
        : (static_cast<uint64_t>(sysM) * sysN * (dimK + 4 * arrayDim - 2));

    const uint32_t busBpc = std::max(1u, busWidthBits / 8);
    const uint64_t bytesRd = static_cast<uint64_t>(weightBytes) * tilesM +
                             static_cast<uint64_t>(actBytes) * tilesN;
    const uint64_t bytesWr = (dataflowMode == 0)
        ? (static_cast<uint64_t>(outElements) * 4 * tilesK)
        : (static_cast<uint64_t>(outElements) * 4);

    const uint64_t rawDma = ((bytesRd + bytesWr + busBpc - 1) / busBpc) + (totalTiles * 12);
    const uint64_t avgDma = std::max<uint64_t>(1, rawDma / std::max<uint64_t>(1, totalTiles));
    const uint64_t avgComp = std::max<uint64_t>(1, totalComp / std::max<uint64_t>(1, totalTiles));
    const uint64_t overlap = (totalTiles > 1) ? (totalTiles - 1) * std::max(avgDma, avgComp) : 0;
    const uint64_t effCycles = avgDma + overlap + avgComp;
    const uint64_t hidCycles = (rawDma + totalComp > effCycles) ? (rawDma + totalComp - effCycles) : 0;

    lastCycles = effCycles;
    stats.numInvocations++;
    stats.npuCycles += effCycles;
    stats.computeCycles += totalComp;
    stats.dmaCycles += rawDma;
    stats.dmaHiddenCycles += hidCycles;
    stats.dmaBytesRead += bytesRd;
    stats.dmaBytesWritten += bytesWr;
    stats.totalMacs += static_cast<uint64_t>(dimM) * dimN * dimK;
    stats.tileCount += totalTiles;

    schedule(finishEvent, curTick() + clockPeriod() * effCycles);
}

void SbrNpu::completeNpuOperation()
{
    const size_t outBytes = pendingOutput.size() * sizeof(int32_t);
    if (outBytes > 0) {
        dmaWriteVirt(outAddr, reinterpret_cast<const uint8_t *>(pendingOutput.data()), outBytes);
    }
    ctrlStatus = NPU_DONE_BIT;
}

Tick SbrNpu::read(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr;
    pkt->makeResponse();
    if (pkt->getSize() == 8) {
        pkt->setLE<uint64_t>((offset == REG_NPU_CYCLES) ? lastCycles : 0);
    } else if (pkt->getSize() == 4) {
        uint32_t val = 0;
        if (offset == REG_CTRL_STATUS) val = ctrlStatus;
        else if (offset == REG_DATAFLOW_MODE) val = dataflowMode;
        else if (offset == REG_DIM_M) val = dimM;
        else if (offset == REG_DIM_N) val = dimN;
        else if (offset == REG_DIM_K) val = dimK;
        else if (offset == REG_NPU_CYCLES) val = static_cast<uint32_t>(lastCycles);
        pkt->setLE<uint32_t>(val);
    }
    return pioDelay;
}

Tick SbrNpu::write(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr;
    pkt->makeResponse();
    if (pkt->getSize() == 8) {
        uint64_t val = pkt->getLE<uint64_t>();
        if (offset == REG_WEIGHT_ADDR) weightAddr = val;
        else if (offset == REG_ACT_ADDR) actAddr = val;
        else if (offset == REG_OUT_ADDR) outAddr = val;
    } else if (pkt->getSize() == 4) {
        uint32_t val = pkt->getLE<uint32_t>();
        if (offset == REG_CTRL_STATUS && (val & NPU_START_BIT) && !(ctrlStatus & NPU_BUSY_BIT))
            startNpuOperation();
        else if (offset == REG_DATAFLOW_MODE) dataflowMode = val;
        else if (offset == REG_DIM_M) dimM = val;
        else if (offset == REG_DIM_N) dimN = val;
        else if (offset == REG_DIM_K) dimK = val;
    }
    return pioDelay;
}

} // namespace gem5
