#include "BanbaOps.h"
#include "BanbaSugar.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::shared {

void GroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct shared group allocator op region\n";
}

} // namespace banba::shared

namespace banba::shared::ga {

void FreeEntryCalcOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  unsigned ldQSize = config.getFifoDepth_L();
  unsigned stQSize = config.getFifoDepth_S();

  // Tentative free load-queue entries: ld_q_head - ld_q_tail
  WrapSubOp ldWrapSub =
      BanbaSugar::createOp<WrapSubOp>(getLdQHead(), getLdQTail(), ldQSize);

  // Tentative free store-queue entries: st_q_head - st_q_tail
  WrapSubOp stWrapSub =
      BanbaSugar::createOp<WrapSubOp>(getStQHead(), getStQTail(), stQSize);

  // Zero-extend the load-queue pointer difference
  BanbaValue ldWrapSubZext = BanbaSugar::zext(ldWrapSub.getDifference(),
                                              config.getLdFreeEntriesWidth());

  // Zero-extend the store-queue pointer difference
  BanbaValue stWrapSubZext = BanbaSugar::zext(stWrapSub.getDifference(),
                                              config.getStFreeEntriesWidth());

  // The number of entries of each queue, as constants
  BanbaValue maxNumLoads(ldQSize, config.getLdFreeEntriesWidth());
  BanbaValue maxNumStores(stQSize, config.getStFreeEntriesWidth());

  // If the load queue is empty, it has all of its entries free, not 0.
  // Otherwise, the wrapped subtraction of the pointers is correct
  BanbaValue ldNumFreeEntries =
      BanbaSugar::select(getLdQEmpty(), maxNumLoads, ldWrapSubZext);

  // If the store queue is empty, it has all of its entries free, not 0.
  // Otherwise, the wrapped subtraction of the pointers is correct
  BanbaValue stNumFreeEntries =
      BanbaSugar::select(getStQEmpty(), maxNumStores, stWrapSubZext);

  // Results
  driveLdNumFreeEntries(ldNumFreeEntries);
  driveStNumFreeEntries(stNumFreeEntries);
}

void ReadyGenOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  // For each group, make a ready calculation and save it in a vector
  SmallVector<Value> gaReadyVec;
  for (unsigned group = 0; group < config.getNumBBs(); ++group) {
    // The number of loads and stores in the group, as constants
    BanbaValue numLoads(config.getNumLoads()[group],
                        config.getLdFreeEntriesWidth());
    BanbaValue numStores(config.getNumStores()[group],
                         config.getStFreeEntriesWidth());

    // Compare the available space in each queue
    // with the space required to allocate the group
    BanbaValue loadsFit = getLdNumFreeEntries() >= numLoads;
    BanbaValue storesFit = getStNumFreeEntries() >= numStores;

    // The group is ready only if there is enough space
    // for both its loads and its stores
    BanbaValue ready = loadsFit & storesFit;

    // Push into the vector, which is later converted to a hw.array
    gaReadyVec.push_back(ready);
  }

  BanbaValue gaReady = BanbaSugar::concatToArray(gaReadyVec);

  // Results
  driveGaReady(gaReady);
}

void PortIdxGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.port_idx_gen op region\n";
}

void OrderMatrixGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.order_matrix_gen op region\n";
}

void CountExtractOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  Value numLoads;
  if (config.getNumBBs() > 1) {
    // ROM of the number of loads in each group
    Value numLoadsRom = BanbaSugar::arrayConstant(config.getNumLoads(),
                                                  config.getLdCountWidth());

    // Mux of the ROM by the one-hot group_allocate
    OneHotMuxOp numLoadsMux =
        BanbaSugar::createOp<OneHotMuxOp>(getGroupAllocate(), numLoadsRom);
    numLoads = numLoadsMux.getSelected();
  } else {
    // With a single group there is nothing to select between:
    // the number of loads is that group's when it allocates, otherwise zero
    BanbaValue groupAllocate = getGroupAllocate();
    BanbaValue groupNumLoads(config.getNumLoads()[0], config.getLdCountWidth());
    BanbaValue zero(0, config.getLdCountWidth());
    numLoads = BanbaSugar::select(groupAllocate[0], groupNumLoads, zero);
  }

  Value numStores;
  if (config.getNumBBs() > 1) {
    // ROM of the number of stores in each group
    Value numStoresRom = BanbaSugar::arrayConstant(config.getNumStores(),
                                                   config.getStCountWidth());

    // Mux of the ROM by the one-hot group_allocate
    OneHotMuxOp numStoresMux =
        BanbaSugar::createOp<OneHotMuxOp>(getGroupAllocate(), numStoresRom);
    numStores = numStoresMux.getSelected();
  } else {
    // With a single group there is nothing to select between:
    // the number of stores is that group's when it allocates, otherwise zero
    BanbaValue groupAllocate = getGroupAllocate();
    BanbaValue groupNumStores(config.getNumStores()[0],
                              config.getStCountWidth());
    BanbaValue zero(0, config.getStCountWidth());
    numStores = BanbaSugar::select(groupAllocate[0], groupNumStores, zero);
  }

  // Results
  driveNumLoads(numLoads);
  driveNumStores(numStores);
}

void AllocatingGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.allocating_gen op region\n";
}

} // namespace banba::shared::ga
