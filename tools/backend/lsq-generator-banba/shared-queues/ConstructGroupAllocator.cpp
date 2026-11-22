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
  llvm::outs() << "Construct shared.ga.free_entry_calc op region\n";
}

void ReadyGenOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  // For each group, make a ready calculation and save it in a vector
  SmallVector<Value> groupAllocateReadyVec;
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
    groupAllocateReadyVec.push_back(ready);
  }

  BanbaValue groupAllocateReady =
      BanbaSugar::concatToArray(groupAllocateReadyVec);

  // Results
  driveGroupAllocateReady(groupAllocateReady);
}

void PortIdxGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.port_idx_gen op region\n";
}

void OrderMatrixGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.order_matrix_gen op region\n";
}

void CountExtractOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.count_extract op region\n";
}

void AllocatingGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.allocating_gen op region\n";
}

} // namespace banba::shared::ga
