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
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  // Convert the one-hot group_allocate from one hw.array value
  // to a C++ list holding each group's allocate bit,
  // used as the mask select of both queues' port index vectors
  BanbaValue groupAllocate = getGroupAllocate();
  SmallVector<BanbaValue> groupAllocateBits;
  for (unsigned group = 0; group < config.getNumBBs(); ++group)
    groupAllocateBits.push_back(groupAllocate[group]);

  // An all-zero port index vector,
  // for masking out port indices which are not allocating.
  // The width is 1 bit even when there is 1 load port
  Value ldQPortIdxZero = BanbaSugar::arrayConstant(
      /*values=*/{}, config.getLdPortIdxWidth(),
      /*numElements=*/config.getFifoDepth_L());

  // For each group,
  // declare its unshifted port indices as a constant vector
  // masked by that group's allocate bit
  SmallVector<Value> ldQMaskedGroupPorts;
  for (unsigned group = 0; group < config.getNumBBs(); ++group) {
    // The unshifted port indices of the group
    Value groupPorts = BanbaSugar::arrayConstant(
        config.getLdPortIdx()[group], config.getLdPortIdxWidth(),
        /*numElements=*/config.getFifoDepth_L());

    // Mask the group's port indices by its allocate bit
    Value maskedGroupPorts = BanbaSugar::select(groupAllocateBits[group],
                                                groupPorts, ldQPortIdxZero);
    ldQMaskedGroupPorts.push_back(maskedGroupPorts);
  }

  // OR reduce the masked vectors
  OrReduceOp ldQPortIdxReduced =
      BanbaSugar::createOp<OrReduceOp>(ldQMaskedGroupPorts);

  // Align the per-entry port indices to match the load queue
  // by shifting them
  CyclicLeftShiftOp ldQPortIdx = BanbaSugar::createOp<CyclicLeftShiftOp>(
      ldQPortIdxReduced.getReduced(), getLdQTail());

  // An all-zero port index vector,
  // for masking out port indices which are not allocating.
  // The width is 1 bit even when there is 1 store port
  Value stQPortIdxZero = BanbaSugar::arrayConstant(
      /*values=*/{}, config.getStPortIdxWidth(),
      /*numElements=*/config.getFifoDepth_S());

  // For each group,
  // declare its unshifted port indices as a constant vector
  // masked by that group's allocate bit
  SmallVector<Value> stQMaskedGroupPorts;
  for (unsigned group = 0; group < config.getNumBBs(); ++group) {
    // The unshifted port indices of the group
    Value groupPorts = BanbaSugar::arrayConstant(
        config.getStPortIdx()[group], config.getStPortIdxWidth(),
        /*numElements=*/config.getFifoDepth_S());

    // Mask the group's port indices by its allocate bit
    Value maskedGroupPorts = BanbaSugar::select(groupAllocateBits[group],
                                                groupPorts, stQPortIdxZero);
    stQMaskedGroupPorts.push_back(maskedGroupPorts);
  }

  // OR reduce the masked vectors
  OrReduceOp stQPortIdxReduced =
      BanbaSugar::createOp<OrReduceOp>(stQMaskedGroupPorts);

  // Align the per-entry port indices to match the store queue
  // by shifting them
  CyclicLeftShiftOp stQPortIdx = BanbaSugar::createOp<CyclicLeftShiftOp>(
      stQPortIdxReduced.getReduced(), getStQTail());

  // Results
  driveLdQPortIdx(ldQPortIdx.getShifted());
  driveStQPortIdx(stQPortIdx.getShifted());
}

void OrderMatrixGenOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();
  Types t(getContext());

  unsigned numGroups = config.getNumBBs();
  unsigned numLdqEntries = config.getFifoDepth_L();
  unsigned numStqEntries = config.getFifoDepth_S();

  // Check all the encodings and see are any nonzero
  bool nonzeroEncoding = false;
  for (unsigned group = 0; group < numGroups; ++group) {
    DenseI32ArrayAttr groupRom = config.getLdOrder()[group];
    for (unsigned ld = 0; ld < (unsigned)groupRom.size(); ++ld) {
      // There is at least one nonzero encoding
      if (groupRom[ld] != 0)
        nonzeroEncoding = true;
    }
  }

  // If so we need the real logic
  if (nonzeroEncoding) {
    // A row of zeros, one bit per store queue entry
    BanbaValue zero(0, numStqEntries);

    // For each group,
    // create one masked order encoding per load it contains:
    // the group's store-before-load encoding when it allocates,
    // otherwise zero.
    // encodings[ld] collects the masked encoding by ld index
    SmallVector<SmallVector<Value>> encodings(numLdqEntries);
    BanbaValue groupAllocate = getGroupAllocate();
    for (unsigned group = 0; group < numGroups; ++group) {
      // Annoyingly structured config input:
      // it has an integer for each load,
      // which is the number of stores older than that load
      DenseI32ArrayAttr groupRom = config.getLdOrder()[group];
      assert((unsigned)groupRom.size() <= numLdqEntries &&
             "group has more loads than load queue entries");

      // This group's allocate bit, the mask select of its encodings
      BanbaValue groupAllocateBit = groupAllocate[group];

      // Create the encoding for each load of this group
      for (unsigned ld = 0; ld < (unsigned)groupRom.size(); ++ld) {
        // If the load is the oldest, the encoding will be zero either way
        // and so we omit it from the OR reduce
        if (groupRom[ld] == 0)
          continue;

        // The number of stores in the group before this load
        unsigned numStoresBefore = groupRom[ld];

        // If N stores in the group are before this load
        // bits 0 to N-1 should be 1, and the rest 0
        // which we use llvm::maskTrailingOnes to set
        BanbaValue encoding(llvm::maskTrailingOnes<uint32_t>(numStoresBefore),
                            numStqEntries);

        // The encoding is only used if the group is allocating
        Value maskedEncoding =
            BanbaSugar::select(groupAllocateBit, encoding, zero);
        encodings[ld].push_back(maskedEncoding);
      }
    }

    // We need to combine the masked encodings
    // by OR reducing along ld index.
    //
    // Per row of the order matrix:
    // align the matrix to the store queue
    // by shifting the bits inside the row.
    // Cast to Nxi1 so the shift is horizontal,
    // then cast back to iN so the final shift is vertical
    SmallVector<Value> rows;
    for (unsigned ld = 0; ld < numLdqEntries; ++ld) {
      if (!encodings[ld].empty()) {
        // It's safe to OR reduce on a 1-element vector
        // but it gives a cleaner output to avoid it
        Value valueToShift;
        if (encodings[ld].size() > 1) {
          // OR reduce the possible encodings
          OrReduceOp reduced =
              BanbaSugar::createOp<OrReduceOp>(encodings[ld]);
          valueToShift = reduced.getReduced();
        } else {
          valueToShift = encodings[ld][0];
        }

        // Cast to Nxi1
        Value rowReducedArray =
            BanbaSugar::castTo(valueToShift, t.Array(numStqEntries, 1));

        // Shift
        CyclicLeftShiftOp rowShift = BanbaSugar::createOp<CyclicLeftShiftOp>(
            rowReducedArray, getStQTail());

        // Cast the row back to iN
        Value rowShifted = BanbaSugar::castToInt(rowShift.getShifted());

        // Push into vector
        // we will later convert to hw.array
        rows.push_back(rowShifted);
      } else {
        // A row with no encodings is constant zero
        // (a rotation of zero is zero, so no shift is needed)
        rows.push_back(zero);
      }
    }

    // Combine the rows to form a true matrix
    Value rowArray = BanbaSugar::concatToArray(rows);

    // Once: align the order matrix to the load queue
    // by shifting the order of the rows
    CyclicLeftShiftOp gaLsOrder =
        BanbaSugar::createOp<CyclicLeftShiftOp>(rowArray, getLdQTail());

    // Results
    driveGaLsOrder(gaLsOrder.getShifted());
  } else {
    // In this kernel,
    // there is never a store before a load inside a basic block,
    // so return a constant zero matrix
    driveGaLsOrder(BanbaSugar::zeroValue(getGaLsOrder().getType()));
  }
}

void CountExtractOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  // ROM of the number of loads in each group
  Value numLoadsRom =
      BanbaSugar::arrayConstant(config.getNumLoads(), config.getLdCountWidth());

  // Mux of the ROM by the one-hot group_allocate
  OneHotMuxOp numLoads =
      BanbaSugar::createOp<OneHotMuxOp>(getGroupAllocate(), numLoadsRom);

  // ROM of the number of stores in each group
  Value numStoresRom = BanbaSugar::arrayConstant(config.getNumStores(),
                                                 config.getStCountWidth());

  // Mux of the ROM by the one-hot group_allocate
  OneHotMuxOp numStores =
      BanbaSugar::createOp<OneHotMuxOp>(getGroupAllocate(), numStoresRom);

  // Results
  driveNumLoads(numLoads.getSelected());
  driveNumStores(numStores.getSelected());
}

void AllocatingGenOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  // The largest number of loads in any group,
  // which num_loads can never exceed
  ArrayRef<int32_t> numLoadsPerGroup = config.getNumLoads();
  unsigned maxNumLoads =
      *std::max_element(numLoadsPerGroup.begin(), numLoadsPerGroup.end());

  // A zero bit, for the entries of either queue
  // that num_loads or num_stores can never reach
  BanbaValue zeroBit(0, 1);

  // For each load queue entry
  // decide if it is in the unshifted allocation mask
  // and save the bit in a vector.
  // Entry i of the mask is 1 if num_loads > i
  SmallVector<Value> ldQAllocatingUnshiftedVec;
  for (unsigned entry = 0; entry < config.getFifoDepth_L(); ++entry) {
    if (entry < maxNumLoads) {
      // Compare num_loads with the entry's index, as a constant
      BanbaValue entryIndex(entry, config.getLdCountWidth());
      BanbaValue allocatingBit = getNumLoads() > entryIndex;

      // Push into vector
      // we will later convert to hw.array
      ldQAllocatingUnshiftedVec.push_back(allocatingBit);
    } else {
      // num_loads can never be greater than this entry's index,
      // so the bit is a constant zero
      ldQAllocatingUnshiftedVec.push_back(zeroBit);
    }
  }

  Value ldQAllocatingUnshifted =
      BanbaSugar::concatToArray(ldQAllocatingUnshiftedVec);

  // Rotate the unshifted mask by the tail pointer
  // so its block of 1s starts at the queue tail
  CyclicLeftShiftOp ldQAllocating = BanbaSugar::createOp<CyclicLeftShiftOp>(
      ldQAllocatingUnshifted, getLdQTail());

  // The largest number of stores in any group,
  // which num_stores can never exceed
  ArrayRef<int32_t> numStoresPerGroup = config.getNumStores();
  unsigned maxNumStores =
      *std::max_element(numStoresPerGroup.begin(), numStoresPerGroup.end());

  // For each store queue entry
  // decide if it is in the unshifted allocation mask
  // and save the bit in a vector.
  // Entry i of the mask is 1 if num_stores > i
  SmallVector<Value> stQAllocatingUnshiftedVec;
  for (unsigned entry = 0; entry < config.getFifoDepth_S(); ++entry) {
    if (entry < maxNumStores) {
      // Compare num_stores with the entry's index, as a constant
      BanbaValue entryIndex(entry, config.getStCountWidth());
      BanbaValue allocatingBit = getNumStores() > entryIndex;

      // Push into vector
      // we will later convert to hw.array
      stQAllocatingUnshiftedVec.push_back(allocatingBit);
    } else {
      // num_stores can never be greater than this entry's index,
      // so the bit is a constant zero
      stQAllocatingUnshiftedVec.push_back(zeroBit);
    }
  }

  Value stQAllocatingUnshifted =
      BanbaSugar::concatToArray(stQAllocatingUnshiftedVec);

  // Rotate the unshifted mask by the tail pointer
  // so its block of 1s starts at the queue tail
  CyclicLeftShiftOp stQAllocating = BanbaSugar::createOp<CyclicLeftShiftOp>(
      stQAllocatingUnshifted, getStQTail());

  // Results
  driveLdQAllocating(ldQAllocating.getShifted());
  driveStQAllocating(stQAllocating.getShifted());
}

} // namespace banba::shared::ga
