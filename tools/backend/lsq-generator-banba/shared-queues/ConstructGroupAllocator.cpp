#include "BanbaOps.h"
#include "BanbaSugar.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::shared {

void GroupAllocatorOp::constructRegion() {
  BanbaSugarScope scope(getOperation());

  // Emit the group allocator into group_allocator.sv
  setOutputFile(getOperation(), "group_allocator.sv");

  BanbaSugar::moduleComment(getOperation(), ModComment);

  ////////////////////////////////////////////////////////////////
  // Declare the sub-components, without specifying inputs
  ////////////////////////////////////////////////////////////////

  auto freeEntryCalc = BanbaSugar::createOpFromConfig<ga::FreeEntryCalcOp>();
  auto readyGen = BanbaSugar::createOpFromConfig<ga::ReadyGenOp>();
  auto portIdxGen = BanbaSugar::createOpFromConfig<ga::PortIdxGenOp>();
  auto orderMatrixGen = BanbaSugar::createOpFromConfig<ga::OrderMatrixGenOp>();
  auto countExtract = BanbaSugar::createOpFromConfig<ga::CountExtractOp>();
  auto allocatingGen = BanbaSugar::createOpFromConfig<ga::AllocatingGenOp>();

  /////////////////////////////////////////////////////////
  // Top level logic
  /////////////////////////////////////////////////////////

  // Mutually exclusive 1-bit signals
  // showing which (if any) group is allocating this cycle.
  // Element-wise AND (returns iN)
  Value groupAllocate =
      (readyGen.getGaReady() & getGaValid()).nameValueInIR("group_allocate");

  /////////////////////////////////////////////////////////
  // Finally, wire the components together
  /////////////////////////////////////////////////////////

  freeEntryCalc.driveLdQHead(getLdQHead());
  freeEntryCalc.driveLdQTail(getLdQTail());
  freeEntryCalc.driveStQHead(getStQHead());
  freeEntryCalc.driveStQTail(getStQTail());
  freeEntryCalc.driveLdQEmpty(getLdQEmpty());
  freeEntryCalc.driveStQEmpty(getStQEmpty());

  readyGen.driveLdNumFreeEntries(freeEntryCalc.getLdNumFreeEntries());
  readyGen.driveStNumFreeEntries(freeEntryCalc.getStNumFreeEntries());

  portIdxGen.driveGroupAllocate(groupAllocate);
  portIdxGen.driveLdQTail(getLdQTail());
  portIdxGen.driveStQTail(getStQTail());

  orderMatrixGen.driveGroupAllocate(groupAllocate);
  orderMatrixGen.driveLdQTail(getLdQTail());
  orderMatrixGen.driveStQTail(getStQTail());

  countExtract.driveGroupAllocate(groupAllocate);

  allocatingGen.driveNumLoads(countExtract.getNumLoads());
  allocatingGen.driveNumStores(countExtract.getNumStores());
  allocatingGen.driveLdQTail(getLdQTail());
  allocatingGen.driveStQTail(getStQTail());

  // Results
  driveGaReady(readyGen.getGaReady());
  driveLdQAllocating(allocatingGen.getLdQAllocating());
  driveNumLoads(countExtract.getNumLoads());
  driveStQAllocating(allocatingGen.getStQAllocating());
  driveNumStores(countExtract.getNumStores());
  driveGaLsOrder(orderMatrixGen.getGaLsOrder());
  driveLdQPortIdx(portIdxGen.getLdQPortIdx());
  driveStQPortIdx(portIdxGen.getStQPortIdx());
}

} // namespace banba::shared

namespace banba::shared::ga {

void FreeEntryCalcOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  BanbaSugar::moduleComment(getOperation(), ModComment);

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

  BanbaSugar::moduleComment(getOperation(), ModComment);

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
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  BanbaSugar::moduleComment(getOperation(), ModComment);

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
  Value ldQPortIdxZero = BanbaSugar::zeroValue(getLdQPortIdx().getType());

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

  // It's safe to OR reduce on a 1-element vector
  // but it gives a cleaner output to avoid it
  Value ldQPortIdxReduced;
  if (ldQMaskedGroupPorts.size() > 1) {
    // OR reduce the masked vectors
    OrReduceOp reduced = BanbaSugar::createOp<OrReduceOp>(ldQMaskedGroupPorts);
    ldQPortIdxReduced = reduced.getReduced();
  } else {
    ldQPortIdxReduced = ldQMaskedGroupPorts[0];
  }

  // Align the per-entry port indices to match the load queue
  // by shifting them
  CyclicLeftShiftOp ldQPortIdx =
      BanbaSugar::createOp<CyclicLeftShiftOp>(ldQPortIdxReduced, getLdQTail());

  // An all-zero port index vector,
  // for masking out port indices which are not allocating.
  // The width is 1 bit even when there is 1 store port
  Value stQPortIdxZero = BanbaSugar::zeroValue(getStQPortIdx().getType());

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

  // It's safe to OR reduce on a 1-element vector
  // but it gives a cleaner output to avoid it
  Value stQPortIdxReduced;
  if (stQMaskedGroupPorts.size() > 1) {
    // OR reduce the masked vectors
    OrReduceOp reduced = BanbaSugar::createOp<OrReduceOp>(stQMaskedGroupPorts);
    stQPortIdxReduced = reduced.getReduced();
  } else {
    stQPortIdxReduced = stQMaskedGroupPorts[0];
  }

  // Align the per-entry port indices to match the store queue
  // by shifting them
  CyclicLeftShiftOp stQPortIdx =
      BanbaSugar::createOp<CyclicLeftShiftOp>(stQPortIdxReduced, getStQTail());

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
    BanbaSugar::moduleComment(getOperation(), ModComment);

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
          OrReduceOp reduced = BanbaSugar::createOp<OrReduceOp>(encodings[ld]);
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
    BanbaSugar::moduleComment(getOperation(), ZeroModComment);

    // In this kernel,
    // there is never a store before a load inside a basic block,
    // so return a constant zero matrix
    driveGaLsOrder(BanbaSugar::zeroValue(getGaLsOrder().getType()));
  }
}

void CountExtractOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  BanbaSugar::moduleComment(getOperation(), ModComment);

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
  BanbaSugarScope scope(getOperation());
  ConfigAttr config = BanbaSugarScope::getConfig();

  BanbaSugar::moduleComment(getOperation(), ModComment);

  // The largest number of loads in any group,
  // which num_loads can never exceed
  ArrayRef<int32_t> numLoadsPerGroup = config.getNumLoads();
  unsigned maxNumLoads =
      *std::max_element(numLoadsPerGroup.begin(), numLoadsPerGroup.end());

  // For each load queue entry that num_loads can reach,
  // decide if it is in the unshifted allocation mask
  // and save the bit in a vector.
  // Entry i of the mask is 1 if num_loads > i
  SmallVector<Value> ldQAllocatingUnshiftedVec;
  for (unsigned entry = 0; entry < maxNumLoads; ++entry) {
    // Compare num_loads with the entry's index, as a constant
    BanbaValue entryIndex(entry, config.getLdCountWidth());
    BanbaValue allocatingBit = getNumLoads() > entryIndex;

    // Push into vector
    // we will later convert to hw.array
    ldQAllocatingUnshiftedVec.push_back(allocatingBit);
  }

  // The mask as an integer, with one bit per load queue entry
  Value ldQAllocatingInt;
  if (maxNumLoads > 0) {
    // num_loads can never be greater than the index of any other entry,
    // so their bits are constant zero: zero-extend the bits in the vector
    BanbaValue reachable = BanbaSugar::castToInt(
        BanbaSugar::concatToArray(ldQAllocatingUnshiftedVec));
    ldQAllocatingInt = BanbaSugar::zext(reachable, config.getFifoDepth_L());
  } else {
    // No group has a load, so the whole mask is constant zero
    ldQAllocatingInt = BanbaValue(0, config.getFifoDepth_L());
  }

  Value ldQAllocatingUnshifted =
      BanbaSugar::castTo(ldQAllocatingInt, getLdQAllocating().getType())
          .nameValueInIR("ld_q_allocating_unshifted");

  // Rotate the unshifted mask by the tail pointer
  // so its block of 1s starts at the queue tail
  CyclicLeftShiftOp ldQAllocating = BanbaSugar::createOp<CyclicLeftShiftOp>(
      ldQAllocatingUnshifted, getLdQTail());

  // The largest number of stores in any group,
  // which num_stores can never exceed
  ArrayRef<int32_t> numStoresPerGroup = config.getNumStores();
  unsigned maxNumStores =
      *std::max_element(numStoresPerGroup.begin(), numStoresPerGroup.end());

  // For each store queue entry that num_stores can reach,
  // decide if it is in the unshifted allocation mask
  // and save the bit in a vector.
  // Entry i of the mask is 1 if num_stores > i
  SmallVector<Value> stQAllocatingUnshiftedVec;
  for (unsigned entry = 0; entry < maxNumStores; ++entry) {
    // Compare num_stores with the entry's index, as a constant
    BanbaValue entryIndex(entry, config.getStCountWidth());
    BanbaValue allocatingBit = getNumStores() > entryIndex;

    // Push into vector
    // we will later convert to hw.array
    stQAllocatingUnshiftedVec.push_back(allocatingBit);
  }

  // The mask as an integer, with one bit per store queue entry
  Value stQAllocatingInt;
  if (maxNumStores > 0) {
    // num_stores can never be greater than the index of any other entry,
    // so their bits are constant zero: zero-extend the bits in the vector
    BanbaValue reachable = BanbaSugar::castToInt(
        BanbaSugar::concatToArray(stQAllocatingUnshiftedVec));
    stQAllocatingInt = BanbaSugar::zext(reachable, config.getFifoDepth_S());
  } else {
    // No group has a store, so the whole mask is constant zero
    stQAllocatingInt = BanbaValue(0, config.getFifoDepth_S());
  }

  Value stQAllocatingUnshifted =
      BanbaSugar::castTo(stQAllocatingInt, getStQAllocating().getType())
          .nameValueInIR("st_q_allocating_unshifted");

  // Rotate the unshifted mask by the tail pointer
  // so its block of 1s starts at the queue tail
  CyclicLeftShiftOp stQAllocating = BanbaSugar::createOp<CyclicLeftShiftOp>(
      stQAllocatingUnshifted, getStQTail());

  // Results
  driveLdQAllocating(ldQAllocating.getShifted());
  driveStQAllocating(stQAllocating.getShifted());
}

} // namespace banba::shared::ga
