//===- MemoryInterfaces.cpp - Memory interface helpers ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements support to work with Handshake memory interfaces.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// MemoryOpLowering
//===----------------------------------------------------------------------===//

void MemoryOpLowering::recordReplacement(Operation *oldOp, Operation *newOp,
                                         bool forwardInterface) {
  copyAttr<MemDependenceArrayAttr>(oldOp, newOp);
  if (forwardInterface)
    copyAttr<MemInterfaceAttr>(oldOp, newOp);
  nameChanges[namer.getName(oldOp)] = namer.getName(newOp);
}

bool MemoryOpLowering::renameDependencies(Operation *topLevelOp) {
  MLIRContext *ctx = topLevelOp->getContext();
  bool anyChange = false;
  topLevelOp->walk([&](Operation *memOp) {
    // We only care about supported load/store memory accesses
    if (!isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
             affine::AffineStoreOp, handshake::LoadOpInterface,
             handshake::StoreOpInterface>(memOp))
      return;

    // Read potential memory dependencies stored on the memory operation
    auto oldMemDeps = getUniqueAttr<MemDependenceArrayAttr>(memOp);
    if (!oldMemDeps)
      return;

    // Copy memory dependence attributes one-by-one, replacing the name of
    // replaced destination memory operations along the way if necessary
    SmallVector<MemDependenceAttr> newMemDeps;
    for (MemDependenceAttr oldDep : oldMemDeps.getDependencies()) {
      StringRef oldName = oldDep.getDstAccess();
      auto replacedName = nameChanges.find(oldName);
      bool opWasReplaced = replacedName != nameChanges.end();
      anyChange |= opWasReplaced;
      if (opWasReplaced) {
        StringAttr newName = StringAttr::get(ctx, replacedName->second);
        newMemDeps.push_back(MemDependenceAttr::get(
            ctx, newName, oldDep.getLoopDepth(), oldDep.getComponents()));
      } else {
        newMemDeps.push_back(oldDep);
      }
    }
    setUniqueAttr(memOp, MemDependenceArrayAttr::get(ctx, newMemDeps));
  });

  return anyChange;
}

//===----------------------------------------------------------------------===//
// MemoryInterfaceBuilder
//===----------------------------------------------------------------------===//

void MemoryInterfaceBuilder::addMCPort(Operation *memOp) {
  std::optional<unsigned> bb = getLogicBB(memOp);
  assert(bb && "MC port must belong to basic block");
  if (isa<handshake::MCLoadOp>(memOp)) {
    ++mcNumLoads;
  } else {
    assert(isa<handshake::MCStoreOp>(memOp) && "invalid MC port");
  }
  mcPorts[*bb].push_back(memOp);
}

void MemoryInterfaceBuilder::addLSQPort(unsigned group, Operation *memOp) {
  if (isa<handshake::LSQLoadOp>(memOp))
    ++lsqNumLoads;
  else
    assert(isa<handshake::LSQStoreOp>(memOp) && "invalid LSQ port");
  lsqPorts[group].push_back(memOp);
}

LogicalResult MemoryInterfaceBuilder::instantiateInterfaces(
    OpBuilder &builder, handshake::MemoryControllerOp &mcOp,
    handshake::LSQOp &lsqOp) {
  BackedgeBuilder edgeBuilder(builder, memref.getLoc());
  return instantiateInterfaces(builder, edgeBuilder, mcOp, lsqOp);
}

LogicalResult MemoryInterfaceBuilder::instantiateInterfaces(
    PatternRewriter &rewriter, handshake::MemoryControllerOp &mcOp,
    handshake::LSQOp &lsqOp) {
  BackedgeBuilder edgeBuilder(rewriter, memref.getLoc());
  return instantiateInterfaces(rewriter, edgeBuilder, mcOp, lsqOp);
}

LogicalResult MemoryInterfaceBuilder::instantiateInterfaces(
    OpBuilder &builder, BackedgeBuilder &edgeBuilder,
    handshake::MemoryControllerOp &mcOp, handshake::LSQOp &lsqOp) {

  // Determine interfaces' inputs
  InterfaceInputs inputs;
  if (failed(determineInterfaceInputs(inputs, builder)))
    return failure();
  if (inputs.mcInputs.empty() && inputs.lsqInputs.empty())
    return success();

  mcOp = nullptr;
  lsqOp = nullptr;

  builder.setInsertionPointToStart(&funcOp.front());
  Location loc = memref.getLoc();

  if (!inputs.mcInputs.empty() && inputs.lsqInputs.empty()) {
    // We only need a memory controller
    mcOp = builder.create<handshake::MemoryControllerOp>(
        loc, memref, memStart, inputs.mcInputs, ctrlEnd, inputs.mcBlocks,
        mcNumLoads);
  } else if (inputs.mcInputs.empty() && !inputs.lsqInputs.empty()) {
    // We only need an LSQ
    lsqOp = builder.create<handshake::LSQOp>(loc, memref, memStart,
                                             inputs.lsqInputs, ctrlEnd,
                                             inputs.lsqGroupSizes, lsqNumLoads);
  } else {
    // We need a MC and an LSQ. They need to be connected with 4 new channels
    // so that the LSQ can forward its loads and stores to the MC. We need
    // load address, store address, and store data channels from the LSQ to
    // the MC and a load data channel from the MC to the LSQ
    MemRefType memrefType = memref.getType().cast<MemRefType>();

    // Create 3 backedges (load address, store address, store data) for the MC
    // inputs that will eventually come from the LSQ.
    MLIRContext *ctx = builder.getContext();
    Type addrType = handshake::ChannelType::getAddrChannel(ctx);
    Backedge ldAddr = edgeBuilder.get(addrType);
    Backedge stAddr = edgeBuilder.get(addrType);
    Backedge stData = edgeBuilder.get(
        handshake::ChannelType::get(memrefType.getElementType()));
    inputs.mcInputs.push_back(ldAddr);
    inputs.mcInputs.push_back(stAddr);
    inputs.mcInputs.push_back(stData);

    // Create the memory controller, adding 1 to its load count so that it
    // generates a load data result for the LSQ
    mcOp = builder.create<handshake::MemoryControllerOp>(
        loc, memref, memStart, inputs.mcInputs, ctrlEnd, inputs.mcBlocks,
        mcNumLoads + 1);

    // Add the MC's load data result to the LSQ's inputs and create the LSQ,
    // passing a flag to the builder so that it generates the necessary
    // outputs that will go to the MC
    inputs.lsqInputs.push_back(mcOp.getOutputs().back());
    lsqOp = builder.create<handshake::LSQOp>(loc, mcOp, inputs.lsqInputs,
                                             inputs.lsqGroupSizes, lsqNumLoads);

    // Resolve the backedges to fully connect the MC and LSQ
    ValueRange lsqMemResults = lsqOp.getOutputs().take_back(3);
    ldAddr.setValue(lsqMemResults[0]);
    stAddr.setValue(lsqMemResults[1]);
    stData.setValue(lsqMemResults[2]);
  }

  // At this point, all load ports are missing their second operand which is the
  // data value coming from a memory interface back to the port
  if (mcOp)
    addMemDataResultToLoads(mcPorts, mcOp);
  if (lsqOp)
    addMemDataResultToLoads(lsqPorts, lsqOp);

  return success();
}

SmallVector<Value, 2>
MemoryInterfaceBuilder::getMemResultsToInterface(Operation *memOp) {
  // For loads, address output go to memory
  if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(memOp))
    return SmallVector<Value, 2>{loadOp.getAddressOutput()};

  // For stores, all outputs (address and data) go to memory
  auto storeOp = dyn_cast<handshake::StoreOpInterface>(memOp);
  assert(storeOp && "input operation must either be load or store");
  return SmallVector<Value, 2>{storeOp->getResults()};
}

Value MemoryInterfaceBuilder::getMCControl(Value ctrl, unsigned numStores,
                                           OpBuilder &builder) {
  assert(isa<handshake::ControlType>(ctrl.getType()) &&
         "control signal must have !handshake.control type");
  if (Operation *defOp = ctrl.getDefiningOp())
    builder.setInsertionPointAfter(defOp);
  else
    builder.setInsertionPointToStart(ctrl.getParentBlock());
  handshake::ConstantOp cstOp = builder.create<handshake::ConstantOp>(
      ctrl.getLoc(), builder.getI32IntegerAttr(numStores), ctrl);
  inheritBBFromValue(ctrl, cstOp);
  return cstOp.getResult();
}

void MemoryInterfaceBuilder::setLoadDataOperand(
    handshake::LoadOpInterface loadOp, Value dataIn) {
  SmallVector<Value, 2> operands;
  operands.push_back(loadOp->getOperand(0));
  operands.push_back(dataIn);
  loadOp->setOperands(operands);
}

LogicalResult
MemoryInterfaceBuilder::determineInterfaceInputs(InterfaceInputs &inputs,
                                                 OpBuilder &builder) {

  // Determine LSQ inputs
  for (auto [group, lsqGroupOps] : lsqPorts) {
    // First, determine the group's control signal, which is dictated by the BB
    // of the first memory port in the group
    Operation *firstOpInGroup = lsqGroupOps.front();
    std::optional<unsigned> block = getLogicBB(firstOpInGroup);
    if (!block)
      return firstOpInGroup->emitError() << "LSQ port must belong to a BB.";
    Value groupCtrl = getCtrl(*block);
    if (!groupCtrl)
      return failure();
    inputs.lsqInputs.push_back(groupCtrl);

    // Them, add all memory port results that go the interface to the list of
    // LSQ inputs
    for (Operation *lsqOp : lsqGroupOps)
      llvm::copy(getMemResultsToInterface(lsqOp),
                 std::back_inserter(inputs.lsqInputs));

    // Add the size of the group to our list
    inputs.lsqGroupSizes.push_back(lsqGroupOps.size());
  }

  if (mcPorts.empty())
    return success();

  // The MC needs control signals from all blocks containing store ports
  // connected to an LSQ, since these requests end up being forwarded to the MC,
  // so we need to know the number of LSQ stores per basic block
  DenseMap<unsigned, unsigned> lsqStoresPerBlock;
  for (auto &[_, lsqGroupOps] : lsqPorts) {
    for (Operation *lsqOp : lsqGroupOps) {
      if (isa<handshake::LSQStoreOp>(lsqOp)) {
        std::optional<unsigned> block = getLogicBB(lsqOp);
        if (!block)
          return lsqOp->emitError() << "LSQ port must belong to a BB.";
        ++lsqStoresPerBlock[*block];
      }
    }
  }

  // Inputs from blocks that have at least one direct load/store access port to
  // the MC are added to the future MC's operands first
  for (auto &[block, mcBlockOps] : mcPorts) {
    // Count the total number of stores in the block, either directly connected
    // to the MC or going through an LSQ
    unsigned numStoresInBlock = lsqStoresPerBlock.lookup(block);
    for (Operation *memOp : mcBlockOps) {
      if (isa<handshake::MCStoreOp>(memOp))
        ++numStoresInBlock;
    }

    // Blocks with at least one store need to provide a control signal fed
    // through a constant indicating the number of stores in the block
    if (numStoresInBlock > 0) {
      Value blockCtrl = getCtrl(block);
      if (!blockCtrl)
        return failure();
      inputs.mcInputs.push_back(
          getMCControl(blockCtrl, numStoresInBlock, builder));
    }

    // Traverse the list of memory operations in the block once more and
    // accumulate memory inputs coming from the block
    for (Operation *mcOp : mcBlockOps)
      llvm::copy(getMemResultsToInterface(mcOp),
                 std::back_inserter(inputs.mcInputs));

    inputs.mcBlocks.push_back(block);
  }

  // Control ports from blocks which do not have memory ports directly
  // connected to the MC but from which the LSQ will forward store requests from
  // are then added to the future MC's operands
  for (auto &[lsqBlock, numStores] : lsqStoresPerBlock) {
    // We only need to do something if the block has stores that have not yet
    // been accounted for
    if (mcPorts.contains(lsqBlock) || numStores == 0)
      continue;

    // Identically to before, blocks with stores need a cntrol signal
    Value blockCtrl = getCtrl(lsqBlock);
    if (!blockCtrl)
      return failure();
    inputs.mcInputs.push_back(getMCControl(blockCtrl, numStores, builder));

    inputs.mcBlocks.push_back(lsqBlock);
  }

  return success();
}

Value MemoryInterfaceBuilder::getCtrl(unsigned block) {
  auto groupCtrl = ctrlVals.find(block);
  if (groupCtrl == ctrlVals.end()) {
    llvm::errs() << "Cannot determine control signal for BB " << block << "\n";
    return nullptr;
  }
  return groupCtrl->second;
}

void MemoryInterfaceBuilder::addMemDataResultToLoads(InterfacePorts &ports,
                                                     Operation *memIfaceOp) {
  unsigned resIdx = 0;
  for (auto &[_, memGroupOps] : ports) {
    for (Operation *memOp : memGroupOps) {
      if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(memOp))
        setLoadDataOperand(loadOp, memIfaceOp->getResult(resIdx++));
    }
  }
}

//===----------------------------------------------------------------------===//
// LSQGenerationInfo
//===----------------------------------------------------------------------===//

LSQGenerationInfo::LSQGenerationInfo(handshake::LSQOp lsqOp, StringRef name)
    : lsqOp(lsqOp), name(name) {
  FuncMemoryPorts lsqPorts = getMemoryPorts(lsqOp);
  fromPorts(lsqPorts);
}

LSQGenerationInfo::LSQGenerationInfo(FuncMemoryPorts &ports, StringRef name)
    : lsqOp(cast<handshake::LSQOp>(ports.memOp)), name(name) {
  fromPorts(ports);
}

void LSQGenerationInfo::fromPorts(FuncMemoryPorts &ports) {
  dataWidth = ports.dataWidth;
  addrWidth = ports.addrWidth;

  numGroups = ports.getNumGroups();
  numLoads = ports.getNumPorts<LoadPort>();
  numStores = ports.getNumPorts<StorePort>();

  unsigned loadIdx = 0, storeIdx = 0;
  for (GroupMemoryPorts &groupPorts : ports.groups) {
    // Number of load and store ports per block
    loadsPerGroup.push_back(groupPorts.getNumPorts<LoadPort>());
    storesPerGroup.push_back(groupPorts.getNumPorts<StorePort>());

    // Compute the ffset of first load/store in the group and indices of each
    // load/store port
    std::optional<unsigned> firstLoadOffset, firstStoreOffset;
    SmallVector<unsigned> groupLoadPorts, groupStorePorts;
    for (auto [portIdx, accessPort] : llvm::enumerate(groupPorts.accessPorts)) {
      if (isa<LoadPort>(accessPort)) {
        if (!firstLoadOffset)
          firstLoadOffset = portIdx;
        groupLoadPorts.push_back(loadIdx++);
      } else {
        assert(isa<StorePort>(accessPort) && "port must be load or store");
        if (!firstStoreOffset)
          firstStoreOffset = portIdx;
        groupStorePorts.push_back(storeIdx++);
      }
    }

    // If there are no loads or no stores in the block, set the corresponding
    // offset to 0
    loadOffsets.push_back(SmallVector<unsigned>{firstLoadOffset.value_or(0)});
    storeOffsets.push_back(SmallVector<unsigned>{firstStoreOffset.value_or(0)});

    loadPorts.push_back(groupLoadPorts);
    storePorts.push_back(groupStorePorts);
  }

  /// Adds as many 0s as necessary to the array so that its size equals the
  /// depth. Asserts if the array size is larger than the depth.
  auto capArray = [&](SmallVector<unsigned> &array, unsigned depth) -> void {
    assert(array.size() <= depth && "array larger than LSQ depth");
    for (size_t i = 0, e = array.size(); i < depth - e; ++i)
      array.push_back(0);
  };

  /// Adds as many 0s as necessary to each nested array so that their size
  /// equals the depth.
  auto capBiArray = [&](SmallVector<SmallVector<unsigned>> &biArray,
                        unsigned depth) -> void {
    for (SmallVector<unsigned> &array : biArray)
      capArray(array, depth);
  };

  // Port offsets and index arrays must have length equal to the depth
  capBiArray(loadOffsets, depth);
  capBiArray(storeOffsets, depth);
  capBiArray(loadPorts, depth);
  capBiArray(storePorts, depth);
}

//===----------------------------------------------------------------------===//
// Misc functions
//===----------------------------------------------------------------------===//

SmallVector<Value> dynamatic::getLSQControlPaths(handshake::LSQOp lsqOp,
                                                 Operation *ctrlOp) {
  // Accumulate all outputs of the control operation that are part of the memory
  // control network
  SmallVector<Value> controlValues;
  // List of control channels to explore, starting from the control operation's
  // results
  SmallVector<Value, 4> controlChannels;
  // Set of control operations already explored from the control operation's
  // results (to avoid looping in the dataflow graph)
  SmallPtrSet<Operation *, 4> controlOps;
  for (OpResult res : ctrlOp->getResults()) {
    // We only care for control-only channels
    if (!isa<handshake::ControlType>(res.getType()))
      continue;

    // Reset the list of control channels to explore and the list of control
    // operations that we have already visited
    controlChannels.clear();
    controlOps.clear();

    controlChannels.push_back(res);
    controlOps.insert(ctrlOp);
    do {
      Value val = controlChannels.pop_back_val();
      for (Operation *succOp : val.getUsers()) {
        // Make sure that we do not loop forever over the same control
        // operations
        if (auto [_, newOp] = controlOps.insert(succOp); !newOp)
          continue;

        if (succOp == lsqOp) {
          // We have found a control path triggering a different group
          // allocation to the LSQ, add it to our list
          controlValues.push_back(res);
          break;
        }
        llvm::TypeSwitch<Operation *, void>(succOp)
            .Case<handshake::ConditionalBranchOp, handshake::BranchOp,
                  handshake::MergeOp, handshake::MuxOp, handshake::ForkOp,
                  handshake::LazyForkOp, handshake::BufferOp>([&](auto) {
              // If the successor just propagates the control path, add
              // all its results to the list of control channels to
              // explore
              for (OpResult succRes : succOp->getResults())
                controlChannels.push_back(succRes);
            })
            .Case<handshake::ControlMergeOp>(
                [&](handshake::ControlMergeOp cmergeOp) {
                  // Only the control merge's data output forwards the input
                  controlChannels.push_back(cmergeOp.getResult());
                });
      }
    } while (!controlChannels.empty());
  }

  return controlValues;
}
