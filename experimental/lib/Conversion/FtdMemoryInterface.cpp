//===- FtdMemoryInterfaces.cpp - FTD Memory interface helpers ----------*- C++
//-*-===//
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

#include <utility>

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Conversion/FtdMemoryInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace experimental {
namespace ftd {

LogicalResult FtdMemoryInterfaceBuilder::instantiateInterfacesWithForks(
    OpBuilder &builder, handshake::MemoryControllerOp &mcOp,
    handshake::LSQOp &lsqOp, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, Value start) {

  // Get the edgeBuilder
  BackedgeBuilder edgeBuilder((PatternRewriter &)builder, memref.getLoc());

  // Connect function
  FConnectLoad connect = [&](LoadOp loadOp, Value dataIn) {
    ((PatternRewriter &)builder).updateRootInPlace(loadOp, [&] {
      loadOp->setOperand(1, dataIn);
    });
  };

  // Determine interfaces' inputs
  InterfaceInputs interfaceInputs;
  if (failed(determineInterfaceInputsWithForks(interfaceInputs, builder, groups,
                                               forksGraph, start)))
    return failure();

  // If we need no inputs both for the standard memory controller and the LSQ,
  // then nothing has to be instantiated.
  if (interfaceInputs.mcInputs.empty() && interfaceInputs.lsqInputs.empty())
    return success();

  mcOp = nullptr;
  lsqOp = nullptr;

  builder.setInsertionPointToStart(&funcOp.front());
  Location loc = memref.getLoc();

  if (!interfaceInputs.mcInputs.empty() && interfaceInputs.lsqInputs.empty()) {
    // We only need a memory controller if the LSQ is not necessary (number of
    // inputs is zero)
    mcOp = builder.create<handshake::MemoryControllerOp>(
        loc, memref, memStart, interfaceInputs.mcInputs, ctrlEnd,
        interfaceInputs.mcBlocks, mcNumLoads);
  } else if (interfaceInputs.mcInputs.empty() &&
             !interfaceInputs.lsqInputs.empty()) {
    // We only need an LSQ if the memory controller is not necessary (number of
    // inputs is zero)
    lsqOp = builder.create<handshake::LSQOp>(
        loc, memref, memStart, interfaceInputs.lsqInputs, ctrlEnd,
        interfaceInputs.lsqGroupSizes, lsqNumLoads);

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
    interfaceInputs.mcInputs.push_back(ldAddr);
    interfaceInputs.mcInputs.push_back(stAddr);
    interfaceInputs.mcInputs.push_back(stData);

    // Create the memory controller, adding 1 to its load count so that it
    // generates a load data result for the LSQ
    mcOp = builder.create<handshake::MemoryControllerOp>(
        loc, memref, memStart, interfaceInputs.mcInputs, ctrlEnd,
        interfaceInputs.mcBlocks, mcNumLoads + 1);

    // Add the MC's load data result to the LSQ's inputs and create the LSQ,
    // passing a flag to the builder so that it generates the necessary
    // outputs that will go to the MC
    interfaceInputs.lsqInputs.push_back(mcOp.getOutputs().back());
    lsqOp = builder.create<handshake::LSQOp>(
        loc, mcOp, interfaceInputs.lsqInputs, interfaceInputs.lsqGroupSizes,
        lsqNumLoads);

    // Resolve the backedges to fully connect the MC and LSQ
    ValueRange lsqMemResults = lsqOp.getOutputs().take_back(3);
    ldAddr.setValue(lsqMemResults[0]);
    stAddr.setValue(lsqMemResults[1]);
    stData.setValue(lsqMemResults[2]);
  }

  // At this point, all load ports are missing their second operand which is the
  // data value coming from a memory interface back to the port
  if (mcOp)
    reconnectLoads(mcPorts, mcOp, connect);
  if (lsqOp)
    reconnectLoads(lsqPorts, lsqOp, connect);

  return success();
}

LogicalResult FtdMemoryInterfaceBuilder::determineInterfaceInputsWithForks(
    InterfaceInputs &inputs, OpBuilder &builder, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, Value start) {

  // Create the fork nodes: for each group among the set of groups
  for (Group *group : groups) {
    Block *bb = group->bb;
    builder.setInsertionPointToStart(bb);

    // Add a lazy fork with two outputs, having the start control value as input
    // and two output ports, one for the LSQ and one for the subsequent buffer
    auto forkOp =
        builder.create<handshake::LazyForkOp>(memref.getLoc(), start, 2);

    // Add the new component to the list of components create for FTD and to the
    // fork graph
    forksGraph[bb] = forkOp;
  }

  // The second output of each lazy fork must be connected to the LSQ, so that
  // they can activate the allocation for the operations of the corresponding
  // basic block

  // For each LSQ port, consider their groups and the related operations
  for (auto &[_, lsqGroupOps] : lsqPorts) {

    // Get the first operations in the group associated to a single LSQ port
    Operation *firstOpInGroup = lsqGroupOps.front();

    // The output of the lazy fork associated to the group containing that
    // operation is now an input for the LSQ port
    Operation *forkNode = forksGraph[firstOpInGroup->getBlock()];
    assert(forkNode && "Fork node is not present in the expected basic block!");
    inputs.lsqInputs.push_back(forkNode->getResult(1));

    // All memory ports resuts that go to the interface are added to the list of
    // LSQ inputs
    for (Operation *lsqOp : lsqGroupOps)
      llvm::copy(getMemResultsToInterface(lsqOp),
                 std::back_inserter(inputs.lsqInputs));

    // Add the size of the group to our list
    inputs.lsqGroupSizes.push_back(lsqGroupOps.size());
  }

  // The rest of the function is identical to `determineInterfaceInputs` from
  // the base class `MemoryInterfaceBuilder`
  if (mcPorts.empty())
    return success();

  // The MC needs control signals from all blocks containing store ports
  // connected to an LSQ, since these requests end up being forwarded to the MC,
  // so we need to know the number of LSQ stores per basic block
  DenseMap<unsigned, unsigned> lsqStoresPerBlock;
  for (auto &[_, lsqGroupOps] : lsqPorts) {
    for (Operation *lsqOp : lsqGroupOps) {
      if (isa<handshake::StoreOp>(lsqOp)) {
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
      if (isa<handshake::StoreOp>(memOp))
        ++numStoresInBlock;
    }

    // Blocks with at least one store need to provide a control signal fed
    // through a constant indicating the number of stores in the block. The
    // output of the corresponding lazy fork is used as control signal of that
    // constant
    if (numStoresInBlock > 0) {
      Block *bb = nullptr;
      for (auto [blockIdx, bbPointer] : llvm::enumerate(funcOp))
        bb = (blockIdx == block) ? &bbPointer : bb;
      Value blockCtrl = forksGraph[bb]->getResult(0);
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

void Group::printDependenices() {
  llvm::dbgs() << "[MEM_GROUP] Group for [";
  bb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "]; predecessors = {";
  for (auto &gp : preds) {
    gp->bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ", ";
  }
  llvm::dbgs() << "}; successors = {";
  for (auto &gp : succs) {
    gp->bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ", ";
  }
  llvm::dbgs() << "} \n";
}

void ProdConsMemDep::printDependency() {
  llvm::dbgs() << "[PROD_CONS_MEM_DEP] Dependency from [";
  prodBb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "] to [";
  consBb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "]";
  if (isBackward)
    llvm::dbgs() << " (backward)";
  llvm::dbgs() << "\n";
}

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
