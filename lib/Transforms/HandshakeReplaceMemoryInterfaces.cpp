//===- HandshakeReplaceMemoryInterfaces.cpp - Replace memories --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-replace-memory-interfaces pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeReplaceMemoryInterfaces.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

/// Replaces memory interfaces in all Handshake functions in the module
/// according to information encoded in the `handshake::MemInterfaceAttr`
/// attribute held by each memory access port.
struct HandshakeReplaceMemoryInterfacesPass
    : public dynamatic::impl::HandshakeReplaceMemoryInterfacesBase<
          HandshakeReplaceMemoryInterfacesPass> {

  void runDynamaticPass() override;

private:
  /// Replace all memory interfaces in the Handshake function.
  LogicalResult replaceInFunction(handshake::FuncOp funcOp);

  /// Replace memory interfaces related to a specific memory region inside the
  /// Handshake function. The last argument maps each basic block ID in the
  /// function to a value inside the function that represents its control.
  LogicalResult replaceForMemRef(handshake::FuncOp funcOp,
                                 TypedValue<mlir::MemRefType> memref,
                                 const DenseMap<unsigned, Value> &ctrlVals);

  /// Replaces the interface's parent function terminator's operand that
  /// represents the memory completion signal associated to the interface's
  /// memory.
  void replaceMemCompletionSignal(MemoryOpInterface masterIface, Value newDone,
                                  OpBuilder &builder) const;
};
} // namespace

void HandshakeReplaceMemoryInterfacesPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();

  // Check that memory access ports are named
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  WalkResult res = modOp.walk([&](Operation *op) {
    if (!isa<handshake::LoadOpInterface, handshake::StoreOpInterface>(op))
      return WalkResult::advance();
    if (!namer.hasName(op)) {
      op->emitError() << "Memory access port must be named.";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return signalPassFailure();

  // Check that all eligible operations within Handshake function belon to a
  // basic block
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (!cannotBelongToCFG(&op) && !getLogicBB(&op)) {
        op.emitError() << "Operation should have basic block "
                          "attribute.";
        return signalPassFailure();
      }
    }
  }

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(replaceInFunction(funcOp)))
      return signalPassFailure();
  }
}

LogicalResult HandshakeReplaceMemoryInterfacesPass::replaceInFunction(
    handshake::FuncOp funcOp) {
  for (BlockArgument arg : funcOp.getArguments()) {
    HandshakeCFG cfg(funcOp);
    DenseMap<unsigned, Value> ctrlVals;
    if (failed(cfg.getControlValues(ctrlVals))) {
      return funcOp.emitError() << "failed to identify control signals for all "
                                   "\"basic blocks\" in the function";
    }

    if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)) {
      if (failed(replaceForMemRef(funcOp, memref, ctrlVals)))
        return failure();
    }
  }
  return success();
}

static constexpr llvm::StringLiteral
    ERR_ATTR("memory operation must have attribute of type "
             "'handshake::MemInterfaceAttr' to encode which memory interface "
             "it should connect to.");

/// Records a memory access port replacement.
static void recordReplacement(Operation *oldOp, Operation *newOp,
                              MemoryOpLowering &memLowering) {
  inheritBB(oldOp, newOp);
  memLowering.recordReplacement(oldOp, newOp, false);
  oldOp->replaceAllUsesWith(newOp->getResults());
  oldOp->erase();
}

/// Replaces a load port of the source type with a load port of a different
/// type.
template <typename DstLoadOp, typename SrcLoadOp>
static DstLoadOp replaceLoad(SrcLoadOp loadOp, Value dataInput,
                             OpBuilder &builder,
                             MemoryOpLowering &memLowering) {
  auto newLoadOp = builder.create<DstLoadOp>(
      loadOp->getLoc(), loadOp.getAddressInput(), dataInput);
  recordReplacement(loadOp, newLoadOp, memLowering);
  return newLoadOp;
}

/// Replaces a store port of the source type with a store port of a different
/// type.
template <typename DstStpreOp, typename SrcStoreOp>
static DstStpreOp replaceStore(SrcStoreOp storeOp, OpBuilder &builder,
                               MemoryOpLowering &memLowering) {
  auto newStoreOp = builder.create<DstStpreOp>(
      storeOp->getLoc(), storeOp.getAddressInput(), storeOp.getDataInput());
  recordReplacement(storeOp, newStoreOp, memLowering);
  return newStoreOp;
}

LogicalResult HandshakeReplaceMemoryInterfacesPass::replaceForMemRef(
    handshake::FuncOp funcOp, TypedValue<mlir::MemRefType> memref,
    const DenseMap<unsigned, Value> &ctrlVals) {
  // There should be at most one memref user in any well-formed function
  auto memrefUsers = memref.getUsers();
  assert(std::distance(memrefUsers.begin(), memrefUsers.end()) <= 1 &&
         "expected at most one memref user");
  if (memrefUsers.empty()) {
    return success();
  }

  // Identify all memory interfaces (master and potential slaves) for the region
  auto masterIface = cast<MemoryOpInterface>(*memrefUsers.begin());
  handshake::MemoryControllerOp mcOp = nullptr;
  handshake::LSQOp lsqOp;
  if (lsqOp = dyn_cast<handshake::LSQOp>(masterIface.getOperation()); !lsqOp) {
    // The master memory interface must be an MC
    mcOp = cast<handshake::MemoryControllerOp>(masterIface.getOperation());

    // There may still be an LSQ slave interface, look for it
    MCPorts ports = mcOp.getPorts();
    if (ports.connectsToLSQ())
      lsqOp = ports.getLSQPort().getLSQOp();
  }

  // Context and builder for creating new operation
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  MemoryOpLowering memLowering(getAnalysis<NameAnalysis>());
  MemoryInterfaceBuilder memBuilder(funcOp, memref, masterIface.getMemStart(),
                                    masterIface.getCtrlEnd(), ctrlVals);
  BackedgeBuilder backedgeBuilder(builder, funcOp->getLoc());

  // Collect all access ports related to the memory region under consideration
  DenseSet<Operation *> regionPorts;
  if (mcOp) {
    MCPorts mcPorts = mcOp.getPorts();
    for (MCBlock &block : mcPorts.getBlocks()) {
      for (MemoryPort &port : block->accessPorts)
        regionPorts.insert(port.portOp);
    }
  }
  if (lsqOp) {
    LSQPorts lsqPorts = lsqOp.getPorts();
    for (LSQGroup &group : lsqPorts.getGroups()) {
      for (MemoryPort &port : group->accessPorts)
        regionPorts.insert(port.portOp);
    }
  }

  using FailOrGroup = FailureOr<std::optional<unsigned>>;
  /// Common logic for replacing access ports currently connected to an MC.
  auto replaceMCPort = [&](Operation *accessOp) -> FailOrGroup {
    auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(accessOp);
    if (!memAttr)
      return accessOp->emitError() << ERR_ATTR;
    if (memAttr.connectsToMC()) {
      memBuilder.addMCPort(accessOp);
      removeDialectAttr<MemInterfaceAttr>(accessOp);
      return {std::nullopt};
    }
    return memAttr.getLsqGroup();
  };

  /// Common logic for replacing access ports currently connected to an LSQ.
  auto replaceLSQPort = [&](Operation *accessOp) -> FailureOr<bool> {
    auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(accessOp);
    if (!memAttr)
      return accessOp->emitError() << ERR_ATTR;
    if (memAttr.connectsToLSQ()) {
      memBuilder.addLSQPort(*memAttr.getLsqGroup(), accessOp);
      removeDialectAttr<MemInterfaceAttr>(accessOp);
      return false;
    }
    return true;
  };

  // It is important to iterate in operation order here when adding ports to the
  // memory builder to maintain the original program order in each memory group
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    // Ignore memory ports that are not related to the memory region under
    // consideration
    if (!regionPorts.contains(&op))
      continue;

    builder.setInsertionPoint(&op);
    llvm::TypeSwitch<Operation *, LogicalResult>(&op)
        .Case<handshake::MCLoadOp>([&](handshake::MCLoadOp loadOp) {
          FailOrGroup connectRes = replaceMCPort(loadOp);
          if (failed(connectRes))
            return failure();
          if (*connectRes) {
            auto op = replaceLoad<LSQLoadOp>(
                loadOp, backedgeBuilder.get(loadOp.getDataInput().getType()),
                builder, memLowering);
            memBuilder.addLSQPort(**connectRes, op);
          }
          return success();
        })
        .Case<handshake::MCStoreOp>([&](handshake::MCStoreOp storeOp) {
          FailOrGroup connectRes = replaceMCPort(storeOp);
          if (failed(connectRes))
            return failure();
          if (*connectRes) {
            auto op = replaceStore<LSQStoreOp>(storeOp, builder, memLowering);
            memBuilder.addLSQPort(**connectRes, op);
          }
          return success();
        })
        .Case<handshake::LSQLoadOp>([&](handshake::LSQLoadOp loadOp) {
          FailureOr<bool> connectRes = replaceLSQPort(loadOp);
          if (failed(connectRes))
            return failure();
          if (*connectRes) {
            auto op = replaceLoad<MCLoadOp>(
                loadOp, backedgeBuilder.get(loadOp.getDataInput().getType()),
                builder, memLowering);
            memBuilder.addMCPort(op);
          }
          return success();
        })
        .Case<handshake::LSQStoreOp>([&](handshake::LSQStoreOp storeOp) {
          FailureOr<bool> connectRes = replaceLSQPort(storeOp);
          if (failed(connectRes))
            return failure();
          if (*connectRes) {
            auto op = replaceStore<MCStoreOp>(storeOp, builder, memLowering);
            memBuilder.addMCPort(op);
          }
          return success();
        });
  }

  // Rename memory accesses referenced by memory dependencies attached to the
  // old and new memory ports
  memLowering.renameDependencies(funcOp);

  // Instantiate new memory interfaces
  handshake::MemoryControllerOp newMCOp;
  handshake::LSQOp newLSQOp;
  if (failed(memBuilder.instantiateInterfaces(builder, newMCOp, newLSQOp)))
    return failure();
  assert(newMCOp || newLSQOp && "no new interface instantiated");

  // The memory completiong signal needs to come from the new interfaces
  Value newMemEnd = newMCOp ? newMCOp.getMemEnd() : newLSQOp.getMemEnd();
  replaceMemCompletionSignal(masterIface, newMemEnd, builder);

  if (mcOp) {
    if (lsqOp) {
      // In case of a master MC and slave LSQ situation, the second to last
      // result of the MC is a load data signal going to the LSQ. It needs to be
      // temporarily replaced with a backedge to allow us to remove the MC
      // before the LSQ
      builder.setInsertionPoint(mcOp);
      Value dataToLSQ = mcOp.getResult(mcOp.getNumResults() - 2);
      dataToLSQ.replaceAllUsesWith(backedgeBuilder.get(dataToLSQ.getType()));
    }
    mcOp.erase();
  }
  if (lsqOp)
    lsqOp.erase();
  return success();
}

void HandshakeReplaceMemoryInterfacesPass::replaceMemCompletionSignal(
    MemoryOpInterface masterIface, Value newDone, OpBuilder &builder) const {
  auto funcOp = masterIface->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "expected parent Handshake function");
  auto endOp = dyn_cast<EndOp>(funcOp.front().getTerminator());

  // Look for the memory completion signal of the memory referenced by the LSQ
  // in the function terminator's operands
  Value done = masterIface.getMemEnd();
  auto indexedTermOperands = llvm::enumerate(endOp->getOperands());
  auto oprdIt = llvm::find_if(indexedTermOperands, [&](auto idxOprd) {
    return idxOprd.value() == done;
  });
  assert(oprdIt != indexedTermOperands.end() && "no memory completion signal");

  auto [idx, _] = *oprdIt;
  endOp->setOperand(idx, newDone);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeReplaceMemoryInterfaces() {
  return std::make_unique<HandshakeReplaceMemoryInterfacesPass>();
}
