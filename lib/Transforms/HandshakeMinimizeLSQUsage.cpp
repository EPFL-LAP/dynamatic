//===- HandshakeMiminizeLSQUsage.cpp - LSQ flow analysis --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-minimize-lsq-usage pass, using the logic
// introduced in https://ieeexplore.ieee.org/document/8977873.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMinimizeLSQUsage.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-minimize-lsq-usage"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

/// Simple pass driver for the LSQ usage minimization pass. Internally uses a
/// single rewrite pattern to optimize LSQs.
struct HandshakeMinimizeLSQUsagePass
    : public dynamatic::impl::HandshakeMiminizeLSQUsageBase<
          HandshakeMinimizeLSQUsagePass> {

  void runDynamaticPass() override;

private:
  /// Groups all the information we need to optimize the LSQ, if it is possible
  /// to do so.
  struct LSQInfo {
    /// Maps LSQ load and store ports to the index of the group they belong to.
    DenseMap<Operation *, unsigned> lsqPortToGroup;
    /// All loads to the LSQ, in group order.
    SetVector<handshake::LSQLoadOp> lsqLoadOps;
    /// All stores to the LSQ, in group order.
    SetVector<handshake::LSQStoreOp> lsqStoreOps;
    /// All accesses to a potential MC connected to the LSQ, in block order.
    SetVector<Operation *> mcOps;
    /// Names of loads to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableLoads;
    /// Names of stores to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableStores;
    /// Maps basic block IDs to their control value, for reconnecting memory
    /// interfaces to the circuit in case the LSQ is optimizable.
    DenseMap<unsigned, Value> ctrlVals;
    /// Whether the LSQ is optimizable.
    bool optimizable = false;

    /// Determines whether the LSQ is optimizable, filling in all struct members
    /// in the process. First, stores the list of loads and stores to the LSQ,
    /// then analyses the DFG to potentially identify accesses that do not need
    /// to go through an LSQ because of control-flow-enforced dependencies, and
    /// finally determines whether the LSQ is optimizable.
    LSQInfo(handshake::LSQOp lsqOp, NameAnalysis &namer);
  };

  /// Reduces the size or erases the LSQ if least some accesses can be
  /// guaranteed to happen in order due to data dependencies within the dataflow
  /// circuit. On success, this may result in the replacement of the LSQ with an
  /// MC, or the displacement of some memory ports from the LSQ to an MC which
  /// may or may not exist prior.
  void tryToOptimizeLSQ(handshake::LSQOp lsqOp);

  /// Replaces the parent function terminator's operand that represents the
  /// memory completion signal associated to the LSQ's memory.
  void replaceMemoryEnd(handshake::LSQOp oldLSQOp, Value newDone,
                        OpBuilder &builder) const;
};
} // namespace

/// Determines whether there exists any RAW dependency between the load and the
/// stores.
static bool hasRAW(handshake::LSQLoadOp loadOp,
                   SetVector<handshake::LSQStoreOp> &storeOps) {
  StringRef loadName = getUniqueName(loadOp);
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp)) {
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (dependency.getDstAccess() == loadName) {
          LLVM_DEBUG(llvm::dbgs()
                     << "\t" << loadName
                     << " cannot be removed from LSQ: RAW dependency with "
                     << getUniqueName(storeOp) << "\n");
          return true;
        }
      }
    }
  }
  return false;
}

/// Determines whether the load is globally in-order independent (GIID) on the
/// store along all non-cyclic CFG paths between them.
static bool isStoreGIIDOnLoad(handshake::LSQLoadOp loadOp,
                              handshake::LSQStoreOp storeOp,
                              HandshakeCFG &cfg) {
  // Identify all CFG paths from the block containing the load to the block
  // containing the store
  handshake::FuncOp funcOp = loadOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "parent of load access must be handshake function");
  SmallVector<CFGPath> allPaths;
  std::optional<unsigned> loadBB = getLogicBB(loadOp);
  std::optional<unsigned> storeBB = getLogicBB(storeOp);
  assert(loadBB && storeBB && "memory accesses must belong to blocks");
  cfg.getNonCyclicPaths(*loadBB, *storeBB, allPaths);

  // There must be a dependence between any operand of the store with the load
  // data result on all CFG paths between them
  Value loadData = loadOp.getDataResult();
  return llvm::all_of(allPaths, [&](CFGPath &path) {
    return isGIID(loadData, storeOp->getOpOperand(0), path) ||
           isGIID(loadData, storeOp->getOpOperand(1), path);
  });
}

/// Determines whether the load is globally in-order independent (GIID) on all
/// stores with which it has a WAR dependency along all non-cyclic CFG paths
/// between them.
static bool hasEnforcedWARs(handshake::LSQLoadOp loadOp,
                            SetVector<handshake::LSQStoreOp> &storeOps,
                            HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::LSQStoreOp> storesByName;
  for (handshake::LSQStoreOp storeOp : storeOps)
    storesByName[getUniqueName(storeOp)] = storeOp;

  // We only need to check stores that depend on the load (WAR dependencies) as
  // others are already provably independent. We may check a single store
  // multiple times if it depends on the load at multiple loop depths
  if (auto deps = getUniqueAttr<MemDependenceArrayAttr>(loadOp)) {
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      handshake::LSQStoreOp storeOp = storesByName[dependency.getDstAccess()];
      assert(storeOp && "unknown store operation");
      if (!isStoreGIIDOnLoad(loadOp, storeOp, cfg)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\t" << getUniqueName(loadOp)
                   << " cannot be removed from LSQ: non-enforced WAR with "
                   << getUniqueName(storeOp) << "\n");
        return false;
      }
    }
  }
  return true;
}

void HandshakeMinimizeLSQUsagePass::runDynamaticPass() {
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

  // We are going do be modifying the IR a lot during LSQ optimization, so we
  // first collect all LSQs in a vector to not have iterator issues
  SmallVector<handshake::LSQOp> lsqs;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
    llvm::copy(funcOp.getOps<handshake::LSQOp>(), std::back_inserter(lsqs));

  // Try to optimize each LSQ independently. We only need to attempt the
  // optimization once for each because they are all independent
  for (handshake::LSQOp lsqOp : lsqs)
    tryToOptimizeLSQ(lsqOp);
}

HandshakeMinimizeLSQUsagePass::LSQInfo::LSQInfo(handshake::LSQOp lsqOp,
                                                NameAnalysis &namer) {

  // Identify load and store accesses to the LSQ
  LSQPorts ports = lsqOp.getPorts();
  for (auto [idx, group] : llvm::enumerate(ports.getGroups())) {
    for (MemoryPort &port : group->accessPorts) {
      if (std::optional<LSQLoadPort> loadPort = dyn_cast<LSQLoadPort>(port)) {
        handshake::LSQLoadOp lsqLoadOp = loadPort->getLSQLoadOp();
        lsqPortToGroup[lsqLoadOp] = idx;
        lsqLoadOps.insert(lsqLoadOp);
      } else if (std::optional<LSQStorePort> storePort =
                     dyn_cast<LSQStorePort>(port)) {
        handshake::LSQStoreOp lsqStoreOp = storePort->getLSQStoreOp();
        lsqPortToGroup[lsqStoreOp] = idx;
        lsqStoreOps.insert(lsqStoreOp);
      }
    }
  }

  // We will need CFG information about the containing Handshake function
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  HandshakeCFG cfg(funcOp);

  // Compute the set of loads that can go directly to an MC inside of an LSQ
  for (handshake::LSQLoadOp loadOp : lsqLoadOps) {
    // Loads with no RAW dependencies and which satisfy the GIID property with
    // all stores may be removed
    if (!hasRAW(loadOp, lsqStoreOps) &&
        hasEnforcedWARs(loadOp, lsqStoreOps, cfg)) {
      LLVM_DEBUG(llvm::dbgs() << "\t" << getUniqueName(loadOp)
                              << " can be removed from LSQ\n");
      removableLoads.insert(getUniqueName(loadOp));
    }
  }
  if (removableLoads.empty())
    return;

  // Compute the set of stores that can go directly to an MC inside of an LSQ
  // now that we know that some loads are out
  DenseSet<handshake::LSQStoreOp> storesStillInSet;
  for (handshake::LSQStoreOp storeOp : lsqStoreOps) {
    auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp);
    if (!deps)
      continue;
    StringRef storeName = getUniqueName(storeOp);
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      // The dependency may be ignored if it's a WAW with itself, otherwise it
      // must be taken into account
      StringRef dstName = dependency.getDstAccess();
      if (storeName != dstName) {
        storesStillInSet.insert(storeOp);
        Operation *dstOp = namer.getOp(dstName);
        if (auto otherStoreOp = dyn_cast<handshake::LSQStoreOp>(dstOp))
          storesStillInSet.insert(otherStoreOp);
      }
    }
  }
  for (handshake::LSQStoreOp storeOp : lsqStoreOps) {
    if (!storesStillInSet.contains(storeOp)) {
      LLVM_DEBUG(llvm::dbgs() << "\t" << getUniqueName(storeOp)
                              << " can be removed from LSQ\n");
      removableStores.insert(getUniqueName(storeOp));
    } else {
      LLVM_DEBUG(llvm::dbgs() << "\t" << getUniqueName(storeOp)
                              << " cannot be removed from LSQ: WAW or RAW "
                                 "dependency with other access\n");
    }
  }

  // We need the control value of each block in the Handshake function to be
  // able to recreate memory interfaces
  if (failed(cfg.getControlValues(ctrlVals))) {
    LLVM_DEBUG(llvm::dbgs() << "\tFailed to get control values from CFG\n");
    return;
  }

  // If the LSQ connects to an MC, memory accesses going directly to the MC will
  // also need to be rerouted
  if (handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC()) {
    MCPorts mcPorts = mcOp.getPorts();
    for (auto [idx, group] : llvm::enumerate(mcPorts.getBlocks())) {
      for (MemoryPort &port : group->accessPorts) {
        if (isa<MCLoadPort, MCStorePort>(port))
          mcOps.insert(port.portOp);
      }
    }
  }

  optimizable = true;
}

void HandshakeMinimizeLSQUsagePass::replaceMemoryEnd(handshake::LSQOp oldLSQOp,
                                                     Value newDone,
                                                     OpBuilder &builder) const {
  auto funcOp = oldLSQOp->getParentOfType<handshake::FuncOp>();
  auto endOp = dyn_cast<EndOp>(funcOp.front().getTerminator());
  handshake::MemoryControllerOp mcOp = oldLSQOp.getConnectedMC();

  // Look for the memory completion signal of the memory referenced by the LSQ
  // in the function terminator's operands
  Value done =
      mcOp ? cast<Value>(mcOp.getMemEnd()) : oldLSQOp->getResults().back();
  auto indexedTermOperands = llvm::enumerate(endOp->getOperands());
  auto oprdIt = llvm::find_if(indexedTermOperands, [&](auto idxOprd) {
    return idxOprd.value() == done;
  });
  assert(oprdIt != indexedTermOperands.end() && "no memory completion signal");

  auto [idx, _] = *oprdIt;
  endOp->setOperand(idx, newDone);
}

void HandshakeMinimizeLSQUsagePass::tryToOptimizeLSQ(handshake::LSQOp lsqOp) {
  LLVM_DEBUG(llvm::dbgs() << "Attempting optimization of LSQ ("
                          << getUniqueName(lsqOp) << ")\n");

  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  // Check whether the LSQ is optimizable
  LSQInfo lsqInfo(lsqOp, namer);
  if (!lsqInfo.optimizable) {
    LLVM_DEBUG(llvm::dbgs() << "\tLSQ cannot be optimized\n");
    return;
  }

  // Context and builder for creating new operation
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Keep track of memory replacements so that dependencies between accesses
  // stay consistent
  MemoryOpLowering memOpLowering(namer);

  // Builder to instantiate new memory interfaces after transforming some of our
  // LSQ ports into MC ports
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  MemRefType memType = cast<MemRefType>(lsqOp.getMemRef().getType());
  MemoryInterfaceBuilder memBuilder(funcOp, lsqOp.getMemRef(),
                                    lsqOp.getMemStart(), lsqOp.getCtrlEnd(),
                                    lsqInfo.ctrlVals);

  // Existing memory ports and memory interface(s) reference each other's
  // results/operands, which makes them un-earasable since it's disallowed to
  // remove an operation whose results still have active uses. Use temporary
  // backedges to replace the to-be-removed memory ports' results in the memory
  // interface(s) operands, which allows us to first delete the memory ports and
  // finally the memory interfaces. All backedges are deleted automatically
  // before the method retuns
  BackedgeBuilder backedgeBuilder(builder, lsqOp.getLoc());

  // Collect all memory accesses that must be rerouted to new memory interfaces.
  // It's important to iterate in operation order here to maintain the original
  // program order in each memory group
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<handshake::LSQLoadOp>([&](handshake::LSQLoadOp lsqLoadOp) {
          if (!lsqInfo.lsqLoadOps.contains(lsqLoadOp))
            return;

          if (!lsqInfo.removableLoads.contains(getUniqueName(lsqLoadOp))) {
            memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqLoadOp], lsqLoadOp);
            return;
          }

          // Replace the LSQ load with an equivalent MC load
          builder.setInsertionPoint(lsqLoadOp);
          auto mcLoadOp = builder.create<handshake::MCLoadOp>(
              lsqLoadOp->getLoc(), memType, lsqLoadOp.getAddressInput());
          inheritBB(lsqLoadOp, mcLoadOp);

          // Record operation replacement (change interface to MC)
          memOpLowering.recordReplacement(lsqLoadOp, mcLoadOp, false);
          setUniqueAttr(mcLoadOp, handshake::MemInterfaceAttr::get(ctx));
          memBuilder.addMCPort(mcLoadOp);

          // Replace the original port operation's results and erase it
          Value addrOut = lsqLoadOp.getAddressOutput();
          addrOut.replaceAllUsesWith(backedgeBuilder.get(addrOut.getType()));
          Value dataOut = lsqLoadOp.getDataResult();
          dataOut.replaceAllUsesWith(mcLoadOp.getDataResult());
          lsqLoadOp.erase();
        })
        .Case<handshake::LSQStoreOp>([&](handshake::LSQStoreOp lsqStoreOp) {
          if (!lsqInfo.lsqStoreOps.contains(lsqStoreOp))
            return;

          if (!lsqInfo.removableStores.contains(getUniqueName(lsqStoreOp))) {
            memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqStoreOp],
                                  lsqStoreOp);
            return;
          }

          // Replace the LSQ store with an equivalent MC store
          builder.setInsertionPoint(lsqStoreOp);
          auto mcStoreOp = builder.create<handshake::MCStoreOp>(
              lsqStoreOp->getLoc(), lsqStoreOp.getAddressInput(),
              lsqStoreOp.getDataInput());
          inheritBB(lsqStoreOp, mcStoreOp);

          // Record operation replacement (change interface to MC)
          memOpLowering.recordReplacement(lsqStoreOp, mcStoreOp, false);
          setUniqueAttr(mcStoreOp, handshake::MemInterfaceAttr::get(ctx));
          memBuilder.addMCPort(mcStoreOp);

          // Replace the original port operation's results and erase it
          Value addrOut = lsqStoreOp.getAddressOutput();
          addrOut.replaceAllUsesWith(backedgeBuilder.get(addrOut.getType()));
          Value dataOut = lsqStoreOp.getDataOutput();
          dataOut.replaceAllUsesWith(backedgeBuilder.get(dataOut.getType()));
          lsqStoreOp->erase();
        })
        .Case<handshake::MCLoadOp>([&](handshake::MCLoadOp mcLoadOp) {
          // The data operand coming from the current memory interface will be
          // replaced during interface creation by the MemoryInterfaceBuilder
          if (lsqInfo.mcOps.contains(mcLoadOp))
            memBuilder.addMCPort(mcLoadOp);
        })
        .Case<handshake::MCStoreOp>([&](handshake::MCStoreOp mcStoreOp) {
          if (lsqInfo.mcOps.contains(mcStoreOp))
            memBuilder.addMCPort(mcStoreOp);
        });
  }

  // Rename memory accesses referenced by memory dependencies attached to the
  // old and new memory ports
  memOpLowering.renameDependencies(lsqOp->getParentOp());

  // Instantiate new memory interfaces
  handshake::MemoryControllerOp newMCOp;
  handshake::LSQOp newLSQOp;
  if (failed(memBuilder.instantiateInterfaces(builder, newMCOp, newLSQOp))) {
    LLVM_DEBUG(llvm::dbgs() << "\tFailed to instantiate memory interfaces\n");
    return;
  }
  assert(newMCOp && "lsq minimization did not generate a new MC");

  LLVM_DEBUG({
    llvm::dbgs() << "\t[SUCCESS] LSQ ";
    if (newLSQOp)
      llvm::dbgs() << "size reduced ";
    else
      llvm::dbgs() << "removed entirely ";
    if (lsqOp.isConnectedToMC())
      llvm::dbgs() << "(previously inexistent MC introduced)\n";
    else
      llvm::dbgs() << "(MC size increased)\n";
  });

  // Replace memory control signals consumed by the end operation
  replaceMemoryEnd(lsqOp, newMCOp.getMemEnd(), builder);

  // If the LSQ is connected to an MC, we delete it first. The second to last
  // result of the MC is a load data signal going to the LSQ, which needs to be
  // temporarily replaced with a backedge to allow us to remove the MC before
  // the LSQ
  if (handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC()) {
    builder.setInsertionPoint(mcOp);
    Value dataToLSQ = mcOp.getResult(mcOp.getNumResults() - 2);
    dataToLSQ.replaceAllUsesWith(backedgeBuilder.get(dataToLSQ.getType()));
    mcOp.erase();
  }

  // Now we can safely delete the original LSQ
  lsqOp.erase();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
