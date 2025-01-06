//===- HandshakeInactivateEnforcedDeps.cpp - LSQ flow analysis ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-inactivate-enforced-deps pass, using the logic
// introduced in https://ieeexplore.ieee.org/document/8977873.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeInactivateEnforcedDeps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-inactivate-enforced-deps"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

using DependencyMap = DenseMap<Operation*, SmallVector<MemDependenceAttr>>;

namespace {

/// Simple pass driver for the Inactivate Enforced Dependencies pass. 
/// Modifies the IR by marking the enforced dependencies as inactive and then
/// setting `handshake::MemInterfaceAttr` attributes on memory ports.
struct HandshakeInactivateEnforcedDepsPass
    : public dynamatic::impl::HandshakeInactivateEnforcedDepsBase<
          HandshakeInactivateEnforcedDepsPass> {

  void runDynamaticPass() override;

  /// Analyzes all memory regions inside a Handshake functions and marks all
  /// operations representing memory accesses to it with the
  /// `handshake::MemInterfaceAttr` attribute.
  void analyzeFunction(handshake::FuncOp funcOp);

  /// Analyzes a specific memory region inside a Handshake function and
  /// determines whether each of its access port should go through an LSQ.
  void analyzeMemRef(handshake::FuncOp funcOp,
                     TypedValue<mlir::MemRefType> memref, HandshakeCFG &cfg);
};
} // namespace

void HandshakeInactivateEnforcedDepsPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();

  // Check that memory access ports are named
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  WalkResult res = modOp.walk([&](Operation *op) {
    if (!isa<handshake::LoadOp, handshake::StoreOp>(op))
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
        op.emitError() << "Operation should have basic block attribute.";
        return signalPassFailure();
      }
    }
  }

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
    analyzeFunction(funcOp);
}

void HandshakeInactivateEnforcedDepsPass::analyzeFunction(handshake::FuncOp funcOp) {
  for (BlockArgument arg : funcOp.getArguments()) {
    HandshakeCFG cfg(funcOp);
    if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg))
      analyzeMemRef(funcOp, memref, cfg);
  }
}

/// Determines whether the load is globally in-order dependent (GIID) on the
/// store along all non-cyclic CFG paths between them.
static bool isStoreGIIDOnLoad(handshake::LoadOp loadOp,
                              handshake::StoreOp storeOp, HandshakeCFG &cfg) {
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

/// returns the inactivated version of the dependency
static MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency){
  MLIRContext* ctx = dependency.getContext();
  return MemDependenceAttr::get(ctx, dependency.getDstAccess(), dependency.getLoopDepth(), dependency.getComponents(), BoolAttr::get(ctx, false));
}


/// Inactivates the dependencies that are enforced by cheking whether the load is
/// globally [Instruction] in-order dependent (GIID) on the store or not
static void inactivateEnforcedWARs(DenseSet<handshake::LoadOp> &loadOps,
                            DenseSet<handshake::StoreOp> &storeOps,
                            DependencyMap &opDeps,
                            HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::StoreOp> storesByName;
  for (handshake::StoreOp storeOp : storeOps)
    storesByName.insert({getUniqueName(storeOp), storeOp});

  // We only need to check stores that depend on the load (WAR dependencies) as
  // others are already provably independent. We may check a single store
  // multiple times if it depends on the load at multiple loop depths
  for (handshake::LoadOp loadOp : loadOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(loadOp)) {
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive().getValue())
          continue;
        auto storeOp = storesByName.at(dependency.getDstAccess());
        // if the laod is GIID which means there is a data dependency,
        // the dependency should be inactivated
        if (isStoreGIIDOnLoad(loadOp, storeOp, cfg))
          opDeps[loadOp].push_back(getInactivatedDependency(dependency));
        else
          opDeps[loadOp].push_back(dependency); 
      }
    }
  }
}


/// replaces the memory dependence array attribute with the dependencies 
/// given in the dictionary `opDeps`
static void changeOpDeps(DependencyMap& opDeps, MLIRContext* ctx){
  for (auto &[op, deps] : opDeps)
    setDialectAttr<MemDependenceArrayAttr>(op, ctx, deps);
}


/// Inactivates the WAW dependencies between an operation and itself
static void inactivateEnforcedWAWs(DenseSet<handshake::StoreOp> &storeOps, DenseMap<Operation*, SmallVector<MemDependenceAttr>>& opDeps){
  for (handshake::StoreOp storeOp : storeOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(storeOp)){
      StringRef storeName = getUniqueName(storeOp);
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive().getValue())
          continue;
        StringRef dstName = dependency.getDstAccess();

        // a WAW dependency between an operation and itself may be ignored.
        if (storeName == dstName)
          opDeps[storeOp].push_back(getInactivatedDependency(dependency));
        else
          opDeps[storeOp].push_back(dependency);
      }
    }
  }
}

// gets the set of all load/store operations and returns the subset of
// operations that are involved in at least one active dependency.
static DenseSet<Operation*> getOpsWithNonEnforcedDeps(DenseSet<Operation*> &loadStoreOps){
  DenseMap<StringRef, Operation*> nameToOpMapping;
  for (Operation *op : loadStoreOps){
    StringRef name = getUniqueName(op);
    nameToOpMapping[name] = op;
  }

  DenseSet<Operation*> opsWithNonEnforcedDeps;
  bool hasAtLeastOneActive;

  for (Operation *op : loadStoreOps){
    hasAtLeastOneActive = false;
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(op)){
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive().getValue())
          continue;
        hasAtLeastOneActive = true;
        Operation *dstOp = nameToOpMapping[dependency.getDstAccess()];
        opsWithNonEnforcedDeps.insert(dstOp);
      }
    }
    if (hasAtLeastOneActive)
        opsWithNonEnforcedDeps.insert(op);
  }
  return opsWithNonEnforcedDeps;
}


/// Mark all accesses with the `MemInterfaceAttr`, indicating whether they
/// should connect to an MC or LSQ depending on their dependencies with other
/// accesses.
static void markLSQPorts(const DenseSet<Operation*> allOps,
                         const DenseSet<Operation*> opsWithNonEnforcedDeps,
                         const DenseMap<Operation *, unsigned> &groupMap,
                         MLIRContext *ctx) {
  for (Operation *op : allOps) {
    if (opsWithNonEnforcedDeps.contains(op))
      setDialectAttr<MemInterfaceAttr>(op, ctx, groupMap.at(op));
    else
      setDialectAttr<MemInterfaceAttr>(op, ctx);
  }
};

void HandshakeInactivateEnforcedDepsPass::analyzeMemRef(
    handshake::FuncOp funcOp, TypedValue<mlir::MemRefType> memref,
    HandshakeCFG &cfg) {
  LLVM_DEBUG({
    unsigned idx = cast<BlockArgument>(memref).getArgNumber();
    StringRef argName = funcOp.getArgName(idx);
    llvm::dbgs() << "Analyzing interfaces for region '" << argName << "'\n";
  });

  // There should be at most one memref user in any well-formed function
  auto memrefUsers = memref.getUsers();
  assert(std::distance(memrefUsers.begin(), memrefUsers.end()) <= 1 &&
         "expected at most one memref user");
  if (memrefUsers.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "\tNo interfaces\n");
    return;
  }
  MLIRContext *ctx = &getContext();

  // Identify all memory interfaces (master and potential slaves) for the region
  Operation *memOp = *memrefUsers.begin();
  handshake::LSQOp lsqOp;
  if (lsqOp = dyn_cast<handshake::LSQOp>(memOp); !lsqOp) {
    // The master memory interface must be an MC
    auto mcOp = cast<handshake::MemoryControllerOp>(memOp);
    // Ports to memory controllers will always remain connected to a memory
    // controller, mark them as such with the memory interface attribute
    MCPorts mcPorts = mcOp.getPorts();
    for (MCBlock &block : mcPorts.getBlocks()) {
      for (MemoryPort &port : block->accessPorts)
        setDialectAttr<MemInterfaceAttr>(port.portOp, ctx);
    }
    // Nothing else to do if the region has no LSQ
    if (!mcPorts.connectsToLSQ()) {
      LLVM_DEBUG(llvm::dbgs() << "\tNo LSQ interface for the region\n");
      return;
    }
    lsqOp = mcPorts.getLSQPort().getLSQOp();
  }

  // Identify load and store accesses to the LSQ
  DenseSet<handshake::LoadOp> lsqLoadOps;
  DenseSet<handshake::StoreOp> lsqStoreOps;
  DenseSet<Operation*> loadStoreOps;
  DenseMap<Operation *, unsigned> groupMap;
  LSQPorts lsqPorts = lsqOp.getPorts();
  for (LSQGroup &group : lsqPorts.getGroups()) {
    for (MemoryPort &port : group->accessPorts) {
      groupMap.insert({port.portOp, group.groupID});
      loadStoreOps.insert(port.portOp);
      if (auto loadOp = dyn_cast<handshake::LoadOp>(port.portOp))
        lsqLoadOps.insert(loadOp);
      else
        lsqStoreOps.insert(cast<handshake::StoreOp>(port.portOp));
    }
  }

  DependencyMap opDeps;
  inactivateEnforcedWARs(lsqLoadOps, lsqStoreOps, opDeps, cfg);
  inactivateEnforcedWAWs(lsqStoreOps, opDeps);
  changeOpDeps(opDeps, ctx);

  DenseSet<Operation*> opsWithNonEnforcedDeps;
  opsWithNonEnforcedDeps = getOpsWithNonEnforcedDeps(loadStoreOps);


  // Tag LSQ access ports with the interface they should actually connect to,
  // which may be different from the one they currently connect to
  markLSQPorts(loadStoreOps, opsWithNonEnforcedDeps, groupMap, ctx);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInactivateEnforcedDeps() {
  return std::make_unique<HandshakeInactivateEnforcedDepsPass>();
}
