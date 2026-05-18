//===- HandshakeInactivateDeps.cpp - Inactivate memory deps -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-inactivate-deps pass, using the logic introduced
// in https://ieeexplore.ieee.org/document/8977873.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "handshake-inactivate-deps"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEINACTIVATEDEPS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

using DependencyMap = DenseMap<Operation *, SmallVector<MemDependenceAttr>>;

namespace {

struct HandshakeInactivateDepsPass
    : public dynamatic::impl::HandshakeInactivateDepsBase<
          HandshakeInactivateDepsPass> {

  using HandshakeInactivateDepsBase::HandshakeInactivateDepsBase;

  void runDynamaticPass() override;

  void analyzeFunction(handshake::FuncOp funcOp);
};

} // namespace

/// Determines whether the store is globally in-order dependent (GIID) on the
/// load along all non-cyclic CFG paths between them.
static bool isStoreGIIDOnLoad(handshake::LoadOp loadOp,
                              handshake::StoreOp storeOp, HandshakeCFG &cfg) {
  handshake::FuncOp funcOp = loadOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "parent of load access must be handshake function");
  SmallVector<CFGPath> allPaths;
  std::optional<unsigned> loadBB = getLogicBB(loadOp);
  std::optional<unsigned> storeBB = getLogicBB(storeOp);
  assert(loadBB && storeBB && "memory accesses must belong to blocks");
  cfg.getNonCyclicPaths(*loadBB, *storeBB, allPaths);

  Value loadData = loadOp.getDataResult();
  return llvm::all_of(allPaths, [&](CFGPath &path) {
    return isGIID(loadData, storeOp->getOpOperand(0), path) ||
           isGIID(loadData, storeOp->getOpOperand(1), path);
  });
}

/// Inactivates WAR dependencies that are enforced by the circuit's data
/// ordering semantics.
static void inactivateEnforcedWARs(DenseSet<handshake::LoadOp> &loadOps,
                                   DenseSet<handshake::StoreOp> &storeOps,
                                   DependencyMap &opDeps, HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::StoreOp> storesByName;
  for (handshake::StoreOp storeOp : storeOps)
    storesByName.insert({getUniqueName(storeOp), storeOp});

  for (handshake::LoadOp loadOp : loadOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(loadOp)) {
      for (MemDependenceAttr dep : deps.getDependencies()) {
        if (!dep.getIsActive())
          continue;
        auto storeOp = storesByName.at(dep.getDstAccess());
        opDeps[loadOp].push_back(
            isStoreGIIDOnLoad(loadOp, storeOp, cfg) ? dep.asInactive() : dep);
      }
    }
  }
}

/// Inactivates WAW dependencies between a store and itself.
static void inactivateEnforcedWAWs(DenseSet<handshake::StoreOp> &storeOps,
                                   DependencyMap &opDeps) {
  for (handshake::StoreOp storeOp : storeOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(storeOp)) {
      StringRef storeName = getUniqueName(storeOp);
      for (MemDependenceAttr dep : deps.getDependencies()) {
        if (dep.getIsActive())
          continue;
        opDeps[storeOp].push_back(
            storeName == dep.getDstAccess() ? dep.asInactive() : dep);
      }
    }
  }
}

/// Replaces each op's dependency array attribute with the updated entries in
/// `opDeps`.
static void changeOpDeps(DependencyMap &opDeps, MLIRContext *ctx) {
  for (auto &[op, deps] : opDeps)
    setDialectAttr<MemDependenceArrayAttr>(op, ctx, deps);
}

void HandshakeInactivateDepsPass::analyzeFunction(handshake::FuncOp funcOp) {
  HandshakeCFG cfg(funcOp);
  DenseSet<handshake::LoadOp> loadOps;
  DenseSet<handshake::StoreOp> storeOps;
  funcOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<handshake::LoadOp>(op))
      loadOps.insert(loadOp);
    else if (auto storeOp = dyn_cast<handshake::StoreOp>(op))
      storeOps.insert(storeOp);
  });

  DependencyMap opDeps;
  inactivateEnforcedWARs(loadOps, storeOps, opDeps, cfg);
  inactivateEnforcedWAWs(storeOps, opDeps);
  changeOpDeps(opDeps, &getContext());
}

void HandshakeInactivateDepsPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();

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
