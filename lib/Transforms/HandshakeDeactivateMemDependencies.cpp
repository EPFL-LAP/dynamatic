//===- HandshakeDeactivateMemDependencies.cpp - Deactivate mem deps
//-*-C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-deactivate-mem-dependencies pass, using the logic
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
#include <fstream>

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEDEACTIVATEMEMDEPENDENCIES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

using DependencyMap = DenseMap<Operation *, SmallVector<MemDependenceAttr>>;

namespace {

struct HandshakeDeactivateMemDependenciesPass
    : public dynamatic::impl::HandshakeDeactivateMemDependenciesBase<
          HandshakeDeactivateMemDependenciesPass> {

  using HandshakeDeactivateMemDependenciesBase::HandshakeDeactivateMemDependenciesBase;

  void runOnOperation() override;

  LogicalResult analyzeFunction(handshake::FuncOp funcOp);
  LogicalResult writeDependenceGraph(ModuleOp modOp);
};

} // namespace

/// Determines whether a WAR (write-after-read) dependency between a load and a
/// store can be deactivated because it is naturally enforced by the circuit.
static FailureOr<bool> isStoreGIIDOnLoad(handshake::LoadOp loadOp,
                                         handshake::StoreOp storeOp,
                                         HandshakeCFG &cfg) {
  std::optional<unsigned> loadBB = getLogicBB(loadOp);
  std::optional<unsigned> storeBB = getLogicBB(storeOp);
  if (!loadBB)
    return loadOp->emitError() << "load op must have basic block attribute";
  if (!storeBB)
    return storeOp->emitError() << "store op must have basic block attribute";

  SmallVector<CFGPath> allPaths;
  cfg.getNonCyclicPaths(*loadBB, *storeBB, allPaths);

  Value loadData = loadOp.getDataResult();
  return llvm::all_of(allPaths, [&](CFGPath &path) {
    return isGIID(loadData, storeOp->getOpOperand(0), path) ||
           isGIID(loadData, storeOp->getOpOperand(1), path);
  });
}

/// Inactivates WAR dependencies that are enforced by the circuit's data
/// ordering semantics.
static LogicalResult
deactivateEnforcedWARs(DenseSet<handshake::LoadOp> &loadOps,
                       DenseSet<handshake::StoreOp> &storeOps,
                       HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::StoreOp> storesByName;
  for (handshake::StoreOp storeOp : storeOps)
    storesByName.insert({getUniqueName(storeOp), storeOp});

  for (handshake::LoadOp loadOp : loadOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(loadOp)) {
      SmallVector<MemDependenceAttr> newDeps;
      newDeps.reserve(deps.getDependencies().size());

      for (MemDependenceAttr dep : deps.getDependencies()) {
        auto storeOp = storesByName.at(dep.getDstAccess());
        FailureOr<bool> giid = isStoreGIIDOnLoad(loadOp, storeOp, cfg);
        if (failed(giid))
          return failure();

        newDeps.push_back(*giid ? MemDependenceAttr::get(
                                      dep.getContext(), dep.getDstAccess(),
                                      dep.getLoopDepth(), dep.getDistance(),
                                      /*isActive=*/false)
                                : dep);
      }

      setDialectAttr<MemDependenceArrayAttr>(
          loadOp, MemDependenceArrayAttr::get(loadOp.getContext(), newDeps));
    }
  }
  return success();
}

/// Inactivates WAW dependencies between a store and itself.
static void deactivateEnforcedWAWs(DenseSet<handshake::StoreOp> &storeOps) {
  for (handshake::StoreOp storeOp : storeOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(storeOp)) {
      SmallVector<MemDependenceAttr> newDeps;
      newDeps.reserve(deps.getDependencies().size());
      StringRef storeName = getUniqueName(storeOp);

      for (MemDependenceAttr dep : deps.getDependencies()) {
        // set the dependency as inactive if the source and destination of the
        // dependency are the same store operation
        newDeps.push_back(
            storeName == dep.getDstAccess()
                ? MemDependenceAttr::get(dep.getContext(), dep.getDstAccess(),
                                         dep.getLoopDepth(), dep.getDistance(),
                                         /*isActive=*/false)
                : dep);
      }
      setDialectAttr<MemDependenceArrayAttr>(
          storeOp, MemDependenceArrayAttr::get(storeOp.getContext(), newDeps));
    }
  }
}

LogicalResult HandshakeDeactivateMemDependenciesPass::analyzeFunction(
    handshake::FuncOp funcOp) {
  HandshakeCFG cfg(funcOp);
  DenseSet<handshake::LoadOp> loadOps;
  DenseSet<handshake::StoreOp> storeOps;
  funcOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<handshake::LoadOp>(op))
      loadOps.insert(loadOp);
    else if (auto storeOp = dyn_cast<handshake::StoreOp>(op))
      storeOps.insert(storeOp);
  });

  if (failed(deactivateEnforcedWARs(loadOps, storeOps, cfg)))
    return failure();
  deactivateEnforcedWAWs(storeOps);
  return success();
}

/// Writes all memory dependences after their activation status has been
/// determined. Inactive dependences are retained in the graph and rendered as
/// dashed edges so that the graph represents both the original analysis result
/// and the edges that still need to be enforced.
LogicalResult HandshakeDeactivateMemDependenciesPass::writeDependenceGraph(
    ModuleOp modOp) {
  std::ofstream file(depGraphFile);
  if (!file.is_open()) {
    modOp.emitError() << "failed to open dependence graph file '"
                      << depGraphFile << "'";
    return failure();
  }

  file << "digraph G {\n";
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    funcOp.walk([&](Operation *op) {
      if (!isa<handshake::LoadOp, handshake::StoreOp>(op))
        return;

      auto deps = getDialectAttr<MemDependenceArrayAttr>(op);
      if (!deps)
        return;

      StringRef srcAccess = getUniqueName(op);
      for (MemDependenceAttr dep : deps.getDependencies()) {
        file << "  \"" << srcAccess.str() << "\" -> \"" << dep.getDstAccess().str()
             << "\" [label=\"depth=" << dep.getLoopDepth()
             << ", distance=" << dep.getDistance() << "\"";
        if (!dep.getIsActive())
          file << ", style=dashed";
        file << "];\n";
      }
    });
  }
  file << "}\n";

  if (!file.good()) {
    modOp.emitError() << "failed to write dependence graph file '"
                      << depGraphFile << "'";
    return failure();
  }
  llvm::errs() << "[INFO] Dependency graph written to " << depGraphFile
               << "\n";
  return success();
}

void HandshakeDeactivateMemDependenciesPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (isa<handshake::LoadOp, handshake::StoreOp>(&op) &&
          !namer.hasName(&op)) {
        op.emitError() << "Memory access port must be named.";
        return signalPassFailure();
      }
    }
  }

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(analyzeFunction(funcOp)))
      return signalPassFailure();
  }

  if (!depGraphFile.empty() && failed(writeDependenceGraph(modOp)))
    return signalPassFailure();
}
