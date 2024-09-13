//===- HandshakeAnalyzeLSQUsage.cpp - LSQ flow analysis ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-analyze-lsq-usage pass, using the logic
// introduced in https://ieeexplore.ieee.org/document/8977873.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeAnalyzeLSQUsage.h"
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

#define DEBUG_TYPE "handshake-analyze-lsq-usage"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

/// Simple pass driver for the LSQ usage analysis pass. Does not modify the IR
/// beyong setting `handshake::MemInterfaceAttr` attributes on memory ports.
struct HandshakeAnalyzeLSQUsagePass
    : public dynamatic::impl::HandshakeAnalyzeLSQUsageBase<
          HandshakeAnalyzeLSQUsagePass> {

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

void HandshakeAnalyzeLSQUsagePass::runDynamaticPass() {
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
        op.emitError() << "Operation should have basic block attribute.";
        return signalPassFailure();
      }
    }
  }

  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
    analyzeFunction(funcOp);
}

void HandshakeAnalyzeLSQUsagePass::analyzeFunction(handshake::FuncOp funcOp) {
  for (BlockArgument arg : funcOp.getArguments()) {
    HandshakeCFG cfg(funcOp);
    if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg))
      analyzeMemRef(funcOp, memref, cfg);
  }
}

/// Determines whether there exists any RAW dependency between the load and the
/// stores.
static bool hasRAW(handshake::LSQLoadOp loadOp,
                   DenseSet<handshake::LSQStoreOp> &storeOps) {
  StringRef loadName = getUniqueName(loadOp);
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(storeOp)) {
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (dependency.getDstAccess() == loadName) {
          LLVM_DEBUG({
            llvm::dbgs() << "\tKeeping '" << loadName
                         << "': RAW dependency with '" << getUniqueName(storeOp)
                         << "'\n";
          });
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
                            DenseSet<handshake::LSQStoreOp> &storeOps,
                            HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::LSQStoreOp> storesByName;
  for (handshake::LSQStoreOp storeOp : storeOps)
    storesByName.insert({getUniqueName(storeOp), storeOp});

  // We only need to check stores that depend on the load (WAR dependencies) as
  // others are already provably independent. We may check a single store
  // multiple times if it depends on the load at multiple loop depths
  if (auto deps = getDialectAttr<MemDependenceArrayAttr>(loadOp)) {
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      auto storeOp = storesByName.at(dependency.getDstAccess());
      if (!isStoreGIIDOnLoad(loadOp, storeOp, cfg)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\tKeeping '" << getUniqueName(loadOp)
                       << "': non-enforced WAR with '" << getUniqueName(storeOp)
                       << "'\n";
        });
        return false;
      }
    }
  }
  return true;
}

/// Mark all accesses with the `MemInterfaceAttr`, indicating whether they
/// should connect to an MC or LSQ depending on their dependencies with other
/// accesses.
template <typename Op>
static void markLSQPorts(const DenseSet<Op> &accesses,
                         const DenseSet<Op> &dependentAccesses,
                         const DenseMap<Operation *, unsigned> &groupMap,
                         MLIRContext *ctx) {
  for (Op accessOp : accesses) {
    if (dependentAccesses.contains(accessOp))
      setDialectAttr<MemInterfaceAttr>(accessOp, ctx, groupMap.at(accessOp));
    else
      setDialectAttr<MemInterfaceAttr>(accessOp, ctx);
  }
};

void HandshakeAnalyzeLSQUsagePass::analyzeMemRef(
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
  DenseSet<handshake::LSQLoadOp> lsqLoadOps;
  DenseSet<handshake::LSQStoreOp> lsqStoreOps;
  DenseMap<Operation *, unsigned> groupMap;
  LSQPorts lsqPorts = lsqOp.getPorts();
  for (LSQGroup &group : lsqPorts.getGroups()) {
    for (MemoryPort &port : group->accessPorts) {
      groupMap.insert({port.portOp, group.groupID});
      if (auto loadOp = dyn_cast<handshake::LSQLoadOp>(port.portOp))
        lsqLoadOps.insert(loadOp);
      else
        lsqStoreOps.insert(cast<handshake::LSQStoreOp>(port.portOp));
    }
  }

  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  // Check whether we can prove independence of some loads w.r.t. other accesses
  DenseSet<handshake::LSQLoadOp> dependentLoads;
  DenseSet<handshake::LSQStoreOp> dependentStores;
  for (handshake::LSQLoadOp loadOp : lsqLoadOps) {
    // Loads with no RAW dependencies and which satisfy the GIID property with
    // all stores they have a dependency with may be removed
    if (hasRAW(loadOp, lsqStoreOps) ||
        !hasEnforcedWARs(loadOp, lsqStoreOps, cfg)) {
      dependentLoads.insert(loadOp);

      // All stores involved in a WAR with the load are still dependent
      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(loadOp)) {
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          Operation *dstOp = namer.getOp(dependency.getDstAccess());
          if (auto storeOp = dyn_cast<LSQStoreOp>(dstOp))
            dependentStores.insert(storeOp);
        }
      }
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << "\tRerouting '" << getUniqueName(loadOp) << "' to MC\n";
      });
    }
  }

  // Stores involed in a RAW ar WAW dependency with another operation are sill
  // dependent
  for (handshake::LSQStoreOp storeOp : lsqStoreOps) {
    auto deps = getDialectAttr<MemDependenceArrayAttr>(storeOp);
    if (!deps)
      continue;

    // Iterate over all RAW and WAW dependencies to determine those which must
    // still be honored by an LSQ
    StringRef storeName = getUniqueName(storeOp);
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      StringRef dstName = dependency.getDstAccess();

      // WAW dependencies on the same operation can be ignored, they are
      // enforced automatically by the dataflow circuit's construction
      if (storeName == dstName)
        continue;

      // The dependency must still be honored
      Operation *dstOp = namer.getOp(dstName);
      dependentStores.insert(storeOp);
      if (auto dstStoreOp = dyn_cast<handshake::LSQStoreOp>(dstOp))
        dependentStores.insert(dstStoreOp);
    }
  }

  LLVM_DEBUG({
    for (handshake::LSQStoreOp storeOp : lsqStoreOps) {
      if (dependentStores.contains(storeOp)) {
        llvm::dbgs() << "\tKeeping '" << getUniqueName(storeOp)
                     << "': WAW or RAW dependency with other access\n";
      } else {
        llvm::dbgs() << "\tRerouting '" << getUniqueName(storeOp)
                     << "' to MC\n";
      }
    }
  });

  // Tag LSQ access ports with the interface they should actually connect to,
  // which may be different from the one they currently connect to
  markLSQPorts(lsqLoadOps, dependentLoads, groupMap, ctx);
  markLSQPorts(lsqStoreOps, dependentStores, groupMap, ctx);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeAnalyzeLSQUsage() {
  return std::make_unique<HandshakeAnalyzeLSQUsagePass>();
}
