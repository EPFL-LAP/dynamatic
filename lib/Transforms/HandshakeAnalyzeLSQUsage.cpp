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

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-analyze-lsq-usage"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEANALYZELSQUSAGE
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

/// Simple pass driver for the LSQ usage analysis pass. Does not modify the IR
/// beyond setting `handshake::MemInterfaceAttr` attributes on memory ports.
/// Requires that `HandshakeInactivateDeps` has already run to mark enforced
/// dependencies as inactive.
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
                     TypedValue<mlir::MemRefType> memref);
};
} // namespace

void HandshakeAnalyzeLSQUsagePass::runDynamaticPass() {
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

  // Check that all eligible operations within Handshake functions belong to a
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
    if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg))
      analyzeMemRef(funcOp, memref);
  }
}

/// Given a set of operations, returns a mapping from each operation to a boolean indicating whether it is
/// involved in at least one active dependency with another operation 
static DenseMap<Operation *, bool>
markOpsWithActiveDependencies(DenseSet<Operation *> &accessOps) {
  DenseMap<Operation *, bool> hasActiveDep;

  DenseMap<StringRef, Operation *> nameToOpMapping;
  for (Operation *op : accessOps) {
    StringRef name = getUniqueName(op);
    nameToOpMapping[name] = op;
  }

  for (Operation *op : accessOps) {
    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(op)) {
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (dependency.getIsActive()) {
          Operation *dstOp = nameToOpMapping[dependency.getDstAccess()];
          hasActiveDep[dstOp] = true;
          hasActiveDep[op] = true;
        }
      }
    }
  }
  return hasActiveDep;
}

void HandshakeAnalyzeLSQUsagePass::analyzeMemRef(
    handshake::FuncOp funcOp, TypedValue<mlir::MemRefType> memref) {
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

  DenseSet<Operation *> lsqAccessOps;
  DenseMap<Operation *, unsigned> groupMap;
  LSQPorts lsqPorts = lsqOp.getPorts();
  for (LSQGroup &group : lsqPorts.getGroups()) {
    for (MemoryPort &port : group->accessPorts) {
      groupMap.insert({port.portOp, group.groupID});
      lsqAccessOps.insert(port.portOp);
    }
  }

  DenseMap<Operation *, bool> isLSQPort = markOpsWithActiveDependencies(lsqAccessOps);

  for (Operation *accessOp : lsqAccessOps) {
    if (isLSQPort.at(accessOp))
      setDialectAttr<MemInterfaceAttr>(accessOp, ctx, groupMap.at(accessOp));
    else
      setDialectAttr<MemInterfaceAttr>(accessOp, ctx);
  }

}
