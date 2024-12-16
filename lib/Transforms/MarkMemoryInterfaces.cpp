//===- MarkMemoryInterfaces.cpp - Mark memory interfaces --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark all memory operations with a `dynamatic::handshake::MemInterfaceAttr`,
// denoting which kind of memory interface it should eventually connect to. Uses
// results of the memory dependence analysis to make this determination.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/MarkMemoryInterfaces.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

/// If the operartions is load/store-like, returns the memref value it
/// references. Otherwise returns nullptr.
static Value getMemrefFromOp(Operation *memOp) {
  return llvm::TypeSwitch<Operation *, Value>(memOp)
      .Case<memref::LoadOp, memref::StoreOp>(
          [&](auto memrefOp) { return memrefOp.getMemRef(); })
      .Case<affine::AffineLoadOp, affine::AffineStoreOp>([&](auto) {
        affine::MemRefAccess access(memOp);
        return access.memref;
      })
      .Default([](auto) { return nullptr; });
}

namespace {

/// Maps memory operations referencing a single memory region to the memory
/// interface they should eventually connect to.
struct RegionInterfaces {
  /// Holds operations that should connect to a memory controller.
  DenseSet<Operation *> connectToMC;
  /// Mpas operations that should connect to a specific LSQ group.
  DenseMap<Operation *, unsigned> connectToLSQ;

  /// Default constructor.
  RegionInterfaces() = default;
  /// Adds the operation to the list that should connect to a memory controller,
  /// unless the operation already belongs to the group that should connect to
  /// an LSQ.
  void connectAccessToMC(Operation *memOp);
  /// Adds the operation to the list that should connect to an LSQ. The group
  /// that the operations should belong to is determined from its parent block
  /// (every operation from the same block will belong to the same LSQ group).
  /// If the operation was already part of the set that should connect to a
  /// memory controller, removes it from there.
  void connectAccessToLSQ(Operation *memOp);

private:
  /// Maps parent blocks of memory operations to a unique group ID. Determines
  /// the LSQ group memory accesses should belong to.
  DenseMap<Block *, unsigned> lsqGroups;
};

/// Maps distinct memory region to their memory interface connectivity needs.
using MemInterfaces = DenseMap<Value, RegionInterfaces>;

/// Simple driver for the memory interface marking pass. Runs for each
/// func-level function in the IR independently.
struct MarkMemoryInterfacesPass
    : public dynamatic::impl::MarkMemoryInterfacesBase<
          MarkMemoryInterfacesPass> {

  void runDynamaticPass() override {
    for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>())
      markMemoryInterfaces(funcOp);
  }

private:
  /// Annotates each memory operation in the IR with the
  /// `dynamatic::handshake::MemInterfaceAttr` attribute, denoting the kind of
  /// memory interface it should eventually connect to. The decision is based on
  /// identified memory dependencies between the memory accesses, represented
  /// using potential `dynamatic::handshake::MemDependenceArrayAttr` attributes
  /// attached to memory operations.
  void markMemoryInterfaces(func::FuncOp funcOp);
};

} // namespace

void RegionInterfaces::connectAccessToMC(Operation *memOp) {
  if (!connectToLSQ.contains(memOp))
    connectToMC.insert(memOp);
}

void RegionInterfaces::connectAccessToLSQ(Operation *memOp) {
  if (connectToMC.contains(memOp))
    connectToMC.erase(memOp);

  Block *block = memOp->getBlock();
  unsigned groupID;

  // Try to find the block's group ID. Failing that, assign a new group ID
  // to the block (the absolute ID doesn't matter, it just has to be different
  // from the others)
  if (auto groupIt = lsqGroups.find(block); groupIt != lsqGroups.end())
    groupID = groupIt->second;
  else
    lsqGroups[block] = groupID = lsqGroups.size();
  connectToLSQ[memOp] = groupID;
}

void MarkMemoryInterfacesPass::markMemoryInterfaces(func::FuncOp funcOp) {
  MemInterfaces interfaces;

  // Find all memory operations and figure out whether they should connect to an
  // MC or an LSQ (if the latter, also figure out which LSQ group)
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  getOperation()->walk([&](Operation *op) {
    Value memref = getMemrefFromOp(op);
    if (!memref)
      return;

    StringRef srcOpName = nameAnalysis.getName(op);
    bool connectToMC = true;
    if (auto allDeps = getDialectAttr<MemDependenceArrayAttr>(op)) {
      for (MemDependenceAttr memDep : allDeps.getDependencies()) {
        // Both the source and destination operation need to connect to an LSQ
        StringRef dstOpName = memDep.getDstAccess();
        if (srcOpName == dstOpName) {
          // This is necessarily a WAW dependency with two instances of the same
          // instruction since we don't record RAR dependencies. Dataflow
          // circuits guarantee that multiple "executions" of the same operation
          // are ordered, so we don't need to enforce it with an LSQ
          continue;
        }

        Operation *dstOp = nameAnalysis.getOp(dstOpName);
        assert(dstOp && "destination memory access does not exist");
        connectToMC = false;
        interfaces[memref].connectAccessToLSQ(op);
        interfaces[memref].connectAccessToLSQ(dstOp);
      }
    }
    if (connectToMC)
      interfaces[memref].connectAccessToMC(op);
  });

  // Set attributes on memory operations to instruct the tell the rest of the
  // pipeline what interface it will eventually connect to
  MLIRContext *ctx = &getContext();
  for (auto &[_, regionInterfaces] : interfaces) {
    for (Operation *mcMemOp : regionInterfaces.connectToMC)
      setDialectAttr<MemInterfaceAttr>(mcMemOp, ctx);
    for (auto &[lsqMemOp, groupID] : regionInterfaces.connectToLSQ)
      setDialectAttr<MemInterfaceAttr>(lsqMemOp, ctx, groupID);
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createMarkMemoryInterfaces() {
  return std::make_unique<MarkMemoryInterfacesPass>();
}
