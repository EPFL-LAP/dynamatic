//===- MarkMemoryDependencies.cpp - Mark mem. deps. in the IR ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --mark-memory-dependencies pass, which identifies all
// dependences between all pairs of memory accesses in each function. All
// information regarding the dependencies is stored in a
// `circt::handshake::MemDependenceArrayAttr` attribute attached to each memory
// operation that is the source of at least one dependency.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/MarkMemoryDependencies.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::affine;
using namespace dynamatic;
using namespace circt;
using namespace circt::handshake;

/// Maps values representing memory regions to load/store operations to it
using MemAccesses = llvm::MapVector<Value, SmallVector<Operation *>>;

/// Maps memory operations their identified memory dependencies with other
/// memory operations.
using MemDependencies = DenseMap<Operation *, SmallVector<MemDependenceAttr>>;

/// Determines whether an operation is akin to a load.
static inline bool isLoadLike(Operation *op) {
  return isa<memref::LoadOp, AffineLoadOp>(op);
}

namespace {

/// Simple driver for memory analysis pass. Runs the pass on every function in
/// the module independently and succeeds whenever the transformation succeeded
/// for every function. The pass assumes that all distinctly named memrefs are
/// disjoint in memory (i.e., they do not alias each other).
struct MarkMemoryDependenciesPass
    : public dynamatic::impl::MarkMemoryDependenciesBase<
          MarkMemoryDependenciesPass> {

  void runDynamaticPass() override {
    for (func::FuncOp funcOp : getOperation().getOps<func::FuncOp>())
      if (failed(analyzeMemAccesses(funcOp)))
        return signalPassFailure();
  }

private:
  /// Analyzes memory accesses in a function and identify all dependencies
  /// between them. Sets a MemDependenceArrayAttr attribute on each operation
  /// that is the source of at least one dependence. Fails if a dependence check
  /// fails; succeeds otherwise.
  LogicalResult analyzeMemAccesses(func::FuncOp funcOp);

  /// Determines whether there are dependences between the source and
  /// destination affine memory accesses at all loop depth that make sense. If
  /// such dependencies exist, add them to source operation's list of
  /// dependencies. Fails if a dependence check between the two operations
  /// fails; succeeds otherwise.
  LogicalResult checkAffineAccessPair(Operation *srcOp, Operation *dstOp,
                                      MemDependencies &opDeps);

  /// Creates a dependency between two memory accesses, unless the accesses are
  /// the same write and outside of all loops or both load-like.
  LogicalResult checkNonAffineAccessPair(Operation *srcOp, Operation *dstOp,
                                         MemDependencies &opDeps);
};

} // namespace

LogicalResult
MarkMemoryDependenciesPass::analyzeMemAccesses(func::FuncOp funcOp) {
  MemAccesses affineAccesses, nonAffineAccesses;

  // Identify all memory accesses in the function
  funcOp->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<memref::LoadOp, memref::StoreOp>([&](auto memrefOp) {
          auto memref = memrefOp.getMemRef();
          nonAffineAccesses[memref].push_back(op);
        })
        .Case<AffineLoadOp, AffineStoreOp>([&](auto) {
          affine::MemRefAccess access(op);
          affineAccesses[access.memref].push_back(op);
        });
  });

  MemDependencies opDeps;

  // Collect dependencies between affine accesses
  for (auto &[memref, accesses] : affineAccesses) {
    for (size_t i = 0, e = accesses.size(); i < e; ++i) {
      for (size_t j = 0; j < e; ++j) {
        if (failed(checkAffineAccessPair(accesses[i], accesses[j], opDeps)))
          return failure();
      }
    }
  }

  // Collect dependencies involving at least one non-affine access
  for (auto &[memref, accesses] : nonAffineAccesses) {
    // Pairs made up of two non-affine accesses
    for (Operation *nonAffineAccess : accesses) {
      for (Operation *otherNonAffineAccess : accesses) {
        if (failed(checkNonAffineAccessPair(nonAffineAccess,
                                            otherNonAffineAccess, opDeps)))
          return failure();
      }

      // Determine whether there are affine accesses to the same memory region
      if (!affineAccesses.contains(memref))
        continue;

      // Pairs made up of one affine access and one non-affine access
      for (Operation *affineAccess : affineAccesses[memref])
        if (failed(checkNonAffineAccessPair(nonAffineAccess, affineAccess,
                                            opDeps)) ||
            failed(checkNonAffineAccessPair(affineAccess, nonAffineAccess,
                                            opDeps)))
          return failure();
    }
  }

  // For each memory operation with dependencies to other memory operations, set
  // the MemDependenceArrayAttr attribute on the operation
  MLIRContext *ctx = &getContext();
  StringRef mnemonic = MemDependenceArrayAttr::getMnemonic();
  for (auto &[op, deps] : opDeps)
    op->setAttr(mnemonic, MemDependenceArrayAttr::get(ctx, deps));
  return success();
}

LogicalResult MarkMemoryDependenciesPass::checkAffineAccessPair(
    Operation *srcOp, Operation *dstOp, MemDependencies &opDeps) {
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  MLIRContext *ctx = &getContext();
  affine::MemRefAccess srcAccess(srcOp), dstAccess(dstOp);
  unsigned numCommonLoops = getNumCommonSurroundingLoops(*srcOp, *dstOp);

  for (unsigned loopDepth = 1; loopDepth <= numCommonLoops + 1; ++loopDepth) {
    SmallVector<DependenceComponent, 2> components;
    DependenceResult result = checkMemrefAccessDependence(
        srcAccess, dstAccess, loopDepth, nullptr, &components);
    StringRef dstName = namer.getName(dstOp);
    if (result.value == DependenceResult::HasDependence) {
      // Add the dependence to the source operation
      opDeps[srcOp].push_back(
          MemDependenceAttr::get(ctx, dstName, loopDepth, components));
    } else if (result.value == DependenceResult::Failure) {
      return srcOp->emitError()
             << "Dependence check failed with memory access '" << dstName
             << "'.";
    }
  }
  return success();
}

LogicalResult MarkMemoryDependenciesPass::checkNonAffineAccessPair(
    Operation *srcOp, Operation *dstOp, MemDependencies &opDeps) {
  // We don't care about RAR dependencies
  if (isLoadLike(srcOp) && isLoadLike(dstOp))
    return success();
  // A write outside of all loops cannot depend on itself
  if (srcOp == dstOp && getNumCommonSurroundingLoops(*srcOp, *dstOp) == 0)
    return failure();

  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  StringRef dstName = namer.getName(dstOp);
  opDeps[srcOp].push_back(MemDependenceAttr::get(
      &getContext(), dstName, 0, ArrayRef<affine::DependenceComponent>{}));

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createMarkMemoryDependencies() {
  return std::make_unique<MarkMemoryDependenciesPass>();
}
