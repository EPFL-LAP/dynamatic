//===- AnalyzeMemoryAccesses.cpp - Analyze memory accesses ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --analyze-memory-accesses pass, which determines all
// dependences between all pairs of memory accesses in each function.
// Information about the dependencies is stored in a MemDependenceArrayAttr
// attribute attached to each memory operation that is the source of at least
// one dependence.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/AnalyzeMemoryAccesses.h"
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

using MemAccesses = llvm::MapVector<Value, SmallVector<Operation *>>;
using OpDependencies = DenseMap<Operation *, SmallVector<MemDependenceAttr>>;

/// Determines whether an operation is akin to a load.
static inline bool isLoadLike(Operation *op) {
  return isa<memref::LoadOp, AffineLoadOp>(op);
}

namespace {

/// Simple driver for memory analysis pass. Runs the pass on every function in
/// the module independently and succeeds whenever the transformation succeeded
/// for every function. The pass assumes that all distinctly named memrefs are
/// disjoint in memory (i.e., they do not alias each other).
struct AnalyzeMemoryAccessesPass
    : public dynamatic::impl::AnalyzeMemoryAccessesBase<
          AnalyzeMemoryAccessesPass> {

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
                                      OpDependencies &opDeps);

  /// Creates a dependency between two memory accesses, unless the accesses are
  /// the same or both load-like.
  LogicalResult checkNonAffineAccessPair(Operation *srcOp, Operation *dstOp,
                                         OpDependencies &opDeps);
};

} // namespace

LogicalResult
AnalyzeMemoryAccessesPass::analyzeMemAccesses(func::FuncOp funcOp) {
  MemAccesses affineAccesses, nonAffineAccesses;
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  // Identify all memory accesses in the function
  funcOp->walk([&](MemoryEffectOpInterface memEffectOp) {
    llvm::TypeSwitch<Operation *, void>(memEffectOp)
        .Case<memref::LoadOp, memref::StoreOp>([&](auto memrefOp) {
          auto memref = memrefOp.getMemRef();
          nonAffineAccesses[memref].push_back(memEffectOp);
        })
        .Case<AffineLoadOp, AffineStoreOp>([&](auto) {
          affine::MemRefAccess access(memEffectOp);
          affineAccesses[access.memref].push_back(memEffectOp);
        });

    // Make sure that the memory operation has a unique name, otherwise name it
    if (!namer.hasName(memEffectOp))
      namer.setName(memEffectOp);
  });

  OpDependencies opDeps;
  MLIRContext *ctx = &getContext();

  // Collect dependencies between affine accesses
  for (auto &[memref, accesses] : affineAccesses)
    for (size_t i = 0, e = accesses.size(); i < e; ++i)
      for (size_t j = 0; j < e; ++j)
        if (failed(checkAffineAccessPair(accesses[i], accesses[j], opDeps)))
          return failure();

  // Collect dependencies involving at least one non-affine access
  for (auto &[memref, accesses] : nonAffineAccesses) {
    SmallVector<Operation *> affAcc;
    if (affineAccesses.contains(memref))
      affAcc = affineAccesses[memref];

    for (size_t i = 0, e = accesses.size(); i < e; ++i) {
      // Pairs of non-affine access
      for (size_t j = 0; j < e; ++j)
        if (failed(checkNonAffineAccessPair(accesses[i], accesses[j], opDeps)))
          return failure();

      // Pairs made up of one affine access and one non-affine access
      for (Operation *otherAccess : affAcc)
        if (failed(
                checkNonAffineAccessPair(accesses[i], otherAccess, opDeps)) ||
            failed(checkNonAffineAccessPair(otherAccess, accesses[i], opDeps)))
          return failure();
    }
  }

  // Set list of dependencies for each memory operation
  for (auto &[op, deps] : opDeps)
    op->setAttr(MemDependenceArrayAttr::getMnemonic(),
                MemDependenceArrayAttr::get(ctx, deps));

  return success();
}

LogicalResult AnalyzeMemoryAccessesPass::checkAffineAccessPair(
    Operation *srcOp, Operation *dstOp, OpDependencies &opDeps) {

  // By construction we know that all operations we wish to know the name of are
  // named so we can safely discard the LogicalResult returned by `getName`
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  MLIRContext *ctx = &getContext();
  affine::MemRefAccess srcAccess(srcOp), dstAccess(dstOp);
  unsigned numCommonLoops = getNumCommonSurroundingLoops(*srcOp, *dstOp);

  for (unsigned loopDepth = 1; loopDepth <= numCommonLoops + 1; ++loopDepth) {
    FlatAffineValueConstraints constraints;
    SmallVector<DependenceComponent, 2> components;
    DependenceResult result = checkMemrefAccessDependence(
        srcAccess, dstAccess, loopDepth, &constraints, &components);
    StringRef dstName;
    (void)namer.getName(dstOp, dstName);
    if (result.value == DependenceResult::HasDependence) {
      // Add the dependence to the list of dependencies attached to the source
      // operation
      opDeps[srcOp].push_back(
          MemDependenceAttr::get(ctx, dstName, loopDepth, components));
    } else if (result.value == DependenceResult::Failure) {
      return srcOp->emitError()
             << "dependence check failed with memory access '" << dstName
             << "'";
    }
  }
  return success();
}

LogicalResult AnalyzeMemoryAccessesPass::checkNonAffineAccessPair(
    Operation *srcOp, Operation *dstOp, OpDependencies &opDeps) {
  // We don't care about self-dependencies or RAR dependencies
  if ((srcOp == dstOp) || (isLoadLike(srcOp) && isLoadLike(dstOp)))
    return success();

  // By construction we know that all operations we wish to know the name of are
  // named so we can safely discard the LogicalResult returned by `getName`
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  StringRef dstName;
  (void)namer.getName(dstOp, dstName);

  opDeps[srcOp].push_back(MemDependenceAttr::get(
      &getContext(), dstName, 0, ArrayRef<affine::DependenceComponent>{}));

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass<false>>
dynamatic::createAnalyzeMemoryAccesses() {
  return std::make_unique<AnalyzeMemoryAccessesPass>();
}
