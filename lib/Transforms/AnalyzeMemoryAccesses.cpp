//===- AnalyzeMemoryAccesses.cpp - Analyze memory accesses ------*- C++ -*-===//
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
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dynamatic;
using namespace circt;
using namespace circt::handshake;

using MemAccesses = llvm::MapVector<Value, SmallVector<Operation *>>;
using OpDependencies = DenseMap<Operation *, SmallVector<MemDependenceAttr>>;

// Determines whether there are dependences between the source and destination
// affine memory accesses at all loop depth that make sense. If such
// dependencies exist, add them to source operation's list of dependencies.
// Fails if a dependence check between the two operations fails; succeeds
// otherwise.
static LogicalResult checkAffineAccessPair(Operation *srcOp, Operation *dstOp,
                                           OpBuilder &builder,
                                           OpDependencies &opDeps) {

  MemRefAccess srcAccess(srcOp), dstAccess(dstOp);
  MLIRContext *ctx = builder.getContext();
  unsigned numCommonLoops = getNumCommonSurroundingLoops(*srcOp, *dstOp);
  for (unsigned loopDepth = 1; loopDepth <= numCommonLoops + 1; ++loopDepth) {
    FlatAffineValueConstraints constraints;
    SmallVector<DependenceComponent, 2> components;
    DependenceResult result = checkMemrefAccessDependence(
        srcAccess, dstAccess, loopDepth, &constraints, &components);
    if (result.value == DependenceResult::HasDependence) {
      // Add the dependence to the list of dependencies attached to the
      // source operation
      auto toAccessName = dstOp->getAttrOfType<MemAccessNameAttr>(
          MemAccessNameAttr::getMnemonic());
      assert(toAccessName && "dstOp must have access name");
      opDeps[srcOp].push_back(
          MemDependenceAttr::get(ctx, toAccessName, loopDepth, components));
    } else if (result.value == DependenceResult::Failure) {
      auto toAccessName = dstOp->getAttrOfType<MemAccessNameAttr>(
          MemAccessNameAttr::getMnemonic());
      assert(toAccessName && "dstOp must have access name");
      return srcOp->emitError()
             << "dependence check failed with memory access '"
             << toAccessName.getName() << "'";
    }
  }
  return success();
}

/// Analyzes memory accesses in a function and identify all dependencies between
/// them. Sets a MemDependenceArrayAttr attribute on each operation that is the
/// source of at least one dependence. Fails if a memory operation doesn't have
/// a MemAccessNameAttr attribute or if a dependence check fails; succeeds
/// otherwise.
static LogicalResult analyzeMemAccesses(func::FuncOp funcOp, MLIRContext *ctx) {

  MemAccesses affineAccesses, memrefAccesses;
  LogicalResult allMemOpsValid = success();
  auto accessNameAttr = MemAccessNameAttr::getMnemonic();

  // Identify all memory accesses in the function
  funcOp->walk([&](Operation *op) {
    auto isMemOp =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case<memref::LoadOp, memref::StoreOp>([&](auto memrefOp) {
              auto memref = memrefOp.getMemRef();
              memrefAccesses[memref].push_back(op);
              return true;
            })
            .Case<AffineLoadOp, AffineStoreOp>([&](auto) {
              MemRefAccess access(op);
              affineAccesses[access.memref].push_back(op);
              return true;
            })
            .Default([&](auto) { return false; });

    // Check that every memory operation has an access name attribute, else emit
    // an operation error and fail the pass later
    if (isMemOp)
      if (auto attr = op->getAttrOfType<MemAccessNameAttr>(accessNameAttr);
          !attr)
        allMemOpsValid = op->emitError() << "memory op doesn't have '"
                                         << accessNameAttr << "' attribute";
  });

  if (failed(allMemOpsValid))
    return funcOp->emitError()
           << "some memory ops in the function do not have an '"
           << accessNameAttr
           << "' attribute, "
              "make sure to run the --name-memory-ops pass "
              "before this one to give a "
              "unique name to each memory operation";

  OpBuilder builder(ctx);
  OpDependencies opDeps;

  // Collect dependencies between affine accesses
  for (auto &[memref, accesses] : affineAccesses)
    for (unsigned i = 0, e = accesses.size(); i < e; ++i)
      for (unsigned j = 0; j < e; ++j)
        if (failed(checkAffineAccessPair(accesses[i], accesses[j], builder,
                                         opDeps)))
          return failure();

  // Set list of dependencies for each memory operation
  for (auto &[op, deps] : opDeps)
    op->setAttr(MemDependenceArrayAttr::getMnemonic(),
                MemDependenceArrayAttr::get(ctx, deps));

  return success();
}

namespace {

/// Simple driver for memory analysis pass. Runs the pass on every function in
/// the module independently and succeeds whenever the transformation succeeded
/// for every function. The pass assumes that all distinctly named memrefs are
/// disjoint in memory (i.e., they do not alias each other).
struct AnalyzeMemoryAccessesPass
    : public AnalyzeMemoryAccessesBase<AnalyzeMemoryAccessesPass> {

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<func::FuncOp>())
      if (failed(analyzeMemAccesses(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createAnalyzeMemoryAccesses() {
  return std::make_unique<AnalyzeMemoryAccessesPass>();
}
