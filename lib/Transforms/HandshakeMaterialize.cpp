//===- HandshakeMaterialize.h - Materialize Handshake IR --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Handshake materialization pass using a mix of simple rewrite
// steps and a couple rewrite patterns applied greedily on the IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Handshake.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

using namespace circt;
using namespace mlir;
using namespace dynamatic;

LogicalResult dynamatic::verifyIRMaterialized(handshake::FuncOp funcOp) {
  auto checkUses = [&](Operation *op, Value val, StringRef desc,
                       unsigned idx) -> LogicalResult {
    auto numUses = std::distance(val.getUses().begin(), val.getUses().end());
    if (numUses == 0)
      return op->emitError() << desc << " " << idx << " has no uses.";
    if (numUses > 1)
      return op->emitError() << desc << " " << idx << " has multiple uses.";
    return success();
  };

  // Check function arguments
  for (BlockArgument funcArg : funcOp.getBody().getArguments()) {
    if (failed(checkUses(funcOp, funcArg, "function argument",
                         funcArg.getArgNumber())))
      return failure();
  }

  // Check results of operations
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    for (OpResult res : op.getResults()) {
      if (failed(checkUses(&op, res, "result", res.getResultNumber())))
        return failure();
    }
  }
  return success();
}

LogicalResult dynamatic::verifyIRMaterialized(mlir::ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(verifyIRMaterialized(funcOp)))
      return failure();
  }
  return success();
}

/// Replaces the first use of `oldVal` by `newVal` in the operation's operands.
/// Asserts if the operation's operands do not contain the old value.
static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      return;
    }
  }
  llvm_unreachable("failed to find operation operand");
}

/// Materializes a value, potentially placing an eager fork (if it has more than
/// one uses) or a sink (if it has no uses) to ensure that it is used exactly
/// once.
static void materializeValue(Value val, OpBuilder &builder) {
  if (val.use_empty()) {
    builder.setInsertionPointAfterValue(val);
    builder.create<handshake::SinkOp>(val.getLoc(), val);
    return;
  }
  if (val.hasOneUse())
    return;

  // The value has multiple uses, collect its owners
  unsigned numUses = std::distance(val.getUses().begin(), val.getUses().end());
  SmallVector<Operation *> valUsers;
  for (OpOperand &oprd : val.getUses())
    valUsers.push_back(oprd.getOwner());

  // Insert a fork with as many results as the value has uses
  builder.setInsertionPointAfterValue(val);
  auto forkOp = builder.create<handshake::ForkOp>(val.getLoc(), val, numUses);
  if (Operation *defOp = val.getDefiningOp())
    inheritBB(defOp, forkOp);

  // Replace original uses of the value with the fork's results
  for (auto [user, forkRes] : llvm::zip_equal(valUsers, forkOp->getResults()))
    replaceFirstUse(user, val, forkRes);
}

/// Transforms the potential eager fork feeding a control signal to the LSQ into
/// a lazy fork if other results of the eager fork are part of the memory
/// control network. This assumes that every value is used exactly once.
static void makeLSQControlForksLazy(handshake::LSQOp lsqOp, Value lsqCtrl,
                                    OpBuilder &builder) {
  Operation *ctrlDefOp = lsqCtrl.getDefiningOp();
  // Nothing to do if the control signal comes from a function argument since we
  // know it is used exclusively by the LSQ. If it already comes from a lazy
  // fork, assume that somebody else did the job correctly before
  if (isa_and_present<handshake::LazyForkOp>(ctrlDefOp))
    return;

  // There can only be other control paths to the same LSQ if the defining
  // operation is a fork
  auto ctrlForkOp = dyn_cast_if_present<handshake::ForkOp>(ctrlDefOp);
  if (!ctrlForkOp)
    return;

  // Find outputs of the control fork which are part of the memory control
  // network. Usually there would be at most two but the implementation is
  // general enough to support any number of control values. If there is a
  // single control path it must be the one between the fork and the LSQ control
  // value passed as argument, in which case there is no need for a lazy fork
  SmallVector<Value> controlValues = getLSQControlPaths(lsqOp, ctrlForkOp);
  assert(!controlValues.empty() && "at least one control path must exist");
  if (controlValues.size() == 1)
    return;

  // We need to ensure that the control fork is lazy if the control path
  // continues to other group allocations to the same LSQ.
  unsigned numControlPaths = controlValues.size();
  unsigned numLazyResults = numControlPaths;
  bool createEagerFork = controlValues.size() < ctrlForkOp.getSize();
  if (createEagerFork) {
    // To minimize the damage to performance, as many outputs of the control
    // fork as possible should remain "eager". We achieve this by creating an
    // eager fork after the lazy fork that handles token duplication outside the
    // memory control network. The lazy fork needs an extra output to feed the
    // eager fork
    ++numLazyResults;
  }
  builder.setInsertionPoint(ctrlForkOp);
  handshake::LazyForkOp lazyForkOp = builder.create<handshake::LazyForkOp>(
      ctrlForkOp->getLoc(), ctrlForkOp.getOperand(), numLazyResults);
  inheritBB(ctrlForkOp, lazyForkOp);

  // Replace the original fork's outputs that are part of the memory control
  // network with the first lazy fork's outputs
  ValueRange lazyForkResults = lazyForkOp->getResults();
  for (auto [oldRes, newRes] : llvm::zip(controlValues, lazyForkResults))
    oldRes.replaceAllUsesWith(newRes);

  if (createEagerFork) {
    // If some of the control fork's outputs go outside the memory control
    // network, create an eager fork fed by the lazy fork's last result
    unsigned numEagerResults = ctrlForkOp.getSize() - numControlPaths;
    handshake::ForkOp eagerForkOp = builder.create<handshake::ForkOp>(
        ctrlForkOp->getLoc(), lazyForkOp->getResults().back(), numEagerResults);
    inheritBB(ctrlForkOp, eagerForkOp);

    // Convert the list of control values to a set for easy lookup
    llvm::SmallSetVector<Value, 4> ctrlValSet(controlValues.begin(),
                                              controlValues.end());

    // Replace the control fork's outputs that do not belong to the memory
    // control network with the eager fork's results
    ValueRange eagerResults = eagerForkOp.getResult();
    auto eagerForkResIt = eagerResults.begin();
    for (OpResult res : ctrlForkOp.getResults()) {
      if (!ctrlValSet.contains(res))
        res.replaceAllUsesWith(*(eagerForkResIt++));
    }
    assert(eagerForkResIt == eagerResults.end() && "did not exhaust iterator");
  }

  // Erase the original control fork, whose results are now unused
  ctrlForkOp->erase();
}

namespace {

/// Removes outputs of forks that do not have real uses. This can result in the
/// size reduction or deletion of fork operations (the latter if none of the
/// fork results have real users) as well as sink users of fork results.
struct MinimizeForkSizes : OpRewritePattern<handshake::ForkOp> {
  using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // Compute the list of fork results that are actually used (erase any sink
    // user along the way)
    SmallVector<Value> usedForkResults;
    for (OpResult res : forkOp.getResults()) {
      if (hasRealUses(res)) {
        usedForkResults.push_back(res);
      } else if (!res.use_empty()) {
        // The value has sink users, delete them as the fork producing their
        // operand will be removed
        for (Operation *sinkUser : llvm::make_early_inc_range(res.getUsers()))
          rewriter.eraseOp(sinkUser);
      }
    }
    // Fail if all fork results are used, since it means that no transformation
    // is requires
    if (usedForkResults.size() == forkOp->getNumResults())
      return failure();

    if (!usedForkResults.empty()) {
      // Create a new fork operation
      rewriter.setInsertionPoint(forkOp);
      handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
          forkOp.getLoc(), forkOp.getOperand(), usedForkResults.size());
      inheritBB(forkOp, newForkOp);

      // Replace results with actual uses of the original fork with results from
      // the new fork
      ValueRange newResults = newForkOp.getResult();
      for (auto [oldRes, newRes] : llvm::zip(usedForkResults, newResults))
        rewriter.replaceAllUsesWith(oldRes, newRes);
    }
    rewriter.eraseOp(forkOp);
    return success();
  }
};

/// Eliminates forks feeding into other forks by replacing both with a single
/// fork operation.
struct EliminateForksToForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The defining operation must be also be a fork for the pattern to apply
    Value forkOprd = forkOp.getOperand();
    auto defForkOp = forkOprd.getDefiningOp<handshake::ForkOp>();
    if (!defForkOp)
      return failure();

    // It is important to take into account whether the matched fork's operand
    // is the single use of the defining fork's corresponding result. If it is
    // not, the new combined fork needs an extra result to replace the defining
    // fork's result with in the other uses
    bool isForkOprdSingleUse = forkOprd.hasOneUse();

    // Create a new combined fork to replace the two others
    unsigned totalNumResults = forkOp.getSize() + defForkOp.getSize();
    if (isForkOprdSingleUse)
      --totalNumResults;
    rewriter.setInsertionPoint(defForkOp);
    handshake::ForkOp newForkOp = rewriter.create<handshake::ForkOp>(
        defForkOp.getLoc(), defForkOp.getOperand(), totalNumResults);
    inheritBB(defForkOp, newForkOp);

    // Replace the defining fork's results with the first results of the new
    // fork (skipping the result feeding the matched fork if it has a single
    // use)
    ValueRange newResults = newForkOp->getResults();
    auto newResIt = newResults.begin();
    for (OpResult defForkRes : defForkOp->getResults()) {
      if (!isForkOprdSingleUse || defForkRes != forkOprd)
        rewriter.replaceAllUsesWith(defForkRes, *(newResIt++));
    }

    // Replace the results of the matched fork with the corresponding results of
    // the new defining fork
    rewriter.replaceOp(forkOp, newResults.take_back(forkOp.getSize()));
    return success();
  }
};

/// Erases forks with a single result unless their operand originates from a
/// lazy fork, in which case they may exist to prevent a combinational cycle.
struct EraseSingleOutputForks : OpRewritePattern<handshake::ForkOp> {
  using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
                                PatternRewriter &rewriter) const override {
    // The fork must have a single result
    if (forkOp.getSize() != 1)
      return failure();

    // The defining operation must not be a lazy fork, otherwise the fork may be
    // here to avoid a combination cycle between the valid and ready wires
    if (forkOp.getOperand().getDefiningOp<handshake::LazyForkOp>())
      return failure();

    // Bypass the fork and succeed
    rewriter.replaceOp(forkOp, forkOp.getOperand());
    return success();
  }
};

/// Driver for the materialization pass, materializing the IR in three
/// sequential steps.
/// 1. First, forks and sinks are inserted within Handshake functions to ensure
/// that every value is used exactly once.
/// 2. Then, unnecessary forks and sinks are erased from the IR in a greedy
/// fashion. If the input IR contained no forks or sinks, this step should do
/// nothing. If the input IR was partially materialized, this may optimize away
/// some of the forks and sinks.
/// 3. Finally, eager forks feeding group allocation signals to LSQs are turned
/// into lazy forks to ensure (together with a correct buffer placement) that
/// multiple group allocations never happen during the same cycle.
struct HandshakeMaterializePass
    : public dynamatic::impl::HandshakeMaterializeBase<
          HandshakeMaterializePass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // First make sure that every value within Handshake functions is used
    // exactly once
    OpBuilder builder(ctx);
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      for (BlockArgument funcArg : funcOp.getBody().getArguments())
        materializeValue(funcArg, builder);
      for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
        for (OpResult res : op.getResults())
          materializeValue(res, builder);
      }
    }

    // Then, greedily optimize forks
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns
        .add<MinimizeForkSizes, EliminateForksToForks, EraseSingleOutputForks>(
            ctx);
    if (failed(
            applyPatternsAndFoldGreedily(modOp, std::move(patterns), config)))
      return signalPassFailure();

    // Finally, make forks to LSQ control ports lazy
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {
        LSQPorts lsqPorts = lsqOp.getPorts();
        ValueRange lsqInputs = lsqOp.getMemOperands();
        for (LSQGroup &group : lsqPorts.getGroups()) {
          Value ctrlValue = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
          makeLSQControlForksLazy(lsqOp, ctrlValue, builder);
        }
      }
    }

    assert(succeeded(verifyIRMaterialized(modOp)) && "IR is not materialized");
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMaterialize() {
  return std::make_unique<HandshakeMaterializePass>();
}
