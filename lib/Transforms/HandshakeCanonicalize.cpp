//===-HandshakeCanonicalize.cpp - Canonicalize Handshake ops ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements rewrite patterns for the Handshake canonicalization pass, whih are
// greedily applied on the IR. These patterns do their best to attach newly
// inserted operations to known basic blocks when enough BB information is
// available. Additionally, the pass preserves the circuit's materialization
// status.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeCanonicalize.h"
#include "dynamatic/Dialect/Handshake/HandshakeCanonicalize.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include <iterator>

using namespace mlir;
using namespace dynamatic;

namespace {

/// Erases unconditional branches (which would eventually lower to simple
/// wires).
struct EraseUnconditionalBranches
    : public OpRewritePattern<handshake::BranchOp> {
  using OpRewritePattern<handshake::BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BranchOp brOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(brOp, brOp.getDataOperand());
    return success();
  }
};

/// Erases merges with a single data operand.
struct EraseSingleInputMerges : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern<handshake::MergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    if (mergeOp->getNumOperands() != 1)
      return failure();

    rewriter.replaceOp(mergeOp, mergeOp.getOperand(0));
    return success();
  }
};

/// Erases muxes with a single data operand. Inserts a sink operation to consume
/// the select operand of erased muxes.
struct EraseSingleInputMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    ValueRange dataOperands = muxOp.getDataOperands();
    if (dataOperands.size() != 1)
      return failure();

    // Insert a sink to consume the mux's select token
    rewriter.setInsertionPoint(muxOp);
    Value select = muxOp.getSelectOperand();
    rewriter.create<handshake::SinkOp>(muxOp->getLoc(), select);

    rewriter.replaceOp(muxOp, dataOperands.front());
    return success();
  }
};

/// Erases control merges with a single data operand. If necessary, inserts a
/// sourced 0 constant to replace any real uses of the index result of erased
/// control merges.
struct EraseSingleInputControlMerges
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    if (cmergeOp->getNumOperands() != 1)
      return failure();

    Value dataRes = cmergeOp.getOperand(0);
    Value indexRes = cmergeOp.getIndex();
    if (hasRealUses(indexRes)) {
      // If the index result has uses, then replace it with a sourced constant
      // with value 0 (the index of the cmerge's single input)
      rewriter.setInsertionPoint(cmergeOp);

      // Create a source operation for the constant
      handshake::SourceOp srcOp = rewriter.create<handshake::SourceOp>(
          cmergeOp->getLoc(), rewriter.getNoneType());
      inheritBB(cmergeOp, srcOp);

      /// NOTE: Sourcing this value may cause problems with very exotic uses of
      /// control merges. Ideally, we would check whether the value is sourcable
      /// first; if not we would connect the constant to the control network
      /// instead.

      // Build the attribute for the constant
      Type indexResType = indexRes.getType();
      handshake::ConstantOp cstOp = rewriter.create<handshake::ConstantOp>(
          cmergeOp.getLoc(), indexResType,
          rewriter.getIntegerAttr(indexResType, 0), srcOp.getResult());
      inheritBB(cmergeOp, cstOp);

      // Replace the cmerge's index result with a constant 0
      rewriter.replaceOp(cmergeOp, {dataRes, cstOp.getResult()});
      return success();
    }

    // Replace the cmerge's data result with its unique operand, erase any sinks
    // consuming the index result, and finally delete the cmerge
    rewriter.replaceAllUsesWith(cmergeOp.getResult(), dataRes);
    eraseSinkUsers(indexRes, rewriter);
    rewriter.eraseOp(cmergeOp);
    return success();
  }
};

/// Downgrades control merges whose index result has no real uses to simpler
/// yet equivalent merges.
struct DowngradeIndexlessControlMerge
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    Value indexRes = cmergeOp.getIndex();
    if (hasRealUses(indexRes))
      return failure();

    // Create a merge operation to replace the cmerge
    rewriter.setInsertionPoint(cmergeOp);
    handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
        cmergeOp.getLoc(), cmergeOp->getOperands());
    inheritBB(cmergeOp, mergeOp);

    // Replace the cmerge's data result with the merge's result, erase any
    // sinks consuming the index result, and finally delete the cmerge
    rewriter.replaceAllUsesWith(cmergeOp.getResult(), mergeOp.getResult());
    eraseSinkUsers(indexRes, rewriter);
    rewriter.eraseOp(cmergeOp);
    return success();
  }
};

// // Rules E
// /// Remove Conditional Branches that have no successors
// // Added an extra check here because this pass comes after materialization so
// // some sinks are inserted
// struct RemoveDoubleSinkBranches
//     : public OpRewritePattern<handshake::ConditionalBranchOp> {
//   using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condBranchOp,
//                                 PatternRewriter &rewriter) const override {
//     Value branchTrueResult = condBranchOp.getTrueResult();
//     Value branchFalseResult = condBranchOp.getFalseResult();

//     // Pattern match fails if the Branch has a true or false successor that
//     is
//     // not a sink OR if it has no successors at all
//     if (!branchTrueResult.getUsers().empty() &&
//         !isa_and_nonnull<handshake::SinkOp>(
//             condBranchOp.getTrueResult().getUsers().begin()))
//       return failure();

//     if (!branchFalseResult.getUsers().empty() &&
//         !isa_and_nonnull<handshake::SinkOp>(
//             condBranchOp.getFalseResult().getUsers().begin()))
//       return failure();

//     rewriter.eraseOp(condBranchOp);
//     // llvm::errs() << "\t***Rules E: remove-double-sink-branch!***\n";

//     return success();
//   }
// };

// /// Erases forks with a single result unless their operand originates from a
// /// lazy fork, in which case they may exist to prevent a combinational cycle.
// struct EraseSingleOutputForks : OpRewritePattern<handshake::ForkOp> {
//   using mlir::OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
//                                 PatternRewriter &rewriter) const override {
//     // The fork must have a single result
//     if (forkOp.getSize() != 1)
//       return failure();

//     // The defining operation must not be a lazy fork, otherwise the fork may
//     be
//     // here to avoid a combination cycle between the valid and ready wires
//     if (forkOp.getOperand().getDefiningOp<handshake::LazyForkOp>())
//       return failure();

//     // Bypass the fork and succeed
//     rewriter.replaceOp(forkOp, forkOp.getOperand());
//     return success();
//   }
// };

// Incomplete because of the complexity of doing it after the materialization...
// Ideally, we should do this in a separate pass from the term rewriting but
// before the materialization
// /// For any Fork having ALL of its outputs feeding the data inputs of
// Suppresses
// /// that (as an extra check) have the same exact condition or the same but
// with
// /// a NOT input, insert a Fork at the output of one of those Branches and let
// it
// /// feed all other outputs and rely on two additional patterns to delete the
// /// original fork and the old supps, respectively
// // Deal with the complexity of deciding if the output should go on the false
// or
// // the true side!!
// struct CombineBranches : public OpRewritePattern<handshake::ForkOp> {
//   using OpRewritePattern<handshake::ForkOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(handshake::ForkOp forkOp,
//                                 PatternRewriter &rewriter) const override {
//     // Pattern match fails if any of the users of the fork is not a Branch
//     DenseSet<handshake::ConditionalBranchOp> branches;
//     for (auto forkUser : forkOp->getResults().getUsers()) {
//       if (!isa_and_nonnull<handshake::ConditionalBranchOp>(forkUser))
//         return failure();
//       branches.insert(cast<handshake::ConditionalBranchOp>(forkUser));
//     }

//     // Pattern match fails if the fork is feeding a Branch condition (not
//     data) for (auto branch : branches) {
//       if (forkOp == branch.getConditionOperand().getDefiningOp())
//         return failure();
//     }

//     // All Branches must be supps, since we are after materialization, we
//     need
//     // to check for Sinks
//     for (auto branch : branches) {
//       assert(std::distance(branch.getTrueResult().getUsers().begin(),
//                            branch.getTrueResult().getUsers().end()) == 1);
//       assert(std::distance(branch.getFalseResult().getUsers().begin(),
//                            branch.getFalseResult().getUsers().end()) == 1);

//       if (!isa_and_nonnull<handshake::SinkOp>(
//               branch.getTrueResult().getUsers().begin()) ||
//           isa_and_nonnull<handshake::SinkOp>(
//               branch.getFalseResult().getUsers().begin()))
//         // if (!branch.getTrueResult().getUsers().empty() ||
//         //     branch.getFalseResult().getUsers().empty())
//         return failure();
//     }

//     // All Branches must have the same original condition
//     // Search for a Branch that does not have a NOT at its condition input
//     Value branchCondition;
//     bool foundBranchWithNoInvertedCond = false;
//     handshake::ConditionalBranchOp brToAccumTo;
//     for (auto branch : branches) {
//       if (!isa_and_nonnull<handshake::NotOp>(
//               branch.getConditionOperand().getDefiningOp()) &&
//           !isa_and_nonnull<handshake::ForkOp>(
//               branch.getConditionOperand().getDefiningOp())) {
//         branchCondition = branch.getConditionOperand();
//         foundBranchWithNoInvertedCond = true;
//         brToAccumTo = branch;
//         break;
//       }
//     }
//     assert(foundBranchWithNoInvertedCond);

//     for (auto branch : branches) {
//       Value correctConditionToCheck;
//       if ((!isa_and_nonnull<handshake::NotOp>(
//                branch.getConditionOperand().getDefiningOp()) &&
//            !isa_and_nonnull<handshake::ForkOp>(
//                branch.getConditionOperand().getDefiningOp())))
//         correctConditionToCheck = branch.getConditionOperand();
//       else
//         correctConditionToCheck =
//             branch.getConditionOperand().getDefiningOp()->getOperand(0);
//       if (correctConditionToCheck != branchCondition)
//         return failure();
//     }

//     for (auto forkRes : forkOp->getResults()) {
//       assert(std::distance(forkRes.getUsers().begin(),
//                            forkRes.getUsers().end()) == 1);
//       handshake::ConditionalBranchOp br =
//           cast<handshake::ConditionalBranchOp>(forkRes.getUsers().begin());

//       if (isa_and_nonnull<handshake::NotOp>(
//               br.getConditionOperand().getDefiningOp()) ||
//           (isa_and_nonnull<handshake::ForkOp>(
//                br.getConditionOperand().getDefiningOp()) &&
//            isa_and_nonnull<handshake::NotOp>(br.getConditionOperand()
//                                                  .getDefiningOp()
//                                                  ->getOperand(0)
//                                                  .getDefiningOp())))
//         rewriter.replaceAllUsesExcept(forkRes, brToAccumTo.getTrueResult(),
//                                       brToAccumTo);

//       else
//         rewriter.replaceAllUsesExcept(forkRes, brToAccumTo.getFalseResult(),
//                                       brToAccumTo);

//       // Note: Will call the materialize pass afterwards to add the
//       unnecessary
//       // forks at the Branches outputs
//     }

//     llvm::errs() << "\t***Combine Branches!***\n";
//     return success();
//   }
// };

/// Simple driver for the Handshake canonicalization pass, based on a greedy
/// pattern rewriter.
struct HandshakeCanonicalizePass
    : public dynamatic::impl::HandshakeCanonicalizeBase<
          HandshakeCanonicalizePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<EraseUnconditionalBranches, EraseSingleInputMerges,
                 EraseSingleInputMuxes, EraseSingleInputControlMerges,
                 DowngradeIndexlessControlMerge>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeCanonicalize() {
  return std::make_unique<HandshakeCanonicalizePass>();
}