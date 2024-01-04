//===- HandshakeMinimizeCstWidth.cpp - Min. constants bitwidth --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the handshake-minimize-cst-width pass, which minimizes the
// bitwidth of all constants. The pass matches on all Handshake constants in the
// IR, determines the minimum bitwidth necessary to hold their value, and
// updates their result/attribute type match to this minimal value.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMinimizeCstWidth.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/ConstantAnalysis.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "BITWIDTH"

STATISTIC(savedBits, "Number of saved bits");

using namespace mlir;
using namespace circt;
using namespace dynamatic;

/// Inserts an extension op after the constant op that extends the constant's
/// integer result to a provided destination type. The function assumes that it
/// makes sense to extend the former type into the latter type.
static void insertExtOp(handshake::ConstantOp toExtend,
                        handshake::ConstantOp toReplace, Type dstType,
                        PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(toExtend);
  auto extOp = rewriter.create<arith::ExtSIOp>(toExtend.getLoc(), dstType,
                                               toExtend.getResult());
  inheritBB(toExtend, extOp);
  rewriter.replaceAllUsesExcept(toReplace.getResult(), extOp.getResult(),
                                extOp);
}

unsigned dynamatic::computeRequiredBitwidth(APInt val) {
  bool isNegative = false;
  if (val.isNegative()) {
    isNegative = true;
    int64_t negVal = val.getSExtValue();
    if (negVal - 1 == 0)
      // The value is the minimum number representable on 64 bits
      return APInt::APINT_BITS_PER_WORD;

    // Flip the sign to make it positive
    val = APInt(APInt::APINT_BITS_PER_WORD, -negVal);
  }

  unsigned log = val.logBase2();
  return val.isPowerOf2() && isNegative ? log + 1 : log + 2;
}

namespace {

/// Minimizes the bitwidth used by Handshake constants as much as possible. If
/// the bitwidth is reduced, inserts an extension operation after the constant
/// so that users of the constant result can keep using the same value type.
struct MinimizeConstantBitwidth
    : public OpRewritePattern<handshake::ConstantOp> {
  using OpRewritePattern<handshake::ConstantOp>::OpRewritePattern;

  MinimizeConstantBitwidth(bool optNegatives, MLIRContext *ctx)
      : OpRewritePattern<handshake::ConstantOp>(ctx),
        optNegatives(optNegatives){};

  LogicalResult matchAndRewrite(handshake::ConstantOp cstOp,
                                PatternRewriter &rewriter) const override {
    // Only consider integer attributes
    auto intAttr = dyn_cast<mlir::IntegerAttr>(cstOp.getValue());
    if (!intAttr)
      return failure();
    IntegerType oldType = cast<IntegerType>(cstOp.getResult().getType());

    // We only support reducing signless values that fit on 64 bits or less
    APInt val = intAttr.getValue();
    if (oldType.getSignedness() != IntegerType::SignednessSemantics::Signless ||
        !val.isSingleWord())
      return failure();

    // Do not optimize negative values
    if (val.isNegative() && !optNegatives)
      return failure();

    // Check if we can reduce the bitwidth
    unsigned newBitwidth = computeRequiredBitwidth(val);
    if (newBitwidth >= oldType.getWidth())
      return failure();

    // Create the new constant attribute
    IntegerType newType = IntegerType::get(cstOp.getContext(), newBitwidth,
                                           oldType.getSignedness());
    IntegerAttr newAttr;
    if (oldType.isUnsigned())
      newAttr = IntegerAttr::get(newType, val.getZExtValue());
    else
      newAttr = IntegerAttr::get(newType, val.getSExtValue());

    // Update the constant's result and attribute type
    rewriter.updateRootInPlace(cstOp, [&] {
      cstOp.getResult().setType(newType);
      cstOp.setValueAttr(newAttr);
    });

    // Check whether bitwidth minimization created a duplicated constant
    if (auto otherCstOp = findEquivalentCst(cstOp)) {
      // Insert an extension operation to compensate for bitwidth minimization
      insertExtOp(otherCstOp, cstOp, oldType, rewriter);
      savedBits += oldType.getWidth();
      return success();
    }

    // Insert an extension operation to compensate for bitwidth minimization
    insertExtOp(cstOp, cstOp, oldType, rewriter);
    savedBits += oldType.getWidth() - newBitwidth;
    return success();
  }

private:
  /// Whether to allow optimization of negative values.
  bool optNegatives;
};

/// Erases redundant extension operations (ones that have the same operand and
/// destination type as another extension operation). This pattern explicitly
/// restricts itself to extension operations that are being fed by constants.
/// Its goal is just to make sure that this pass doesn't create extraneous
/// useless extensions.
struct EraseRedundantExtension : public OpRewritePattern<arith::ExtSIOp> {
  using OpRewritePattern<arith::ExtSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtSIOp extOp,
                                PatternRewriter &rewriter) const override {
    // Only match on extension operations that extend constants
    Value extOperand = extOp.getIn();
    Operation *defOp = extOperand.getDefiningOp();
    if (!defOp || !isa<handshake::ConstantOp>(defOp))
      return failure();

    // Get the enclosing function
    auto funcOp = extOp->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "extension operation should have parent function");

    // Try to find an equivalent extension operation
    Type extDstType = extOp.getOut().getType();
    for (auto otherExtOp : funcOp.getOps<arith::ExtSIOp>()) {
      // Don't match ourself
      if (extOp == otherExtOp)
        continue;

      if (extOperand == otherExtOp.getIn() &&
          extDstType == otherExtOp.getOut().getType()) {
        // Replace uses of the current extension with the equivalent one (same
        // operand and same result type) we found
        rewriter.replaceOp(extOp, otherExtOp.getOut());
        return success();
      }
    }

    return failure();
  }
};

/// Driver for the constant bitwidth reduction pass. A greedy pattern rewriter
/// matches on all Handshake constants and minimizes their bitwidth.
struct HandshakeMinimizeCstWidthPass
    : public dynamatic::impl::HandshakeMinimizeCstWidthBase<
          HandshakeMinimizeCstWidthPass> {

  HandshakeMinimizeCstWidthPass(bool optNegatives) {
    this->optNegatives = optNegatives;
  }

  void runDynamaticPass() override {
    auto *ctx = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<MinimizeConstantBitwidth>(optNegatives, ctx);
    patterns.add<EraseRedundantExtension>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Number of saved bits is " << savedBits << "\n");
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeCstWidth(bool optNegatives) {
  return std::make_unique<HandshakeMinimizeCstWidthPass>(optNegatives);
}
