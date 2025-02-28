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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "BITWIDTH"

STATISTIC(savedBits, "Number of saved bits");

using namespace mlir;
using namespace dynamatic;

/// Determines whether the control value of two constants can be considered
/// equivalent.
static bool areCstCtrlEquivalent(Value ctrl, Value otherCtrl) {
  if (ctrl == otherCtrl)
    return true;

  // Both controls are equivalent if they originate from sources in the same
  // block
  Operation *defOp = ctrl.getDefiningOp();
  if (!defOp || !isa<handshake::SourceOp>(defOp))
    return false;
  Operation *otherDefOp = otherCtrl.getDefiningOp();
  if (!otherDefOp || !isa<handshake::SourceOp>(otherDefOp))
    return false;
  std::optional<unsigned> block = getLogicBB(defOp);
  std::optional<unsigned> otherBlock = getLogicBB(otherDefOp);
  return block.has_value() && otherBlock.has_value() &&
         block.value() == otherBlock.value();
}

handshake::ConstantOp static findEquivalentCst(TypedAttr valueAttr,
                                               Value ctrl) {
  auto funcOp = cast<handshake::FuncOp>(ctrl.getParentBlock()->getParentOp());
  for (auto cstOp : funcOp.getOps<handshake::ConstantOp>()) {
    // The constant operation needs to have the same value attribute and the
    // same control
    auto cstAttr = cstOp.getValue();
    if (cstAttr == valueAttr && areCstCtrlEquivalent(ctrl, cstOp.getCtrl()))
      return cstOp;
  }
  return nullptr;
}

handshake::ConstantOp
dynamatic::findEquivalentCst(handshake::ConstantOp cstOp) {
  auto cstAttr = cstOp.getValue();
  auto funcOp = cstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "constant should have parent function");

  for (auto otherCstOp : funcOp.getOps<handshake::ConstantOp>()) {
    // Don't match ourself
    if (cstOp == otherCstOp)
      continue;

    // The constant operation needs to have the same value attribute and the
    // same control
    auto otherCstAttr = otherCstOp.getValue();
    if (otherCstAttr == cstAttr &&
        areCstCtrlEquivalent(cstOp.getCtrl(), otherCstOp.getCtrl()))
      return otherCstOp;
  }

  return nullptr;
}

/// Inserts an extension op after the constant op that extends the constant's
/// integer result to a provided destination type. The function assumes that it
/// makes sense to extend the former type into the latter type.
static handshake::ExtSIOp insertExtOp(handshake::ConstantOp toExtend,
                                      handshake::ConstantOp toReplace,
                                      PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(toExtend);
  auto extOp = rewriter.create<handshake::ExtSIOp>(
      toExtend.getLoc(), toReplace.getResult().getType(), toExtend.getResult());
  inheritBB(toExtend, extOp);
  return extOp;
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
        optNegatives(optNegatives) {};

  LogicalResult matchAndRewrite(handshake::ConstantOp cstOp,
                                PatternRewriter &rewriter) const override {
    // Only consider integer attributes
    auto intAttr = dyn_cast<mlir::IntegerAttr>(cstOp.getValue());
    if (!intAttr)
      return failure();
    handshake::ChannelType channelType = cstOp.getResult().getType();
    IntegerType dataType = cast<IntegerType>(channelType.getDataType());

    // We only support reducing signless values that fit on 64 bits or less
    APInt val = intAttr.getValue();
    if (dataType.getSignedness() !=
            IntegerType::SignednessSemantics::Signless ||
        !val.isSingleWord())
      return failure();

    // Do not optimize negative values
    if (val.isNegative() && !optNegatives)
      return failure();

    // Check if we can reduce the bitwidth
    unsigned newWidth = computeRequiredBitwidth(val);
    if (newWidth >= dataType.getWidth())
      return failure();

    // Create the new constant value
    IntegerAttr newAttr = IntegerAttr::get(
        IntegerType::get(getContext(), newWidth, dataType.getSignedness()),
        val.trunc(newWidth));

    if (auto otherCstOp = findEquivalentCst(newAttr, cstOp.getCtrl())) {
      // Use the other constant's result and simply erase the matched constant
      rewriter.replaceOp(cstOp, insertExtOp(otherCstOp, cstOp, rewriter));
      return success();
    }

    // Create a new constant to replace the matched one with
    auto newCstOp = rewriter.create<handshake::ConstantOp>(
        cstOp->getLoc(), newAttr, cstOp.getCtrl());
    inheritBB(cstOp, newCstOp);
    rewriter.replaceOp(cstOp, insertExtOp(newCstOp, cstOp, rewriter));
    return success();
  }

private:
  /// Whether to allow optimization of negative values.
  bool optNegatives;
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
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();

    llvm::errs() << "Number of saved bits is " << savedBits << "\n";
    LLVM_DEBUG(llvm::dbgs() << "Number of saved bits is " << savedBits << "\n");
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeCstWidth(bool optNegatives) {
  return std::make_unique<HandshakeMinimizeCstWidthPass>(optNegatives);
}
