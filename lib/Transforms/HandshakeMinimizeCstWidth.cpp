//===- HandshakeMinimizeCstWidth.cpp - Min. constants bitwidth --*- C++ -*-===//
//
// Implements the handshake-minimize-cst-width pass, which minimizes the
// bitwidth of all constants. The pass matches on all Handshake constants in the
// IR, determines the minimum bitwidth necessary to hold their value, and
// updates their result/attribute type match to this minimal value.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMinimizeCstWidth.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "BITWIDTH"

STATISTIC(savedBits, "Number of saved bits");

/// Inserts an extension op after the constant op that extends the constant's
/// integer result to a provided destination type. The function assumes that it
/// makes sense to extend the former type into the latter type.
static void insertExtOp(handshake::ConstantOp cstOp, Type dstType,
                        PatternRewriter &rewriter) {
  // Insert an extension operation to keep the same type for users of the
  // constant's result
  rewriter.setInsertionPointAfter(cstOp);
  auto extOp = rewriter.create<arith::ExtSIOp>(cstOp.getLoc(), dstType,
                                               cstOp.getResult());
  inheritBB(cstOp, extOp);

  // Replace uses of the constant result with the extension op's result
  rewriter.replaceAllUsesExcept(cstOp.getResult(), extOp.getResult(), extOp);
}

// Checks whether the bitwidth minimization process of a constant op (with the
// resulting value attribute passed as argument) resulted in a duplicated
// constant (with same control and same value attribute). If so, erases the
// duplicated constant and makes the circuit use another equivalent constant
// instead.
static LogicalResult checkForDuplicateCst(handshake::ConstantOp cstOp,
                                          IntegerAttr cstAttr,
                                          PatternRewriter &rewriter) {
  auto funcOp = cstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "constant should have parent function");

  for (auto otherCstOp : funcOp.getOps<handshake::ConstantOp>()) {
    if (cstOp == otherCstOp)
      continue;

    // The constant operation needs to have the same value attribute and the
    // same control
    auto otherIntAttr = dyn_cast<IntegerAttr>(otherCstOp.getValue());
    if (!otherIntAttr)
      continue;
    if (otherIntAttr != cstAttr || otherCstOp.getCtrl() != cstOp.getCtrl())
      continue;

    // Try to find an extension operation whose result matches our constant's
    // original result type
    Type cstResType = cstOp.getResult().getType();
    for (auto *otherCstUser : otherCstOp->getUsers()) {
      if (auto extOp = dyn_cast<arith::ExtSIOp>(otherCstUser))
        if (cstResType == extOp.getResult().getType()) {
          rewriter.updateRootInPlace(
              cstOp, [&] { rewriter.replaceOp(cstOp, extOp.getResult()); });
          return success();
        }
    }

    // Insert an extension operation to compensate for bitwidth minimization
    insertExtOp(otherCstOp, cstResType, rewriter);
    return success();
  }
  return failure();
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

    // Check whether bitwidth minimization created a duplicated constant
    if (succeeded(checkForDuplicateCst(cstOp, newAttr, rewriter)))
      return success();

    // Update the constant's result and attribute type
    rewriter.updateRootInPlace(cstOp, [&] {
      cstOp.getResult().setType(newType);
      cstOp.setValueAttr(newAttr);
    });

    // Insert an extension operation to compensate for bitwidth minimization
    insertExtOp(cstOp, oldType, rewriter);

    // Accumulate the number of bits saved by the pass and return
    savedBits += oldType.getWidth() - newBitwidth;
    return success();
  }
};

/// Driver for the constant bitwidth reduction pass. A greedy pattern rewriter
/// matches on all Handshake constants and minimizes their bitwidth.
struct HandshakeMinimizeCstWidthPass
    : public dynamatic::impl::HandshakeMinimizeCstWidthBase<
          HandshakeMinimizeCstWidthPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<MinimizeConstantBitwidth>(ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Number of saved bits is " << savedBits << "\n");
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeMinimizeCstWidth() {
  return std::make_unique<HandshakeMinimizeCstWidthPass>();
}
