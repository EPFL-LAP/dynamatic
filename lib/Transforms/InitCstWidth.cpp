//===- InitCstWidth.cpp - Reduce the constant bits width --------*- C++ -*-===//
//
// This file contains the implementation of the init-cstwidth pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/InitCstWidth.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "BITWIDTH" 

static LogicalResult initCstOpBitsWidth(handshake::FuncOp funcOp,
                                        MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<handshake::ConstantOp> cstOps;

  int savedBits = 0;

  for (auto op :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ConstantOp>())) {
    unsigned cstBitWidth = CPP_MAX_WIDTH;
    IntegerType::SignednessSemantics ifSign =
        IntegerType::SignednessSemantics::Signless;
    // skip the bool value constant operation
    if (!isa<mlir::IntegerAttr>(op.getValue()))
      continue;

    // get the attribute value
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(op.getValue())) {
      if (int cstVal = intAttr.getValue().getZExtValue(); cstVal > 0)
        cstBitWidth = log2(cstVal) + 2;
      else if (int cstVal = intAttr.getValue().getZExtValue(); cstVal < 0) {
        cstBitWidth = log2(-cstVal) + 2;
      } else
        cstBitWidth = 2;
    }

    if (cstBitWidth < op.getResult().getType().getIntOrFloatBitWidth()) {
      // Get the new type of calculated bitwidth
      Type newType = getNewType(op.getResult(), cstBitWidth, ifSign);

      // Update the constant operator for both ValueAttr and result Type
      builder.setInsertionPointAfter(op);
      handshake::ConstantOp newCstOp = builder.create<handshake::ConstantOp>(
          op.getLoc(), newType, op.getValue(), op.getCtrl());

      // Determine the proper representation of the constant value
      int intVal = op.getValue().cast<IntegerAttr>().getInt();
      intVal =
          ((1 << op.getValue().getType().getIntOrFloatBitWidth()) - 1 + intVal);
      newCstOp.setValueAttr(IntegerAttr::get(newType, intVal));
      // save the original bb
      newCstOp->setAttr(BB_ATTR, op->getAttr(BB_ATTR));

      // recursively replace the uses of the old constant operation with the new
      // one Value opVal = op.getResult();
      savedBits +=
          op.getResult().getType().getIntOrFloatBitWidth() - cstBitWidth;
      auto extOp = builder.create<mlir::arith::ExtSIOp>(
          newCstOp.getLoc(), op.getResult().getType(), newCstOp.getResult());
      auto a = op->getAttrs();

      if (failed(containsAttr(op, BB_ATTR)))
        return failure();        
      extOp->setAttr(BB_ATTR, op->getAttr(BB_ATTR));

      // replace the constant operation (default width)
      // with the extension of new constant operation (optimized width)
      op->replaceAllUsesWith(extOp);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Number of saved bits is " << savedBits << "\n");

  return success();
}

struct HandshakeInitCstWidthPass
    : public HandshakeInitCstWidthBase<HandshakeInitCstWidthPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(initCstOpBitsWidth(funcOp, ctx)))
        return signalPassFailure();
  };
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createInitCstWidthPass() {
  return std::make_unique<HandshakeInitCstWidthPass>();
}
