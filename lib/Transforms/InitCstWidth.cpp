//===- InitCstWidth.cpp - Reduce the constant bits width -------*- C++ -*-===//
//
// This file contains the implementation of the init-cstwidth pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/InitCstWidth.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "dynamatic/Transforms/ForwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/IndentedOstream.h"

static LogicalResult initCstOpBitsWidth(handshake::FuncOp funcOp,
                        //  ConversionPatternRewriter &rewriter){
                        MLIRContext *ctx)  {
  OpBuilder builder(ctx);
  SmallVector<handshake::ConstantOp> cstOps;

  for (auto constOp : funcOp.getOps<handshake::ConstantOp>()) {
    cstOps.push_back(constOp);
  }

  for (auto op : cstOps){
    unsigned cstBitWidth = cpp_max_width;
     IntegerType::SignednessSemantics ifSign = IntegerType::SignednessSemantics::Signless;
    // skip the bool value constant operation
    if (auto ValAttr = op.getValue(); isa<BoolAttr>(ValAttr))
      continue;

    // get the attribute value
    if (auto IntAttr = op.getValue().dyn_cast<mlir::IntegerAttr>()){
      if (int cstVal = IntAttr.getValue().getZExtValue() ; cstVal>0)
        cstBitWidth = log2(cstVal)+2;
      else if (int cstVal = IntAttr.getValue().getZExtValue() ; cstVal<0){
        cstBitWidth = log2(-cstVal)+2;
      }
      else
        cstBitWidth = 2;
    }
      
    // Get the new type of calculated bitwidth
    Type newType = getNewType(op.getResult(), cstBitWidth, ifSign);

    // Update the constant operator for both ValueAttr and result Type
    builder.setInsertionPointAfter(op);
    auto newResult = builder.create<handshake::ConstantOp>(op.getLoc(), 
                                                      newType,
                                                      op.getValue(),
                                                      op.getCtrl());

    // Determine the proper representation of the constant value
    int intVal = op.getValue().cast<IntegerAttr>().getInt();
    intVal = ((1 << op.getValue().getType().getIntOrFloatBitWidth())-1 + intVal);
    newResult.setValueAttr(IntegerAttr::get(newType, intVal));
    // save the original bb
    newResult->setAttr("bb", op->getAttr("bb")); 

    // recursively replace the uses of the old constant operation with the new one
    op->replaceAllUsesWith(newResult);

    SmallVector<Operation *> userOps;
    for (auto &user : newResult.getResult().getUses())
      userOps.push_back(user.getOwner());

    SmallVector<Operation *> vecOp;
    for (auto updateOp : userOps){
      vecOp.insert(vecOp.end(), newResult);
      forward::updateUserType(updateOp, newType, vecOp, ctx);
      vecOp.clear();
    }
      // llvm::errs() <<"\n\n";

    op->erase();
  }
  
  return success();
}

struct HandshakeInitCstWidthPass
    : public HandshakeInitCstWidthBase<HandshakeInitCstWidthPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(initCstOpBitsWidth(funcOp, ctx)))
        return signalPassFailure();
  };
  
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createInitCstWidthPass() {
  return std::make_unique<HandshakeInitCstWidthPass>();
}

