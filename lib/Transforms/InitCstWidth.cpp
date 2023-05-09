//===- InitCstWidth.cpp - Reduce the constant bits width --------*- C++ -*-===//
//
// This file contains the implementation of the init-cstwidth pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/InitCstWidth.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"

static LogicalResult initCstOpBitsWidth(handshake::FuncOp funcOp,
                        MLIRContext *ctx)  {
  OpBuilder builder(ctx);
  SmallVector<handshake::ConstantOp> cstOps;

  int savedBits = 0;

  for (auto op : llvm::make_early_inc_range(funcOp.getOps<handshake::ConstantOp>())) {
    unsigned cstBitWidth = CPP_MAX_WIDTH;
     IntegerType::SignednessSemantics ifSign = IntegerType::SignednessSemantics::Signless;
    // skip the bool value constant operation
    if (isa<BoolAttr>(op.getValue()))
      continue;

    // get the attribute value
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(op.getValue())){
      if (int cstVal = intAttr.getValue().getZExtValue() ; cstVal>0)
        cstBitWidth = log2(cstVal)+2;
      else if (int cstVal = intAttr.getValue().getZExtValue() ; cstVal<0){
        cstBitWidth = log2(-cstVal)+2;
      }
      else
        cstBitWidth = 2;
    }
      
    if (cstBitWidth < op.getResult().getType().getIntOrFloatBitWidth()) {
      // Get the new type of calculated bitwidth
      Type newType = getNewType(op.getResult(), cstBitWidth, ifSign);

      // Update the constant operator for both ValueAttr and result Type
      builder.setInsertionPointAfter(op);
      handshake::ConstantOp newCstOp = builder.create<handshake::ConstantOp>(op.getLoc(), 
                                                        newType,
                                                        op.getValue(),
                                                        op.getCtrl());

      // Determine the proper representation of the constant value
      int intVal = op.getValue().cast<IntegerAttr>().getInt();
      intVal = ((1 << op.getValue().getType().getIntOrFloatBitWidth())-1 + intVal);
      newCstOp.setValueAttr(IntegerAttr::get(newType, intVal));
      // save the original bb
      newCstOp->setAttr("bb", op->getAttr("bb")); 

      // recursively replace the uses of the old constant operation with the new one
      // Value opVal = op.getResult();
      savedBits += op.getResult().getType().getIntOrFloatBitWidth() - cstBitWidth;
      auto extOp = builder.create<mlir::arith::ExtSIOp>(newCstOp.getLoc(),
                                                        op.getResult().getType(),
                                                        newCstOp.getResult()); 

      // replace the constant operation (default width) 
      // with new constant operation (optimized width)
      op->replaceAllUsesWith(newCstOp);

      // update the user of constant operation 
      SmallVector<Operation *> userOps;
      for (auto &user : newCstOp.getResult().getUses())
          userOps.push_back(user.getOwner());

      for (auto updateOp : userOps)
        if (!isa<mlir::arith::ExtSIOp>(*updateOp))
          updateOp->replaceUsesOfWith(newCstOp.getResult(), extOp->getResult(0));
        

      builder.clearInsertionPoint();
      op->erase();
    }
    

  }
  llvm::errs() << "Constant saved bits " << savedBits << "\n";
  
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

