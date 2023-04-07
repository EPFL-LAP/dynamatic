//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/BuiltinDialect.h"
// #include "mlir/IR//BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

#include "mlir/IR/MLIRContext.h"

// #include "mlir/IR/BuiltinAttributes.h"

// #include "TypeDetail.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/IndentedOstream.h"


using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace mlir::detail;
using namespace dynamatic;

const unsigned cpp_max_width = 32;

IntegerType getNewType(Value opType, unsigned bitswidth, bool signless=false){                 
  IntegerType::SignednessSemantics ifSign = 
  IntegerType::SignednessSemantics::Signless;
  if (!signless)
    if (auto validType = opType.getType() ; isa<IntegerType>(validType))
      ifSign = dyn_cast<IntegerType>(validType).getSignedness();

  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

IntegerType getNewType(Value opType, unsigned bitswidth,  
                      IntegerType::SignednessSemantics ifSign){
  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

static void constrcutFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (SmallVector<Value> vecOperands)>> 
                     &mapOpNameWidth){

  mapOpNameWidth[StringRef("arith.addi")] = [](SmallVector<Value> vecOperands){
    return std::min(cpp_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth())+1);
  };
  mapOpNameWidth[StringRef("arith.subi")] = [](SmallVector<Value> vecOperands){
    return std::min(cpp_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth())+1);
  };
};
static void updateUserType(Operation *newResult, Type newType,
                          MLIRContext *ctx){
  
        llvm::errs() << "----------recursive--------------"  << '\n';
  // registerDialect<MLIRStandardOpDialect>();
  OpBuilder builder(ctx);
  llvm::errs() << "New Operation : " << newResult->getName() << " \n";

  bool success = true;
  if (isa<handshake::BranchOp>(newResult)){
    if(handshake::BranchOp brOp = dyn_cast<handshake::BranchOp>(newResult)){
      if (brOp.getResult().getType() == newType){
        return;
      } else {
        brOp.getResult().setType(newType);
        for(auto &user : brOp.getResult().getUses()) 
          updateUserType(user.getOwner(), newType, ctx);
      }
    }
  } 
  else if (isa<handshake::MuxOp>(newResult)){
    if (handshake::MuxOp muxOp = dyn_cast<handshake::MuxOp>(newResult))
      // add the extension operation
      // start from the second operand (i=1), as the first one is the select index
      for (int i = 1; i < muxOp.getNumOperands(); ++i){
        auto Operand = muxOp.getOperand(i);
        if (Operand.getType() != muxOp.getResult().getType()){
          builder.setInsertionPoint(newResult);
          auto extOp = builder.create<mlir::arith::ExtSIOp>(newResult->getLoc(),
                                              muxOp.getResult().getType(),
                                              Operand); 
          muxOp.setOperand(i, extOp.getResult());
        }
      }
  }
  else if (isa<handshake::DynamaticStoreOp>(newResult)){
    SmallVector<Type> inferredReturnTypes;
    if (handshake::DynamaticStoreOp dstoreOp = dyn_cast<handshake::DynamaticStoreOp>(newResult)){
      for (int i = 0; i < dstoreOp.getNumOperands(); ++i){
        auto Operand = dstoreOp.getOperand(i);
        if (Operand.getType() != dstoreOp.getResult(i).getType()){
      // llvm::errs() << "DynamaticStoreOp : " << dstoreOp.getResult(i) << " \n";

          builder.setInsertionPoint(newResult);
          auto extOp = builder.create<mlir::arith::ExtSIOp>(newResult->getLoc(),
                                              dstoreOp.getResult(i).getType(),
                                              Operand); 
          dstoreOp.setOperand(i, extOp.getResult());
          inferredReturnTypes.push_back(extOp.getResult().getType());
          

          for(auto &user : dstoreOp.getResult(i).getUses()) 
            updateUserType(user.getOwner(), newType, ctx);
      llvm::errs() << "DynamaticStoreOp result: " << dstoreOp.getResult(i) << " \n";

        }

      }
      dstoreOp.inferReturnTypes(ctx, dstoreOp.getLoc(), 
                                    dstoreOp.getOperands(),
                                    newResult->getAttrDictionary(), newResult->getRegions(),
                                    inferredReturnTypes);}
  }
  else if (isa<handshake::DynamaticReturnOp>(newResult)){
    if (handshake::DynamaticReturnOp dreturnOp = dyn_cast<handshake::DynamaticReturnOp>(newResult)){
      for (int i=0; i<dreturnOp.getNumOperands(); ++i){
        auto Operand = dreturnOp.getOperand(i);
        if (Operand.getType() != dreturnOp.getResult(i).getType()){
          builder.setInsertionPoint(newResult);
          auto extOp = builder.create<mlir::arith::ExtSIOp>(newResult->getLoc(),
                                              dreturnOp.getResult(i).getType(),
                                              Operand); 
          dreturnOp.setOperand(i, extOp.getResult());
        }
      }
      llvm::errs() << "DynamaticReturnOp : " << dreturnOp << " \n";
    }
  }
}


static LogicalResult initIndexType(handshake::FuncOp funcOp, MLIRContext *ctx){
  OpBuilder builder(ctx);
  for (Operation &op : funcOp.getOps()){
    for (int i=0; i<op.getNumResults(); ++i){

      auto result = op.getResult(i);

      if (isa<IndexType>(result.getType())){
        unsigned indexWidth = IndexType::kInternalStorageBitWidth;
        llvm::errs() << "Current index width : " << indexWidth << " \n";
        result.setType(IntegerType::get(ctx, indexWidth));
        // For constant operation, change the value attribute to match the new type
        if (isa<handshake::ConstantOp>(op)){
          handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(op);
          cstOp.setValueAttr(IntegerAttr::get(IntegerType::get(ctx, indexWidth), 
                      cstOp.getValue().cast<IntegerAttr>().getInt()));
        }
        
        builder.setInsertionPoint(&op);
        Value newVal = builder.clone(op)->getResult(i);
        newVal.setType(IntegerType::get(ctx, indexWidth));
        result.replaceAllUsesWith(newVal);
        // op.erase();
      }
    }
    
  }
  return success();
}

static LogicalResult initCstOpBitsWidth(ArrayRef<handshake::ConstantOp> cstOps,
                        //  ConversionPatternRewriter &rewriter){
                        MLIRContext *ctx)  {
  OpBuilder builder(ctx);
  for (auto op : cstOps){

    unsigned cstBitWidth = cpp_max_width;
     IntegerType::SignednessSemantics ifSign = IntegerType::SignednessSemantics::Signless;
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
    intVal = ((1 << op.getValue().getType().getIntOrFloatBitWidth()-1) + intVal);
    newResult.setValueAttr(IntegerAttr::get(newType, intVal));

    // save the original bb
    newResult->setAttr("bb", op->getAttr("bb")); 

    // recursively replace the uses of the old constant operation with the new one
    op->replaceAllUsesWith(newResult);
    for (auto &user : newResult.getResult().getUses()){
      auto userOp = user.getOwner();
      llvm::errs() << "User Op : " << userOp->getName() << '\n';
      updateUserType(userOp, newType, ctx);
    }

    op->erase();
    llvm::errs() << "-----------------------------" << "\n\n";
  }
    
  return success();
}

static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);

  using forward_func  = std::function<unsigned (SmallVector<Value> vecOperands)>;

    DenseMap<StringRef, forward_func> mapOpNameWidth;
    constrcutFuncMap(mapOpNameWidth);

    SmallVector<handshake::ConstantOp> cstOps;
    for (auto constOp : funcOp.getOps<handshake::ConstantOp>()) {
      cstOps.push_back(constOp);
    }
    // adapt the Index type to the Integer type
    if (failed(initIndexType(funcOp, ctx)))
      return failure();
    // initialize bits information we know
    return initCstOpBitsWidth(cstOps, ctx);
    if (failed(initCstOpBitsWidth(cstOps, ctx)))
      return failure();

    // bool changed = true;
    // while (changed)
    // changed = false;

    // Forward process
    for (auto &op : funcOp.getOps()){
      llvm::errs() << "op : " << op << '\n';
      if (isa<handshake::ConstantOp>(op))
      continue;
      // get the name of the operator
      const auto opName = op.getName().getStringRef();

      // get the type attribute of the operators;
      SmallVector<Value> vecOperands;
      for (auto Operand : op.getOperands()){
        vecOperands.push_back(Operand);
      }

      // functions implemented for forward pass
      // input: opType, vecOperands
      // return newOpType: type attributes of the results
      if (0 < op.getNumResults()){
        int newWidth = 32;
        if (mapOpNameWidth.find(opName) != mapOpNameWidth.end()){
          // get the new bit width of the result operator
          newWidth = mapOpNameWidth[opName](vecOperands);
          llvm::errs() << "change forward resultOp Type\n";
          // if the new type can be optimized, update the type
          if(Type newOpResultType = getNewType(op.getResult(0), newWidth, true);  
              newOpResultType != op.getResult(0).getType()){
                // changed |= true;
                op.getResult(0).setType(newOpResultType);
                updateUserType(&op, newOpResultType, ctx);
              }

        }
      }
    }

    // Backward Process
    
    
    return success();
}

struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    // BitsOptimForwardTarget target(*ctx);
    RewritePatternSet patterns{ctx};
    // patterns.add<BitsOptimForward>(target, ctx);

    ModuleOp m = getOperation();

    // Apply the patterns to the module.
    // if (failed(applyPartialConversion(m, target, std::move(patterns))))
    //   signalPassFailure();

    llvm::errs() << "Attemp to debug\n";
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(rewriteBitsWidths(funcOp, ctx)))
        return signalPassFailure();

    llvm::errs() << "End of debug\n";

  };

};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}