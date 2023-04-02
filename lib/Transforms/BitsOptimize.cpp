//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinDialect.h"
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


static LogicalResult convertType(handshake::FuncOp funcOp,
                          MLIRContext *ctx) {
  OpBuilder builder(ctx);
  bool optimizeProcess = true;
  for (auto &op : funcOp.getOps()){
    llvm::errs() << op << '\n';
    if (isa<handshake::ConstantOp>(op)){
      // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html
      // first get the operands type
      const auto opName = op.getName();
      llvm::errs() << opName << '\n';
      // get the type attribute of the operators;
      SmallVector<Value> vecOperands;
      for (auto Operand : op.getOperands()){
        vecOperands.push_back(Operand);
      }

      // functions to be implemented for forward pass
      // input: opType, vecOperands
      // return newOpType: type attributes of the results
      IntegerType::SignednessSemantics ifSign;
      IntegerType newOutType;
      llvm::errs() << "Number of operators : " << op.getNumOperands() << '\n';
      for (auto optimOp : op.getOperands()){
        llvm::errs() << "Operator type " << optimOp.getType() << " ; ";
        if (auto validType = optimOp.getType() ; isa<IntegerType>(validType))
          ifSign = dyn_cast<IntegerType>(validType).getSignedness();
        newOutType = IntegerType::get(ctx, 16,ifSign);
      }
      llvm::errs() << " \n";
      newOutType = IntegerType::get(ctx, 16,ifSign);

      // change the operator types with the output from the forward function
      // https://mlir.llvm.org/doxygen/IR_2BuiltinTypes_8cpp_source.html

      // change results type
      // https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
      // optimOp.setType(newOpType)

      llvm::errs() << "Number of result Operators : " << op.getNumResults() << '\n';
      for (auto resultOp : op.getResults()){
        llvm::errs() << "Result type : " << resultOp.getType() << " \n ";
        if (!isa<IndexType>(resultOp.getType()) && isa<IntegerType>(resultOp.getType())) {
          // First change attribute value type to match the output result;
          if (auto valueAttr = op.getAttrOfType<mlir::IntegerAttr>("value")){
            auto constanVal = valueAttr.getValue().getZExtValue();
            op.setAttr("value", builder.getIntegerAttr(newOutType, constanVal));
          }
          resultOp.setType(newOutType);
          llvm::errs() << op << '\n'; // This would create a new operation, should change to conversionPattern
          llvm::errs() << " ------- UPdate operators -----------\n ";
          break; //currently only set the first result   
          // TODO : update values for all the users
        }
      }

    }
  }

  return success();
}

IntegerType getNewType(Value opType, unsigned bitswidth){                 
  IntegerType::SignednessSemantics ifSign;
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
    return 16; // for debugging, the latter expression can be hold after index is implemented
    // return std::min(cpp_max_width,
    //             std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
    //                     vecOperands[1].getType().getIntOrFloatBitWidth()));
  };
  mapOpNameWidth[StringRef("arith.subi")] = [](SmallVector<Value> vecOperands){
    return 16; 
    // return std::min(cpp_max_width,
    //             std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
    //                     vecOperands[1].getType().getIntOrFloatBitWidth()));
  };
};

static LogicalResult initCstOpBitsWidth(ArrayRef<handshake::ConstantOp> cstOps,
                         ConversionPatternRewriter &rewriter){
   
  for (auto op : cstOps){
    if (op.getValue() == nullptr)
      continue;
    unsigned cstBitWidth = cpp_max_width;
     IntegerType::SignednessSemantics ifSign = IntegerType::SignednessSemantics::Unsigned;
    // get the attribute value
    if (auto IntAttr = op.getValue().dyn_cast<mlir::IntegerAttr>()){
      if (int cstVal = IntAttr.getValue().getSExtValue() ; cstVal>0)
        cstBitWidth = log2(cstVal)+1;
      else if (int cstVal = IntAttr.getValue().getSExtValue() ; cstVal<0){
        ifSign = IntegerType::SignednessSemantics::Signed;
        cstBitWidth = log2(-cstVal)+1;
      }
      else
        cstBitWidth = 1;
    }
      
    // Get the new type of calculated bitwidth
    Value resultOp = op.getResult();
    Type newType = getNewType(resultOp, cstBitWidth, ifSign);

    // update the constant operator for both ValueAttr and result Type
    // if (op.)
    rewriter.setInsertionPointAfter(op);
    auto newResult = 
                rewriter.create<handshake::ConstantOp>(op.getLoc(), 
                                                      newType,
                                                      op.getValue(),
                                                      op.getCtrl());
      // Determine the sign of the ValAttr
    if (auto validType = op.getValue().getType() ; isa<IntegerType>(validType)){
      if (auto ifSign = dyn_cast<IntegerType>(validType).getSignedness();             ifSign==IntegerType::SignednessSemantics::Signed)
        newResult.setValueAttr(IntegerAttr::get(newType, 
                    op.getValue().cast<IntegerAttr>().getSInt()));
    } else
      newResult.setValueAttr(IntegerAttr::get(newType, 
                      op.getValue().cast<IntegerAttr>().getInt()));
    // save the original bb
    newResult->setAttr("bb", op->getAttr("bb")); 
    rewriter.replaceAllUsesWith(op.getResult(), newResult.getResult());

    // update the users of the constant operator
    // For the operation with only 1 operand, replace with the new result
    // op.getResult().replaceUsesWithIf(newResult.getResult(), 
    //                                 [&](OpOperand &operand) {
    //                                   return opek,rand.getOperandNumber()<=1;
    //                                 });
    // For the operation with more than 1 operands,  insert arith::uiext before the operation
    // rewriter.eraseOp(op);
  }
  // for (auto op : cstOps)
  //   rewriter.eraseOp(op);
    
  return success();
}

static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp,
                         ConversionPatternRewriter &rewriter) {

  using forward_func  = std::function<unsigned (SmallVector<Value> vecOperands)>;

  DenseMap<StringRef, forward_func> mapOpNameWidth;
  constrcutFuncMap(mapOpNameWidth);

  SmallVector<handshake::ConstantOp> cstOps;
  for (auto constOp : funcOp.getOps<handshake::ConstantOp>()) {
    cstOps.push_back(constOp);
  }
  // first initialize bits information we know
  initCstOpBitsWidth(cstOps, rewriter);

  // forward process
  for (auto &op : funcOp.getOps()){
    if (isa<handshake::ConstantOp>(op))
      continue;
    llvm::errs() << op << '\n';
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
      if (mapOpNameWidth.find(opName) != mapOpNameWidth.end())
        newWidth = mapOpNameWidth[opName](vecOperands);
      auto newOpType = getNewType(op.getResult(0), newWidth);
    
      // rewriter.setInsertionPointAfter(&op);
      // Operation *newOp = op.clone(); 
      // llvm::errs() << "New op : " << newOp->getName() << '\n';
      // Value newOpValue = newOp->getResult(0);
      // llvm::errs() << "New op value : " << newOpValue << "\n\n";
      // newOpValue.setType(newOpType);
      // rewriter.replaceAllUsesWith(op.getResult(0), newOp->getResult(0));
      // rewriter.eraseOp(&op);
      // llvm::errs() << "New type : " << newOpType << '\n';

      // for (auto optimOp : op.getOpResults())
      //   llvm::errs() << optimOp << "\n\n";
      // Value optimOp = op.getOpResults();
      // change results type
      // https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
      // optimOp.setType(newOpType)
      // llvm::errs() << op.getOperand(0) << '\n';
    }
    
  }
  return success();
}

namespace{
/// Custom conversion target used to mark functions dynamically legal after
/// we've applied the conversion pattern to them.
class BitsOptimForwardTarget : public ConversionTarget {
public:
  explicit BitsOptimForwardTarget(MLIRContext &context)
      : ConversionTarget(context) {
    // addLegalOp<arith::AddIOp>();
    // addLegalDialect<mlir::arith::ArithDialect>();
    // addIllegalDialect<handshake::HandshakeDialect>();
    addLegalDialect<mlir::memref::MemRefDialect>();
    addLegalOp<handshake::ConstantOp>();
    // addIllegalOp<handshake::ConstantOp>();
    // more supported Operations need to be marked as legal
  }
};


struct BitsOptimForward : public OpConversionPattern<handshake::FuncOp> {

  BitsOptimForward(BitsOptimForwardTarget &target, MLIRContext *ctx)
      : OpConversionPattern<handshake::FuncOp>(ctx), target(target) {}


  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "Attemp to debug forward\n";
    // Convert legal operator type
    LogicalResult res = failure();
    rewriter.updateRootInPlace(op,
                               [&] { res = rewriteBitsWidths(op, rewriter); });

    return res;
  }

private:

  BitsOptimForwardTarget &target;
  
};

struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    BitsOptimForwardTarget target(*ctx);
    RewritePatternSet patterns{ctx};
    patterns.add<BitsOptimForward>(target, ctx);

    ModuleOp m = getOperation();

    // Apply the patterns to the module.
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    llvm::errs() << "Attemp to debug\n";
    // for (auto funcOp : m.getOps<handshake::FuncOp>())
    //   if(failed(convertType(funcOp, ctx)))
    //     return signalPassFailure();

    llvm::errs() << "End of debug\n";

  };

};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}