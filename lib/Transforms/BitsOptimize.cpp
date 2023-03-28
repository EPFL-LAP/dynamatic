//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
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


/// A type converter for convert bit widths of integer type 
class WidthTypeConverter : public TypeConverter {
public:
  WidthTypeConverter(){
    /// addConversion
    /// addTargetMaterialization
  }


};

static LogicalResult convertType(handshake::FuncOp funcOp,
                          MLIRContext *ctx) {
  bool optimizeProcess = true;

  for (auto &op : funcOp.getOps()){
    llvm::errs() << op << '\n';
    
    // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html
    // first get the operands type
    const auto &opType = op.getOpOperands();
    // llvm::errs() << opType << '\n';

    // get get the type attribute of the operators;
    SmallVector<Value> vecOperands;
    for (auto Operand : op.getOperands()){
      vecOperands.push_back(Operand);
    }

    // functions to be implemented for forward pass
    // input: opType, vecOperands
    // return newOpType: type attributes of the results
    IntegerType::SignednessSemantics ifSign;
    llvm::errs() << "Number of operators : " << op.getNumOperands() << '\n';
    for (auto optimOp : op.getOperands()){
      llvm::errs() << "Operator type " << optimOp.getType() << " ; ";
      if (auto validType = optimOp.getType() ; isa<IntegerType>(validType))
        ifSign = dyn_cast<IntegerType>(validType).getSignedness();
    }
    llvm::errs() << " \n";
    auto newOutType = IntegerType::get(ctx, 16,ifSign);

    // https://mlir.llvm.org/doxygen/IR_2BuiltinTypes_8cpp_source.html



    llvm::errs() << "Number of result Operators : " << op.getNumResults() << '\n';
    for (auto resultOp : op.getResults()){
      llvm::errs() << "Result type " << resultOp.getType() << " ; ";
      if (isa<IntegerType>(resultOp.getType()))
        resultOp.setType(newOutType);
        break; //currently only set the first result
    }
      
    llvm::errs() << " \n\n ";
    // Value optimOp = op.getOpResults();
    // change results type
    // https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
    // optimOp.setType(newOpType)
    // llvm::errs() << op.getOperand(0) << '\n';
  }
  return success();
}

static LogicalResult convertType(handshake::FuncOp funcOp,
                         PatternRewriter &rewriter) {
  for (auto &op : funcOp.getOps()){
    llvm::errs() << op << '\n';
    
    // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html
    // first get the operands type
    const auto &opType = op.getOpOperands();

    // get get the type attribute of the operators;
    SmallVector<Value> vecOperands;
    for (auto Operand : op.getOperands()){
      vecOperands.push_back(Operand);
    }

    // functions to be implemented for forward pass
    // input: opType, vecOperands
    // return newOpType: type attributes of the results

    for (auto optimOp : op.getOpResults())
      llvm::errs() << optimOp << "\n\n";
    // Value optimOp = op.getOpResults();
    // change results type
    // https://mlir.llvm.org/doxygen/classmlir_1_1Value.html
    // optimOp.setType(newOpType)
    // llvm::errs() << op.getOperand(0) << '\n';
  }
  return success();
}

/// Custom conversion target used to mark functions dynamically legal after
/// we've applied the conversion pattern to them.
class BitsOptimForwardTarget : public ConversionTarget {
public:
  explicit BitsOptimForwardTarget(MLIRContext &context)
      : ConversionTarget(context) {
    addLegalOp<arith::AddIOp>();
    addLegalDialect<mlir::arith::ArithDialect>();
    addLegalDialect<handshake::HandshakeDialect>();
    addLegalOp<handshake::ConstantOp>();
    // more supported Operations need to be marked as legal
  }
};

struct BitsOptimForward : public OpConversionPattern<handshake::FuncOp> {
  // using OpConversionPattern::OpConversionPattern;

  BitsOptimForward(BitsOptimForwardTarget &target, MLIRContext *ctx)
      : OpConversionPattern<handshake::FuncOp>(ctx), target(target) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "Attemp to debug forward\n";
    // Convert legal operator type
    LogicalResult res = failure();
    rewriter.updateRootInPlace(op,
                               [&] { res = convertType(op, rewriter); });

    return res;
  }

private:
  BitsOptimForwardTarget &target;
};

namespace {
struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp m = getOperation();

    llvm::errs() << "Attemp to debug\n";
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(convertType(funcOp, ctx)))
        return signalPassFailure();

    // BitsOptimForwardTarget target(*ctx);
    // RewritePatternSet patterns{ctx};
    // patterns.add<BitsOptimForward>(target, ctx);

    llvm::errs() << "End of debug\n";

  };

};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}