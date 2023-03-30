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

const unsigned defaultWidth = 32;


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

  // Backward ...
  // reverse the order of the operators
  // create a new vector to store the operators 
    // for (auto &op : llvm::reverse(llvm::make_early_inc_range(funcOp.getOps()))){
    //   llvm::errs() << op << '\n';
    // }
  // for (auto &op : llvm::make_early_inc_range(funcOp.getOps()))
  //     llvm::errs() << op << '\n';
  // auto &blocks = funcOp.getRegion().getBlocks();
  // for (auto bb=blocks.rbegin(), bbend=blocks.rend(); bb!=bbend; bb++){
  //   llvm::errs() << bb << '/n';
  // }

  return success();
}

// Compute the new width of the output
IntegerType getNewType(StringRef opName, SmallVector<Value> vecOperands,
                      // DenseMap mapOfNewWidths,
                       ConversionPatternRewriter &rewriter){
  using forward_func  = std::function<IntegerType(StringRef opName, 
                                                  SmallVector<Value> vecOperands)>;

  DenseMap<StringRef, forward_func> mapOpNameWidth;
  // determine the width of the output
  int newWidth = defaultWidth;
  // opName.data();
  if (opName.equals("arith.addi"))
    newWidth = std::min(defaultWidth,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth()));

  // determine the sign of the output                        
  IntegerType::SignednessSemantics ifSign;
  for (auto operands : vecOperands){
    if (auto validType = operands.getType() ; isa<IntegerType>(validType))
      ifSign = dyn_cast<IntegerType>(validType).getSignedness();
      if (ifSign==IntegerType::SignednessSemantics::Signed)
        break;
  }

  return IntegerType::get(rewriter.getContext(), newWidth,ifSign);
}

static void constrcutFuncMap(DenseMap<StringRef, 
                     std::function<int(SmallVector<Value> vecOperands)>> 
                     &mapOpNameWidth,
                     ConversionPatternRewriter &rewriter){

  mapOpNameWidth[StringRef("arith.addi")] = [](SmallVector<Value> vecOperands){
    return 16;
  };
};

static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp,
                         ConversionPatternRewriter &rewriter) {
  using forward_func  = std::function<int(SmallVector<Value> vecOperands)>;
  DenseMap<StringRef, forward_func> mapOpNameWidth;
  constrcutFuncMap(mapOpNameWidth, rewriter);

  for (auto &op : funcOp.getOps()){
    llvm::errs() << op << '\n';
    // https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html
    // first get the operands type
    const auto opName = op.getName().getStringRef();
    llvm::errs() << opName << '\n';

    // get get the type attribute of the operators;
    SmallVector<Value> vecOperands;
    for (auto Operand : op.getOperands()){
      vecOperands.push_back(Operand);
    }

    auto newOpType = getNewType(opName, vecOperands, rewriter);
    llvm::errs() << "New type : " << newOpType << '\n';

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
    // addLegalOp<arith::AddIOp>();
    addLegalDialect<mlir::arith::ArithDialect>();
    // addLegalDialect<handshake::HandshakeDialect>();
    addLegalOp<handshake::ConstantOp>();
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

namespace {
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