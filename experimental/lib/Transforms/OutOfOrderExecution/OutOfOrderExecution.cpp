//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the out-of-order execution methodology
// https://dl.acm.org/doi/10.1145/3626202.3637556
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderExecution.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

Type channelifyType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<IndexType, IntegerType, FloatType>(
          [](auto type) { return handshake::ChannelType::get(type); })
      .Case<MemRefType>([](MemRefType memrefType) {
        if (!isa<IndexType>(memrefType.getElementType()))
          return memrefType;
        OpBuilder builder(memrefType.getContext());
        IntegerType elemType = builder.getIntegerType(32);
        return MemRefType::get(memrefType.getShape(), elemType);
      })
      .Case<handshake::ChannelType, handshake::ControlType>(
          [](auto type) { return type; })

      .Default([](auto type) { return nullptr; });
}

namespace {

static LogicalResult applyOutOfOrderExecution(handshake::FuncOp funcOp,
                                              MLIRContext *ctx) {

  ConversionPatternRewriter rewriter(ctx);
  /*for (Operation &mux : funcOp.getOps<handshake::MuxOp>()) {
    
    mux->getOperand(1).setType(channelifyType(mux->getOperand(1).getType()));
    mux->getOperand(2).setType(channelifyType(mux->getOperand(2).getType()));
    mux->getResult(0).setType(channelifyType(mux->getResult(0).getType()));
  }*/
  for (Operation &op : funcOp.getOps()) {
    // hange all the inputs into channels
    for(Value operand: op.getOperands()){
      operand.setType(channelifyType(operand.getType()));
    }

    rewriter.setInsertionPointAfterValue(op.getOperand(0));
    //auto FifoOp = rewriter.create<handshake::FreeTagsFifoOp>(op.getOperand(0).getLoc());
  }
  int i = 0;
  for (auto op : funcOp.getOps<handshake::LoadOp>()) {
    i++;
  }
  for (auto op : funcOp.getOps<handshake::MulIOp>()) {
    i++;
  }
  llvm::outs() << i;
  return success();
}

struct OutOfOrderExecutionPass
    : public dynamatic::experimental::impl::OutOfOrderExecutionBase<
          OutOfOrderExecutionPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<handshake::FuncOp>())
      if (failed(applyOutOfOrderExecution(funcOp, ctx)))
        signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}