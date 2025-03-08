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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace handshake;

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

  
  for (auto op : funcOp.getOps<handshake::LoadOp>()) {
    auto startValue = (Value)funcOp.getArguments().back();

    Value addrInput = op.getAddressInput();
    auto channel = cast<TypedValue<handshake::ChannelType>>(addrInput);
    //handshake::ChannelType c;
    //handshake::ExtraSignal s = c.getExtraSignals().front();
    
    rewriter.setInsertionPoint(op);
    
    // Tag the address input of the load of
    handshake::TaggerOp taggerOp =
       rewriter.create<handshake::TaggerOp>(op.getLoc(), addrInput.getType(), addrInput, startValue);

    
    //Connect the tagger to the load
    op.getOperation()->replaceUsesOfWith(addrInput, taggerOp.getDataOut().front());

    UntaggerOp untaggerOp = rewriter.create<handshake::TaggerOp>(op.getLoc(), op.getDataResult());

    FreeTagsFifoOp fifo = rewriter.create<handshake::TaggerOp>(op.getLoc(), untaggerOp.getTagOut());

    op.getOperation()->replaceUsesOfWith(op.getDataOutput(), untaggerOp.getDataOut().front());

    taggerOp.getOperation()->replaceUsesOfWith(startValue, fifo.getTagOut());
  }
  
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