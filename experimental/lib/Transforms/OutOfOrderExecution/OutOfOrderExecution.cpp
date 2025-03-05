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
    Value addrInput = op.getAddressInput();
    auto channel = cast<TypedValue<handshake::ChannelType>>(addrInput);
    //handshake::ChannelType c;
    //handshake::ExtraSignal s = c.getExtraSignals().front();
    
    rewriter.setInsertionPoint(op);
    
    
    // Tag the address input of the load of
    handshake::TaggerOp taggerOp =
       rewriter.create<handshake::TaggerOp>(op.getLoc(), addrInput.getType(), addrInput, addrInput);
    
    Value data = taggerOp.getDataOut().front();
    Value tag = taggerOp.getTagOut();

    // Define the channel type with the data and the tag
    auto channelType = handshake::ChannelType::get(ctx, 
      data.getType(), 
       {/* Extra signal */ handshake::ExtraSignal("tag1", tag.getType(),true)});

    handshake::BundleOp bundleOp = 
       rewriter.create<handshake::BundleOp>(op.getLoc(), data, tag, channelType);

    auto r = bundleOp.getChannelLike();

    //handshake::LoadOpTagged

    // Create a control signal first
    //Value validSignal = ...;  // Assume this is available (i1)
    //auto bundleCtrlOp = builder.create<handshake::BundleOp>(op.getLoc(), validSignal);
    //Value ctrl = bundleCtrlOp.getResult(0);  // !handshake.control

    

    /*
  

    handshake::LoadOpTagged taggedLoad = rewriter.create<handshake::LoadOpTagged>(op.getLoc(), op.getAddressInput(), bundleOp.getChannelLike(), op.getDataInput());

    SmallVector<Type, 2> resultTypes;
    resultTypes.push_back(op.getDataOutput().getType());
    resultTypes.push_back(taggerOp.getTagOut().getType());

    handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(op.getLoc());*/
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