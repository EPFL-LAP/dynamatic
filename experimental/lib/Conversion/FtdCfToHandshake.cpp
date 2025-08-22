//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the fast token delivery methodology
// https://ieeexplore.ieee.org/abstract/document/10035134, together with the
// straight LSQ allocation https://dl.acm.org/doi/abs/10.1145/3543622.3573050.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<ControlDependenceAnalysis>(),
        getAnalysis<gsa::GSAAnalysis>(), getAnalysis<NameAnalysis>(), converter,
        ctx);

    patterns
        .add<ConvertCalls,
             FtdConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
             FtdConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
             FtdOneToOneConversion<arith::AddFOp, handshake::AddFOp>,
             FtdOneToOneConversion<arith::AddIOp, handshake::AddIOp>,
             FtdOneToOneConversion<arith::AndIOp, handshake::AndIOp>,
             FtdOneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
             FtdOneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
             FtdOneToOneConversion<arith::DivFOp, handshake::DivFOp>,
             FtdOneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
             FtdOneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
             FtdOneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
             FtdOneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
             FtdOneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
             FtdOneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
             FtdOneToOneConversion<arith::MulFOp, handshake::MulFOp>,
             FtdOneToOneConversion<arith::MulIOp, handshake::MulIOp>,
             FtdOneToOneConversion<arith::NegFOp, handshake::NegFOp>,
             FtdOneToOneConversion<arith::OrIOp, handshake::OrIOp>,
             FtdOneToOneConversion<arith::SelectOp, handshake::SelectOp>,
             FtdOneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
             FtdOneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
             FtdOneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
             FtdOneToOneConversion<arith::SubFOp, handshake::SubFOp>,
             FtdOneToOneConversion<arith::SubIOp, handshake::SubIOp>,
             FtdOneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
             FtdOneToOneConversion<arith::TruncFOp, handshake::TruncFOp>,
             FtdOneToOneConversion<arith::XOrIOp, handshake::XOrIOp>,
             FtdOneToOneConversion<arith::SIToFPOp, handshake::SIToFPOp>,
             FtdOneToOneConversion<arith::FPToSIOp, handshake::FPToSIOp>,
             FtdOneToOneConversion<arith::ExtFOp, handshake::ExtFOp>,
             FtdOneToOneConversion<math::AbsFOp, handshake::AbsFOp>>(
            getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

static void channelifyMuxes(handshake::FuncOp &funcOp) {
  // Considering each mux that was added, the inputs and output values must be
  // channellified
  for (handshake::MuxOp muxOp : funcOp.getOps<handshake::MuxOp>()) {
    assert(muxOp.getDataOperands().size() == 2 &&
           "Multiplexers should have two data inputs");
    muxOp.getDataOperands()[0].setType(
        channelifyType(muxOp.getDataOperands()[0].getType()));
    muxOp.getDataOperands()[1].setType(
        channelifyType(muxOp.getDataOperands()[1].getType()));
    muxOp.getDataResult().setType(
        channelifyType(muxOp.getDataResult().getType()));
  }
}

/// Converts undefined operations (LLVM::UndefOp) with a default "0"
/// constant triggered by the start signal of the corresponding function.
/// This is usually associated to uninitialized variables in the code
static LogicalResult convertUndefinedValues(ConversionPatternRewriter &rewriter,
                                            handshake::FuncOp &funcOp,
                                            NameAnalysis &namer) {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each undefined value
  auto undefinedValues = funcOp.getBody().getOps<LLVM::UndefOp>();

  for (auto undefOp : undefinedValues) {
    // Create an attribute of the appropriate type for the constant
    auto resType = undefOp.getRes().getType();
    TypedAttr cstAttr;
    if (isa<IndexType>(resType)) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    } else if (isa<IntegerType>(resType)) {
      cstAttr = rewriter.getIntegerAttr(resType, 0);
    } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
      cstAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    }

    // Create a constant with a default value and replace the undefined value
    rewriter.setInsertionPoint(undefOp);
    auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(),
                                                        cstAttr, startValue);
    cstOp->setDialectAttrs(undefOp->getAttrDictionary());
    undefOp.getResult().replaceAllUsesWith(cstOp.getResult());
    namer.replaceOp(cstOp, cstOp);
    rewriter.replaceOp(undefOp, cstOp.getResult());
  }

  return success();
}

/// Convers arith-level constants to handshake-level constants. Constants are
/// triggered by the start value of the corresponding function. The FTD
/// algorithm is then in charge of connecting the constants to the rest of the
/// network, in order for them to be re-generated
static LogicalResult convertConstants(ConversionPatternRewriter &rewriter,
                                      handshake::FuncOp &funcOp,
                                      NameAnalysis &namer) {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();
  llvm::DenseMap<Block *, Value> sourcesPerBlock;

  // For each constant
  auto constants = funcOp.getBody().getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : constants) {

    rewriter.setInsertionPoint(cstOp);

    // This variable will work as activation value for the constant. If the
    // constant is considered as sourcable, this will be the output of a source
    // component, otherwise it remains startValue
    auto controlValue = startValue;

    // Continue the conversion by obtaining the size of the constnat
    TypedAttr valueAttr = cstOp.getValue();

    if (isa<IndexType>(valueAttr.getType())) {
      auto intType = rewriter.getIntegerType(32);
      valueAttr = IntegerAttr::get(
          intType, cast<IntegerAttr>(valueAttr).getValue().trunc(32));
    }

    auto newCstOp = rewriter.create<handshake::ConstantOp>(
        cstOp.getLoc(), valueAttr, controlValue);

    newCstOp->setDialectAttrs(cstOp->getDialectAttrs());

    // Replace the constant and the usage of its result
    namer.replaceOp(cstOp, newCstOp);
    cstOp.getResult().replaceAllUsesWith(newCstOp.getResult());
    rewriter.replaceOp(cstOp, newCstOp->getResults());
  }
  return success();
}

template <typename SrcOp, typename DstOp>
LogicalResult FtdOneToOneConversion<SrcOp, DstOp>::matchAndRewrite(
    SrcOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(srcOp);
  SmallVector<Type> newTypes;
  for (Type resType : srcOp->getResultTypes())
    newTypes.push_back(channelifyType(resType));
  auto newOp =
      rewriter.create<DstOp>(srcOp->getLoc(), newTypes, adaptor.getOperands(),
                             srcOp->getAttrDictionary().getValue());

  // /!\ This is the main difference from the base function. Without such
  // replacement, a "null operand found" error is present at the end of the
  // transformation pass in almost any test. This is due to the way FTD tweaks
  // the coexistence of `cf` and `handshake` dialect to obtain a final circuit:
  // without such explicit replacement, deleted operations still provide values
  // to new operations. However, this should be fixed by understanding what is
  // causing MLIR to complain.
  for (auto [from, to] : llvm::zip(srcOp->getResults(), newOp->getResults()))
    from.replaceAllUsesWith(to);

  this->namer.replaceOp(srcOp, newOp);
  rewriter.replaceOp(srcOp, newOp);
  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // Add the muxes as obtained by the GSA analysis pass. This requires the start
  // value, as init merges need it as one of their output. However, the start
  // value is not available yet here, so a backedge is adopted instead.
  BackedgeBuilder edgeBuilderStart(rewriter, lowerFuncOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());
  if (failed(addGsaGates(lowerFuncOp.getRegion(), rewriter, gsaAnalysis,
                         startValueBackedge)))
    return failure();

  // First lower the parent function itself, without modifying its body
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // When GSA-MU functions are translated into multiplexers, an `init merge`
  // is created to feed them. This merge requires the start value of the
  // function as one of its data inputs. However, the start value was not
  // present yet when `addGsaGates` is called, thus we need to reconnect
  // it.
  startValueBackedge.setValue((Value)funcOp.getArguments().back());

  // Stores mapping from each value that passes through a merge-like
  // operation to the data result of that merge operation
  ArgReplacements argReplacements;

  // Currently, the following 2 functions do nothing but construct the network
  // of CMerges in complete isolation from the rest of the components
  // implementing the operations
  // In particular, the addMergeOps relies on adding Merges for every block
  // argument but because we removed all "real" arguments, we are only left
  // with the Start value as an argument for every block
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo, true)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports
  // get tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);

  // Create the memory interface according to the algorithm from FPGA'23. This
  // functions introduce new data dependencies that are then passed to FTD for
  // correctly delivering data between them like any real data dependencies
  if (failed(verifyAndCreateMemInterfaces(funcOp, rewriter, memInfo)))
    return failure();

  // Convert the constants and undefined values from the `arith` dialect to
  // the `handshake` dialect, while also using the start value as their
  // control value
  if (failed(::convertConstants(rewriter, funcOp, namer)) ||
      failed(::convertUndefinedValues(rewriter, funcOp, namer)))
    return failure();

  if (funcOp.getBlocks().size() != 1) {

    // Add muxes for regeneration of values in loop
    addRegen(funcOp, rewriter);
    channelifyMuxes(funcOp);

    // Add suppression blocks between each pair of producer and consumer
    addSupp(funcOp, rewriter);
  }

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  // Annotate the IR with the CFG information
  cfg::annotateCFG(funcOp, rewriter, namer);

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}

template <typename CastOp, typename ExtOp>
LogicalResult FtdConvertIndexCast<CastOp, ExtOp>::matchAndRewrite(
    CastOp castOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto getWidth = [](Type type) -> unsigned {
    // In Fast Token Delivery the type of the element might be already a
    // channel, rather than a simple type. In this case, the type should be
    // extracted. We also make sure that no extra bits are present at this
    // compilation stage.
    if (auto dataType = dyn_cast<handshake::ChannelType>(type)) {
      assert(dataType.getNumExtraSignals() == 0 &&
             "expected type to have no extra signals");
      type = dataType.getDataType();
    }
    if (isa<IndexType>(type))
      return 32;
    return type.getIntOrFloatBitWidth();
  };

  unsigned srcWidth = getWidth(castOp.getOperand().getType());
  unsigned dstWidth = getWidth(castOp.getResult().getType());
  Type dstType = handshake::ChannelType::get(rewriter.getIntegerType(dstWidth));
  Operation *newOp;
  if (srcWidth < dstWidth) {
    // This is an extension
    newOp =
        rewriter.create<ExtOp>(castOp.getLoc(), dstType, adaptor.getOperands(),
                               castOp->getAttrDictionary().getValue());
  } else {
    // This is a truncation
    newOp = rewriter.create<handshake::TruncIOp>(
        castOp.getLoc(), dstType, adaptor.getOperands(),
        castOp->getAttrDictionary().getValue());
  }
  this->namer.replaceOp(castOp, newOp);
  rewriter.replaceOp(castOp, newOp);

  // /!\ This is again the main difference from the normal flow. See the comment
  // in FtdOneToOneConversion.
  castOp.getResult().replaceAllUsesWith(newOp->getResult(0));
  return success();
}

std::unique_ptr<dynamatic::DynamaticPass> ftd::createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}
