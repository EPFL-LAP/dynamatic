//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake -----*- C++
//-*-===//
//
// Implements the fast token delivery methodology as depicted in FPGA'22,
// together with the fast LSQ allocation as depicted in FPGA'23.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
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

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // Print a message on stdout to prove that the pass is running
    // llvm::outs() << "FTD pass is running as expected";

    // Put all non-external functions into maximal SSA form
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      if (!funcOp.isExternal()) {
        FuncSSAStrategy strategy;
        if (failed(dynamatic::maximizeSSA(funcOp.getBody(), strategy)))
          return signalPassFailure();
      }
    }

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);
    patterns.add<experimental::ftd::FtdLowerFuncToHandshake, ConvertConstants,
                 ConvertCalls, ConvertUndefinedValues,
                 ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
                 ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
                 OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
                 OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
                 OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
                 OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
                 OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
                 OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
                 OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
                 OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
                 OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
                 OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
                 OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
                 OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
                 OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
                 OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
                 OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
                 OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
                 OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
                 OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
                 OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
                 OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
                 OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
                 OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>>(
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

namespace dynamatic {
namespace experimental {
namespace ftd {

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

LogicalResult FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // First lower the parent function itself, without modifying its body (except
  // the block arguments and terminators)
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // Stores mapping from each value that passes through a merge-like operation
  // to the data result of that merge operation
  ArgReplacements argReplacements;
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports get
  // tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);
  if (failed(verifyAndCreateMemInterfaces(funcOp, rewriter, memInfo)))
    return failure();

  idBasicBlocks(funcOp, rewriter);
  return flattenAndTerminate(funcOp, rewriter, argReplacements);
}

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
