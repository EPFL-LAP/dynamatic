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
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

// [START Boilerplate code for the MLIR pass]
#include "experimental/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

struct AllocaOpConversion : public DynOpConversionPattern<memref::AllocaOp> {
  using DynOpConversionPattern<memref::AllocaOp>::DynOpConversionPattern;

  // Construct a dense element attribute with everything zeroes.
  DenseElementsAttr getZeroAttr(ShapedType type) const {
    auto elemType = type.getElementType();
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      return DenseElementsAttr::get(type, APInt(intTy.getWidth(), 0));
    }
    if (auto floatTy = dyn_cast<FloatType>(type)) {
      if (floatTy.isF16())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEhalf(), /*negative=*/false));
      if (floatTy.isBF16())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::BFloat(), /*negative=*/false));
      if (floatTy.isF32())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEsingle(), /*negative=*/false));
      if (floatTy.isF64())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEdouble(), /*negative=*/false));
      llvm::report_fatal_error("Unhandled float element type!");
    }
    llvm::report_fatal_error("Unknown base element type!");
  }

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    // HACK: By default, we initialize the memory with all zeros. According to
    // the C standard, this only happens for arrays.
    rewriter.replaceOpWithNewOp<handshake::RAMOp>(op, op.getType(),
                                                  getZeroAttr(op.getType()));
    return success();
  }
};

struct GetGlobalOpConversion
    : public DynOpConversionPattern<memref::GetGlobalOp> {
  using DynOpConversionPattern<memref::GetGlobalOp>::DynOpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    // clang-format off
    // Example:
    //  memref.global "external" constant @internal_array : memref<...> = dense<...>
    //  ....
    //  %4 = memref.get_global @internal_array : memref<...>
    //
    // In this case, we remove the global constant and rewrite the addressof
    // node into a RAMOp (and we put an attribute to describe its constant
    // value).
    // clang-format on
    SymbolTableCollection symbolTableCollection;

    auto symNameOfGetGlobal = op.getNameAttr();

    memref::GlobalOp global;
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    moduleOp.walk([&global, symNameOfGetGlobal](memref::GlobalOp gbl) {
      if (gbl.getSymName() == symNameOfGetGlobal.getValue()) {
        global = gbl;
      }
    });

    if (!global) {
      // No corresponding Global (maybe emit pass failure is better)
      return failure();
    }

    /// The initial value doesn't have any type constraints. Therefore we need
    /// to check if it is stored as dense elements.
    mlir::Attribute initValueAttr = global.getInitialValueAttr();
    if (auto denseAttr = initValueAttr.dyn_cast<DenseElementsAttr>()) {
      rewriter.replaceOpWithNewOp<handshake::RAMOp>(op, op.getType(),
                                                    denseAttr);
    } else {
      llvm::report_fatal_error(
          "The initial value must be denoted in DenseElementsAttr.");
    }
    return success();
  }
};

// TODO: Here we simply erase all the global variables and attach the initial
// values to the RAMOps inside the handshake function.
struct GlobalOpConversion : public DynOpConversionPattern<memref::GlobalOp> {
  using DynOpConversionPattern<memref::GlobalOp>::DynOpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Per-block edge information captured from CF-level IR before conversion.
struct BlockEdgeInfo {
  bool isConditional = false;
  bool hasSuccessors = false;
  unsigned trueSuccIdx = 0;
  unsigned falseSuccIdx = 0;
  unsigned uncondSuccIdx = 0;
};

/// Complete CFG topology of one function, captured before conversion.
struct OriginalCFGInfo {
  unsigned numBlocks = 0;
  SmallVector<BlockEdgeInfo> blockEdges;
};

/// Walk every func::FuncOp in the module and capture its CFG topology.
/// Must be called BEFORE applyFullConversion.
static DenseMap<StringRef, OriginalCFGInfo>
captureAllCFGTopologies(ModuleOp moduleOp) {
  DenseMap<StringRef, OriginalCFGInfo> result;

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (funcOp.isExternal() || funcOp.getSymName().startswith("__init"))
      continue;

    Region &region = funcOp.getBody();
    if (region.empty())
      continue;

    OriginalCFGInfo info;

    DenseMap<Block *, unsigned> blockIdx;
    for (auto [idx, block] : llvm::enumerate(region))
      blockIdx[&block] = idx;

    info.numBlocks = blockIdx.size();
    info.blockEdges.resize(info.numBlocks);

    for (auto &[block, idx] : blockIdx) {
      BlockEdgeInfo &edge = info.blockEdges[idx];
      Operation *term = block->getTerminator();

      if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        edge.isConditional = true;
        edge.hasSuccessors = true;
        edge.trueSuccIdx = blockIdx.lookup(condBr.getTrueDest());
        edge.falseSuccIdx = blockIdx.lookup(condBr.getFalseDest());
      } else if (auto br = dyn_cast<cf::BranchOp>(term)) {
        edge.hasSuccessors = true;
        edge.uncondSuccIdx = blockIdx.lookup(br.getDest());
      }
    }

    result[funcOp.getSymName()] = std::move(info);
  }

  return result;
}

/// Create a ShadowCFG for a given handshake::FuncOp using the topology
/// captured before conversion.
static ftd::ShadowCFG buildShadowCFG(OpBuilder &builder,
                                     handshake::FuncOp realFuncOp,
                                     const OriginalCFGInfo &info) {
  ftd::ShadowCFG shadow;
  Location loc = realFuncOp.getLoc();

  // 1. Create a temporary func::FuncOp with blocks + CF terminators
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(realFuncOp);
    auto funcType = builder.getFunctionType({}, {});
    shadow.shadowFunc =
        builder.create<func::FuncOp>(loc, "__ftd_shadow_cfg__", funcType);

    Region &R = shadow.shadowFunc.getBody();
    SmallVector<Block *> blocks;
    for (unsigned i = 0; i < info.numBlocks; ++i)
      blocks.push_back(builder.createBlock(&R, R.end()));

    for (unsigned i = 0; i < info.numBlocks; ++i) {
      const BlockEdgeInfo &edge = info.blockEdges[i];
      builder.setInsertionPointToEnd(blocks[i]);

      if (edge.isConditional) {
        auto dummyCond =
            builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
        builder.create<cf::CondBranchOp>(
            loc, dummyCond, blocks[edge.trueSuccIdx], ValueRange{},
            blocks[edge.falseSuccIdx], ValueRange{});
      } else if (edge.hasSuccessors) {
        builder.create<cf::BranchOp>(loc, blocks[edge.uncondSuccIdx]);
      } else {
        builder.create<func::ReturnOp>(loc);
      }
    }
  }

  // 2. Scan the real funcOp to map BB index -> real condition Value
  realFuncOp.walk([&](handshake::ConditionalBranchOp brOp) {
    if (brOp->hasAttr("ftd.skip"))
      return;
    auto bbAttr = brOp->getAttrOfType<IntegerAttr>("handshake.bb");
    if (!bbAttr)
      return;
    unsigned bbIdx = bbAttr.getUInt();
    if (!shadow.conditionMap.contains(bbIdx))
      shadow.conditionMap[bbIdx] = brOp.getConditionOperand();
  });

  return shadow;
}

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    // Capture CFG topology before conversion flattens everything.
    auto cfgTopologies = captureAllCFGTopologies(modOp);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<ControlDependenceAnalysis>(),
        getAnalysis<gsa::GSAAnalysis>(), getAnalysis<NameAnalysis>(), converter,
        ctx);

    patterns.add<
        // LowerFuncToHandshake,
        ConvertConstants, AllocaOpConversion, ConvertCalls,
        ConvertUndefinedValues, GetGlobalOpConversion, GlobalOpConversion,
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
        OneToOneConversion<arith::RemSIOp, handshake::RemSIOp>,
        OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
        OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
        OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
        OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
        OneToOneConversion<arith::MaxSIOp, handshake::MaxSIOp>,
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
        OneToOneConversion<arith::TruncFOp, handshake::TruncFOp>,
        OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>,
        OneToOneConversion<arith::SIToFPOp, handshake::SIToFPOp>,
        OneToOneConversion<arith::UIToFPOp, handshake::UIToFPOp>,
        OneToOneConversion<arith::FPToSIOp, handshake::FPToSIOp>,
        OneToOneConversion<arith::ExtFOp, handshake::ExtFOp>,
        OneToOneConversion<math::AbsFOp, handshake::AbsFOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
      // If the call is to __init, consider it legal for now.
      // This allows the pass to continue so that __placeholder conversion can
      // later erase these __init calls.
      // All other func.CallOp (not calling __init) remain illegal due to the
      // addIllegalDialect rule above and must be converted by a pattern.
      if (auto calledFn = dyn_cast_or_null<func::FuncOp>(
              SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()))) {
        return calledFn.getSymName().startswith("__init");
      }
      // If symbol lookup fails or it's not a func::FuncOp, treat as default
      // (illegal)
      return false;
    });
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return op.getSymName().startswith("__init"); });

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();

    // Clean up: Remove the definition of each __init* function, but only if it
    // has no remaining uses. This is safe because all valid calls to __init*
    // were tracked and deleted earlier.
    for (auto func : llvm::make_early_inc_range(modOp.getOps<func::FuncOp>())) {
      if (func.getSymName().startswith("__init")) {
        assert(func.use_empty() &&
               "__init function should not have users after transformation");
        func.erase();
      }
    }

    for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
      mlir::OpBuilder builder(funcOp.getContext());

      auto topoIt = cfgTopologies.find(funcOp.getName());
      if (topoIt == cfgTopologies.end())
        continue;
      const OriginalCFGInfo &info = topoIt->second;

      if (info.numBlocks <= 1)
        continue;

      // Build the shadow CFG — one struct, everything inside.
      ftd::ShadowCFG shadow = buildShadowCFG(builder, funcOp, info);

      ftd::resolveCondPlaceholders(funcOp, builder, shadow);

      // Populate conditionMap from XorOp placeholders
      for (auto xorOp : funcOp.getOps<handshake::XOrIOp>()) {
        if (!xorOp->hasAttr("ftd.cvar"))
          continue;
        auto bbAttr = xorOp->getAttrOfType<IntegerAttr>("handshake.bb");
        if (!bbAttr)
          continue;
        shadow.conditionMap[bbAttr.getUInt()] = xorOp.getResult();
      }

      ftd::addRegen(funcOp, builder, shadow);
      ftd::addSupp(funcOp, builder, shadow);
      ftd::finalizeCondPlaceholders(funcOp);

      shadow.destroy();
    }
  }
};
} // namespace

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

// static void channelifyMuxes(handshake::FuncOp &funcOp) {
//   // Considering each mux that was added, the inputs and output values must
//   be
//   // channellified
//   for (handshake::MuxOp muxOp : funcOp.getOps<handshake::MuxOp>()) {
//     assert(muxOp.getDataOperands().size() == 2 &&
//            "Multiplexers should have two data inputs");
//     muxOp.getDataOperands()[0].setType(
//         channelifyType(muxOp.getDataOperands()[0].getType()));
//     muxOp.getDataOperands()[1].setType(
//         channelifyType(muxOp.getDataOperands()[1].getType()));
//     muxOp.getDataResult().setType(
//         channelifyType(muxOp.getDataResult().getType()));
//   }
// }

LogicalResult ftd::FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {
  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  ftd::createAllCondPlaceholders(lowerFuncOp.getRegion(), rewriter);

  // Structure used inside addGsaGates to temporarily map a cf value to a
  // backedge until the proper handshake values are created; in which case, the
  // backedge is replaced with the corresponding hanshake values
  static DenseMap<Value, SmallVector<Backedge, 2>> pendingMuxOperands;

  // Add the muxes as obtained by the GSA analysis pass. This requires the
  // start value, as init merges need it as one of their output. However,
  // the start value is not available yet here, so a backedge is adopted
  // instead.
  BackedgeBuilder edgeBuilderStart(rewriter, lowerFuncOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());
  if (failed(addGsaGates(lowerFuncOp.getRegion(), rewriter, gsaAnalysis,
                         startValueBackedge, &pendingMuxOperands)))
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

  for (auto &[originalValue, backedges] : pendingMuxOperands) {
    Value newVal = rewriter.getRemappedValue(originalValue);
    assert(newVal && "Failed to remap GSA mux operand!");

    for (Backedge &be : backedges)
      be.setValue(newVal);
  }
  pendingMuxOperands.clear();

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

  // addBranchOps only creates handshake::ConditionalBranchOp for live-out
  // values.  If a conditional block has no live-outs, no ConditionalBranchOp
  // is created and the condition value is lost after flattening.
  // Create one using the block's control signal so that buildShadowCFG's
  // walk can always find the condition.
  for (Block &block : funcOp) {
    auto condBr = dyn_cast<cf::CondBranchOp>(block.getTerminator());
    if (!condBr)
      continue;

    bool hasHandshakeCondBr = llvm::any_of(block, [](Operation &op) {
      return isa<handshake::ConditionalBranchOp>(&op);
    });

    if (!hasHandshakeCondBr) {
      Value cond = rewriter.getRemappedValue(condBr.getCondition());
      assert(cond && "Failed to remap condition");
      Value ctrl = block.getArguments().back();
      rewriter.setInsertionPoint(condBr);
      rewriter.create<handshake::ConditionalBranchOp>(
          condBr.getLoc(), cond, ctrl);
    }
  }

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo)))
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

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  // Annotate the IR with the CFG information
  cfg::annotateCFG(funcOp, rewriter, namer);

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}
