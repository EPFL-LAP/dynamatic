//===- StandardToHandshakeFPGA18.cpp - FPGA18's elastic pass ----*- C++ -*-===//
//
// This file contains the implementation of the elastic pass, as described in
// https://www.epfl.ch/labs/lap/wp-content/uploads/2018/11/JosipovicFeb18_DynamicallyScheduledHighLevelSynthesis_FPGA18.pdf.
// The implementation relies for some parts on CIRCT's standard-to-handshake
// conversion pass, but brings siginificant changes related to memory interface
// management and return network creation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::memref;
using namespace dynamatic;

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return failure();

// ============================================================================
// Helper functions
// ============================================================================

/// Determines whether an operation is akin to a load or store memory operation.
static bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, mlir::AffineReadOpInterface,
             mlir::AffineWriteOpInterface>(op);
}

/// Determines whether an operation is akin to a memory allocation operation.
static bool isAllocOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

/// Determines whether a memref type is suitable for covnersion in the context
/// of this pass.
static bool isValidMemrefType(Location loc, mlir::MemRefType type) {
  if (type.getNumDynamicDims() != 0 || type.getShape().size() != 1) {
    emitError(loc) << "memref's must be both statically sized and "
                      "unidimensional.";
    return false;
  }
  return true;
}

/// Extracts the memref argument to a memory operation and puts it in out.
/// Returns an error whenever the passed operation is not a memory operation.
static LogicalResult getOpMemRef(Operation *op, Value &out) {
  out = Value();
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    out = memOp.getMemRef();
  else if (auto memOp = dyn_cast<memref::StoreOp>(op))
    out = memOp.getMemRef();
  else if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
    MemRefAccess access(op);
    out = access.memref;
  }
  if (out != Value())
    return success();
  return op->emitOpError("Unknown Op type");
}

/// Adds a new operation (along with its parent block) to memory interfaces
/// identified so far in the function. If the operation references a so far
/// unencoutnered memory interface, the latter is added to the set of known
/// interfaces first.
static void
addOpToMemInterfaces(HandshakeLoweringFPGA18::MemInterfacesInfo &memInfo,
                     Value memref, Operation *op) {

  Block *opBlock = op->getBlock();

  // Search for the memory interface represented by memref
  for (auto &[interface, blockOps] : memInfo)
    if (memref == interface) {
      // Search for the block the operation belongs to
      for (auto &[block, ops] : blockOps)
        if (opBlock == block) {
          // Add the operation to the block
          ops.push_back(op);
          return;
        }

      // Add a new block to the memory interface, along with the memory
      // operation within the block
      std::vector<Operation *> newOps;
      newOps.push_back(op);
      blockOps.push_back(std::make_pair(opBlock, newOps));
      return;
    }

  // Add a new memory interface, along with the new block and the memory
  // operation within it
  std::vector<Operation *> newOps;
  newOps.push_back(op);
  HandshakeLoweringFPGA18::MemBlockOps newBlock;
  newBlock.push_back(std::make_pair(opBlock, newOps));
  memInfo.push_back(std::make_pair(memref, newBlock));
}

/// Returns load/store results which are to be given as operands to a
/// handshake::MemoryControllerOp.
static SmallVector<Value, 8> getResultsToMemory(Operation *op) {

  if (auto loadOp = dyn_cast<handshake::DynamaticLoadOp>(op)) {
    // For load, get all address outputs/indices
    // (load also has one data output which goes to successor operation)
    SmallVector<Value, 8> results(loadOp.getAddressResults());
    return results;
  } else {
    // For store, all outputs (data and address indices) go to memory
    auto storeOp = dyn_cast<handshake::DynamaticStoreOp>(op);
    assert(storeOp && "input operation must either be load or store");
    SmallVector<Value, 8> results(storeOp.getResults());
    return results;
  }
}

/// Adds a value to the list of operands of an operation.
static void addValueToOperands(Operation *op, Value val) {
  SmallVector<Value, 8> results(op->getOperands());
  results.push_back(val);
  op->setOperands(results);
}

/// Returns the list of data inputs to be passed as operands to the
/// handshake::EndOp of a handshake::FuncOp. In the case of a single return
/// statement, this is simply the return's outputs. In the case of multiple
/// returns, this is the list of individually merged outputs of all returns.
/// In the latter case, the function inserts the required handshake::MergeOp's
/// in the region.
static SmallVector<Value, 8>
mergeFunctionResults(Region &r, ConversionPatternRewriter &rewriter,
                     SmallVector<Operation *, 4> &newReturnOps,
                     std::optional<size_t> endNetworkId) {
  auto entryBlock = &r.front();
  if (newReturnOps.size() == 1) {
    // No need to merge results in case of single return
    return SmallVector<Value, 8>(newReturnOps[0]->getResults());
  }

  // Return values from multiple returns need to be merged together
  SmallVector<Value, 8> results;
  Location loc = entryBlock->getOperations().back().getLoc();
  rewriter.setInsertionPointToEnd(entryBlock);
  for (unsigned i = 0, e = newReturnOps[0]->getNumResults(); i < e; i++) {
    SmallVector<Value, 4> mergeOperands;
    for (auto *retOp : newReturnOps) {
      mergeOperands.push_back(retOp->getResult(i));
    }
    auto mergeOp = rewriter.create<handshake::MergeOp>(loc, mergeOperands);
    results.push_back(mergeOp.getResult());
    // Merge operation inherits from the bb atttribute of the latest (in program
    // order) return operation
    if (endNetworkId.has_value())
      mergeOp->setAttr(BB_ATTR,
                       rewriter.getUI32IntegerAttr(endNetworkId.value()));
  }
  return results;
}

/// Returns the control signals from memory controllers to be passed as operands
/// to the handshake::EndOp of a handshake::FuncOp.
static SmallVector<Value, 8> getFunctionEndControls(Region &r) {
  SmallVector<Value, 8> controls;
  for (auto op : r.getOps<handshake::MemoryControllerOp>()) {
    auto memOp = dyn_cast<handshake::MemoryControllerOp>(&op);
    controls.push_back(memOp->getDone());
  }
  return controls;
}

// ============================================================================
// Concrete lowering steps
// ============================================================================

LogicalResult HandshakeLoweringFPGA18::createControlOnlyNetwork(
    ConversionPatternRewriter &rewriter) {

  // Add start point of the control-only path to the entry block's arguments
  Block *entryBlock = &r.front();
  startCtrl =
      entryBlock->addArgument(rewriter.getNoneType(), rewriter.getUnknownLoc());
  setBlockEntryControl(entryBlock, startCtrl);

  // Add a control-only argument to each block
  for (auto &block : r.getBlocks())
    if (!block.isEntryBlock())
      setBlockEntryControl(&block, block.addArgument(startCtrl.getType(),
                                                     rewriter.getUnknownLoc()));

  // Modify branch-like block terminators to forward control value through
  // all blocks
  for (auto &block : r.getBlocks())
    if (auto op = dyn_cast<BranchOpInterface>(block.getTerminator()); op)
      for (unsigned i = 0, e = op->getNumSuccessors(); i < e; i++)
        op.getSuccessorOperands(i).append(getBlockEntryControl(&block));

  return success();
}

LogicalResult HandshakeLoweringFPGA18::replaceMemoryOps(
    ConversionPatternRewriter &rewriter,
    HandshakeLoweringFPGA18::MemInterfacesInfo &memInfo) {
  std::vector<Operation *> opsToErase;

  // Make sure to record external memories passed as function arguments, even if
  // they aren't used by any memory operation
  for (auto arg : r.getArguments()) {
    auto memrefType = dyn_cast<mlir::MemRefType>(arg.getType());
    if (!memrefType)
      continue;

    // Ensure that this is a valid memref-typed value.
    if (!isValidMemrefType(arg.getLoc(), memrefType))
      return failure();

    SmallVector<std::pair<Block *, std::vector<Operation *>>> emptyOps;
    memInfo.push_back(std::make_pair(arg, emptyOps));
  }

  // Replace load and store ops with the corresponding handshake ops
  // Need to traverse ops in blocks to store them in memRefOps in program
  // order
  for (Operation &op : r.getOps()) {
    if (!isMemoryOp(&op))
      continue;

    // For now, we don´t support memory allocations within the kernels
    if (isAllocOp(&op)) {
      op.emitOpError("allocation operations not supported");
      return failure();
    }

    rewriter.setInsertionPoint(&op);
    Value memref;
    if (getOpMemRef(&op, memref).failed())
      return failure();
    Operation *newOp = nullptr;

    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](auto loadOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc
          SmallVector<Value, 8> operands(loadOp.getIndices());

          newOp = rewriter.create<handshake::DynamaticLoadOp>(op.getLoc(),
                                                              memref, operands);

          // Replace uses of old load result with data result of new load
          op.getResult(0).replaceAllUsesWith(
              dyn_cast<handshake::DynamaticLoadOp>(newOp).getDataResult());
        })
        .Case<memref::StoreOp>([&](auto storeOp) {
          // Get operands which correspond to address indices
          // This will add all operands except alloc and data
          SmallVector<Value, 8> operands(storeOp.getIndices());

          // Create new op where operands are store data and address indices
          newOp = rewriter.create<handshake::DynamaticStoreOp>(
              op.getLoc(), storeOp.getValueToStore(), operands);
        })
        .Case<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(
            [&](auto) {
              // Get essential memref access inforamtion.
              MemRefAccess access(&op);
              // The address of an affine load/store operation can be a result
              // of an affine map, which is a linear combination of constants
              // and parameters. Therefore, we should extract the affine map of
              // each address and expand it into proper expressions that
              // calculate the result.
              mlir::AffineMap map;
              if (auto loadOp = dyn_cast<mlir::AffineReadOpInterface>(op))
                map = loadOp.getAffineMap();
              else
                map = dyn_cast<mlir::AffineWriteOpInterface>(op).getAffineMap();

              // The returned object from expandAffineMap is an optional list of
              // the expansion results from the given affine map, which are the
              // actual address indices that can be used as operands for
              // handshake LoadOp/StoreOp. The following processing requires it
              // to be a valid result.
              auto operands =
                  expandAffineMap(rewriter, op.getLoc(), map, access.indices);
              assert(operands && "Address operands of affine memref access "
                                 "cannot be reduced.");

              if (isa<mlir::AffineReadOpInterface>(op)) {
                auto loadOp = rewriter.create<handshake::DynamaticLoadOp>(
                    op.getLoc(), access.memref, *operands);
                newOp = loadOp;
                op.getResult(0).replaceAllUsesWith(loadOp.getDataResult());
              } else {
                newOp = rewriter.create<handshake::DynamaticStoreOp>(
                    op.getLoc(), op.getOperand(0), *operands);
              }
            })
        .Default([&](auto) {
          op.emitOpError("Load/store operation cannot be handled.");
        });

    // Record operation along the memory interface it uses
    addOpToMemInterfaces(memInfo, memref, newOp);

    // Old memory operation should be erased
    opsToErase.push_back(&op);
  }

  // Erase old memory operations
  for (auto *op : opsToErase) {
    op->eraseOperands(0, op->getNumOperands());
    rewriter.eraseOp(op);
  }

  return success();
}

LogicalResult
HandshakeLoweringFPGA18::connectToMemory(ConversionPatternRewriter &rewriter,
                                         MemInterfacesInfo &memInfo) {

  // Connect memories (externally defined by memref block argument or locally
  // defined by an allocation operation) to their respective loads and stores.
  unsigned memCount = 0;
  for (auto &[memref, memBlockOps] : memInfo) {

    // Derive memory interface operands from operations interacting with it.
    // - memControls for control signal coming from each basic block from which
    // the memory is interacted with
    // - memDataInputs for data inputs coming from results of loads (addresses)
    // and stores (addresses and data)
    std::vector<Value> memControls, memDataInputs;
    unsigned ldCount = 0, stCount = 0;
    for (auto &[block, memOps] : memBlockOps) {
      unsigned blockStCount = 0;

      for (auto *op : memOps) {
        // Add results of memory operation to memory interface operands
        SmallVector<Value, 8> results = getResultsToMemory(op);
        memDataInputs.insert(memDataInputs.end(), results.begin(),
                             results.end());
        // Keep track of the number of loads and stores
        if (isa<handshake::DynamaticLoadOp>(op))
          ldCount++;
        else
          blockStCount++;
      }

      // Keep track of the total number of stores to the memory interface
      stCount += blockStCount;

      // Add control signal from block, fed through a constant indicating the
      // number of stores in the block (to eventually indicate block completion
      // to the end node)
      auto blockCtrl = getBlockEntryControl(block);
      rewriter.setInsertionPointAfter(blockCtrl.getDefiningOp());
      auto cstNumStore = rewriter.create<handshake::ConstantOp>(
          blockCtrl.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(blockStCount), blockCtrl);
      memControls.push_back(cstNumStore.getResult());
    }

    // A memory is external if the memref that defines it is provided as a
    // function (block) argument.
    bool isExternalMemory = memref.isa<BlockArgument>();

    // Combine all memory operands together
    std::vector<Value> memOperands;
    std::copy(memControls.begin(), memControls.end(),
              std::back_inserter(memOperands));
    std::copy(memDataInputs.begin(), memDataInputs.end(),
              std::back_inserter(memOperands));

    // Create memory interface at the top of the function
    Block *entryBlock = &r.front();
    rewriter.setInsertionPointToStart(entryBlock);
    auto memInterface = rewriter.create<handshake::MemoryControllerOp>(
        entryBlock->front().getLoc(), memref, memOperands, memBlockOps.size(),
        ldCount, stCount, isExternalMemory, memCount++);

    // Add data result from memory to each load operation's operands
    unsigned memResultIdx = 0;
    for (auto &[block, memOps] : memBlockOps)
      for (auto *op : memOps)
        if (isa<handshake::DynamaticLoadOp>(op))
          addValueToOperands(op, memInterface->getResult(memResultIdx++));
  }

  return success();
}

LogicalResult
HandshakeLoweringFPGA18::idBasicBlocks(ConversionPatternRewriter &rewriter) {
  for (auto &indexAndBlock : llvm::enumerate(r))
    for (auto &op : indexAndBlock.value())
      // Memory interfaces do not naturally belong to any block, so they do not
      // get an attribute
      if (!isa<handshake::MemoryControllerOp>(op))
        op.setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(indexAndBlock.index()));
  return success();
}

LogicalResult HandshakeLoweringFPGA18::createReturnNetwork(
    ConversionPatternRewriter &rewriter, bool idBasicBlocks) {

  auto *entryBlock = &r.front();
  auto &entryBlockOps = entryBlock->getOperations();

  // Move all operations to entry block. While doing so, delete all block
  // terminators and create a handshake-level return operation for each
  // existing
  // func-level return operation
  SmallVector<Operation *> terminatorsToErase;
  SmallVector<Operation *, 4> newReturnOps;
  for (auto &block : r) {
    Operation &termOp = block.back();
    if (isa<func::ReturnOp>(termOp)) {
      SmallVector<Value, 8> operands(termOp.getOperands());
      // When the enclosing function only returns a control value (no data
      // results), return statements must take exactly one control-only input
      if (operands.empty())
        operands.push_back(getBlockEntryControl(&block));

      // Insert new return operation next to the old one
      rewriter.setInsertionPoint(&termOp);
      auto newRet = rewriter.create<handshake::DynamaticReturnOp>(
          termOp.getLoc(), operands);
      newReturnOps.push_back(newRet);

      if (idBasicBlocks)
        // New return operation belongs in the same basic block as the old one
        newRet->setAttr(BB_ATTR, termOp.getAttr(BB_ATTR));
    }
    terminatorsToErase.push_back(&termOp);
    entryBlockOps.splice(entryBlockOps.end(), block.getOperations());
  }
  assert(!newReturnOps.empty() && "function must have at least one return");

  // When identifying basic blocks, the end node is either put in the same block
  // as the function's single return statement or, in the case of multiple
  // return statements, it is put in a "fake block" along with the merges that
  // feed it its data inputs
  std::optional<size_t> endNetworkID{};
  if (idBasicBlocks)
    endNetworkID = (newReturnOps.size() > 1)
                       ? r.getBlocks().size()
                       : newReturnOps[0]
                             ->getAttrOfType<mlir::IntegerAttr>(BB_ATTR)
                             .getValue()
                             .getZExtValue();

  // Erase all blocks except the entry block
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(r, 1))) {
    block.clear();
    block.dropAllDefinedValueUses();
    block.eraseArguments(0, block.getNumArguments());
    block.erase();
  }

  // Erase all leftover block terminators
  for (auto *op : terminatorsToErase)
    op->erase();

  // Insert an end node at the end of the function that merges results from
  // all handshake-level return operations and wait for all memory controllers
  // to signal completion
  SmallVector<Value, 8> endOperands;
  endOperands.append(
      mergeFunctionResults(r, rewriter, newReturnOps, endNetworkID));
  endOperands.append(getFunctionEndControls(r));
  rewriter.setInsertionPointToEnd(entryBlock);
  auto endOp = rewriter.create<handshake::EndOp>(entryBlockOps.back().getLoc(),
                                                 endOperands);
  if (endNetworkID.has_value())
    endOp->setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(endNetworkID.value()));

  return success();
}

// ============================================================================
// Lowering strategy
// ============================================================================
namespace {

/// Conversion target for lowering a func::FuncOp to a handshake::FuncOp
class LowerFuncOpTarget : public ConversionTarget {
public:
  explicit LowerFuncOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredFuncs.clear();
    addLegalDialect<handshake::HandshakeDialect>();
    addLegalDialect<func::FuncDialect>();
    addLegalDialect<arith::ArithDialect>();
    addIllegalDialect<scf::SCFDialect>();
    addIllegalDialect<mlir::AffineDialect>();

    // The root operation to be replaced is marked dynamically legal based on
    // the lowering status of the given operation, see PartialLowerOp. This is
    // to make the operation go from illegal to legal after partial lowering
    addDynamicallyLegalOp<func::FuncOp>(
        [&](const auto &op) { return loweredFuncs[op]; });
  }
  DenseMap<Operation *, bool> loweredFuncs;
};

/// Conversion pattern for partially lowering a func::FuncOp to a
/// handshake::FuncOp. Lowering is achieved by a provided partial lowering
/// function.
struct PartialLowerFuncOp : public OpConversionPattern<func::FuncOp> {
  using PartialLoweringFunc =
      std::function<LogicalResult(func::FuncOp, ConversionPatternRewriter &)>;

public:
  PartialLowerFuncOp(LowerFuncOpTarget &target, MLIRContext *context,
                     const PartialLoweringFunc &fun)
      : OpConversionPattern<func::FuncOp>(context), target(target),
        loweringFunc(fun) {}
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion scheme requires the matched root operation to be
    // replaced or updated if the match was successful. Calling
    // updateRootInPlace ensures that happens even if loweringFUnc doesn't
    // modify the root operation
    LogicalResult res = failure();
    rewriter.updateRootInPlace(op, [&] { res = loweringFunc(op, rewriter); });

    // Signal to the conversion target that the function was successfully
    // partially lowered
    target.loweredFuncs[op] = true;

    // Success status of conversion pattern determined by success of partial
    // lowering function
    return res;
  };

private:
  /// The conversion target for this pattern
  LowerFuncOpTarget &target;
  /// The rewrite function
  PartialLoweringFunc loweringFunc;
};

/// Strategy class for SSA maximization during std-to-handshake conversion.
/// Block arguments of type MemRefType and allocation operations are not
/// considered for SSA maximization.
class HandshakeLoweringSSAStrategy : public SSAMaximizationStrategy {
  /// Filters out block arguments of type MemRefType
  bool maximizeArgument(BlockArgument arg) override {
    return !arg.getType().isa<mlir::MemRefType>();
  }

  /// Filters out allocation operations
  bool maximizeOp(Operation *op) override { return !isAllocOp(op); }
};
} // namespace

/// Converts every value in the region into maximal SSA form, unless the value
/// is a block argument of type MemRefType or the result of an allocation
/// operation.
static LogicalResult maximizeSSANoMem(Region &r,
                                      ConversionPatternRewriter &rewriter) {
  HandshakeLoweringSSAStrategy strategy;
  return maximizeSSA(r, strategy, rewriter);
}

/// Convenience function for running lowerToHandshake with a partial
/// handshake::FuncOp lowering function.
static LogicalResult
partiallyLowerOp(const PartialLowerFuncOp::PartialLoweringFunc &loweringFunc,
                 MLIRContext *ctx, func::FuncOp op) {

  RewritePatternSet patterns(ctx);
  auto target = LowerFuncOpTarget(*ctx);
  patterns.add<PartialLowerFuncOp>(target, ctx, loweringFunc);
  return applyPartialConversion(op, target, std::move(patterns));
}

/// Lowers the region referenced by the handshake lowering strategy following a
/// fixed sequence of steps, some implemented in this file and some in CIRCT's
/// standard-to-handshake conversion pass.
static LogicalResult lowerRegion(HandshakeLoweringFPGA18 &hl,
                                 bool idBasicBlocks) {

  auto &baseHl = static_cast<HandshakeLowering &>(hl);

  HandshakeLoweringFPGA18::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::replaceMemoryOps,
                                memInfo)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::createControlOnlyNetwork)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::replaceCallOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addBranchOps)))
    return failure();

  bool sourceConstants = true;
  if (failed(runPartialLowering(baseHl,
                                &HandshakeLowering::connectConstantsToControl,
                                sourceConstants)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::connectToMemory,
                                memInfo)))
    return failure();

  if (idBasicBlocks &&
      failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::createReturnNetwork, idBasicBlocks)))
    return failure();

  return success();
}

/// Fully lowers a func::FuncOp to a handshake::FuncOp.
static LogicalResult lowerFuncOp(func::FuncOp funcOp, MLIRContext *ctx,
                                 bool idBasicBlocks) {
  // Only retain those attributes that are not constructed by build
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == funcOp.getFunctionTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &argType : funcOp.getArgumentTypes())
    argTypes.push_back(argType);

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;
  for (auto resType : funcOp.getResultTypes())
    resTypes.push_back(resType);

  handshake::FuncOp newFuncOp;

  bool funcIsExternal = funcOp.isExternal();

  // Add control input/output to function arguments/results and create a
  // handshake::FuncOp of appropriate type
  returnOnError(partiallyLowerOp(
      [&](func::FuncOp funcOp, PatternRewriter &rewriter) {
        auto noneType = rewriter.getNoneType();
        resTypes.push_back(noneType);
        argTypes.push_back(noneType);
        auto func_type = rewriter.getFunctionType(argTypes, resTypes);
        newFuncOp = rewriter.create<handshake::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), func_type, attributes);
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());
        if (!funcIsExternal)
          newFuncOp.resolveArgAndResNames();
        return success();
      },
      ctx, funcOp));

  // Delete the original function
  funcOp->erase();

  // Apply SSA maximization
  returnOnError(
      partiallyLowerRegion(maximizeSSANoMem, ctx, newFuncOp.getBody()));

  if (!funcIsExternal) {
    // Lower the region inside the function
    HandshakeLoweringFPGA18 hl(newFuncOp.getBody());
    returnOnError(lowerRegion(hl, idBasicBlocks));
  }

  return success();
}

namespace {
/// FPGA18's elastic pass. Runs elastic pass on every function (func::FuncOp) of
/// the module it is applied on. Succeeds whenever all functions in the module
/// were succesfully lowered to handshake.
struct StandardToHandshakeFPGA18Pass
    : public StandardToHandshakeFPGA18Base<StandardToHandshakeFPGA18Pass> {

  StandardToHandshakeFPGA18Pass(bool idBasicBlocks) {
    this->idBasicBlocks = idBasicBlocks;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Lower every function individually
    for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>()))
      if (failed(lowerFuncOp(funcOp, &getContext(), idBasicBlocks)))
        return signalPassFailure();

    // Legalize the resulting functions by performing any simple conversion
    for (auto handshakeFunc : m.getOps<handshake::FuncOp>())
      if (failed(postDataflowConvert(handshakeFunc)))
        return signalPassFailure();
  }
};
} // namespace

dynamatic::HandshakeBlocks
dynamatic::getHandshakeBlocks(handshake::FuncOp funcOp) {
  dynamatic::HandshakeBlocks handshakeBlocks;
  for (auto &op : funcOp.getOps())
    if (auto bbAttr = op.getAttrOfType<mlir::IntegerAttr>(BB_ATTR); bbAttr)
      handshakeBlocks.blocks[bbAttr.getValue().getZExtValue()].push_back(&op);
    else
      handshakeBlocks.outOfBlocks.push_back(&op);
  return handshakeBlocks;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createStandardToHandshakeFPGA18Pass(bool idBasicBlocks) {
  return std::make_unique<StandardToHandshakeFPGA18Pass>(idBasicBlocks);
}
