//===- StandardToHandshakeFPL22.cpp - FPL22's elastic pass ----*- C++ -*-===//
//
// This file contains the implementation of the elastic pass, as described in
// https://www.epfl.ch/labs/lap/wp-content/uploads/2022/09/ElakhrasAug22_UnleashingParallelismInElasticCircuitsWithFasterTokenDelivery_FPL22.pdf
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/StandardToHandshakeFPL22.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace dynamatic;
using namespace dynamatic::experimental;

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return failure();

// ============================================================================
// Helper functions
// ============================================================================

/// Determines whether an operation is akin to a load or store memory operation.
static bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, AffineReadOpInterface,
             AffineWriteOpInterface>(op);
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
  else if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
    affine::MemRefAccess access(op);
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
addOpToMemInterfaces(HandshakeLoweringFPL22::MemInterfacesInfo &memInfo,
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
  HandshakeLoweringFPL22::MemBlockOps newBlock;
  newBlock.push_back(std::make_pair(opBlock, newOps));
  memInfo.push_back(std::make_pair(memref, newBlock));
}

/// Returns load/store results which are to be given as operands to a
/// handshake::MemoryControllerOp.
static SmallVector<Value, 2> getResultsToMemory(Operation *op) {

  if (auto loadOp = dyn_cast<handshake::DynamaticLoadOp>(op)) {
    // For load, get address output
    SmallVector<Value, 2> results;
    results.push_back(loadOp.getAddressResult());
    return results;
  } else {
    // For store, all outputs (data and address) go to memory
    auto storeOp = dyn_cast<handshake::DynamaticStoreOp>(op);
    assert(storeOp && "input operation must either be load or store");
    SmallVector<Value, 2> results(storeOp.getResults());
    return results;
  }
}

/// Adds the data input (from memory interface) to the list of load operands.
static void addLoadDataOperand(Operation *op, Value dataIn) {
  assert(op->getNumOperands() == 1 &&
         "load must have single address operand at this point");
  SmallVector<Value, 2> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(dataIn);
  op->setOperands(operands);
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

LogicalResult
HandshakeLoweringFPL22::createStartCtrl(ConversionPatternRewriter &rewriter) {

  // Add start point of the control-only path to the entry block's arguments
  Block *entryBlock = &r.front();
  startCtrl =
      entryBlock->addArgument(rewriter.getNoneType(), rewriter.getUnknownLoc());

  // Connect each block to startCtrl
  for (auto &block : r.getBlocks())
    setBlockEntryControl(&block, startCtrl);

  return success();
}

LogicalResult HandshakeLoweringFPL22::replaceMemoryOps(
    ConversionPatternRewriter &rewriter,
    HandshakeLoweringFPL22::MemInterfacesInfo &memInfo) {
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

    // Replace memref operation with corresponding handshake operation
    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](memref::LoadOp loadOp) {
          auto indices = loadOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          newOp = rewriter.create<handshake::DynamaticLoadOp>(
              op.getLoc(), memref, indices[0]);

          // Replace uses of old load result with data result of new load
          op.getResult(0).replaceAllUsesWith(
              dyn_cast<handshake::DynamaticLoadOp>(newOp).getDataResult());
        })
        .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
          auto indices = storeOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          newOp = rewriter.create<handshake::DynamaticStoreOp>(
              op.getLoc(), storeOp.getValueToStore(), indices[0]);
        })
        .Default([&](auto) {
          return op.emitOpError("Load/store operation cannot be handled.");
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
HandshakeLoweringFPL22::connectToMemory(ConversionPatternRewriter &rewriter,
                                        MemInterfacesInfo &memInfo) {

  // Connect memories (externally defined by memref block argument) to their
  // respective loads and stores
  unsigned memCount = 0;
  for (auto &[memref, memBlockOps] : memInfo) {

    // Derive memory interface inputs from operations interacting with it
    SmallVector<Value> memInputs;
    SmallVector<SmallVector<AccessTypeEnum>> accesses;

    for (auto &[block, memOps] : memBlockOps) {

      // Traverse the list of operations once to determine the ordering of loads
      // and stores
      unsigned stCount = 0;
      SmallVector<AccessTypeEnum> blockAccesses;
      for (auto *op : memOps) {
        if (isa<handshake::DynamaticLoadOp>(op))
          blockAccesses.push_back(AccessTypeEnum::Load);
        else {
          blockAccesses.push_back(AccessTypeEnum::Store);
          stCount++;
        }
      }
      accesses.push_back(blockAccesses);

      if (stCount > 0) {
        // Add control signal from block, fed through a constant indicating the
        // number of stores in the block (to eventually indicate block
        // completion to the end node)
        auto blockCtrl = getBlockEntryControl(block);
        rewriter.setInsertionPointAfter(blockCtrl.getDefiningOp());
        auto cstNumStore = rewriter.create<handshake::ConstantOp>(
            blockCtrl.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(stCount), blockCtrl);
        memInputs.push_back(cstNumStore.getResult());
      }

      // Traverse the list of operations once more and accumulate memory inputs
      // coming from the block
      for (auto *op : memOps) {
        // Add results of memory operation to memory interface operands
        SmallVector<Value, 2> results = getResultsToMemory(op);
        memInputs.insert(memInputs.end(), results.begin(), results.end());
      }
    }

    // Create memory interface at the top of the function
    Block *entryBlock = &r.front();
    rewriter.setInsertionPointToStart(entryBlock);
    auto memInterface = rewriter.create<handshake::MemoryControllerOp>(
        entryBlock->front().getLoc(), memref, memInputs, accesses, memCount++);

    // Add data result from memory to each load operation's operands
    unsigned memResultIdx = 0;
    for (auto &[block, memOps] : memBlockOps)
      for (auto *op : memOps)
        if (isa<handshake::DynamaticLoadOp>(op))
          addLoadDataOperand(op, memInterface->getResult(memResultIdx++));
  }

  return success();
}

LogicalResult HandshakeLoweringFPL22::replaceUndefinedValues(
    ConversionPatternRewriter &rewriter) {
  for (auto &block : r) {
    for (auto undefOp : block.getOps<mlir::LLVM::UndefOp>()) {
      // Create an attribute of the appropriate type for the constant
      auto resType = undefOp.getRes().getType();
      TypedAttr cstAttr = llvm::TypeSwitch<Type, TypedAttr>(resType)
                              .Case<IndexType>([&](auto type) {
                                return rewriter.getIndexAttr(0);
                              })
                              .Case<IntegerType>([&](auto type) {
                                return rewriter.getIntegerAttr(type, 0);
                              })
                              .Case<FloatType>([&](auto type) {
                                return rewriter.getFloatAttr(type, 0.0);
                              })
                              .Default([&](auto type) { return nullptr; });
      if (!cstAttr)
        return undefOp->emitError() << "operation has unsupported result type";

      // Create a constant with a default value and replace the undefined value
      rewriter.setInsertionPoint(undefOp);
      auto cstOp = rewriter.create<handshake::ConstantOp>(
          undefOp.getLoc(), resType, cstAttr, getBlockEntryControl(&block));
      rewriter.replaceOp(undefOp, cstOp.getResult());
    }
  }
  return success();
}

LogicalResult
HandshakeLoweringFPL22::idBasicBlocks(ConversionPatternRewriter &rewriter) {
  for (auto indexAndBlock : llvm::enumerate(r))
    for (auto &op : indexAndBlock.value())
      // Memory interfaces do not naturally belong to any block, so they do
      // not get an attribute
      if (!isa<handshake::MemoryControllerOp>(op))
        op.setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(indexAndBlock.index()));
  return success();
}

LogicalResult
HandshakeLoweringFPL22::createReturnNetwork(ConversionPatternRewriter &rewriter,
                                            bool idBasicBlocks) {

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

  // When identifying basic blocks, the end node is either put in the same
  // block as the function's single return statement or, in the case of
  // multiple return statements, it is put in a "fake block" along with the
  // merges that feed it its data inputs
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
    addIllegalDialect<affine::AffineDialect>();

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

static void removeBlockOperands(Region &f) {
  // Remove all block arguments, they are no longer used
  // eraseArguments also removes corresponding branch operands
  for (Block &block : f) {
    if (!block.isEntryBlock()) {
      int x = block.getNumArguments() - 1;
      for (int i = x; i >= 0; --i)
        block.eraseArgument(i);
    }
  }
}

// Get value from predBlock which will be set as operand of op (merge)
static Value getMergeOperand(HandshakeLowering::MergeOpInfo mergeInfo,
                             Block *predBlock, bool isFirstOperand) {
  // The input value to the merge operations
  Value srcVal = mergeInfo.val;
  // The block the merge operation belongs to
  Block *block = mergeInfo.op->getBlock();

  // The block terminator is either a cf-level branch or cf-level conditional
  // branch. In either case, identify the value passed to the block using its
  // index in the list of block arguments
  unsigned index = srcVal.cast<BlockArgument>().getArgNumber();
  Operation *termOp = predBlock->getTerminator();
  if (mlir::cf::CondBranchOp br = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    // Block should be one of the two destinations of the conditional branch
    auto *trueDest = br.getTrueDest(), *falseDest = br.getFalseDest();
    if (block == trueDest) {
      if (!isFirstOperand && trueDest == falseDest)
        return br.getFalseOperand(index);
      return br.getTrueOperand(index);
    }
    assert(block == falseDest);
    return br.getFalseOperand(index);
  }
  if (isa<mlir::cf::BranchOp>(termOp))
    return termOp->getOperand(index);
  return nullptr;
}

static unsigned getBlockPredecessorCount(Block *block) {
  // Returns number of block predecessors
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

LogicalResult HandshakeLoweringFPL22::handleTokenMissmatch(
    BackedgeBuilder &edgeBuilder, ConversionPatternRewriter &rewriter) {
  // Each consumer Block should only contain one MergeOp for a Value produced in
  // another Block.
  DenseMap<Block *, DenseMap<Value, handshake::MergeOp>> mapConsumerBlocks;

  // Run loop analysis
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&r);
  // CFGLoop nodes become invalid after CFGLoopInfo is destroyed.
  CFGLoopInfo li(domTree);
  DenseMap<Block *, BlockLoopInfo> blockToLoopInfoMap = findLoopDetails(li, r);

  // Iterate through all producer-consumer pairs (traversing edges in DFG)
  for (auto &block : r.getBlocks()) {
    for (auto &producerOp : block.getOperations()) {
      for (const auto &producerOpResult : producerOp.getResults()) {
        for (const auto &consumerOp : producerOpResult.getUsers()) {
          // llvm::outs() << producerOp.getName();
          // llvm::outs() << " --> ";
          // llvm::outs() << consumerOp->getName();
          // llvm::outs() << "\n";

          // Check if consumer Block is already added to the map
          auto blockIt = mapConsumerBlocks.find(consumerOp->getBlock());
          if (blockIt == mapConsumerBlocks.end()) {
            DenseMap<Value, handshake::MergeOp> dm;
            mapConsumerBlocks[consumerOp->getBlock()] = dm;
          }

          // Check if merge for producers result is already inserted at the
          // beginning of the consumer's block
          auto valueIt =
              mapConsumerBlocks[consumerOp->getBlock()].find(producerOpResult);
          if (valueIt != mapConsumerBlocks[consumerOp->getBlock()].end())
            continue;

          // Find common loop for producer's and consumer's blocks
          CFGLoop *producersInnermostLoop =
              blockToLoopInfoMap[producerOp.getBlock()].loop;
          CFGLoop *consumersInnermostLoop =
              blockToLoopInfoMap[consumerOp->getBlock()].loop;
          CFGLoop *commonLoop =
              findLCALoop(producersInnermostLoop, consumersInnermostLoop);

          int commonLoopDepth = commonLoop ? commonLoop->getLoopDepth() : 0;
          int consumerLoopDepth = consumersInnermostLoop
                                      ? consumersInnermostLoop->getLoopDepth()
                                      : 0;

          int numOfMerges = consumerLoopDepth - commonLoopDepth;

          if (numOfMerges <= 0)
            // There is no need to insert merge operation(s) because there is no
            // token missmatch
            continue;

          rewriter.setInsertionPointToStart(consumerOp->getBlock());
          auto insertLoc = consumerOp->getBlock()->front().getLoc();
          SmallVector<Value> operands;
          operands.push_back(producerOpResult);

          // Merge should take its own output as one of the inputs
          auto backedge = edgeBuilder.get(producerOpResult.getType());
          operands.push_back(Value(backedge));

          // Create MergeOp and resolve the backedge
          Operation *mergeOp = rewriter.create<handshake::MergeOp>(insertLoc, operands);
          backedge.setValue(mergeOp->getResult(0));

          // // Replace uses of producer's operation result in consumer's block
          // // with the merge output
          // for (Operation &opp : *(consumerOp->getBlock()))
          //   if (!isa<MergeLikeOpInterface>(opp)) {
          //     opp.replaceUsesOfWith(producerOpResult, mergeOp->getResult(0));
          //   }
        }
      }
    }
  }

  return success();
}

HandshakeLowering::MergeOpInfo
HandshakeLoweringFPL22::insertMerge(Block *block, Value val,
                                    BackedgeBuilder &edgeBuilder,
                                    ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getBlockPredecessorCount(block);
  auto insertLoc = block->front().getLoc();
  SmallVector<Backedge> dataEdges;
  SmallVector<Value> operands;

  // For consistency within the entry block, replace the latter's entry
  // control with the output of a MergeOp that takes the control-only
  // network's start point as input. This makes it so that only the
  // MergeOp's output is used as a control within the entry block, instead
  // of a combination of the MergeOp's output and the function/block control
  // argument. Taking this step out should have no impact on functionality
  // but would make the resulting IR less "regular"
  if (block == &r.front() && val == getBlockEntryControl(block)) {
    Operation *mergeOp;

    operands.push_back(val);
    mergeOp = rewriter.create<handshake::MergeOp>(insertLoc, operands);

    for (auto &block : r.getBlocks())
      setBlockEntryControl(&block, mergeOp->getResult(0));

    return MergeOpInfo{mergeOp, val, dataEdges};
  }

  // Insert "dummy" MergeOp's for blocks with less than two predecessors
  if (numPredecessors <= 1) {
    if (numPredecessors == 0) {
      // All of the entry block's block arguments get passed through a dummy
      // MergeOp. There is no need for a backedge here as the unique operand can
      // be resolved immediately
      operands.push_back(val);
    } else {
      // The value incoming from the single block predecessor will be resolved
      // later during merge reconnection
      auto edge = edgeBuilder.get(val.getType());
      dataEdges.push_back(edge);
      operands.push_back(Value(edge));
    }
    auto merge = rewriter.create<handshake::MergeOp>(insertLoc, operands);
    return MergeOpInfo{merge, val, dataEdges};
  }

  // Create a backedge for for each data operand. The index operand will
  // eventually resolve to the current block's control merge index output, while
  // data operands will resolve to their respective values from each block
  // predecessor
  for (unsigned i = 0; i < numPredecessors; i++) {
    auto edge = edgeBuilder.get(val.getType());
    dataEdges.push_back(edge);
    operands.push_back(Value(edge));
  }
  auto merge = rewriter.create<handshake::MergeOp>(insertLoc, operands);
  return MergeOpInfo{merge, val, dataEdges};
}

HandshakeLowering::BlockOps
HandshakeLoweringFPL22::insertMergeOps(HandshakeLowering::ValueMap &mergePairs,
                                       BackedgeBuilder &edgeBuilder,
                                       ConversionPatternRewriter &rewriter) {
  HandshakeLowering::BlockOps blockMerges;
  for (Block &block : r) {
    rewriter.setInsertionPointToStart(&block);

    // All of the block's live-ins are passed explictly through block arguments
    // thanks to prior SSA maximization
    for (auto &arg : block.getArguments()) {
      // No merges on memref block arguments; these are handled separately
      if (arg.getType().isa<mlir::MemRefType>())
        continue;

      auto mergeInfo = HandshakeLoweringFPL22::insertMerge(
          &block, arg, edgeBuilder, rewriter);
      blockMerges[&block].push_back(mergeInfo);
      mergePairs[arg] = mergeInfo.op->getResult(0);
    }
  }
  return blockMerges;
}

static void reconnectMergeOps(Region &r,
                              HandshakeLowering::BlockOps blockMerges,
                              HandshakeLowering::ValueMap &mergePairs) {
  // At this point all merge-like operations have backedges as operands.
  // We here replace all backedge values with appropriate value from
  // predecessor block. The predecessor can either be a merge, the original
  // defining value, or a branch operand.

  for (Block &block : r) {
    for (auto &mergeInfo : blockMerges[&block]) {
      size_t operandIdx = 0;
      // Set appropriate operand from each predecessor block
      for (auto *predBlock : block.getPredecessors()) {
        Value mgOperand =
            getMergeOperand(mergeInfo, predBlock, operandIdx == 0);
        assert(mgOperand != nullptr);
        if (!mgOperand.getDefiningOp()) {
          assert(mergePairs.count(mgOperand));
          mgOperand = mergePairs[mgOperand];
        }
        mergeInfo.dataEdges[operandIdx].setValue(mgOperand);
        operandIdx++;
      }

      // Reconnect all operands originating from livein defining value through
      // corresponding merge of that block
      for (Operation &opp : block)
        if (!isa<MergeLikeOpInterface>(opp))
          opp.replaceUsesOfWith(mergeInfo.val, mergeInfo.op->getResult(0));
    }
  }

  removeBlockOperands(r);
}

LogicalResult
HandshakeLoweringFPL22::addMergeOps(ConversionPatternRewriter &rewriter) {
  // Stores mapping from each value that pass through a merge operation to the
  // first result of that merge operation
  ValueMap mergePairs;

  // Create backedge builder to manage operands of merge operations between
  // insertion and reconnection
  BackedgeBuilder edgeBuilder{rewriter, r.front().front().getLoc()};

  // Iterate through all producer-consumer pairs and see if a token
  // missmatch occurs. One example of token missmatch is when a producers is
  // outside the loop and consumer is inside. In that case token produced once
  // needs to be consumed multiple times. This is solved by adding merge
  // operations where merge result is also one of the input operands.
  if (failed(handleTokenMissmatch(edgeBuilder, rewriter)))
    return failure();

  // Insert merge operations (with backedges instead of actual operands)
  BlockOps mergeOps =
      HandshakeLoweringFPL22::insertMergeOps(mergePairs, edgeBuilder, rewriter);

  // Reconnect merge operations with values incoming from predecessor blocks
  // and resolve all backedges that were created during merge insertion
  reconnectMergeOps(r, mergeOps, mergePairs);

  return success();
}

/// Lowers the region referenced by the handshake lowering strategy following
/// a fixed sequence of steps, some implemented in this file and some in
/// CIRCT's standard-to-handshake conversion pass.
static LogicalResult lowerRegion(HandshakeLoweringFPL22 &hl,
                                 bool idBasicBlocks) {

  auto &baseHl = static_cast<HandshakeLowering &>(hl);

  HandshakeLoweringFPL22::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::replaceMemoryOps,
                                memInfo)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::createStartCtrl)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::replaceCallOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addBranchOps)))
    return failure();

  bool sourceConstants = false;
  if (failed(runPartialLowering(baseHl,
                                &HandshakeLowering::connectConstantsToControl,
                                sourceConstants)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::connectToMemory,
                                memInfo)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPL22::replaceUndefinedValues)))
    return failure();

  if (idBasicBlocks &&
      failed(runPartialLowering(hl, &HandshakeLoweringFPL22::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPL22::createReturnNetwork, idBasicBlocks)))
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
        if (resTypes.empty())
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

  if (!funcIsExternal) {
    // Lower the region inside the function
    HandshakeLoweringFPL22 hl(newFuncOp.getBody());
    returnOnError(lowerRegion(hl, idBasicBlocks));
  }

  return success();
}

namespace {
/// FPL22's elastic pass. Runs elastic pass on every function (func::FuncOp)
/// of the module it is applied on. Succeeds whenever all functions in the
/// module were succesfully lowered to handshake.
struct StandardToHandshakeFPL22Pass
    : public dynamatic::experimental::impl::StandardToHandshakeFPL22Base<
          StandardToHandshakeFPL22Pass> {

  StandardToHandshakeFPL22Pass(bool idBasicBlocks) {
    this->idBasicBlocks = idBasicBlocks;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Lower every function individually
    for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>()))
      if (failed(lowerFuncOp(funcOp, &getContext(), idBasicBlocks)))
        return signalPassFailure();
  }
};
} // namespace

DenseMap<Block *, BlockLoopInfo>
dynamatic::experimental::findLoopDetails(CFGLoopInfo &li, Region &funcReg) {
  // Finding all loops.
  std::vector<CFGLoop *> loops;
  for (auto &block : funcReg.getBlocks()) {
    CFGLoop *loop = li.getLoopFor(&block);

    while (loop) {
      auto pos = std::find(loops.begin(), loops.end(), loop);
      if (pos == loops.end())
        loops.push_back(loop);
      loop = loop->getParentLoop();
    }
  }

  // Iterating over blocks of each loop, and attaching loop info.
  DenseMap<Block *, BlockLoopInfo> blockToLoopInfoMap;
  for (auto &block : funcReg.getBlocks()) {
    BlockLoopInfo bli;

    blockToLoopInfoMap.insert(std::make_pair(&block, bli));
  }

  for (auto &loop : loops) {
    Block *loopHeader = loop->getHeader();
    blockToLoopInfoMap[loopHeader].isHeader = true;
    blockToLoopInfoMap[loopHeader].loop = loop;

    llvm::SmallVector<Block *> exitBlocks;
    loop->getExitingBlocks(exitBlocks);
    for (auto &block : exitBlocks) {
      blockToLoopInfoMap[block].isExit = true;
      blockToLoopInfoMap[block].loop = loop;
    }

    // A latch block is a block that contains a branch back to the header.
    llvm::SmallVector<Block *> latchBlocks;
    loop->getLoopLatches(latchBlocks);
    for (auto &block : latchBlocks) {
      blockToLoopInfoMap[block].isLatch = true;
      blockToLoopInfoMap[block].loop = loop;
    }
  }

  return blockToLoopInfoMap;
}

CFGLoop *dynamatic::experimental::findLCALoop(CFGLoop *innermostLoopOfBB1,
                                              CFGLoop *innermostLoopOfBB2) {
  std::set<CFGLoop *> loopsOfB1;

  // Traverse upwards from block 1 innermost loop and store the loop ancestors
  // in the set.
  for (CFGLoop *currLoop = innermostLoopOfBB1; currLoop;
       currLoop = currLoop->getParentLoop())
    loopsOfB1.insert(currLoop);

  // // Traverse upwards from block 2 innermost loop until a common loop is
  // found.
  for (CFGLoop *currLoop = innermostLoopOfBB2; currLoop != nullptr;
       currLoop = currLoop->getParentLoop())
    if (loopsOfB1.find(currLoop) != loopsOfB1.end())
      return currLoop;

  return nullptr;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::experimental::createStandardToHandshakeFPL22Pass(
    bool idBasicBlocks) {
  return std::make_unique<StandardToHandshakeFPL22Pass>(idBasicBlocks);
}
