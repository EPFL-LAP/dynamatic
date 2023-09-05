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

using TokenMissmatchMergeOps =
    std::vector<HandshakeLoweringFPL22::TokenMissmatchMergeOp>;

// ============================================================================
// Helper functions
// ============================================================================

static Value
getResultForStayingInLoop(mlir::cf::CondBranchOp loopExitBranchOp,
                          handshake::ConditionalBranchOp condBranch,
                          CFGLoop *loop) {
  if (loop->contains(loopExitBranchOp.getTrueDest()))
    return condBranch.getTrueResult();
  assert(loop->contains(loopExitBranchOp.getFalseDest()));
  return condBranch.getFalseResult();
}

static OperandRange getBranchOperands(Operation *termOp) {
  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp))
    return condBranchOp.getOperands().drop_front();
  assert(isa<mlir::cf::BranchOp>(termOp) && "unsupported block terminator");
  return termOp->getOperands();
}

// Return the appropriate branch result based on successor block which uses it
static Value getSuccResult(Operation *termOp, Operation *newOp,
                           Block *succBlock) {
  // For conditional block, check if result goes to true or to false successor
  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    if (condBranchOp.getTrueDest() == succBlock)
      return dyn_cast<handshake::ConditionalBranchOp>(newOp).getTrueResult();
    assert(condBranchOp.getFalseDest() == succBlock);
    return dyn_cast<handshake::ConditionalBranchOp>(newOp).getFalseResult();
  }
  // If the block is unconditional, newOp has only one result
  return newOp->getResult(0);
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

} // namespace

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

static void producedValueDataflowAnalysis(
    Value producedValue,
    DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap) {
  std::set<Block *> visitedBlocks;
  SmallVector<Value, 4> producedValues;
  producedValues.push_back(producedValue);

  // Check if Value is already added to the map
  if (valueIsConsumedInBlocksMap.find(producedValue) ==
      valueIsConsumedInBlocksMap.end()) {
    std::set<Block *> blockSet;
    valueIsConsumedInBlocksMap[producedValue] = blockSet;
  }

  while (!producedValues.empty()) {
    Value &val = producedValues.front();

    std::set<Operation *> visitedUserOps;
    for (const auto &consumerOp : val.getUsers()) {
      if (visitedUserOps.find(consumerOp) != visitedUserOps.end())
        // If user operation is already visited
        continue;

      visitedUserOps.insert(consumerOp);

      // Check if consumer operation is branch and the Value is not used
      // in any other operation of that block. Note that branch operation
      // is always the terminator operation of a block.
      if (isa<BranchOpInterface>(*consumerOp) &&
          visitedBlocks.find(consumerOp->getBlock()) == visitedBlocks.end()) {

        visitedBlocks.insert(consumerOp->getBlock());

        // Values are passed through a branch to successor blocks.
        // Therefore, the op operands are mapped to the coresponding block
        // arguments.
        if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(consumerOp)) {
          // If Value is used as a condition of a CondBranchOp, it is
          // considered used in the block.
          if (condBranchOp.getCondition() == val) {
            valueIsConsumedInBlocksMap[producedValue].insert(
                consumerOp->getBlock());
            continue;
          }

          // Get block arguments of a true dest block
          for (size_t idx = 0; idx < condBranchOp.getTrueDestOperands().size();
               idx++) {
            BlockArgument argOperand =
                condBranchOp.getTrueDest()->getArgument(idx);
            if (argOperand)
              producedValues.push_back(argOperand);
          }
          // Get block arguments of a false dest block
          for (size_t idx = 0; idx < condBranchOp.getFalseDestOperands().size();
               idx++) {
            BlockArgument argOperand =
                condBranchOp.getFalseDest()->getArgument(idx);
            if (argOperand)
              producedValues.push_back(argOperand);
          }
        } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(consumerOp)) {
          for (size_t idx = 0; idx < branchOp.getOperands().size(); idx++) {
            BlockArgument argOperand = branchOp.getDest()->getArgument(idx);
            if (argOperand)
              producedValues.push_back(argOperand);
          }
        }
      } else if (!isa<BranchOpInterface>(*consumerOp)) {
        // Value is used by a non-branch Operation, so it is used in the
        // Operation's parent Block
        valueIsConsumedInBlocksMap[producedValue].insert(
            consumerOp->getBlock());
      }
    }

    producedValues.erase(&val);
  }
}

// Analysis that returns a data structure that maps each Value to the Blocks
// where it is actually consumed.
static DenseMap<Value, std::set<Block *>> runDataflowAnalysis(Region &r) {
  DenseMap<Value, std::set<Block *>> valueIsConsumedInBlocksMap;

  // Traversing entry block arguments
  for (auto &blockArg : r.getBlocks().front().getArguments())
    producedValueDataflowAnalysis(blockArg, valueIsConsumedInBlocksMap);

  // Traverse operation result values
  for (auto &block : r.getBlocks())
    for (auto &producerOp : block.getOperations())
      for (const auto &producerOpResult : producerOp.getResults())
        producedValueDataflowAnalysis(producerOpResult,
                                      valueIsConsumedInBlocksMap);

  return valueIsConsumedInBlocksMap;
}

// Inserting a branch for a token missmatch prevention merge, and connecting the
// branch output to the merge input.
static void
resolveMergeBackedges(TokenMissmatchMergeOps &preventTokenMissmatchMerges,
                      ConversionPatternRewriter &rewriter) {

  for (auto &m : preventTokenMissmatchMerges) {
    rewriter.setInsertionPointAfter(m.mergeOp);
    auto insertLoc = m.mergeOp->getLoc();

    // Branch condition is the exit condition of the loop. Current
    // implementation supports loop with only one exit.
    Block *exitBlock = m.loop->getExitingBlock();

    Operation *loopExitBranchOp = exitBlock->getTerminator();
    Value condValue = nullptr;
    if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(loopExitBranchOp))
      condValue = condBranchOp.getCondition();
    else
      continue;

    auto condBranch = rewriter.create<handshake::ConditionalBranchOp>(
        insertLoc, condValue, m.mergeOp->getResult(0));

    // Resolve the backedge
    m.mergeBackedge.setValue(getResultForStayingInLoop(
        dyn_cast<mlir::cf::CondBranchOp>(loopExitBranchOp), condBranch,
        m.loop));
  }
}

static void processProducedValues(
    Value producedValue,
    DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap,
    TokenMissmatchMergeOps &preventTokenMissmatchMerges,
    BackedgeBuilder &edgeBuilder, ConversionPatternRewriter &rewriter,
    DenseMap<Block *, BlockLoopInfo> blockToLoopInfoMap,
    DenseMap<Block *, DenseMap<Value, Operation *>> &mapLoopHeaderBlocks) {

  // Saving all producedValue user operations before any merge insertion
  std::set<Operation *> opSet;
  for (const auto &consumerOp : producedValue.getUsers())
    opSet.insert(consumerOp);

  for (const auto &consumerOp : opSet) {
    if (isa<MergeLikeOpInterface>(*consumerOp))
      continue;

    // If producedValue is not used in consumerOp's block, don't insert
    // the merge
    if (valueIsConsumedInBlocksMap[producedValue].find(
            consumerOp->getBlock()) ==
        valueIsConsumedInBlocksMap[producedValue].end())
      continue;

    // Find common loop for producer's and consumer's blocks
    CFGLoop *producersInnermostLoop =
        blockToLoopInfoMap[producedValue.getParentBlock()].loop;
    CFGLoop *consumersInnermostLoop =
        blockToLoopInfoMap[consumerOp->getBlock()].loop;
    CFGLoop *commonLoop =
        findLCALoop(producersInnermostLoop, consumersInnermostLoop);

    int commonLoopDepth = commonLoop ? commonLoop->getLoopDepth() : 0;
    int consumerLoopDepth =
        consumersInnermostLoop ? consumersInnermostLoop->getLoopDepth() : 0;

    int numOfMerges = consumerLoopDepth - commonLoopDepth;

    if (numOfMerges < 0)
      // Token missmatch when token produced multiple times needs to be consumed
      // once
      continue;

    SmallVector<Backedge, 2> prevMergeInputBackedges;
    for (CFGLoop *currLoop = consumersInnermostLoop; currLoop != commonLoop;
         currLoop = currLoop->getParentLoop()) {

      Block *loopHeader = currLoop->getHeader();

      // Check if consumer Block is already added to the map
      auto blockIt = mapLoopHeaderBlocks.find(loopHeader);
      if (blockIt == mapLoopHeaderBlocks.end()) {
        DenseMap<Value, Operation *> dm;
        mapLoopHeaderBlocks[loopHeader] = dm;
      }

      // Check if merge for producers result is already inserted at the
      // beginning of the consumer's block
      auto valueIt = mapLoopHeaderBlocks[loopHeader].find(producedValue);
      if (valueIt != mapLoopHeaderBlocks[loopHeader].end()) {
        // Connect merge for coresponding loop header block to the last added
        // merge in the nested loop, if exists
        if (!prevMergeInputBackedges.empty()) {
          Backedge &backedge = prevMergeInputBackedges.front();
          Operation *mergeOp = mapLoopHeaderBlocks[loopHeader][producedValue];
          backedge.setValue(mergeOp->getResult(0));
          prevMergeInputBackedges.erase(&backedge);
        }
        continue;
      }

      rewriter.setInsertionPointToStart(loopHeader);
      auto insertLoc = loopHeader->front().getLoc();
      SmallVector<Value> operands;

      // Creating a backedge to first input, that might be a
      // producedValue or an output of the mergeOp from outer loop
      auto firstInputBackedge = edgeBuilder.get(producedValue.getType());
      prevMergeInputBackedges.push_back(firstInputBackedge);
      operands.push_back(Value(firstInputBackedge));

      // This backedge needs to be set to a branch output that will be inserted
      // in a separate backedge resolving step.
      auto secondInputBackedge = edgeBuilder.get(producedValue.getType());
      operands.push_back(Value(secondInputBackedge));

      // Create MergeOp
      Operation *mergeOp =
          rewriter.create<handshake::MergeOp>(insertLoc, operands);
      mapLoopHeaderBlocks[loopHeader][producedValue] = mergeOp;

      preventTokenMissmatchMerges.push_back(
          HandshakeLoweringFPL22::TokenMissmatchMergeOp{
              mergeOp, secondInputBackedge, currLoop});

      // Replace uses of producer's operation result in all loop blocks
      // with the merge output
      for (Block *block : currLoop->getBlocks()) {
        if (blockToLoopInfoMap[block].loop != currLoop)
          continue;
        for (Operation &opp : block->getOperations())
          if (!isa<MergeLikeOpInterface>(opp)) {
            opp.replaceUsesOfWith(producedValue, mergeOp->getResult(0));
          }
      }

      // Connect new merge to previous merge, if exists.
      // Note that prevMergeInputBackedges containes backedge from current
      // merge, and optionally the one from the previous merge.
      if (prevMergeInputBackedges.size() > 1) {
        Backedge &backedge = prevMergeInputBackedges.front();
        backedge.setValue(mergeOp->getResult(0));
        prevMergeInputBackedges.erase(&backedge);
      }
    }

    // Connect producer to last added merge
    if (!prevMergeInputBackedges.empty()) {
      Backedge &backedge = prevMergeInputBackedges.front();
      backedge.setValue(producedValue);
      prevMergeInputBackedges.erase(&backedge);
    }
  }
}

TokenMissmatchMergeOps HandshakeLoweringFPL22::handleTokenMissmatch(
    DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap,
    DenseMap<Block *, BlockLoopInfo> &blockToLoopInfoMap,
    BackedgeBuilder &edgeBuilder, ConversionPatternRewriter &rewriter) {

  TokenMissmatchMergeOps preventTokenMissmatchMerges;

  // Each loop header Block should only contain one MergeOp for a Value produced
  // in another Block outside the loop.
  DenseMap<Block *, DenseMap<Value, Operation *>> mapLoopHeaderBlocks;

  // Iterate through all producer-consumer pairs (traversing edges in DFG)
  for (auto &block : r.getBlocks()) {
    // Process Block arguments
    for (auto &blockArg : block.getArguments())
      // MemRef values are handled separately
      if (blockArg.isUsedOutsideOfBlock(&block) &&
          !blockArg.getType().isa<mlir::MemRefType>())
        processProducedValues(
            blockArg, valueIsConsumedInBlocksMap, preventTokenMissmatchMerges,
            edgeBuilder, rewriter, blockToLoopInfoMap, mapLoopHeaderBlocks);

    // Process Values produced in Operations
    for (auto &producerOp : block.getOperations()) {
      if (isa<MergeLikeOpInterface>(producerOp))
        continue;

      for (const auto &producerOpResult : producerOp.getResults())
        // MemRef values are handled separately
        if (!producerOpResult.getType().isa<mlir::MemRefType>())
          processProducedValues(producerOpResult, valueIsConsumedInBlocksMap,
                                preventTokenMissmatchMerges, edgeBuilder,
                                rewriter, blockToLoopInfoMap,
                                mapLoopHeaderBlocks);
    }
  }

  return preventTokenMissmatchMerges;
}

HandshakeLowering::MergeOpInfo
HandshakeLoweringFPL22::insertMerge(Block *block, Value val,
                                    BackedgeBuilder &edgeBuilder,
                                    ConversionPatternRewriter &rewriter) {
  unsigned numPredecessors = getBlockPredecessorCount(block);
  auto insertLoc = block->front().getLoc();
  SmallVector<Backedge> dataEdges;
  SmallVector<Value> operands;

  // Insert "dummy" MergeOp's for blocks with less than two predecessors
  if (numPredecessors == 0) {
    // All of the entry block's block arguments get passed through a dummy
    // MergeOp. There is no need for a backedge here as the unique operand can
    // be resolved immediately
    operands.push_back(val);

    Operation *mergeOp =
        rewriter.create<handshake::MergeOp>(insertLoc, operands);

    // For consistency within the entry block, replace the latter's entry
    // control with the output of a MergeOp that takes the control-only
    // network's start point as input. This makes it so that only the
    // MergeOp's output is used as a control within the entry block, instead
    // of a combination of the MergeOp's output and the function/block control
    // argument. Taking this step out should have no impact on functionality
    // but would make the resulting IR less "regular"
    if (block == &r.front() && val == getBlockEntryControl(block))
      for (auto &block : r.getBlocks())
        setBlockEntryControl(&block, mergeOp->getResult(0));

    // Reconnect entry block arguments in merge operations of other blocks.
    for (Block &b : r)
      if (&b != block)
        for (Operation &opp : b)
          if (isa<MergeLikeOpInterface>(opp))
            opp.replaceUsesOfWith(val, mergeOp->getResult(0));

    return MergeOpInfo{mergeOp, val, dataEdges};
  }
  if (numPredecessors == 1) {
    // The value incoming from the single block predecessor will be resolved
    // later during merge reconnection
    auto edge = edgeBuilder.get(val.getType());
    dataEdges.push_back(edge);
    operands.push_back(Value(edge));

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

    // Inserts SSA merges
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

static bool
isTokenMissmatchMergeOp(TokenMissmatchMergeOps &preventTokenMissmatchMerges,
                        Operation *opp) {
  for (const auto &mergeOpInfo : preventTokenMissmatchMerges)
    if (mergeOpInfo.mergeOp == opp)
      return true;
  return false;
}

static void
reconnectSSAMergeOps(Region &r, HandshakeLowering::BlockOps blockMerges,
                     HandshakeLowering::ValueMap &mergePairs,
                     TokenMissmatchMergeOps &preventTokenMissmatchMerges) {
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

      // Reconnect all operands
      for (Block &b : r)
        for (Operation &opp : b)
          if (!isa<MergeLikeOpInterface>(opp) ||
              isTokenMissmatchMergeOp(preventTokenMissmatchMerges, &opp))
            opp.replaceUsesOfWith(mergeInfo.val, mergeInfo.op->getResult(0));
    }
  }

  removeBlockOperands(r);
}

LogicalResult HandshakeLoweringFPL22::addMergeOps(
    ConversionPatternRewriter &rewriter,
    DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap) {

  // Merge operations added for token missmatch prevention. Backedges will be
  // resolved during the branch insertion for token missmarch prevention merges.
  TokenMissmatchMergeOps preventTokenMissmatchMerges;

  // Create backedge builder to manage operands of merge operations between
  // insertion and reconnection
  BackedgeBuilder edgeBuilder{rewriter, r.front().front().getLoc()};

  // Run loop analysis
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&r);
  // CFGLoop nodes become invalid after CFGLoopInfo is destroyed.
  CFGLoopInfo li(domTree);
  DenseMap<Block *, BlockLoopInfo> blockToLoopInfoMap = findLoopDetails(li, r);

  // Iterate through all producer-consumer pairs and see if a token
  // missmatch occurs. One example of token missmatch is when a producers is
  // outside the loop and consumer is inside. In that case token produced once
  // needs to be consumed multiple times. This is solved by adding merge
  // operations where merge result is also one of the input operands.
  preventTokenMissmatchMerges = handleTokenMissmatch(
      valueIsConsumedInBlocksMap, blockToLoopInfoMap, edgeBuilder, rewriter);

  // Inserting a branch for a token missmatch prevention merge, and connecting
  // the branch output to the merge input.
  resolveMergeBackedges(preventTokenMissmatchMerges, rewriter);

  // Stores mapping from each value that pass through a merge operation to the
  // first result of that merge operation
  ValueMap mergePairs;

  // Insert merge operations (with backedges instead of actual operands)
  BlockOps mergeOps =
      HandshakeLoweringFPL22::insertMergeOps(mergePairs, edgeBuilder, rewriter);

  // Reconnect merge operations with values incoming from predecessor blocks
  // and resolve all backedges that were created during merge insertion
  reconnectSSAMergeOps(r, mergeOps, mergePairs, preventTokenMissmatchMerges);

  return success();
}

LogicalResult HandshakeLoweringFPL22::lowerBranchesToHandshake(
    ConversionPatternRewriter &rewriter,
    DenseMap<Value, std::set<Block *>> &valueIsConsumedInBlocksMap) {

  for (Block &block : r) {
    Operation *termOp = block.getTerminator();
    rewriter.setInsertionPoint(termOp);

    Value condValue = nullptr;
    if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp))
      condValue = condBranchOp.getCondition();
    else if (isa<mlir::func::ReturnOp>(termOp))
      continue;

    // Insert a branch-like operation for every branch operand that will be
    // passed as a block argument and replace the original branch operand value
    // in successor blocks with the result(s) of the new operation
    DenseMap<Value, Operation *> branches;
    for (Value val : getBranchOperands(termOp)) {

      // If producedValue is not used in block, don't insert the branch
      if (valueIsConsumedInBlocksMap[val].find(&block) ==
          valueIsConsumedInBlocksMap[val].end())
        continue;

      // Create a branch-like operation for the branch operand, or re-use one
      // created earlier for that same value
      Operation *newOp = nullptr;
      if (auto branchOp = branches.find(val); branchOp != branches.end())
        newOp = branchOp->getSecond();
      else {
        if (condValue)
          newOp = rewriter.create<handshake::ConditionalBranchOp>(
              termOp->getLoc(), condValue, val);
        else
          newOp = rewriter.create<handshake::BranchOp>(termOp->getLoc(), val);
        branches.insert(std::make_pair(val, newOp));
      }

      for (int j = 0, e = block.getNumSuccessors(); j < e; ++j) {
        Block *succ = block.getSuccessor(j);

        // Look for the merge-like operation in the successor block that takes
        // as input the original branch operand, and replace the latter with a
        // result of the newly inserted branch operation
        for (auto *user : val.getUsers()) {
          if (user->getBlock() == succ &&
              isa<handshake::MergeLikeOpInterface>(user)) {
            user->replaceUsesOfWith(val, getSuccResult(termOp, newOp, succ));
            break;
          }
        }
      }
    }
  }

  return success();
}

/// Lowers the region referenced by the handshake lowering strategy following
/// a fixed sequence of steps, some implemented in this file and some in
/// CIRCT's standard-to-handshake conversion pass.
static LogicalResult lowerRegion(HandshakeLoweringFPL22 &hl,
                                 bool idBasicBlocks) {

  auto &fpga18Hl = static_cast<HandshakeLoweringFPGA18 &>(hl);
  auto &baseHl = static_cast<HandshakeLowering &>(fpga18Hl);

  HandshakeLoweringFPGA18::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(
          fpga18Hl, &HandshakeLoweringFPGA18::replaceMemoryOps, memInfo)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::createStartCtrl)))
    return failure();

  bool sourceConstants = false;
  if (failed(runPartialLowering(baseHl,
                                &HandshakeLowering::connectConstantsToControl,
                                sourceConstants)))
    return failure();

  // Run dataflow analysis
  DenseMap<Value, std::set<Block *>> valueIsConsumedInBlocksMap =
      runDataflowAnalysis(hl.getRegion());

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPL22::addMergeOps,
                                valueIsConsumedInBlocksMap)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::replaceCallOps)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPL22::lowerBranchesToHandshake,
          valueIsConsumedInBlocksMap)))
    return failure();

  if (failed(runPartialLowering(
          fpga18Hl, &HandshakeLoweringFPGA18::connectToMemory, memInfo)))
    return failure();

  if (failed(runPartialLowering(
          fpga18Hl, &HandshakeLoweringFPGA18::replaceUndefinedValues)))
    return failure();

  if (idBasicBlocks && failed(runPartialLowering(
                           fpga18Hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(fpga18Hl,
                                &HandshakeLoweringFPGA18::createReturnNetwork,
                                idBasicBlocks)))
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
