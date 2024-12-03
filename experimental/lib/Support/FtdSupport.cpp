//===- FtdSupport.cpp - FTD conversion support -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FtdSupport.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/HandshakeSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;

/// Different types of loop suppression.
enum BranchToLoopType {
  MoreProducerThanConsumers,
  SelfRegeneration,
  BackwardRelationship
};

constexpr llvm::StringLiteral FTD_OP_TO_SKIP("ftd.skip");
constexpr llvm::StringLiteral FTD_SUPP_BRANCH("ftd.supp");
constexpr llvm::StringLiteral FTD_EXPLICIT_PHI("ftd.phi");
constexpr llvm::StringLiteral NEW_PHI("nphi");
constexpr llvm::StringLiteral FTD_INIT_MERGE("ftd.imerge");
constexpr llvm::StringLiteral FTD_REGEN("ftd.regen");
constexpr llvm::StringLiteral FTD_REGEN_DONE("ftd.rd");
constexpr llvm::StringLiteral FTD_SUPP_DONE("ftd.sd");

/// Given a block, get its immediate dominator if exists
static Block *getImmediateDominator(Region &region, Block *bb) {
  // Avoid a situation with no blocks in the region
  if (region.getBlocks().empty())
    return nullptr;
  // The first block in the CFG has both non predecessors and no dominators
  if (bb->hasNoPredecessors())
    return nullptr;
  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&region);
  return domTree.getNode(bb)->getIDom()->getBlock();
}

/// Get the dominance frontier of each block in the region
static DenseMap<Block *, DenseSet<Block *>>
getDominanceFrontier(Region &region) {

  // This algorithm comes from the following paper:
  // Cooper, Keith D., Timothy J. Harvey and Ken Kennedy. “AS imple, Fast
  // Dominance Algorithm.” (1999).

  DenseMap<Block *, DenseSet<Block *>> result;

  // Create an empty set of reach available block
  for (Block &bb : region.getBlocks())
    result.insert({&bb, DenseSet<Block *>()});

  for (Block &bb : region.getBlocks()) {

    // Get the predecessors of the block
    auto predecessors = bb.getPredecessors();

    // Count the number of predecessors
    int numberOfPredecessors = 0;
    for (auto *pred : predecessors)
      if (pred)
        numberOfPredecessors++;

    // Skip if the node has none or only one predecessors
    if (numberOfPredecessors < 2)
      continue;

    // Run the algorihm as explained in the paper
    for (auto *pred : predecessors) {
      Block *runner = pred;
      // Runer performs a bottom up traversal of the dominator tree
      while (runner != getImmediateDominator(region, &bb)) {
        result[runner].insert(&bb);
        runner = getImmediateDominator(region, runner);
      }
    }
  }

  return result;
}

/// Get a list of all the loops in which the consumer is but the producer is
/// not, starting from the innermost.
static SmallVector<CFGLoop *> getLoopsConsNotInProd(Block *cons, Block *prod,
                                                    mlir::CFGLoopInfo &li) {
  SmallVector<CFGLoop *> result;

  // Get all the loops in which the consumer is but the producer is
  // not, starting from the innermost
  for (CFGLoop *loop = li.getLoopFor(cons); loop;
       loop = loop->getParentLoop()) {
    if (!loop->contains(prod))
      result.push_back(loop);
  }

  // Reverse to the get the loops from outermost to innermost
  std::reverse(result.begin(), result.end());
  return result;
};

/// Given two sets containing object of type `Block*`, remove the common
/// entries
static void eliminateCommonBlocks(DenseSet<Block *> &s1,
                                  DenseSet<Block *> &s2) {

  std::vector<Block *> intersection;
  for (auto &e1 : s1) {
    if (s2.contains(e1))
      intersection.push_back(e1);
  }

  for (auto &bb : intersection) {
    s1.erase(bb);
    s2.erase(bb);
  }
}

/// Given an operation, returns true if the operation is a conditional branch
/// which terminates a for loop
static bool isBranchLoopExit(Operation *op, CFGLoopInfo &li) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    if (CFGLoop *loop = li.getLoopFor(op->getBlock()); loop) {
      llvm::SmallVector<Block *> exitBlocks;
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
static bool isaMergeLoop(Operation *merge, CFGLoopInfo &li) {

  if (merge->getNumOperands() == 1)
    return false;

  Block *bb1 = merge->getOperand(0).getParentBlock();
  if (merge->getOperand(0).getDefiningOp()) {
    auto *op1 = merge->getOperand(0).getDefiningOp();
    while (llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(op1) &&
           op1->getBlock() == merge->getBlock()) {
      auto op = dyn_cast<handshake::ConditionalBranchOp>(op1);
      if (op.getOperand(1).getDefiningOp()) {
        op1 = op.getOperand(1).getDefiningOp();
        bb1 = op1->getBlock();
      } else {
        break;
      }
    }
  }

  Block *bb2 = merge->getOperand(1).getParentBlock();
  if (merge->getOperand(1).getDefiningOp()) {
    auto *op2 = merge->getOperand(1).getDefiningOp();
    while (llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(op2) &&
           op2->getBlock() == merge->getBlock()) {
      auto op = dyn_cast<handshake::ConditionalBranchOp>(op2);
      if (op.getOperand(1).getDefiningOp()) {
        op2 = op.getOperand(1).getDefiningOp();
        bb2 = op2->getBlock();
      } else {
        break;
      }
    }
  }

  return li.getLoopFor(bb1) != li.getLoopFor(bb2);
}

/// The boolean condition to either generate or suppress a token are computed
/// by considering all the paths from the producer (`start`) to the consumer
/// (`end`). "Each path identifies a Boolean product of elementary conditions
/// expressing the reaching of the target BB from the corresponding member of
/// the set; the product of all such paths are added".
static BoolExpression *enumeratePaths(Block *start, Block *end,
                                      const ftd::BlockIndexing &bi,
                                      const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end, bi);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    DenseSet<unsigned> tempCofactorSet;
    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm =
        getPathExpression(path, tempCofactorSet, bi, controlDeps, false);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

/// Get a boolean expression representing the exit condition of the current
/// loop block
static BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                                 CFGLoopInfo &li,
                                                 const ftd::BlockIndexing &bi) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(bi.getBlockCondition(loopExit));
  auto *terminatorOperation = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(terminatorOperation) &&
         "Terminator condition of a loop exit must be a conditional branch.");
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);

  // If the destination of the false outcome is not the block, then the
  // condition must be negated
  if (li.getLoopFor(condBranch.getFalseDest()) != loop)
    blockCond->boolNegate();

  return blockCond;
}

/// Run the cryton algorithm to determine, give a set of values, in which blocks
/// should we add a merge in order for those values to be merged
static DenseSet<Block *>
runCrytonAlgorithm(Region &funcRegion, DenseMap<Block *, Value> &inputBlocks) {
  // Get dominance frontier
  auto dominanceFrontier = getDominanceFrontier(funcRegion);

  // Temporary data structures to run the Cryton algorithm for phi positioning
  DenseMap<Block *, bool> work;
  DenseMap<Block *, bool> hasAlready;
  SmallVector<Block *> w;

  DenseSet<Block *> result;

  // Initialize data structures to run the Cryton algorithm
  for (auto &bb : funcRegion.getBlocks()) {
    work.insert({&bb, false});
    hasAlready.insert({&bb, false});
  }

  for (auto &[bb, val] : inputBlocks)
    w.push_back(bb), work[bb] = true;

  // Until there are no more elements in `w`
  while (w.size() != 0) {

    // Pop the top of `w`
    auto *x = w.back();
    w.pop_back();

    // Get the dominance frontier of `w`
    auto xFrontier = dominanceFrontier[x];

    // For each of its elements
    for (auto &y : xFrontier) {

      // Add the block in the dominance frontier to the list of blocks which
      // require a new phi. If it was not analyzed yet, also add it to `w`
      if (!hasAlready[y]) {
        result.insert(y);
        hasAlready[y] = true;
        if (!work[y])
          work[y] = true, w.push_back(y);
      }
    }
  }

  return result;
}

FailureOr<DenseMap<Block *, Value>>
ftd::createPhiNetwork(Region &funcRegion, ConversionPatternRewriter &rewriter,
                      SmallVector<Value> &vals) {

  if (vals.empty()) {
    llvm::errs() << "Input of \"createPhiNetwork\" is empty";
    return failure();
  }

  // Type of the inputs
  Type valueType = vals[0].getType();
  // All the input values associated to one block
  DenseMap<Block *, SmallVector<Value>> valuesPerBlock;
  // Associate for each block the value that is dominated by all the others in
  // the same block
  DenseMap<Block *, Value> inputBlocks;
  // Backedge builder to insert new merges
  BackedgeBuilder edgeBuilder(rewriter, funcRegion.getLoc());
  // Backedge corresponding to each phi
  DenseMap<Block *, Backedge> resultPerPhi;
  // Operands of each merge
  DenseMap<Block *, SmallVector<Value>> operandsPerPhi;
  // Which value should be the input of each input value
  DenseMap<Block *, Value> inputPerBlock;

  // Check that all the values have the same type, then collet them according to
  // their input blocks
  for (auto &val : vals) {
    if (val.getType() != valueType) {
      llvm::errs() << "All values must have the same type\n";
      return failure();
    }
    auto *bb = val.getParentBlock();
    valuesPerBlock[bb].push_back(val);
  }

  // Sort the vectors of values in each block according to their dominance and
  // get only the last input value for each block. This is necessary in case in
  // the input sets there is more than one value per blocks
  for (auto &[bb, vals] : valuesPerBlock) {
    mlir::DominanceInfo domInfo;
    std::sort(vals.begin(), vals.end(), [&](Value a, Value b) -> bool {
      if (!a.getDefiningOp())
        return true;
      if (!b.getDefiningOp())
        return false;
      return domInfo.dominates(a.getDefiningOp(), b.getDefiningOp());
    });
    inputBlocks.insert({bb, vals[vals.size() - 1]});
  }

  // In which block a new phi is necessary
  DenseSet<Block *> blocksToAddPhi =
      runCrytonAlgorithm(funcRegion, inputBlocks);

  // A backedge is created for each block in `blocksToAddPhi`, and it will
  // contain the value used as placeholder for the phi
  for (auto &bb : blocksToAddPhi) {
    Backedge mergeResult = edgeBuilder.get(valueType, bb->front().getLoc());
    operandsPerPhi.insert({bb, SmallVector<Value>()});
    resultPerPhi.insert({bb, mergeResult});
  }

  // For each phi, we need one input for every predecessor of the block
  for (auto &bb : blocksToAddPhi) {

    // Avoid to cover a predecessor twice
    llvm::DenseSet<Block *> coveredPred;
    auto predecessors = bb->getPredecessors();

    for (Block *pred : predecessors) {
      if (coveredPred.contains(pred))
        continue;
      coveredPred.insert(pred);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      Block *predecessorOrDominator = nullptr;
      Value valueToUse = nullptr;

      do {
        predecessorOrDominator =
            !predecessorOrDominator
                ? pred
                : getImmediateDominator(funcRegion, predecessorOrDominator);

        if (inputBlocks.contains(predecessorOrDominator))
          valueToUse = inputBlocks[predecessorOrDominator];
        else if (resultPerPhi.contains(predecessorOrDominator))
          valueToUse = resultPerPhi.find(predecessorOrDominator)->getSecond();

      } while (!valueToUse);

      operandsPerPhi[bb].push_back(valueToUse);
    }
  }

  // Create the merge and then replace the values
  DenseMap<Block *, handshake::MergeOp> newMergePerPhi;

  for (auto *bb : blocksToAddPhi) {
    rewriter.setInsertionPointToStart(bb);
    auto mergeOp = rewriter.create<handshake::MergeOp>(bb->front().getLoc(),
                                                       operandsPerPhi[bb]);
    mergeOp->setAttr(NEW_PHI, rewriter.getUnitAttr());
    newMergePerPhi.insert({bb, mergeOp});
  }

  for (auto *bb : blocksToAddPhi)
    resultPerPhi.find(bb)->getSecond().setValue(newMergePerPhi[bb].getResult());

  // For each block, find the incoming value of the network
  for (Block &bb : funcRegion.getBlocks()) {

    Value foundValue = nullptr;
    Block *blockOrDominator = &bb;

    if (blocksToAddPhi.contains(&bb)) {
      inputPerBlock[&bb] = newMergePerPhi[&bb].getResult();
      continue;
    }

    do {
      if (!blockOrDominator->hasNoPredecessors())
        blockOrDominator = getImmediateDominator(funcRegion, blockOrDominator);

      if (inputBlocks.contains(blockOrDominator)) {
        foundValue = inputBlocks[blockOrDominator];
      } else if (blocksToAddPhi.contains(blockOrDominator)) {
        foundValue = newMergePerPhi[blockOrDominator].getResult();
      }

    } while (!foundValue);

    inputPerBlock[&bb] = foundValue;
  }

  return inputPerBlock;
}

void ftd::addRegenOperandConsumer(ConversionPatternRewriter &rewriter,
                                  handshake::FuncOp &funcOp,
                                  Operation *consumerOp, Value operand) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  auto startValue = (Value)funcOp.getArguments().back();

  // Skip if the consumer was added by this function, if it is an init merge, if
  // it comes from the explicit phi process or if it is an operation to skip
  if (consumerOp->hasAttr(FTD_REGEN) || consumerOp->hasAttr(FTD_EXPLICIT_PHI) ||
      consumerOp->hasAttr(FTD_INIT_MERGE) ||
      consumerOp->hasAttr(FTD_OP_TO_SKIP))
    return;

  // Skip if the consumer has to do with memory operations, c-merge networks or
  // if it is a conditional branch.
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(consumerOp))
    return;

  mlir::Operation *producerOp = operand.getDefiningOp();

  // Skip if the producer was added by this function or if it is an op to skip
  if (producerOp &&
      (producerOp->hasAttr(FTD_REGEN) || producerOp->hasAttr(FTD_OP_TO_SKIP)))
    return;

  // Skip if the producer has to do with memory operations
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(producerOp) ||
      llvm::isa_and_nonnull<MemRefType>(operand.getType()))
    return;

  // Last regenerated value
  Value regeneratedValue = operand;

  // Get all the loops for which we need to regenerate the
  // corresponding value
  SmallVector<CFGLoop *> loops = getLoopsConsNotInProd(
      consumerOp->getBlock(), operand.getParentBlock(), loopInfo);

  auto cstType = rewriter.getIntegerType(1);
  auto cstAttr = IntegerAttr::get(cstType, 0);
  unsigned numberOfLoops = loops.size();

  // For each of the loop, from the outermost to the innermost
  for (unsigned i = 0; i < numberOfLoops; i++) {

    // If we are in the innermost loop (thus the iterator is at its end)
    // and the consumer is a loop merge, stop
    if (i == numberOfLoops - 1 && consumerOp->hasAttr(NEW_PHI))
      break;

    // Add the merge to the network, by substituting the operand with
    // the output of the merge, and forwarding the output of the merge
    // to its inputs.
    //
    rewriter.setInsertionPointToStart(loops[i]->getHeader());

    // The type of the input must be channelified
    regeneratedValue.setType(channelifyType(regeneratedValue.getType()));

    // Create an INIT merge to provide the select of the multiplexer
    auto constOp = rewriter.create<handshake::ConstantOp>(consumerOp->getLoc(),
                                                          cstAttr, startValue);
    constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
    Value conditionValue =
        loops[i]->getExitingBlock()->getTerminator()->getOperand(0);

    SmallVector<Value> mergeOperands;
    mergeOperands.push_back(constOp.getResult());
    mergeOperands.push_back(conditionValue);
    auto initMergeOp = rewriter.create<handshake::MergeOp>(consumerOp->getLoc(),
                                                           mergeOperands);
    initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

    // Create the multiplexer
    auto selectSignal = initMergeOp->getResult(0);
    selectSignal.setType(channelifyType(selectSignal.getType()));

    SmallVector<Value> muxOperands;
    muxOperands.push_back(regeneratedValue);
    muxOperands.push_back(regeneratedValue);

    auto muxOp = rewriter.create<handshake::MuxOp>(regeneratedValue.getLoc(),
                                                   regeneratedValue.getType(),
                                                   selectSignal, muxOperands);

    // The new producer operand is the output of the multiplxer
    regeneratedValue = muxOp.getResult();

    // Set the output of the mux as its input as well
    muxOp->setOperand(2, muxOp->getResult(0));
    muxOp->setAttr(FTD_REGEN, rewriter.getUnitAttr());
  }

  consumerOp->replaceUsesOfWith(operand, regeneratedValue);
}

/// Get a value out of the input boolean expression
static Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                                   experimental::boolean::BoolExpression *expr,
                                   Block *block, const ftd::BlockIndexing &bi) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  auto condition =
      bi.getBlockFromCondition(singleCond->id)->getTerminator()->getOperand(0);
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        ftd::channelifyType(condition.getType()), condition);
    notOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
    return notOp->getResult(0);
  }
  condition.setType(ftd::channelifyType(condition.getType()));
  return condition;
}

/// Get a circuit out a boolean expression, depending on the different kinds
/// of expressions you might have
static Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                     BoolExpression *expr, Block *block,
                                     const ftd::BlockIndexing &bi) {

  // Variable case
  if (expr->type == ExpressionType::Variable)
    return boolVariableToCircuit(rewriter, expr, block, bi);

  // Constant case (either 0 or 1)
  rewriter.setInsertionPointToStart(block);
  auto sourceOp = rewriter.create<handshake::SourceOp>(
      block->getOperations().front().getLoc());
  Value cnstTrigger = sourceOp.getResult();

  auto intType = rewriter.getIntegerType(1);
  auto cstAttr = rewriter.getIntegerAttr(
      intType, (expr->type == ExpressionType::One ? 1 : 0));

  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->getOperations().front().getLoc(), cstAttr, cnstTrigger);

  constOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return constOp.getResult();
}

/// Convert a `BDD` object as obtained from the bdd expansion to a
/// circuit
static Value bddToCircuit(ConversionPatternRewriter &rewriter, BDD *bdd,
                          Block *block, const ftd::BlockIndexing &bi) {
  if (!bdd->inputs.has_value())
    return boolExpressionToCircuit(rewriter, bdd->boolVariable, block, bi);

  rewriter.setInsertionPointToStart(block);

  // Get the two operands by recursively calling `bddToCircuit` (it possibly
  // creates other muxes in a hierarchical way)
  SmallVector<Value> muxOperands;
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().first, block, bi));
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().second, block, bi));
  Value muxCond =
      boolExpressionToCircuit(rewriter, bdd->boolVariable, block, bi);

  // Create the multiplxer and add it to the rest of the circuit
  auto muxOp = rewriter.create<handshake::MuxOp>(
      block->getOperations().front().getLoc(), muxCond, muxOperands);
  muxOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return muxOp.getResult();
}

using PairOperandConsumer = std::pair<Value, Operation *>;

/// Insert a branch to the correct position, taking into account whether it
/// should work to suppress the over-production of tokens or self-regeneration
static Value addSuppressionInLoop(ConversionPatternRewriter &rewriter,
                                  CFGLoop *loop, Operation *consumer,
                                  Value connection, BranchToLoopType btlt,
                                  CFGLoopInfo &li,
                                  std::vector<PairOperandConsumer> &toCover,
                                  const ftd::BlockIndexing &bi) {

  handshake::ConditionalBranchOp branchOp;

  // Case in which there is only one termination block
  if (Block *loopExit = loop->getExitingBlock(); loopExit) {

    // Do not add the branch in case of a while loop with backward edge
    if (btlt == BackwardRelationship &&
        bi.greaterIndex(connection.getParentBlock(), loopExit))
      return connection;

    // Get the termination operation, which is supposed to be conditional
    // branch.
    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    // A conditional branch is now to be added next to the loop terminator, so
    // that the token can be suppressed
    auto *exitCondition = getBlockLoopExitCondition(loopExit, loop, li, bi);
    auto conditionValue =
        boolVariableToCircuit(rewriter, exitCondition, loopExit, bi);

    rewriter.setInsertionPointToStart(loopExit);

    // Since only one output is used, the other one will be connected to sink
    // in the materialization pass, as we expect from a suppress branch
    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().back().getLoc(),
        ftd::getBranchResultTypes(connection.getType()), conditionValue,
        connection);

  } else {

    std::vector<std::string> cofactorList;
    SmallVector<Block *> exitBlocks;
    loop->getExitingBlocks(exitBlocks);
    loopExit = exitBlocks.front();

    BoolExpression *fLoopExit = BoolExpression::boolZero();

    // Get the list of all the cofactors related to possible exit conditions
    for (Block *exitBlock : exitBlocks) {
      BoolExpression *blockCond =
          getBlockLoopExitCondition(exitBlock, loop, li, bi);
      fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
      cofactorList.push_back(bi.getBlockCondition(exitBlock));
    }

    // Sort the cofactors alphabetically
    std::sort(cofactorList.begin(), cofactorList.end());

    // Apply a BDD expansion to the loop exit expression and the list of
    // cofactors
    BDD *bdd = buildBDD(fLoopExit, cofactorList);

    // Convert the boolean expression obtained through bdd to a circuit
    Value branchCond = bddToCircuit(rewriter, bdd, loopExit, bi);

    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    rewriter.setInsertionPointToStart(loopExit);

    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().front().getLoc(),
        ftd::getBranchResultTypes(connection.getType()), branchCond,
        connection);
  }

  Value newConnection = btlt == MoreProducerThanConsumers
                            ? branchOp.getTrueResult()
                            : branchOp.getFalseResult();

  // If we are handling a case with more producers than consumers, the new
  // branch must undergo the `addSupp` function so we add it to our structure
  // to be able to loop over it
  if (btlt == MoreProducerThanConsumers) {
    branchOp->setAttr(FTD_SUPP_BRANCH, rewriter.getUnitAttr());
    toCover.emplace_back(newConnection, consumer);
  }

  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
static void insertDirectSuppression(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    Operation *consumer, Value connection, const ftd::BlockIndexing &bi,
    ControlDependenceAnalysis::BlockControlDepsMap &cdAnalysis) {

  Block *entryBlock = &funcOp.getBody().front();
  Block *producerBlock = connection.getParentBlock();
  Block *consumerBlock = consumer->getBlock();

  // Get the control dependencies from the producer
  DenseSet<Block *> prodControlDeps =
      cdAnalysis[producerBlock].forwardControlDeps;

  // Get the control dependencies from the consumer
  DenseSet<Block *> consControlDeps =
      cdAnalysis[consumer->getBlock()].forwardControlDeps;

  // Get rid of common entries in the two sets
  eliminateCommonBlocks(prodControlDeps, consControlDeps);

  // Compute the activation function of producer and consumer
  BoolExpression *fProd =
      enumeratePaths(entryBlock, producerBlock, bi, prodControlDeps);
  BoolExpression *fCons =
      enumeratePaths(entryBlock, consumerBlock, bi, consControlDeps);

  /// f_supp = f_prod and not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fSup, cofactorList);
    Value branchCond = bddToCircuit(rewriter, bdd, consumer->getBlock(), bi);

    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), ftd::getBranchResultTypes(connection.getType()),
        branchCond, connection);

    // Take into account the possiblity of a mux to get the condition input also
    // as data input. In this case, a branch needs to be created, but only the
    // corresponding data input is affected. The conditions below take into
    // account this possibility.
    for (auto &use : connection.getUses()) {
      if (use.getOwner() != consumer)
        continue;
      if (llvm::isa<handshake::MuxOp>(consumer) && use.getOperandNumber() == 0)
        continue;
      use.set(branchOp.getFalseResult());
    }
    connection = branchOp.getFalseResult();
  }

  // The condition related to the select signal of the consumer mux must be
  // added if the following conditions hold: The consumer is a mux; The
  // mux was a GAMMA from GSA analysis; The input of the mux (i.e., coming
  // from the producer) is a data input.
  if (llvm::isa<handshake::MuxOp>(consumer) &&
      consumer->hasAttr(FTD_EXPLICIT_PHI) &&
      (consumer->getOperand(1) == connection ||
       consumer->getOperand(2) == connection) &&
      consumer->getOperand(0).getParentBlock() != consumer->getBlock() &&
      consumer->getBlock() != producerBlock) {

    auto selectOperand = consumer->getOperand(0);
    BoolExpression *selectOperandCondition = BoolExpression::parseSop(
        bi.getBlockCondition(selectOperand.getDefiningOp()->getBlock()));

    // The condition must be taken into account for `fCons` only if the
    // producer is not control dependent from the block which produces the
    // condition of the mux
    if (!prodControlDeps.contains(selectOperand.getParentBlock())) {
      if (consumer->getOperand(1) == connection)
        selectOperandCondition = selectOperandCondition->boolNegate();

      std::set<std::string> blocks = selectOperandCondition->getVariables();
      std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
      BDD *bdd = buildBDD(selectOperandCondition, cofactorList);
      Value branchCond = bddToCircuit(rewriter, bdd, consumer->getBlock(), bi);

      rewriter.setInsertionPointToStart(consumer->getBlock());
      auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
          consumer->getLoc(), ftd::getBranchResultTypes(connection.getType()),
          branchCond, connection);

      for (auto &use : connection.getUses()) {
        if (use.getOwner() != consumer)
          continue;
        if (llvm::isa<handshake::MuxOp>(consumer) &&
            use.getOperandNumber() == 0)
          continue;
        use.set(branchOp.getTrueResult());
      }
    }
  }
}

void ftd::addSuppOperandConsumer(ConversionPatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp,
                                 Operation *consumerOp, Value operand) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  BlockIndexing bi(region);
  auto cda = ControlDependenceAnalysis(region).getAllBlockDeps();

  // Skip the prod-cons if the producer is part of the operations related to
  // the BDD expansion or INIT merges
  if (consumerOp->hasAttr(FTD_OP_TO_SKIP) ||
      consumerOp->hasAttr(FTD_INIT_MERGE))
    return;

  // Do not take into account conditional branch
  if (llvm::isa<handshake::ConditionalBranchOp>(consumerOp))
    return;

  Block *consumerBlock = consumerOp->getBlock();
  Block *producerBlock = operand.getParentBlock();

  // If the consumer and the producer are in the same block without the
  // consumer being a multiplxer skip because no delivery is needed
  if (consumerBlock == producerBlock &&
      !llvm::isa<handshake::MuxOp>(consumerOp))
    return;

  if (Operation *producerOp = operand.getDefiningOp(); producerOp) {

    // Skip the prod-cons if the consumer is part of the operations
    // related to the BDD expansion or INIT merges
    if (producerOp->hasAttr(FTD_OP_TO_SKIP) ||
        producerOp->hasAttr(FTD_INIT_MERGE))
      return;

    // TODO: Group the conditions of memory and the conditions of Branches
    // in 1 function?
    // Skip if either the producer of the consumer are
    // related to memory operations, or if the consumer is a conditional
    // branch
    if (llvm::isa_and_nonnull<handshake::MemoryControllerOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::MemoryControllerOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::LSQOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::LSQOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::ControlMergeOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(consumerOp) ||
        llvm::isa_and_nonnull<cf::CondBranchOp>(consumerOp) ||
        llvm::isa_and_nonnull<cf::BranchOp>(consumerOp) ||
        (llvm::isa<memref::LoadOp>(consumerOp) &&
         !llvm::isa<handshake::LoadOp>(consumerOp)) ||
        (llvm::isa<memref::StoreOp>(consumerOp) &&
         !llvm::isa<handshake::StoreOp>(consumerOp)) ||
        llvm::isa<mlir::MemRefType>(operand.getType()))
      return;

    // The next step is to identify the relationship between the producer
    // and consumer in hand: Are they in the same loop or at different
    // loop levels? Are they connected through a bwd edge?

    // Set true if the producer is in a loop which does not contains
    // the consumer
    bool producingGtUsing =
        loopInfo.getLoopFor(producerBlock) &&
        !loopInfo.getLoopFor(producerBlock)->contains(consumerBlock);

    auto *consumerLoop = loopInfo.getLoopFor(consumerBlock);
    std::vector<PairOperandConsumer> newToCover;

    // Set to true if the consumer uses its own result
    bool selfRegeneration =
        llvm::any_of(consumerOp->getResults(),
                     [&operand](const Value &v) { return v == operand; });

    // We need to suppress all the tokens produced within a loop and
    // used outside each time the loop is not terminated. This should be
    // done for as many loops there are
    if (producingGtUsing && !isBranchLoopExit(producerOp, loopInfo)) {
      Value con = operand;
      for (CFGLoop *loop = loopInfo.getLoopFor(producerBlock); loop;
           loop = loop->getParentLoop()) {

        // For each loop containing the producer but not the consumer, add
        // the branch
        if (!loop->contains(consumerBlock))
          con = addSuppressionInLoop(rewriter, loop, consumerOp, con,
                                     MoreProducerThanConsumers, loopInfo,
                                     newToCover, bi);
      }

      for (auto &pair : newToCover)
        addSuppOperandConsumer(rewriter, funcOp, pair.second, pair.first);

      return;
    }

    // We need to suppress a token if the consumer is the producer itself
    // within a loop
    if (selfRegeneration && consumerLoop &&
        !producerOp->hasAttr(FTD_SUPP_BRANCH)) {
      addSuppressionInLoop(rewriter, consumerLoop, consumerOp, operand,
                           SelfRegeneration, loopInfo, newToCover, bi);
      return;
    }

    // We need to suppress a token if the consumer comes before the
    // producer (backward edge)
    if ((bi.greaterIndex(producerBlock, consumerBlock) ||
         (llvm::isa<handshake::MuxOp>(consumerOp) &&
          producerBlock == consumerBlock &&
          isaMergeLoop(consumerOp, loopInfo))) &&
        consumerLoop) {
      addSuppressionInLoop(rewriter, consumerLoop, consumerOp, operand,
                           BackwardRelationship, loopInfo, newToCover, bi);
      return;
    }
  }

  // Handle the suppression in all the other cases (inlcuding the operand being
  // a function arguement)
  insertDirectSuppression(rewriter, funcOp, consumerOp, operand, bi, cda);
}

void ftd::addSupp(handshake::FuncOp &funcOp,
                  ConversionPatternRewriter &rewriter) {

  // Set of original operations in the IR
  std::vector<Operation *> consumersToCover;
  for (Operation &consumerOp : funcOp.getOps())
    consumersToCover.push_back(&consumerOp);

  for (auto *consumerOp : consumersToCover) {
    if (consumerOp->hasAttr(FTD_SUPP_DONE))
      continue;
    consumerOp->setAttr(FTD_SUPP_DONE, rewriter.getUnitAttr());

    for (auto operand : consumerOp->getOperands())
      addSuppOperandConsumer(rewriter, funcOp, consumerOp, operand);
  }
}

void ftd::addRegen(handshake::FuncOp &funcOp,
                   ConversionPatternRewriter &rewriter) {

  // Set of original operations in the IR
  std::vector<Operation *> consumersToCover;
  for (Operation &consumerOp : funcOp.getOps())
    consumersToCover.push_back(&consumerOp);

  // For each producer/consumer relationship
  for (Operation *consumerOp : consumersToCover) {
    if (consumerOp->hasAttr(FTD_REGEN_DONE))
      continue;
    consumerOp->setAttr(FTD_REGEN_DONE, rewriter.getUnitAttr());

    for (Value operand : consumerOp->getOperands())
      addRegenOperandConsumer(rewriter, funcOp, consumerOp, operand);
  }
}

LogicalResult experimental::ftd::addGsaGates(
    Region &region, ConversionPatternRewriter &rewriter,
    const gsa::GSAAnalysis &gsa, Backedge startValue, bool removeTerminators) {

  using namespace experimental::gsa;

  // The function instantiates the GAMMA and MU gates as provided by the GSA
  // analysis pass. A GAMMA function is translated into a multiplxer driven by
  // single control signal and fed by two operands; a MU function is
  // translated into a multiplxer driven by an init (it is currently
  // implemented as a Merge fed by a constant triggered from Start once and
  // from the loop condition thereafter). The input of one of these functions
  // might be another GSA function, and it's possible that the function was
  // not instantiated yet. For this reason, we keep track of the missing
  // operands, and reconnect them later on.
  //
  // To simplify the way GSA functions are handled, each of them has an unique
  // index.

  struct MissingGsa {
    // Index of the GSA function to modify
    unsigned phiIndex;
    // Index of the GSA function providing the result
    unsigned edgeIndex;
    // Index of the operand to modify
    unsigned operandInput;

    MissingGsa(unsigned pi, unsigned ei, unsigned oi)
        : phiIndex(pi), edgeIndex(ei), operandInput(oi) {}
  };

  if (region.getBlocks().size() == 1)
    return success();

  // List of missing GSA functions
  SmallVector<MissingGsa> missingGsaList;
  // List of gammas with only one input
  DenseSet<Operation *> oneInputGammaList;
  // Maps the index of each GSA function to each real operation
  DenseMap<unsigned, Operation *> gsaList;

  // For each block excluding the first one, which has no gsa
  for (Block &block : llvm::drop_begin(region)) {

    // For each GSA function
    ArrayRef<Gate *> phis = gsa.getGatesPerBlock(&block);
    for (Gate *phi : phis) {

      // TODO: No point of this skipping if we have a guarantee that the
      // phi->gsaGateFunction is exclusively either MuGate or GammaGate...
      // Skip if it's an SSA phi that has more than 2 inputs (not yet broken
      // down to multiple GSA gates)
      if (phi->gsaGateFunction == PhiGate)
        continue;

      Location loc = block.front().getLoc();
      rewriter.setInsertionPointToStart(&block);
      SmallVector<Value> operands;

      // Maintain the index of the current operand
      unsigned operandIndex = 0;
      // Checks whether one index is empty
      int nullOperand = -1;

      // For each of its operand
      for (auto *operand : phi->operands) {
        // If the input is another GSA function, then a dummy value is used as
        // operand and the operations will be reconnected later on.
        // If the input is empty, we keep track of its index.
        // In the other cases, we already have the operand of the function.
        if (operand->isTypeGate()) {
          Gate *g = std::get<Gate *>(operand->input);
          operands.emplace_back(g->result);
          missingGsaList.emplace_back(
              MissingGsa(phi->index, g->index, operandIndex));
        } else if (operand->isTypeEmpty()) {
          nullOperand = operandIndex;
          operands.emplace_back(nullptr);
        } else {
          auto val = std::get<Value>(operand->input);
          operands.emplace_back(val);
        }
        operandIndex++;
      }

      // The condition value is provided by the `condition` field of the phi
      rewriter.setInsertionPointAfterValue(phi->result);
      Value conditionValue =
          phi->conditionBlock->getTerminator()->getOperand(0);

      // If the function is MU, then we create a merge
      // and use its result as condition
      if (phi->gsaGateFunction == MuGate) {
        mlir::DominanceInfo domInfo;
        mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

        // The inputs of the merge are the condition value and a `false`
        // constant driven by the start value of the function. This will
        // created later on, so we use a dummy value.
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(conditionValue);
        mergeOperands.push_back(conditionValue);

        auto initMergeOp =
            rewriter.create<handshake::MergeOp>(loc, mergeOperands);

        initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

        // Replace the new condition value
        conditionValue = initMergeOp->getResult(0);
        conditionValue.setType(channelifyType(conditionValue.getType()));

        // Add the activation constant driven by the backedge value, which will
        // be then updated with the real start value, once available
        auto cstType = rewriter.getIntegerType(1);
        auto cstAttr = IntegerAttr::get(cstType, 0);
        rewriter.setInsertionPointToStart(initMergeOp->getBlock());
        auto constOp = rewriter.create<handshake::ConstantOp>(
            initMergeOp->getLoc(), cstAttr, startValue);
        constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
        initMergeOp->setOperand(0, constOp.getResult());
      }

      // When a single input gamma is encountered, a mux is inserted as a
      // placeholder to perform the gamma/mu allocation flow. In the end,
      // these muxes are erased from the IR
      if (nullOperand >= 0) {
        operands[0] = operands[1 - nullOperand];
        operands[1] = operands[1 - nullOperand];
      }

      // Create the multiplexer
      auto mux = rewriter.create<handshake::MuxOp>(loc, phi->result.getType(),
                                                   conditionValue, operands);

      // The one input gamma is marked at an operation to skip in the IR and
      // later removed
      if (nullOperand >= 0)
        oneInputGammaList.insert(mux);

      if (phi->isRoot)
        rewriter.replaceAllUsesWith(phi->result, mux.getResult());

      gsaList.insert({phi->index, mux});
      mux->setAttr(FTD_EXPLICIT_PHI, rewriter.getUnitAttr());
    }
  }

  // For each of the GSA missing inputs, perform a replacement
  for (auto &missingMerge : missingGsaList) {

    auto *operandMerge = gsaList[missingMerge.phiIndex];
    auto *resultMerge = gsaList[missingMerge.edgeIndex];

    operandMerge->setOperand(missingMerge.operandInput + 1,
                             resultMerge->getResult(0));

    // In case of a one-input gamma, the other input must be replaced as well,
    // to avoid errors when the block arguments are erased later on
    if (oneInputGammaList.contains(operandMerge))
      operandMerge->setOperand(2 - missingMerge.operandInput,
                               resultMerge->getResult(0));
  }

  // Get rid of the multiplexers adopted as place-holders of one input gamma
  for (auto &op : llvm::make_early_inc_range(oneInputGammaList)) {
    int operandToUse = llvm::isa_and_nonnull<handshake::MuxOp>(
                           op->getOperand(1).getDefiningOp())
                           ? 1
                           : 2;
    op->getResult(0).replaceAllUsesWith(op->getOperand(operandToUse));
    rewriter.eraseOp(op);
  }

  if (!removeTerminators)
    return success();

  // Remove all the block arguments for all the non starting blocks
  for (Block &block : llvm::drop_begin(region))
    block.eraseArguments(0, block.getArguments().size());

  // Each terminator must be replaced so that it does not provide any block
  // arguments (possibly only the final control argument)
  for (Block &block : region) {
    if (Operation *terminator = block.getTerminator(); terminator) {
      rewriter.setInsertionPointAfter(terminator);
      if (auto cbr = dyn_cast<cf::CondBranchOp>(terminator); cbr) {
        while (!cbr.getTrueOperands().empty())
          cbr.eraseTrueOperand(0);
        while (!cbr.getFalseOperands().empty())
          cbr.eraseFalseOperand(0);
      } else if (auto br = dyn_cast<cf::BranchOp>(terminator); br) {
        while (!br.getOperands().empty())
          br.eraseOperand(0);
      }
    }
  }

  return success();
}

LogicalResult ftd::replaceMergeToGSA(handshake::FuncOp funcOp,
                                     ConversionPatternRewriter &rewriter) {
  auto startValue = (Value)funcOp.getArguments().back();

  // Create a backedge for the start value, to be sued during the merges to
  // muxes conversion
  BackedgeBuilder edgeBuilderStart(rewriter, funcOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());

  // For each merge that was signed with the `NEW_PHI` attribute, substitute
  // it with its GSA equivalent
  for (handshake::MergeOp merge : funcOp.getOps<handshake::MergeOp>()) {
    if (!merge->hasAttr(NEW_PHI))
      continue;
    gsa::GSAAnalysis gsa(merge, funcOp.getRegion());
    if (failed(ftd::addGsaGates(funcOp.getRegion(), rewriter, gsa,
                                startValueBackedge, false)))
      return failure();

    // Get rid of the merge
    rewriter.eraseOp(merge);
  }

  // Replace the backedge
  startValueBackedge.setValue(startValue);

  return success();
}
