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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;

unsigned ftd::getBlockIndex(Block *bb) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  bb->printAsOperand(os1);
  std::string block1id = os1.str();
  return std::stoi(block1id.substr(3));
}

bool ftd::lessThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) < getBlockIndex(block2);
}

bool ftd::greaterThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) > getBlockIndex(block2);
}

bool ftd::isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || isSameLoop(loop1->getParentLoop(), loop2) ||
          isSameLoop(loop1, loop2->getParentLoop()) ||
          isSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

bool ftd::isSameLoopBlocks(Block *source, Block *dest,
                           const mlir::CFGLoopInfo &li) {
  return isSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

std::string ftd::getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(ftd::getBlockIndex(block));
  return blockCondition;
}

bool ftd::isHandhsakeLSQOperation(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

void ftd::eliminateCommonBlocks(DenseSet<Block *> &s1, DenseSet<Block *> &s2) {

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

/// Helper recursive function to get the innermost common loop
static CFGLoop *checkInnermostCommonLoop(CFGLoop *loop1, CFGLoop *loop2) {

  // None of them is a loop
  if (!loop1 || !loop2)
    return nullptr;

  // Same loop
  if (loop1 == loop2)
    return loop1;

  // Check whether the parent loop of `loop1` is `loop2`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop1->getParentLoop(), loop2); pl)
    return pl;

  // Check whether the parent loop of `loop2` is `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(), loop1); pl)
    return pl;

  // Check whether the parent loop of `loop1` is identical to the parent loop
  // of `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(),
                                             loop1->getParentLoop());
      pl)
    return pl;

  // If no common loop is found, return nullptr
  return nullptr;
}

CFGLoop *ftd::getInnermostCommonLoop(Block *block1, Block *block2,
                                     mlir::CFGLoopInfo &li) {
  return checkInnermostCommonLoop(li.getLoopFor(block1), li.getLoopFor(block2));
}

bool ftd::isBranchLoopExit(Operation *op, CFGLoopInfo &li) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    if (CFGLoop *loop = li.getLoopFor(op->getBlock()); loop) {
      llvm::SmallVector<Block *> exitBlocks;
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS
static void dfsAllPaths(Block *start, Block *end, std::vector<Block *> &path,
                        std::unordered_set<Block *> &visited,
                        std::vector<std::vector<Block *>> &allPaths,
                        Block *blockToTraverse,
                        const std::vector<Block *> &blocksToAvoid,
                        bool blockToTraverseFound) {

  // The current block is part of the current path
  path.push_back(start);
  // The current block has been visited
  visited.insert(start);

  bool blockFound = (!blockToTraverse || start == blockToTraverse);

  // If we are at the end of the path, then add it to the list of paths
  if (start == end && (blockFound || blockToTraverseFound)) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (Block *successor : start->getSuccessors()) {

      // Do not run DFS if the successor is in the list of blocks to traverse
      bool incorrectPath = false;
      for (auto *toAvoid : blocksToAvoid) {
        if (toAvoid == successor &&
            ftd::getBlockIndex(toAvoid) > ftd::getBlockIndex(blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      if (visited.find(successor) == visited.end()) {
        dfsAllPaths(successor, end, path, visited, allPaths, blockToTraverse,
                    blocksToAvoid, blockFound || blockToTraverseFound);
      }
    }
  }

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

/// Recursive function which allows to obtain all the paths from operation
/// `start` to operation `end` using a DFS
static void dfsAllPaths(Operation *current, Operation *end,
                        std::unordered_set<Operation *> &visited,
                        std::vector<Operation *> &path,
                        std::vector<std::vector<Operation *>> &allPaths) {
  visited.insert(current);
  path.push_back(current);

  if (current == end) {
    // If the current operation is the end, add the path to allPaths
    allPaths.push_back(path);
  } else {
    // Otherwise, explore the successors
    for (auto result : current->getResults()) {
      for (auto *successor : result.getUsers()) {
        if (visited.find(successor) == visited.end()) {
          dfsAllPaths(successor, end, visited, path, allPaths);
        }
      }
    }
  }

  // Backtrack
  path.pop_back();
  visited.erase(current);
}

std::vector<std::vector<Operation *>> ftd::findAllPaths(Operation *start,
                                                        Operation *end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::unordered_set<Operation *> visited;
  std::vector<Operation *> path;
  dfsAllPaths(start, end, visited, path, allPaths);
  return allPaths;
}

std::vector<std::vector<Block *>>
ftd::findAllPaths(Block *start, Block *end, Block *blockToTraverse,
                  ArrayRef<Block *> blocksToAvoid) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, blockToTraverse,
              blocksToAvoid, false);
  return allPaths;
}

/// Helper recursive function for getPostDominantSuccessor
static Block *getPostDominantSuccessor(Block *prod, Block *cons,
                                       std::unordered_set<Block *> &visited,
                                       PostDominanceInfo &postDomInfo) {

  // If the producer is not valid, return, otherwise insert it among the
  // visited ones.
  if (!prod)
    return nullptr;

  visited.insert(prod);

  // For each successor of the producer
  for (Block *successor : prod->getSuccessors()) {

    // Check if the successor post-dominates cons
    if (successor != cons && postDomInfo.postDominates(successor, cons))
      return successor;

    // If not visited, recursively search successors of the current successor
    if (visited.find(successor) == visited.end()) {
      Block *result =
          getPostDominantSuccessor(successor, cons, visited, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

Block *ftd::getPostDominantSuccessor(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  PostDominanceInfo postDomInfo;
  return ::getPostDominantSuccessor(prod, cons, visited, postDomInfo);
}

/// Helper recursive function for getPredecessorDominatingAndPostDominating
static Block *getPredecessorDominatingAndPostDominating(
    Block *producer, Block *consumer, std::unordered_set<Block *> &visited,
    DominanceInfo &domInfo, PostDominanceInfo &postDomInfo) {

  // If the consumer is not valid, return, otherwise insert it in the visited
  // ones
  if (!consumer)
    return nullptr;
  visited.insert(consumer);

  // For each predecessor of the consumer
  for (Block *predecessor : consumer->getPredecessors()) {

    // If the current predecessor is not the producer itself, and this block
    // both dominates the consumer and post-dominates the producer, return it
    if (predecessor != producer &&
        postDomInfo.postDominates(predecessor, producer) &&
        domInfo.dominates(predecessor, consumer))
      return predecessor;

    // If not visited, recursively search predecessors of the current
    // predecessor
    if (visited.find(predecessor) == visited.end()) {
      Block *result = getPredecessorDominatingAndPostDominating(
          producer, predecessor, visited, domInfo, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

Block *ftd::getPredecessorDominatingAndPostDominating(Block *prod,
                                                      Block *cons) {
  std::unordered_set<Block *> visited;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  return ::getPredecessorDominatingAndPostDominating(prod, cons, visited,
                                                     domInfo, postDomInfo);
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool ftd::isaMergeLoop(Operation *merge, CFGLoopInfo &li) {

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

boolean::BoolExpression *
ftd::getPathExpression(ArrayRef<Block *> path,
                       DenseSet<unsigned> &blockIndexSet,
                       const DenseMap<Block *, unsigned> &mapBlockToIndex,
                       const DenseSet<Block *> &deps, const bool ignoreDeps) {

  // Start with a boolean expression of one
  boolean::BoolExpression *exp = boolean::BoolExpression::boolOne();

  // Cover each pair of adjacent blocks
  unsigned pathSize = path.size();
  for (unsigned i = 0; i < pathSize - 1; i++) {
    Block *firstBlock = path[i];
    Block *secondBlock = path[i + 1];

    // Skip pair if the first block has only one successor, thus no conditional
    // branch
    if (firstBlock->getSuccessors().size() == 1)
      continue;

    if (ignoreDeps || deps.contains(firstBlock)) {

      // Get last operation of the block, also called `terminator`
      Operation *terminatorOp = firstBlock->getTerminator();

      if (isa<cf::CondBranchOp>(terminatorOp)) {
        unsigned blockIndex = mapBlockToIndex.lookup(firstBlock);
        std::string blockCondition = "c" + std::to_string(blockIndex);

        // Get a boolean condition out of the block condition
        boolean::BoolExpression *pathCondition =
            boolean::BoolExpression::parseSop(blockCondition);

        // Possibly add the condition to the list of cofactors
        if (!blockIndexSet.contains(blockIndex))
          blockIndexSet.insert(blockIndex);

        // Negate the condition if `secondBlock` is reached when the condition
        // is false
        auto condOp = dyn_cast<cf::CondBranchOp>(terminatorOp);
        if (condOp.getFalseDest() == secondBlock)
          pathCondition->boolNegate();

        // And the condition with the rest path
        exp = boolean::BoolExpression::boolAnd(exp, pathCondition);
      }
    }
  }

  // Minimize the condition and return
  return exp;
}

BoolExpression *
ftd::enumeratePaths(Block *start, Block *end,
                    const DenseMap<Block *, unsigned> &mapBlockToIndex,
                    const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    DenseSet<unsigned> tempCofactorSet;
    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm = getPathExpression(
        path, tempCofactorSet, mapBlockToIndex, controlDeps, false);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

Type ftd::channelifyType(Type type) {
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

BoolExpression *ftd::getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                               CFGLoopInfo &li) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(getBlockCondition(loopExit));
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

SmallVector<Type> ftd::getBranchResultTypes(Type inputType) {
  SmallVector<Type> handshakeResultTypes;
  handshakeResultTypes.push_back(channelifyType(inputType));
  handshakeResultTypes.push_back(channelifyType(inputType));
  return handshakeResultTypes;
}

Block *ftd::getImmediateDominator(Region &region, Block *bb) {
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

DenseMap<Block *, DenseSet<Block *>> ftd::getDominanceFrontier(Region &region) {

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

#define PRINT_DEBUG
#ifdef PRINT_DEBUG
  for (auto &entry : result) {
    llvm::dbgs() << "[DOM FRONT] Domination frontier of ";
    entry.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ": ";
    llvm::dbgs() << "{";
    for (auto &dom : entry.second) {
      dom->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " ";
    }
    llvm::dbgs() << "}\n";
  }
#endif

  return result;
}

FailureOr<DenseMap<Block *, Value>>
ftd::insertPhi(Region &funcRegion, ConversionPatternRewriter &rewriter,
               SmallVector<Value> &vals) {

  auto dominanceFrontier = getDominanceFrontier(funcRegion);

  // The number of values to be considered cannot be empty
  if (vals.empty())
    return funcRegion.getParentOp()->emitError()
           << "The number values provided in `insertPhi` "
              "must be larger than 2\n";

  // All the values provided must have the same type.
  // As an additional constraint, all the values should be in a different basic
  // block
  DenseSet<Block *> foundBlocks;
  for (auto &val : vals) {
    if (val.getType() != vals[0].getType())
      return funcRegion.getParentOp()->emitError()
             << "The values provided to `addPhi` do not all have the same type";
    if (foundBlocks.contains(val.getParentBlock()))
      return funcRegion.getParentOp()->emitError()
             << "Some of the values provided to `addPhi` "
                "belong to the same basic block";
    foundBlocks.insert(val.getParentBlock());
  }

  llvm::dbgs() << "[NEW PHI] Producers in: {";
  for (auto &val : vals) {
    val.getParentBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " ";
  }
  llvm::dbgs() << "}\n";

  // Temporary data structures to run the Cryton algorithm for phi positioning
  DenseMap<Block *, bool> work;
  DenseMap<Block *, bool> hasAlready;
  SmallVector<Block *> w;

  // Initialize data structures
  for (auto &bb : funcRegion.getBlocks()) {
    work.insert({&bb, false});
    hasAlready.insert({&bb, false});
  }

  for (auto val : vals) {
    w.push_back(val.getParentBlock());
    work[val.getParentBlock()] = true;
  }

  // This vector ends up containig the blocks in which a new argument is to be
  // added
  DenseSet<Block *> blocksToAddPhi;

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
        blocksToAddPhi.insert(y);
        hasAlready[y] = true;
        if (!work[y])
          work[y] = true, w.push_back(y);
      }
    }
  }

  // Get a location to insert the new phis
  auto loc = UnknownLoc::get(vals[0].getContext());
  for (auto &bb : blocksToAddPhi)
    bb->addArgument(vals[0].getType(), loc);

  llvm::dbgs() << "[NEW PHI] Insertion in { ";
  for (auto &bb : blocksToAddPhi) {
    bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " ";
  }
  llvm::dbgs() << "}\n";

  if (blocksToAddPhi.empty())
    return success();

  // Since a new block argument was added for each block in `blocksToAddPhi`,
  // new values must be provided to them together with the branches (either
  // conditional or non-conditional). The values might come from one of the
  // input operations or from another of the added block arguments. In order to
  // find the correct value, we first anlayzed the predecessor of each node: if
  // it has a redefinition of the value, then we use it, otherwise we move the
  // anlaysis to its immediate dominator. Since a definition of the value must
  // always exist, BB0 must define the value as well.

  for (auto &bb : blocksToAddPhi) {

    // For each bb in `blocksToAddPhi`, we need to modify the terminator of each
    // of its predecessors
    auto predecessors = bb->getPredecessors();

    for (auto *pred : predecessors) {

      auto *terminator = pred->getTerminator();
      rewriter.setInsertionPointAfter(terminator);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      auto *predecessorOrDominator = pred;

      Value valueToUse = nullptr;

      while (valueToUse == nullptr) {

        // For each of the values provided as input
        for (auto &val : vals) {

          // If the block of the current `predecessorOrDominator` contains a
          // definition of the value, then we use it in the terminator
          if (val.getParentBlock() == predecessorOrDominator) {
            valueToUse = val;
            break;
          }
        }

        if (valueToUse == nullptr) {
          // Go through the blocks having a new arugment for the value
          for (auto &phibb : blocksToAddPhi) {
            if (predecessorOrDominator == phibb) {
              valueToUse = phibb->getArgument(phibb->getNumArguments() - 1);
              break;
            }
          }
        }

        if (valueToUse) {

          // Case in which the terminator is a branch
          if (llvm::isa_and_nonnull<cf::BranchOp>(terminator)) {
            auto branch = cast<cf::BranchOp>(terminator);
            SmallVector<Value> operands = branch.getDestOperands();
            operands.push_back(valueToUse);
            auto newBranch = rewriter.create<cf::BranchOp>(
                branch->getLoc(), branch.getDest(), operands);
            rewriter.replaceOp(branch, newBranch);
          }

          // Case in which the terminator is a conditional branch
          if (llvm::isa_and_nonnull<cf::CondBranchOp>(terminator)) {
            auto branch = cast<cf::CondBranchOp>(terminator);
            SmallVector<Value> trueOperands = branch.getTrueDestOperands();
            SmallVector<Value> falseOperands = branch.getFalseDestOperands();

            if (branch.getTrueDest() == bb)
              trueOperands.push_back(valueToUse);
            else
              falseOperands.push_back(valueToUse);

            auto newBranch = rewriter.create<cf::CondBranchOp>(
                branch->getLoc(), branch.getCondition(), branch.getTrueDest(),
                trueOperands, branch.getFalseDest(), falseOperands);
            rewriter.replaceOp(branch, newBranch);
            break;
          }
        }

        // Terminate if the value was found
        if (valueToUse != nullptr ||
            predecessorOrDominator->hasNoPredecessors())
          break;

        // Move to the immediate dominator
        predecessorOrDominator =
            getImmediateDominator(funcRegion, predecessorOrDominator);
      }

      if (!valueToUse)
        return funcRegion.getParentOp()->emitError()
               << "A branch could not be modified, because no definition of "
                  "the value was found\n";
    }
  }

  DenseMap<Block *, Value> result;

  for (auto &bb : funcRegion.getBlocks()) {

    if (blocksToAddPhi.contains(&bb)) {
      result.insert({&bb, bb.getArgument(bb.getNumArguments() - 1)});
      continue;
    }

    auto predecessors = bb.getPredecessors();

    for (auto *pred : predecessors) {

      auto *terminator = pred->getTerminator();
      rewriter.setInsertionPointAfter(terminator);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      auto *predecessorOrDominator = pred;

      Value valueToUse = nullptr;

      while (valueToUse == nullptr) {

        // For each of the values provided as input
        for (auto &val : vals) {

          // If the block of the current `predecessorOrDominator` contains a
          // definition of the value, then we use it in the terminator
          if (val.getParentBlock() == predecessorOrDominator) {
            valueToUse = val;
            break;
          }
        }

        if (valueToUse == nullptr) {
          // Go through the blocks having a new arugment for the value
          for (auto &phibb : blocksToAddPhi) {
            if (predecessorOrDominator == phibb) {
              valueToUse = phibb->getArgument(phibb->getNumArguments() - 1);
              break;
            }
          }
        }

        if (valueToUse)
          result.insert({&bb, valueToUse});

        // Terminate if the value was found
        if (valueToUse != nullptr ||
            predecessorOrDominator->hasNoPredecessors())
          break;

        // Move to the immediate dominator
        predecessorOrDominator =
            getImmediateDominator(funcRegion, predecessorOrDominator);
      }

      if (!valueToUse)
        return funcRegion.getParentOp()->emitError()
               << "Cannot find defintion of a value for a block";
    }
  }

#define PRINT_DEBUG
#ifdef PRINT_DEBUG
  for (auto &entry : result) {
    llvm::dbgs() << "[PHI RESULT] In ";
    entry.first->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " value is ";
    entry.second.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
#endif

  return result;
}

SmallVector<CFGLoop *> ftd::getLoopsConsNotInProd(Block *cons, Block *prod,
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

LogicalResult ftd::addRegenToConsumer(ConversionPatternRewriter &rewriter,
                                      handshake::FuncOp &funcOp,
                                      Operation *consumerOp) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  auto startValue = (Value)funcOp.getArguments().back();

  // Skip if the consumer was added by this function, if it is an init merge, if
  // it comes from the explicit phi process or if it is an operation to skip
  if (consumerOp->hasAttr(FTD_REGEN) || consumerOp->hasAttr(FTD_EXPLICIT_PHI) ||
      consumerOp->hasAttr(FTD_INIT_MERGE) ||
      consumerOp->hasAttr(FTD_OP_TO_SKIP))
    return success();

  // Skip if the consumer has to do with memory operations or with che C-network
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp))
    return success();

  // Consider all the operands of the consumer
  for (Value operand : consumerOp->getOperands()) {

    mlir::Operation *producerOp = operand.getDefiningOp();

    // Skip if the producer was added by this function or if it is an op to skip
    if (producerOp &&
        (producerOp->hasAttr(FTD_REGEN) || producerOp->hasAttr(FTD_OP_TO_SKIP)))
      continue;

    // Skip if the producer has to do with memory operations
    if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(producerOp) ||
        llvm::isa_and_nonnull<MemRefType>(operand.getType()))
      continue;

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
      if (i == numberOfLoops - 1 && consumerOp->hasAttr(FTD_MEM_DEP))
        break;

      // Add the merge to the network, by substituting the operand with
      // the output of the merge, and forwarding the output of the merge
      // to its inputs.
      //
      rewriter.setInsertionPointToStart(loops[i]->getHeader());

      // The type of the input must be channelified
      regeneratedValue.setType(channelifyType(regeneratedValue.getType()));

      // Create an INIT merge to provide the select of the multiplexer
      auto constOp = rewriter.create<handshake::ConstantOp>(
          consumerOp->getLoc(), cstAttr, startValue);
      constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
      Value conditionValue =
          loops[i]->getExitingBlock()->getTerminator()->getOperand(0);

      SmallVector<Value> mergeOperands;
      mergeOperands.push_back(constOp.getResult());
      mergeOperands.push_back(conditionValue);
      auto initMergeOp = rewriter.create<handshake::MergeOp>(
          consumerOp->getLoc(), mergeOperands);
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

  return success();
}

dynamatic::experimental::ftd::BlockIndexing::BlockIndexing(Region &region) {
  mlir::DominanceInfo domInfo;

  // Create a vector with all the blocks
  SmallVector<Block *> allBlocks;
  for (Block &bb : region.getBlocks())
    allBlocks.push_back(&bb);

  // Sort the vector according to the dominance information
  std::sort(allBlocks.begin(), allBlocks.end(),
            [&](Block *a, Block *b) { return domInfo.dominates(a, b); });

  // Associate a smalled index in the map to the blocks at higer levels of the
  // dominance tree
  unsigned bbIndex = 0;
  for (Block *bb : allBlocks)
    blockIndexing.insert({bbIndex++, bb});
}

Block *
dynamatic::experimental::ftd::BlockIndexing::getBlockFromIndex(unsigned index) {
  auto it = blockIndexing.find(index);
  return (it == blockIndexing.end()) ? nullptr : it->getSecond();
}

Block *dynamatic::experimental::ftd::BlockIndexing::getBlockFromCondition(
    const std::string &condition) {
  std::string conditionNumber = condition;
  conditionNumber.erase(0, 1);
  unsigned index = std::stoi(conditionNumber);
  return this->getBlockFromIndex(index);
}

unsigned
dynamatic::experimental::ftd::BlockIndexing::getIndexFromBlock(Block *bb) {
  for (auto const &[i, b] : blockIndexing) {
    if (bb == b)
      return i;
  }
  return -1;
}

bool dynamatic::experimental::ftd::BlockIndexing::greaterIndex(Block *bb1,
                                                               Block *bb2) {
  return getIndexFromBlock(bb1) > getIndexFromBlock(bb2);
}

/// Get a value out of the input boolean expression
static Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                                   experimental::boolean::BoolExpression *expr,
                                   Block *block, ftd::BlockIndexing &bi) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  auto condition =
      bi.getBlockFromCondition(singleCond->id)->getTerminator()->getOperand(0);
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        ftd::channelifyType(condition.getType()), condition);
    notOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());
    return notOp->getResult(0);
  }
  condition.setType(ftd::channelifyType(condition.getType()));
  return condition;
}

/// Get a circuit out a boolean expression, depending on the different kinds
/// of expressions you might have
static Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                     BoolExpression *expr, Block *block,
                                     ftd::BlockIndexing &bi) {

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

  constOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return constOp.getResult();
}

/// Convert a `BDD` object as obtained from the bdd expansion to a
/// circuit
static Value bddToCircuit(ConversionPatternRewriter &rewriter, BDD *bdd,
                          Block *block, ftd::BlockIndexing &bi) {
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
  muxOp->setAttr(ftd::FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return muxOp.getResult();
}

/// Insert a branch to the correct position, taking into account whether it
/// should work to suppress the over-production of tokens or self-regeneration
static Value addSuppressionInLoop(ConversionPatternRewriter &rewriter,
                                  CFGLoop *loop, Operation *consumer,
                                  Value connection, ftd::BranchToLoopType btlt,
                                  CFGLoopInfo &li,
                                  std::vector<Operation *> &producersToCover,
                                  ftd::BlockIndexing &bi) {

  handshake::ConditionalBranchOp branchOp;

  // Case in which there is only one termination block
  if (Block *loopExit = loop->getExitingBlock(); loopExit) {

    // Do not add the branch in case of a while loop with backward edge
    if (btlt == ftd::BackwardRelationship &&
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
    auto *exitCondition = ftd::getBlockLoopExitCondition(loopExit, loop, li);
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
          ftd::getBlockLoopExitCondition(exitBlock, loop, li);
      fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
      cofactorList.push_back(ftd::getBlockCondition(exitBlock));
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

  // If we are handling a case with more producers than consumers, the new
  // branch must undergo the `addSupp` function so we add it to our structure
  // to be able to loop over it
  if (btlt == ftd::MoreProducerThanConsumers) {
    branchOp->setAttr(ftd::FTD_SUPP_BRANCH, rewriter.getUnitAttr());
    producersToCover.push_back(branchOp);
  }

  Value newConnection = btlt == ftd::MoreProducerThanConsumers
                            ? branchOp.getTrueResult()
                            : branchOp.getFalseResult();

  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
static LogicalResult insertDirectSuppression(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    Operation *consumer, Value connection, ftd::BlockIndexing &bi,
    ControlDependenceAnalysis::BlockControlDepsMap &cdAnalysis) {

  Block *entryBlock = &funcOp.getBody().front();
  Block *producerBlock = connection.getParentBlock();

  DenseMap<Block *, unsigned> indexPerBlock;
  for (auto &bb : funcOp.getBlocks()) {
    indexPerBlock.insert({&bb, ftd::getBlockIndex(&bb)});
  }

  // Get the control dependencies from the producer
  DenseSet<Block *> prodControlDeps =
      cdAnalysis[producerBlock].forwardControlDeps;

  // Get the control dependencies from the consumer
  DenseSet<Block *> consControlDeps =
      cdAnalysis[consumer->getBlock()].forwardControlDeps;

  // Get rid of common entries in the two sets
  ftd::eliminateCommonBlocks(prodControlDeps, consControlDeps);

  // Compute the activation function of producer and consumer
  BoolExpression *fProd = ftd::enumeratePaths(entryBlock, producerBlock,
                                              indexPerBlock, prodControlDeps);
  BoolExpression *fCons = ftd::enumeratePaths(entryBlock, consumer->getBlock(),
                                              indexPerBlock, consControlDeps);

  // The condition related to the select signal of the consumer mux must be
  // added if the following conditions hold: The consumer is a mux; The
  // mux was a GAMMA from GSA analysis; The input of the mux (i.e., coming
  // from the producer) is a data input.
  if (llvm::isa_and_nonnull<handshake::MuxOp>(consumer) &&
      consumer->hasAttr(ftd::FTD_EXPLICIT_PHI) &&
      consumer->getOperand(0) != connection &&
      consumer->getOperand(0).getParentBlock() != consumer->getBlock() &&
      consumer->getBlock() != producerBlock) {

    auto selectOperand = consumer->getOperand(0);
    BoolExpression *selectOperandCondition = BoolExpression::parseSop(
        ftd::getBlockCondition(selectOperand.getDefiningOp()->getBlock()));

    // The condition must be taken into account for `fCons` only if the
    // producer is not control dependent from the block which produces the
    // condition of the mux
    if (!prodControlDeps.contains(selectOperand.getParentBlock())) {
      if (consumer->getOperand(1) == connection)
        fCons = BoolExpression::boolAnd(fCons,
                                        selectOperandCondition->boolNegate());
      else
        fCons = BoolExpression::boolAnd(fCons, selectOperandCondition);
    }
  }

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
    consumer->replaceUsesOfWith(connection, branchOp.getFalseResult());
  }

  return success();
}

LogicalResult
ftd::addSuppToProducer(ConversionPatternRewriter &rewriter,
                       handshake::FuncOp &funcOp, Operation *producerOp,
                       ftd::BlockIndexing &bi,
                       std::vector<Operation *> &producersToCover,
                       ControlDependenceAnalysis::BlockControlDepsMap &cda) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  Block *producerBlock = producerOp->getBlock();

  // Skip the prod-cons if the producer is part of the operations related to
  // the BDD expansion or INIT merges
  if (producerOp->hasAttr(ftd::FTD_OP_TO_SKIP) ||
      producerOp->hasAttr(ftd::FTD_INIT_MERGE))
    return success();

  // Consider all the consumers of each value of the producer
  for (Value result : producerOp->getResults()) {

    std::vector<Operation *> users(result.getUsers().begin(),
                                   result.getUsers().end());
    users.erase(unique(users.begin(), users.end()), users.end());

    for (Operation *consumerOp : users) {
      Block *consumerBlock = consumerOp->getBlock();

      // If the consumer and the producer are in the same block without the
      // consumer being a multiplxer skip because no delivery is needed
      if (consumerBlock == producerBlock && !isa<handshake::MuxOp>(consumerOp))
        continue;

      // Skip the prod-cons if the consumer is part of the operations
      // related to the BDD expansion or INIT merges
      if (consumerOp->hasAttr(ftd::FTD_OP_TO_SKIP) ||
          consumerOp->hasAttr(ftd::FTD_INIT_MERGE))
        continue;

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
          llvm::isa<handshake::ConditionalBranchOp>(consumerOp) ||
          llvm::isa<cf::CondBranchOp>(consumerOp) ||
          llvm::isa<cf::BranchOp>(consumerOp) ||
          (llvm::isa<memref::LoadOp>(consumerOp) &&
           !llvm::isa<handshake::LSQLoadOp>(consumerOp)) ||
          (llvm::isa<memref::StoreOp>(consumerOp) &&
           !llvm::isa<handshake::LSQStoreOp>(consumerOp)) ||
          (llvm::isa<memref::LoadOp>(consumerOp) &&
           !llvm::isa<handshake::MCLoadOp>(consumerOp)) ||
          (llvm::isa<memref::StoreOp>(consumerOp) &&
           !llvm::isa<handshake::MCStoreOp>(consumerOp)) ||
          llvm::isa<mlir::MemRefType>(result.getType()))
        continue;

      // The next step is to identify the relationship between the producer
      // and consumer in hand: Are they in the same loop or at different
      // loop levels? Are they connected through a bwd edge?

      // Set true if the producer is in a loop which does not contains
      // the consumer
      bool producingGtUsing =
          loopInfo.getLoopFor(producerBlock) &&
          !loopInfo.getLoopFor(producerBlock)->contains(consumerBlock);

      auto *consumerLoop = loopInfo.getLoopFor(consumerBlock);

      // Set to true if the consumer uses its own result
      bool selfRegeneration =
          llvm::any_of(consumerOp->getResults(),
                       [&result](const Value &v) { return v == result; });

      // We need to suppress all the tokens produced within a loop and
      // used outside each time the loop is not terminated. This should be
      // done for as many loops there are
      if (producingGtUsing && !ftd::isBranchLoopExit(producerOp, loopInfo)) {
        Value con = result;
        for (CFGLoop *loop = loopInfo.getLoopFor(producerBlock); loop;
             loop = loop->getParentLoop()) {

          // For each loop containing the producer but not the consumer, add
          // the branch
          if (!loop->contains(consumerBlock))
            con = addSuppressionInLoop(rewriter, loop, consumerOp, con,
                                       ftd::MoreProducerThanConsumers, loopInfo,
                                       producersToCover, bi);
        }
      }

      // We need to suppress a token if the consumer is the producer itself
      // within a loop
      else if (selfRegeneration && consumerLoop &&
               !producerOp->hasAttr(ftd::FTD_SUPP_BRANCH)) {
        addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                             ftd::SelfRegeneration, loopInfo, producersToCover,
                             bi);
      }

      // We need to suppress a token if the consumer comes before the
      // producer (backward edge)
      else if ((bi.greaterIndex(producerBlock, consumerBlock) ||
                (isa<handshake::MuxOp>(consumerOp) &&
                 producerBlock == consumerBlock &&
                 ftd::isaMergeLoop(consumerOp, loopInfo))) &&
               consumerLoop) {
        addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                             ftd::BackwardRelationship, loopInfo,
                             producersToCover, bi);
      }

      // If no loop is involved, then there is a direct relationship between
      // consumer and producer
      else if (failed(insertDirectSuppression(rewriter, funcOp, consumerOp,
                                              result, bi, cda)))
        return failure();
    }
  }

  // Once that we have considered all the consumers of the results of a
  // producer, we consider the operands of the producer. Some of these
  // operands might be the arguments of the functions, and these might need
  // to be suppressed as well.

  // Do not take into account conditional branch
  if (llvm::isa<handshake::ConditionalBranchOp>(producerOp))
    return success();

  // For all the operands of the operation, take into account only the
  // start value if exists
  for (Value operand : producerOp->getOperands()) {
    // The arguments of a function do not have a defining operation
    if (operand.getDefiningOp())
      continue;

    // Skip if we are in block 0 and no multiplexer is involved
    if (operand.getParentBlock() == producerBlock &&
        !isa<handshake::MuxOp>(producerOp))
      continue;

    // Handle the suppression
    if (failed(insertDirectSuppression(rewriter, funcOp, producerOp, operand,
                                       bi, cda)))
      return failure();
  }

  return success();
}
