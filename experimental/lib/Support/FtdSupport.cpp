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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic::experimental::boolean;

namespace dynamatic {
namespace experimental {
namespace ftd {

int getBlockIndex(Block *bb) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  bb->printAsOperand(os1);
  std::string block1id = os1.str();
  return std::stoi(block1id.substr(3));
}

bool lessThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) < getBlockIndex(block2);
}

bool greaterThanBlocks(Block *block1, Block *block2) {
  return getBlockIndex(block1) > getBlockIndex(block2);
}

bool isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || isSameLoop(loop1->getParentLoop(), loop2) ||
          isSameLoop(loop1, loop2->getParentLoop()) ||
          isSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li) {
  return isSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

std::string getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(ftd::getBlockIndex(block));
  return blockCondition;
}

bool isHandhsakeLSQOperation(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

void eliminateCommonBlocks(DenseSet<Block *> &s1, DenseSet<Block *> &s2) {

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

CFGLoop *getInnermostCommonLoop(Block *block1, Block *block2,
                                mlir::CFGLoopInfo &li) {
  return checkInnermostCommonLoop(li.getLoopFor(block1), li.getLoopFor(block2));
}

bool isBranchLoopExit(Operation *op, CFGLoopInfo &li) {
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
            getBlockIndex(toAvoid) > getBlockIndex(blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      if (visited.find(successor) == visited.end())
        dfsAllPaths(successor, end, path, visited, allPaths, blockToTraverse,
                    blocksToAvoid, blockFound || blockToTraverseFound);
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

std::vector<std::vector<Operation *>> findAllPaths(Operation *start,
                                                   Operation *end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::unordered_set<Operation *> visited;
  std::vector<Operation *> path;
  dfsAllPaths(start, end, visited, path, allPaths);
  return allPaths;
}

std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, Block *blockToTraverse,
             const std::vector<Block *> &blocksToAvoid) {
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

Block *getPostDominantSuccessor(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  PostDominanceInfo postDomInfo;
  return getPostDominantSuccessor(prod, cons, visited, postDomInfo);
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

Block *getPredecessorDominatingAndPostDominating(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  return getPredecessorDominatingAndPostDominating(prod, cons, visited, domInfo,
                                                   postDomInfo);
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool isaMergeLoop(Operation *merge, CFGLoopInfo &li) {

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
getPathExpression(const std::vector<Block *> &path,
                  std::vector<std::string> &cofactorList,
                  const DenseSet<Block *> &deps, const bool ignoreDeps) {

  // Start with a boolean expression of one
  boolean::BoolExpression *exp = boolean::BoolExpression::boolOne();

  // Cover each pair of adjacent blocks
  for (int i = 0; i < (int)path.size() - 1; i++) {
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
        auto blockCondition = getBlockCondition(firstBlock);

        // Get a boolean condition out of the block condition
        boolean::BoolExpression *pathCondition =
            boolean::BoolExpression::parseSop(blockCondition);

        // Possibly add the condition to the list of cofactors
        if (std::find(cofactorList.begin(), cofactorList.end(),
                      blockCondition) == cofactorList.end())
          cofactorList.push_back(blockCondition);

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

BoolExpression *enumeratePaths(Block *start, Block *end,
                               const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    std::vector<std::string> tempCofactorList;
    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm =
        getPathExpression(path, tempCofactorList, controlDeps, false);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

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

BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
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

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic
