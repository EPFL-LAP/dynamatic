//===- FtdSupport.cpp - FTD conversion support -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA analysis pass. All the functions are about
// analyzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

std::string ftd::BlockIndexing::getBlockCondition(Block *block) const {
  return "c" + std::to_string(getIndexFromBlock(block).value_or(0));
}

ftd::BlockIndexing::BlockIndexing(Region &region) {
  mlir::DominanceInfo domInfo;

  // Create a vector with all the blocks
  SmallVector<Block *> allBlocks;
  for (Block &bb : region.getBlocks())
    allBlocks.push_back(&bb);

  // Sort the vector according to the dominance information, so that a block
  // comes before each dominators.
  llvm::sort(allBlocks.begin(), allBlocks.end(),
             [&](Block *a, Block *b) { return domInfo.dominates(a, b); });

  // Associate a smalled index in the map to the blocks at higer levels of the
  // dominance tree
  for (auto [blockID, bb] : llvm::enumerate(allBlocks)) {
    indexToBlock.insert({blockID, bb});
    blockToIndex.insert({bb, blockID});
  }
}

std::optional<Block *>
ftd::BlockIndexing::getBlockFromIndex(unsigned index) const {
  auto it = indexToBlock.find(index);
  if (it == indexToBlock.end())
    return {};
  return it->getSecond();
}

std::optional<Block *>
ftd::BlockIndexing::getBlockFromCondition(StringRef condition) const {
  std::string conditionNumber = condition.str();
  conditionNumber.erase(0, 1);
  StringRef conditionRef = conditionNumber;
  unsigned index = 0;
  if (conditionRef.getAsInteger(0, index))
    return {};
  return getBlockFromIndex(index);
}

std::optional<unsigned> ftd::BlockIndexing::getIndexFromBlock(Block *bb) const {
  auto it = blockToIndex.find(bb);
  if (it == blockToIndex.end())
    return {};
  return it->getSecond();
}

bool dynamatic::experimental::ftd::BlockIndexing::isGreater(Block *bb1,
                                                            Block *bb2) const {
  auto index1 = getIndexFromBlock(bb1);
  auto index2 = getIndexFromBlock(bb2);
  if (!index1.has_value() || !index2.has_value())
    return false;
  return index1 > index2;
}

bool ftd::BlockIndexing::isLess(Block *bb1, Block *bb2) const {
  auto index1 = getIndexFromBlock(bb1);
  auto index2 = getIndexFromBlock(bb2);
  if (!index1.has_value() || !index2.has_value())
    return false;
  return index1 < index2;
}

/// Recursively check whether 2 blocks belong to the same loop, starting
/// from the inner-most loops.
static bool isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
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

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS.
static void dfsAllPaths(Block *start, Block *end, std::vector<Block *> &path,
                        std::unordered_set<Block *> &visited,
                        std::vector<std::vector<Block *>> &allPaths,
                        Block *blockToTraverse,
                        const std::vector<Block *> &blocksToAvoid,
                        const ftd::BlockIndexing &bi,
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
        if (toAvoid == successor && bi.isGreater(toAvoid, blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      if (!visited.count(successor)) {
        dfsAllPaths(successor, end, path, visited, allPaths, blockToTraverse,
                    blocksToAvoid, bi, blockFound || blockToTraverseFound);
      }
    }
  }

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

std::vector<std::vector<Block *>>
ftd::findAllPaths(Block *start, Block *end, const BlockIndexing &bi,
                  Block *blockToTraverse, ArrayRef<Block *> blocksToAvoid) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, blockToTraverse,
              blocksToAvoid, bi, false);
  return allPaths;
}

boolean::BoolExpression *
ftd::getPathExpression(ArrayRef<Block *> path,
                       DenseSet<unsigned> &blockIndexSet,
                       const BlockIndexing &bi, const DenseSet<Block *> &deps,
                       const bool ignoreDeps) {

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
        auto blockIndexOptional = bi.getIndexFromBlock(firstBlock);
        if (!blockIndexOptional.has_value()) {
          llvm::errs() << "The block index of a block cannot be obtained\n";
          continue;
        }
        unsigned blockIndex = blockIndexOptional.value();
        std::string blockCondition = bi.getBlockCondition(firstBlock);

        // Get a boolean condition out of the block condition
        boolean::BoolExpression *pathCondition =
            boolean::BoolExpression::parseSop(blockCondition);

        // Possibly add the condition to the list of cofactors
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

SmallVector<Type> ftd::getListTypes(Type inputType, unsigned size) {
  return SmallVector<Type>(size, channelifyType(inputType));
}
