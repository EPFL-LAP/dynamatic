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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <unordered_set>

using namespace mlir;
using namespace dynamatic::experimental::boolean;

namespace dynamatic {
namespace experimental {
namespace ftd {

std::string getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(ftd::getBlockIndex(block));
  return blockCondition;
}

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

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS, possibly traversing `blockToTraverse` and not
/// traversing `blocksToAvoid`.
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

    if (!ignoreDeps && !deps.contains(firstBlock))
      continue;

    // Get last operation of the block, also called `terminator`
    Operation *terminatorOp = firstBlock->getTerminator();

    if (!isa<cf::CondBranchOp>(terminatorOp))
      continue;

    auto blockCondition = getBlockCondition(firstBlock);

    // Get a boolean condition out of the block condition
    boolean::BoolExpression *pathCondition =
        boolean::BoolExpression::parseSop(blockCondition);

    // Possibly add the condition to the list of cofactors
    if (std::find(cofactorList.begin(), cofactorList.end(), blockCondition) ==
        cofactorList.end())
      cofactorList.push_back(blockCondition);

    // Negate the condition if `secondBlock` is reached when the condition
    // is false
    auto condOp = dyn_cast<cf::CondBranchOp>(terminatorOp);
    if (condOp.getFalseDest() == secondBlock)
      pathCondition->boolNegate();

    // And the condition with the rest path
    exp = boolean::BoolExpression::boolAnd(exp, pathCondition);
  }

  return exp;
}

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic
