//===- SortBlocksInProgramOrder.cpp - Sort blocks ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the block sorting pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include <algorithm>
#include <utility>

using namespace mlir;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_SORTBLOCKSINPROGRAMORDER
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

/// Returns the blocks of the region in reverse postorder, starting from its
/// entry block. Blocks which the entry cannot reach come last.
static SmallVector<Block *> getBlocksInProgramOrder(Region &region) {
  SmallVector<Block *> programOrder;
  if (region.empty())
    return programOrder;

  // Iterative DFS, each entry of the stack holding a block together with the
  // index of its next successor to explore
  DenseSet<Block *> visited;
  SmallVector<std::pair<Block *, unsigned>> stack;
  Block *entry = &region.front();
  visited.insert(entry);
  stack.push_back({entry, 0});

  while (!stack.empty()) {
    Block *block = stack.back().first;
    unsigned nextSuccessor = stack.back().second;

    if (nextSuccessor < block->getNumSuccessors()) {
      stack.back().second = nextSuccessor + 1;
      Block *successor = block->getSuccessor(nextSuccessor);
      if (visited.insert(successor).second)
        stack.push_back({successor, 0});
      continue;
    }

    // All the successors were explored, thus the block is in postorder position
    programOrder.push_back(block);
    stack.pop_back();
  }

  // The reverse of the postorder is the wanted order
  std::reverse(programOrder.begin(), programOrder.end());

  for (Block &block : region.getBlocks())
    if (!visited.contains(&block))
      programOrder.push_back(&block);

  return programOrder;
}

/// Sorts the blocks of a function in program order.
static void sortBlocksInProgramOrder(func::FuncOp funcOp) {
  SmallVector<Block *> order = getBlocksInProgramOrder(funcOp.getBody());

  // Moving each block right before the next one, from the last to the first,
  // leaves the region in that same order
  for (size_t idx = order.size(); idx-- > 1;)
    order[idx - 1]->moveBefore(order[idx]);
}

namespace {

/// Simple driver for the block sorting pass. Runs the pass on every function in
/// the module independently.
struct SortBlocksInProgramOrderPass
    : public dynamatic::impl::SortBlocksInProgramOrderBase<
          SortBlocksInProgramOrderPass> {

  using SortBlocksInProgramOrderBase::SortBlocksInProgramOrderBase;
  void runDynamaticPass() override {
    ModuleOp m = getOperation();
    // Process every function individually
    for (auto funcOp : m.getOps<func::FuncOp>())
      sortBlocksInProgramOrder(funcOp);
  };
};
} // namespace
