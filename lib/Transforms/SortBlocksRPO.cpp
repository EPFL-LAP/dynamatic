//===- SortBlocksRPO.cpp - Sort blocks --------------------------*- C++ -*-===//
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
#define GEN_PASS_DEF_SORTBLOCKSRPO
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

/// Returns the blocks of the region in reverse post order, starting from its
/// entry block. Blocks which the entry cannot reach come last.
static SmallVector<Block *> getBlocksInRPO(Region &region) {
  SmallVector<Block *> rpo;
  if (region.empty())
    return rpo;

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

    // If there is another successor to explore, remember to continue with the
    // following successor when returning to this block.
    if (nextSuccessor < block->getNumSuccessors()) {
      stack.back().second = nextSuccessor + 1;
      Block *successor = block->getSuccessor(nextSuccessor);

      // Visit the successor if it has not been reached through another path.
      if (visited.insert(successor).second)
        stack.push_back({successor, 0});
      continue;
    }

    // All the successors were explored, thus the block is in postorder position
    rpo.push_back(block);
    stack.pop_back();
  }

  // The reverse of the postorder is the wanted order
  std::reverse(rpo.begin(), rpo.end());

  for (Block &block : region.getBlocks())
    if (!visited.contains(&block))
      rpo.push_back(&block);

  return rpo;
}

/// Sorts the blocks of a function in reverse post order.
static void sortBlocksInRPO(func::FuncOp funcOp) {
  SmallVector<Block *> rpo = getBlocksInRPO(funcOp.getBody());

  // Moving each block right before the next one, from the last to the first,
  // leaves the region in that same order
  for (size_t idx = rpo.size(); idx > 1; --idx)
    rpo[idx - 2]->moveBefore(rpo[idx - 1]);
}

namespace {

/// Simple driver for the block sorting pass. Runs the pass on every function in
/// the module independently.
struct SortBlocksRPOPass
    : public dynamatic::impl::SortBlocksRPOBase<SortBlocksRPOPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Process every function individually
    for (auto funcOp : m.getOps<func::FuncOp>())
      sortBlocksInRPO(funcOp);
  };
};
} // namespace
