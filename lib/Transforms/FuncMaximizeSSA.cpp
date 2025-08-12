//===- FuncMaximizeSSA.cpp - Maximal SSA form within functions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the SSA maximization pass as well as utilities
// for converting a function with standard control flow into maximal SSA form.
//
// This if largely inherited from CIRCT, with minor modifications.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace dynamatic;

/// Determines the block to which the value belongs.
static Block *getDefiningBlock(Value value) {
  // Value is either a block argument...
  if (BlockArgument blockArg = dyn_cast<BlockArgument>(value); blockArg)
    return blockArg.getParentBlock();

  // ... or an operation's result
  Operation *defOp = value.getDefiningOp();
  assert(defOp);
  return defOp->getBlock();
}

/// Adds the value to the successor operands of the predecessor block's
/// branch-like terminator's branch(es) to the successor block. Fails if the
/// predecessor's block terminator is not a branch-like operation.
static LogicalResult addArgToTerminator(Block *succBlock, Block *predBlock,
                                        Value value) {

  // Identify terminator branching instruction in predecessor block
  auto branchOp = dyn_cast<BranchOpInterface>(predBlock->getTerminator());
  if (!branchOp)
    return predBlock->getTerminator()->emitError(
        "Expected terminator operation of block to be a "
        "branch-like operation");

  // In the predecessor block's terminator, find all successors that equal
  // the block and add the value to the list of operands it's passed
  for (auto [idx, branchBlock] : llvm::enumerate(branchOp->getSuccessors()))
    if (succBlock == branchBlock)
      branchOp.getSuccessorOperands(idx).append(value);

  return success();
}

bool dynamatic::isRegionSSAMaximized(Region &region) {
  // Check whether all operands used within each block are also defined within
  // the same block
  for (auto &block : region.getBlocks())
    for (auto &op : block.getOperations())
      for (auto operand : op.getOperands())
        if (getDefiningBlock(operand) != &block)
          return false;

  return true;
}

bool dynamatic::SSAMaximizationStrategy::maximizeBlock(Block &block) {
  return true;
}
bool dynamatic::SSAMaximizationStrategy::maximizeArgument(BlockArgument arg) {
  return true;
}
bool dynamatic::SSAMaximizationStrategy::maximizeOp(Operation &op) {
  return !isa<
      // clang-format off
      memref::AllocOp,
      memref::AllocaOp,
      memref::GetGlobalOp
      // clang-format on
      >(op);
}
bool dynamatic::SSAMaximizationStrategy::maximizeResult(OpResult res) {
  return true;
}

LogicalResult dynamatic::maximizeSSA(Value value) {

  // Identify the basic block in which the value is defined
  Block *defBlock = getDefiningBlock(value);
  Location loc = UnknownLoc::get(value.getContext());

  // Identify all basic blocks in which the value is used (excluding the
  // value-defining block)
  DenseSet<Block *> blocksUsing;
  for (OpOperand &use : value.getUses()) {
    Block *block = use.getOwner()->getBlock();
    if (block != defBlock)
      blocksUsing.insert(block);
  }

  // Prepare a stack to iterate over the list of basic blocks that must be
  // modified for the value to be in maximum SSA form. At all points,
  // blocksUsing is a non-strict superset of the elements contained in
  // blocksToVisit
  SmallVector<Block *> blocksToVisit(blocksUsing.begin(), blocksUsing.end());

  // Backtrack from all blocks using the value to the value-defining basic
  // block, adding a new block argument for the value along the way. Keep
  // track of which blocks have already been modified to avoid visiting a
  // block more than once while backtracking (possible due to branching
  // control flow)
  DenseMap<Block *, BlockArgument> blockToArg;
  while (!blocksToVisit.empty()) {
    // Pop the basic block at the top of the stack
    Block *block = blocksToVisit.pop_back_val();

    // Add an argument to the block to hold the value
    blockToArg[block] = block->addArgument(value.getType(), loc);

    // In all unique block predecessors, modify the terminator branching
    // instruction to also pass the value to the block
    SmallPtrSet<Block *, 8> uniquePredecessors;
    for (Block *predBlock : block->getPredecessors()) {
      // If we have already visited the block predecessor, skip it. It's
      // possible to get duplicate block predecessors when there exists a
      // conditional branch with both branches going to the same block e.g.,
      // cf.cond_br %cond, ^bbx, ^bbx
      if (auto [_, newPredecessor] = uniquePredecessors.insert(predBlock);
          !newPredecessor) {
        continue;
      }

      // Modify the terminator instruction
      if (failed(addArgToTerminator(block, predBlock, value)))
        return failure();

      // Now the predecessor block is using the value, so we must also make sure
      // to visit it
      if (predBlock != defBlock)
        if (auto [_, blockNewlyUsing] = blocksUsing.insert(predBlock);
            blockNewlyUsing)
          blocksToVisit.push_back(predBlock);
    }
  }

  // Replace all uses of the value with the newly added block arguments
  SmallVector<Operation *> users;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getBlock() != defBlock)
      users.push_back(owner);
  }
  for (Operation *user : users)
    user->replaceUsesOfWith(value, blockToArg[user->getBlock()]);

  return success();
}

LogicalResult dynamatic::maximizeSSA(Operation &op,
                                     SSAMaximizationStrategy &strategy) {
  // Apply SSA maximization on each of the operation's results
  for (OpResult res : op.getResults()) {
    if (strategy.maximizeResult(res)) {
      if (failed(maximizeSSA(res)))
        return failure();
    }
  }
  return success();
}

LogicalResult dynamatic::maximizeSSA(Block &block,
                                     SSAMaximizationStrategy &strategy) {
  // Apply SSA maximization on each of the block's arguments
  for (BlockArgument arg : block.getArguments()) {
    if (strategy.maximizeArgument(arg)) {
      if (failed(maximizeSSA(arg)))
        return failure();
    }
  }
  // Apply SSA maximization on each of the block's operations
  for (Operation &op : block.getOperations()) {
    if (strategy.maximizeOp(op)) {
      if (failed(maximizeSSA(op, strategy)))
        return failure();
    }
  }

  return success();
}

LogicalResult dynamatic::maximizeSSA(Region &region,
                                     SSAMaximizationStrategy &strategy) {
  // Apply SSA maximization on each of the region's block
  for (Block &block : region.getBlocks()) {
    if (strategy.maximizeBlock(block)) {
      if (failed(maximizeSSA(block, strategy)))
        return failure();
    }
  }
  return success();
}

namespace {

struct FuncMaximizeSSAPass
    : public dynamatic::impl::FuncMaximizeSSABase<FuncMaximizeSSAPass> {
public:
  void runOnOperation() override {
    SSAMaximizationStrategy strategy;
    if (failed(maximizeSSA(getOperation().getRegion(), strategy)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
dynamatic::createFuncMaximizeSSA() {
  return std::make_unique<FuncMaximizeSSAPass>();
}