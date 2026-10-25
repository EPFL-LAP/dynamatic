//===- BaseCfg.h - CFG representation for operand routing 2 -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CFG representation that routing is done on, and the base CFG every
// routing's CFG is built from.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_OPERANDROUTING2_BASECFG_H
#define EXPERIMENTAL_CONVERSION_OPERANDROUTING2_BASECFG_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace dynamatic {
namespace experimental {
namespace operand_routing2 {

using mlir::SetVector;

// As we run SSA maximization on the IR before routing
// (to make things easier rather than any academic reason)
// all block args of a basic block are block inputs
// (apart from block args which are actually function args)
using BlockInput = BlockArgument;

/// MLIRs block argument approach means
/// we define a live-out along an edge between two blocks
/// as a tuple of
/// 1. the value as it is in the pred block
/// 2. the block input it becomes in the succ block
struct LiveOut {
  Value liveOut;
  BlockInput blockInput;
};

/// A CFG edge annotated with live-outs
struct CfgEdge {
  // pred block
  Block *from;
  // succ block
  Block *to;
  // if the pred block has multiple successors,
  // is this edge taken when the condition is true?
  // or when the condition is false?
  std::optional<bool> condition;
  // values which remain live when this edge is taken
  SmallVector<LiveOut> liveOuts;

  /// The live-out on this edge that becomes the given block input
  /// in the successor.
  const LiveOut *getLiveOut(BlockInput blockInput) const;
};

/// A block use: a value used in a block that does not produce it.
///
/// We iterate over the collected block uses, as each block use must be routed.
/// Every operation in that block that uses the block use then takes it
/// from the routing.
///
/// The block use is not, however, actively part of the routing algorithm.
///
/// With SSA maximized, the set of block uses for a program is a subset of
/// the block arguments: a block argument is a block use if
/// 1) the basic block actually uses the value, rather than simply
/// forwarding it to a branch operation and
/// 2) the block arg is not a function arg (we treat function args as being
/// produced in the entry block for routing purposes)
using BlockUse = BlockArgument;

/// A CFG marked with which blocks produce or consume data
/// so it can be safely optimized
/// and therefore produce a task-parallel routing
struct Cfg {
  SetVector<Block *> blocks;
  SmallVector<CfgEdge> edges;
  SetVector<Block *> needed;
  Block *entry;
  Block *exit;

  // the block uses this CFG will be used to route to:
  // with SSA maximized, each is a block argument of the basic block using it
  SetVector<BlockUse> blockUses;

  /// The edges leaving a block.
  SmallVector<CfgEdge *> getOutEdges(Block *block);
  SmallVector<const CfgEdge *> getOutEdges(Block *block) const;

  /// The edges entering a block.
  SmallVector<CfgEdge *> getInEdges(Block *block);
  SmallVector<const CfgEdge *> getInEdges(Block *block) const;

  /// Removes an edge.
  /// Every pointer to an edge is invalid afterwards.
  void removeEdge(CfgEdge *edge);
};

/// The function's control flow and nothing else:
/// every edge between its blocks,
/// with the side it leaves on,
/// no live-outs, and no needed blocks.
///
/// All CFGs used for routing are built from this base.
Cfg getBaseCfg(Region &body);

/// The position of a basic block in its region, which is its bb id.
unsigned getBlockNumber(Block *block);

} // namespace operand_routing2
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_OPERANDROUTING2_BASECFG_H
