//===- FtdSupport.h - FTD conversion support -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA analysis pass. All the functions are about
// analyzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H
#define DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Class to associate an index to each block, so that if block Bi dominates
/// block Bj then i < j. While this is guaranteed by the MLIR CFG construction,
/// it cannot really be given for granted, thus it is more convenient to have a
/// custom indexing.
class BlockIndexing {

  /// Map to store the connection between indices and blocks.
  DenseMap<unsigned, Block *> indexToBlock;

  /// Map to store the connection between blocks and indices.
  DenseMap<Block *, unsigned> blockToIndex;

public:
  /// Build the map out of a region.
  BlockIndexing(mlir::Region &region);

  /// Get a block out of an index.
  std::optional<Block *> getBlockFromIndex(unsigned index) const;

  /// Get a block out of a string condition in the format `cX` where X is a
  /// number.
  std::optional<Block *> getBlockFromCondition(StringRef condition) const;

  /// Get the index of a block.
  std::optional<unsigned> getIndexFromBlock(Block *bb) const;

  /// Return true if the index of bb1 is greater than then index of bb2.
  bool isGreater(Block *bb1, Block *bb2) const;

  /// Return true if the index of bb1 is smaller than then index of bb2.
  bool isLess(Block *bb1, Block *bb2) const;

  /// Given a block whose name is `^BBN` (where N is an integer) return a string
  /// in the format `cN`, used to identify the condition which allows the block
  /// to be executed. The adopted index is retrieved from the BlockIndexing.
  std::string getBlockCondition(Block *block) const;
};

/// Checks if the source and destination are in a loop
bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li);

/// Gets all the paths from block `start` to block `end` using a DFS search.
/// If `blockToTraverse` is non null, then we want the paths having
/// `blockToTraverse` in the path; filters paths that do not contain blocks in
/// `blocksToAvoid`.
std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, const BlockIndexing &bi,
             Block *blockToTraverse = nullptr,
             ArrayRef<Block *> blocksToAvoid = {});

/// Given a sequence of block, find a boolean expression defining the conditions
/// for which the path is traversed. If one edge is unconditional, then no
/// condition is added; otherwise, the condition for the conditional branch is
/// added (either direct or negated). The list of blocks whose condition is
/// considered is saved in `blockIndexSet`. If `ignoreDeps` is false, then a
/// condition is added only if the source block was in the set `deps`.
boolean::BoolExpression *
getPathExpression(ArrayRef<Block *> path, DenseSet<unsigned> &blockIndexSet,
                  const BlockIndexing &bi,
                  const DenseSet<Block *> &deps = DenseSet<Block *>(),
                  bool ignoreDeps = true);

/// Return the channelified version of the input type.
Type channelifyType(Type type);

/// Get an array of `size` elements all identical to the
SmallVector<Type> getListTypes(Type inputType, unsigned size = 2);

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H
