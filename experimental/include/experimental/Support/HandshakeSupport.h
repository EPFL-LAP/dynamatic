#ifndef DYNAMATIC_HANDSHAKE_SUPPORT_SUPPORT_H
#define DYNAMATIC_HANDSHAKE_SUPPORT_SUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Class to associate an index to each block, that if block Bi dominates block
/// Bj then i < j. While this is guaranteed by the MLIR CFG construction, it
/// cannot really be given for granted, thus it is more convenient to make a
/// custom one.
class BlockIndexing {

  /// Map to store the connection between indexes and blocks.
  DenseMap<unsigned, Block *> indexToBlock;

  /// Map to store the connection between blocks and indexes.
  DenseMap<Block *, unsigned> blockToIndex;

public:
  /// Build the map.
  BlockIndexing(mlir::Region &region);

  /// Get a block out of an index.
  Block *getBlockFromIndex(unsigned index) const;

  /// Get a block out of a string condition in the format `cX` where X is a
  /// number.
  Block *getBlockFromCondition(const std::string &condition) const;

  /// Get the index of a block.
  unsigned getIndexFromBlock(Block *bb) const;

  /// Return true if the index of bb1 is greater than then index of bb2.
  bool greaterIndex(Block *bb1, Block *bb2) const;

  /// Return true if the index of bb1 is smaller than then index of bb2.
  bool lessIndex(Block *bb1, Block *bb2) const;

  /// Given a block whose name is `^BBN` (where N is an integer) return a string
  /// in the format `cN`, used to identify the condition which allows the block
  /// to be executed.
  std::string getBlockCondition(Block *block) const;
};

/// checks if the source and destination are in a loop
bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li);

/// Gets all the paths from block `start` to block `end` using a dfs search.
/// If `blockToTraverse` is non null, then we want the paths having that block
/// in the path; if `blocksToAvoid` is non empty, then we want the paths which
/// do not cross those paths.
std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, const BlockIndexing &bi,
             Block *blockToTraverse = nullptr,
             ArrayRef<Block *> blocksToAvoid = std::vector<Block *>());

/// Get the boolean condition determining when a path is executed. While
/// covering each block in the path, add the cofactor of each block to the
/// list of cofactors if not already covered
boolean::BoolExpression *
getPathExpression(ArrayRef<Block *> path, DenseSet<unsigned> &blockIndexSet,
                  const BlockIndexing &bi,
                  const DenseSet<Block *> &deps = DenseSet<Block *>(),
                  bool ignoreDeps = true);

/// Return the channelified version of the input type
Type channelifyType(Type type);

/// Get an array of two types containing the result type of a branch
/// operation, channelifying the input type
SmallVector<Type> getBranchResultTypes(Type inputType);

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif
