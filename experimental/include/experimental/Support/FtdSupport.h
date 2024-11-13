//===- FtdSupport.h --- FTD conversion support -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_SUPPORT_H
#define DYNAMATIC_SUPPORT_FTD_SUPPORT_H

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Different types of loop suppression.
enum BranchToLoopType {
  MoreProducerThanConsumers,
  SelfRegeneration,
  BackwardRelationship
};

/// Class to associate an index to each block, that if block Bi dominates block
/// Bj then i < j. While this is guaranteed by the MLIR CFG construction, it
/// cannot really be given for granted, thus it is more convenient to make a
/// custom one.
class BlockIndexing {

  /// Map to store the connection between blocks and unsigned numbers.
  DenseMap<unsigned, Block *> blockIndexing;

public:
  /// Build the map.
  BlockIndexing(mlir::Region &region);

  /// Get a block out of an index.
  Block *getBlockFromIndex(unsigned index);

  /// Get a block out of a string condition in the format `cX` where X is a
  /// number.
  Block *getBlockFromCondition(const std::string &condition);

  /// Get the index of a block.
  unsigned getIndexFromBlock(Block *bb);

  /// Return true if the index of bb1 is smaller than then index of bb2.
  bool greaterIndex(Block *bb1, Block *bb2);
};

constexpr llvm::StringLiteral FTD_OP_TO_SKIP("ftd.skip");
constexpr llvm::StringLiteral FTD_SUPP_BRANCH("ftd.supp");
constexpr llvm::StringLiteral FTD_EXPLICIT_PHI("ftd.phi");
constexpr llvm::StringLiteral FTD_MEM_DEP("ftd.memdep");
constexpr llvm::StringLiteral FTD_INIT_MERGE("ftd.imerge");
constexpr llvm::StringLiteral FTD_REGEN("ftd.regen");

/// Get the index of a basic block
unsigned getBlockIndex(Block *bb);

/// Check whether the index of `block1` is less than the one of `block2`
bool lessThanBlocks(Block *block1, Block *block2);

/// Check whether the index of `block1` is greater than the one of `block2`
bool greaterThanBlocks(Block *block1, Block *block2);

/// Recursively check weather 2 blocks belong to the same loop, starting
/// from the inner-most loops
bool isSameLoop(const mlir::CFGLoop *loop1, const mlir::CFGLoop *loop2);

/// checks if the source and destination are in a loop
bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li);

/// Given a block whose name is `^BBN` (where N is an integer) return a string
/// in the format `cN`, used to identify the condition which allows the block
/// to be executed.
std::string getBlockCondition(Block *block);

/// Returns true if the provided operation is either of they `LSQLoad` or
/// `LSQStore`
bool isHandhsakeLSQOperation(Operation *op);

/// Given two sets containing object of type `Block*`, remove the common entries
void eliminateCommonBlocks(DenseSet<Block *> &s1, DenseSet<Block *> &s2);

/// Given two blocks, return a reference to the innermost common loop. The
/// result is `nullptr` if the two blocks are not within a loop
mlir::CFGLoop *getInnermostCommonLoop(Block *block1, Block *block2,
                                      mlir::CFGLoopInfo &li);

/// Given an operation, returns true if the operation is a conditional branch
/// which terminates a for loop
bool isBranchLoopExit(Operation *op, mlir::CFGLoopInfo &li);

/// Gets all the paths from operation `start` to operation `end` using a dfs
/// search
std::vector<std::vector<Operation *>> findAllPaths(Operation *start,
                                                   Operation *end);

/// Gets all the paths from block `start` to block `end` using a dfs search. If
/// `blockToTraverse` is non null, then we want the paths having that block in
/// the path; if `blocksToAvoid` is non empty, then we want the paths which do
/// not cross those paths.
std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, Block *blockToTraverse = nullptr,
             ArrayRef<Block *> blocksToAvoid = std::vector<Block *>());

/// Given a pair of consumer and producer, we are interested in a basic block
/// which is a successor of the producer and post-dominates the consumer.
/// If this block exists, the MERGE/GENERATE block can be put right after it,
/// since all paths between the producer and the consumer pass through it.
Block *getPostDominantSuccessor(Block *prod, Block *cons);

/// Given a pair of consumer and producer, we are interested in a basic block
/// which both dominates the consumer and post-dominates the producer. If this
/// block exists, the MERGE/GENERATE block can be put right after it, since
/// all paths between the producer and the consumer pass through it.
Block *getPredecessorDominatingAndPostDominating(Block *prod, Block *cons);

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool isaMergeLoop(Operation *merge, mlir::CFGLoopInfo &li);

/// Get the boolean condition determining when a path is executed. While
/// covering each block in the path, add the cofactor of each block to the list
/// of cofactors if not already covered
boolean::BoolExpression *
getPathExpression(ArrayRef<Block *> path, DenseSet<unsigned> &blockIndexSet,
                  const DenseMap<Block *, unsigned> &mapBlockToIndex,
                  const DenseSet<Block *> &deps = DenseSet<Block *>(),
                  bool ignoreDeps = true);

/// The boolean condition to either generate or suppress a token are computed
/// by considering all the paths from the producer (`start`) to the consumer
/// (`end`). "Each path identifies a Boolean product of elementary conditions
/// expressing the reaching of the target BB from the corresponding member of
/// the set; the product of all such paths are added".
boolean::BoolExpression *
enumeratePaths(Block *start, Block *end,
               const DenseMap<Block *, unsigned> &mapBlockToIndex,
               const DenseSet<Block *> &controlDeps);

/// Return the channelified version of the input type
Type channelifyType(Type type);

/// Get a boolean expression representing the exit condition of the current
/// loop block
boolean::BoolExpression *getBlockLoopExitCondition(Block *loopExit,
                                                   mlir::CFGLoop *loop,
                                                   mlir::CFGLoopInfo &li);

/// Get an array of two types containing the result type of a branch operation,
/// channelifying the input type
SmallVector<Type> getBranchResultTypes(Type inputType);

/// Given a block, get its immediate dominator if exists
Block *getImmediateDominator(Region &region, Block *bb);

/// Get the dominance frontier of each block in the region
DenseMap<Block *, DenseSet<Block *>> getDominanceFrontier(Region &region);

/// Given a set of values defining the same value in different blocks of a CFG,
/// modify the SSA representation to connect the values through some phi. The
/// final representation is coherent, so that in each block you can obtain one
/// only definition for the value. The resulting map provides such association,
/// with the value as it is available at the beginning of the block.
FailureOr<DenseMap<Block *, Value>>
insertPhi(Region &funcRegion, ConversionPatternRewriter &rewriter,
          SmallVector<Value> &vals);

/// Get a list of all the loops in which the consumer is but the producer is
/// not, starting from the innermost.
SmallVector<mlir::CFGLoop *> getLoopsConsNotInProd(Block *cons, Block *prod,
                                                   mlir::CFGLoopInfo &li);

/// Add some regen multiplexers to all the operands of a given consumer whenever
/// it is necessary according to the CFG structure of the input function
LogicalResult addRegenToConsumer(ConversionPatternRewriter &rewriter,
                                 dynamatic::handshake::FuncOp &funcOp,
                                 Operation *consumerOp);

/// Add suppression mechanism to all the inputs and outputs of a producer
LogicalResult
addSuppToProducer(ConversionPatternRewriter &rewriter,
                  handshake::FuncOp &funcOp, Operation *producerOp,
                  ftd::BlockIndexing &bi,
                  std::vector<Operation *> &producersToCover,
                  ControlDependenceAnalysis::BlockControlDepsMap &cda);

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_SUPPORT_H
