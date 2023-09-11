//===- LogicBB.h - Infrastructure for working with logical BBs --*- C++ -*-===//
//
// This file declares the infrastructure useful for handling logical basic
// blocks (logical BBs) in Handshake functions. These are not basic blocks in
// the MLIR sense since Handshake function only have a single block. They
// instead map to the original basic blocks that the std-level IR possessed
// before conversion to Handshake-level.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Operation attribute to identify the basic block the operation originated
/// from in the std-level IR.
const std::string BB_ATTR = "bb";

/// ID of entry basic block of every Handshake function.
const unsigned ENTRY_BB = 0;

/// This struct groups the operations of a handshake::FuncOp in "blocks" based
/// on the "bb" attribute potentially attached to each operation.
struct LogicBBs {
  /// Maps each block ID to the operations (in program order) that are tagged
  /// with it.
  mlir::DenseMap<unsigned, mlir::SmallVector<mlir::Operation *>> blocks;
  /// List of operations (in program order) that do not belong to any block.
  mlir::SmallVector<mlir::Operation *> outOfBlocks;
};

/// Groups the operations of a function into "blocks" based on the "bb"
/// attribute of each operation.
LogicBBs getLogicBBs(circt::handshake::FuncOp funcOp);

/// If the source operation belongs to a logical BB, makes the destination
/// operation part of the same BB and returns true; otherwise return false.
bool inheritBB(Operation *srcOp, Operation *dstOp);

/// If the source value is the result of an operation that belongs to a logical
/// BB, makes the destination operation part of the same BB and returns true;
/// otherwise return false. If the source value is a block argument, make the
/// destination operation part of the entry BB.
bool inheritBBFromValue(Value val, Operation *dstOp);

/// Thin wrapper around an attribute access to the "bb" attribute.
std::optional<unsigned> getLogicBB(Operation *op);

/// Attempts to backtrack through forks and bitwidth modification operations
/// till reaching a branch-like operation. On success, returns the branch-like
/// operation that was backtracked to (or the passed operation if it was itself
/// branch-like); otherwise, returns nullptr.
Operation *backtrackToBranch(Operation *op);

/// Attempts to follow the def-use chains of all the operation's results through
/// forks and bitwidth modification operations till reaching merge-like
/// operations that all belong to the same basic block. On success, returns one
/// of the merge-like operations reached by a def-use chain (or the passed
/// operation if it was itself merge-like); otherwise, returns nullptr.
Operation *followToMerge(Operation *op);

/// Determines whether the value is a backedge i.e., whether the channel
/// corresponding to the value is located between a branch-like operation and a
/// merge-like operation, where the merge-like operation happens semantically
/// "before" the branch-like operation. This function can only correctly
/// identify backedges if the circuit's branches and merges are associated to
/// basic blocks (otherwise it will always return false). `user` must be one of
/// `val`'s users.
bool isBackedge(Value val, Operation *user);

/// Determines whether the value is a backedge. The value must have a single
/// user (the function will assert if that is not the case).
bool isBackedge(Value val);

} // namespace dynamatic