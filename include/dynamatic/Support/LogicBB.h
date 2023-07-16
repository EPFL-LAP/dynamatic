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

/// Thin wrapper around an attribute access to the "bb" attribute.
std::optional<unsigned> getLogicBB(Operation *op);

} // namespace dynamatic