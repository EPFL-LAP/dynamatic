//===- HandshakeInferBasicBlocks.h - Infer ops basic blocks -----*- C++ -*-===//
//
// This file declares the --handshake-infer-basic-blocks pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H
#define DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Tries to infer the basic block of an operation (which must have a
/// handshake::FuncOp as immediate parent operation). The function tries to
/// backtrack all of the operation's operands till reaching dataflow
/// predecessors with known basic blocks. Returns the inferred basic block ID or
/// an empty optional value when the basic block could not be inferred.
std::optional<unsigned> inferOpBasicBlock(Operation *op);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeInferBasicBlocksPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H
