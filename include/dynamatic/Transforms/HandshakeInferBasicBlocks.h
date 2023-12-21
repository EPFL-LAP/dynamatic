//===- HandshakeInferBasicBlocks.h - Infer ops basic blocks -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-infer-basic-blocks pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H
#define DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Tries to infer the logical basic block of an operation by looking at the
/// basic block to which the operation's predecessors and successors belong to.
/// In case the inference logic produces a different basic block between
/// successors and predecessors, the former has priority. This has the side
/// effect of attaching components located "between" branch-like and merge-like
/// operations to the block to which merge-like operations belong. On success,
/// `logicBB` contains the inferred basic block for the provided operation;
/// otherwise, it is undefined. Note that the function never sets the basic
/// block attribute on the operation.
LogicalResult inferLogicBB(Operation *op, unsigned &logicBB);

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeInferBasicBlocksPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_INFERBASICBLOCKS_H
