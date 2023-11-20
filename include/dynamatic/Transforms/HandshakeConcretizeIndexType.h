//===- HandshakeConcretizeIndexType.h - Index -> Integer --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-concretize-index-type pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKECONCRETIZEINDEXTYPE
#define GEN_PASS_DEF_HANDSHAKECONCRETIZEINDEXTYPE
#include "dynamatic/Transforms/Passes.h.inc"

const std::string ERR_RUN_CONCRETIZATION =
    "Run the --handshake-conretize-index-type pass before to concretize all "
    "index types in the IR.";

/// Verifies that none of an operation's operands and results has an IndexType.
LogicalResult verifyAllIndexConcretized(Operation *op);

/// Verifies that none of a Handshake function's arguments and results, as well
/// as any of its nested operations operands and results, has an IndexType.
LogicalResult verifyAllIndexConcretized(circt::handshake::FuncOp funcOp);

/// Verifies that none of the Handshake functions in a module has an argument
/// ar result with an IndexType, or that any operation in the body of a
/// Handshake function has an IndexType operand or result.
LogicalResult verifyAllIndexConcretized(mlir::ModuleOp modOp);

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeConcretizeIndexType(unsigned width = 64);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECONCRETIZEINDEXTYPE_H