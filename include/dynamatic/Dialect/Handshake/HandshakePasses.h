//===- HandshakePasses.h - Handshake pass entry points ----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHandshakeToDotPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dynamatic/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
