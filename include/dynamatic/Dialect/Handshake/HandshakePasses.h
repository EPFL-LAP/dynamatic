//===- HandshakePasses.h - Handshake passes registration --------*- C++ -*-===//
//
// This file contains declarations to register handshake passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

#include "dynamatic/Dialect/Handshake/HandshakeToDot.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dynamatic/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
