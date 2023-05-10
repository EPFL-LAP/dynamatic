//===- PassDetails.h - Handshake passes class details -----------*- C++ -*-===//
//
// This is the header file for all transformation passes defined in Dynamatic.
// It contains forward declarations needed by tranformation passes and includes
// auto-generated base class definitions for all tranformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSDETAILS_H
#define DYNAMATIC_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // namespace memref
} // namespace mlir

namespace circt {
namespace handshake {
class HandshakeDialect;
} // namespace handshake
} // namespace circt

namespace dynamatic {

#define GEN_PASS_CLASSES
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PASSDETAILS_H
