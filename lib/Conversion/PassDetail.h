//===- PassDetail.h - Conversion Pass class details -----------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithDialect;
} // namespace arith

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace scf {
class SCFDialect;
} // namespace scf

namespace func {
class FuncDialect;
class FuncOp;
} // namespace func
} // namespace mlir

namespace circt {
namespace handshake {
class HandshakeDialect;
class FuncOp;
} // namespace handshake

} // namespace circt

namespace dynamatic {

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "dynamatic/Conversion/Passes.h.inc"

} // namespace dynamatic

#endif // CONVERSION_PASSDETAIL_H
