//===- PassDetails.h - Conversion passes class details ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the header file for all conversion passes defined in Dynamatic. It
// contains forward declarations needed by conversion passes and includes
// auto-generated base class definitions for all conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_PASSDETAILS_H
#define DYNAMATIC_CONVERSION_PASSDETAILS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace affine {
class AffineDialect;
} // namespace affine

namespace arith {
class ArithDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // namespace memref

namespace vector {
class VectorDialect;
} // namespace vector

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

namespace hw {
class HWDialect;
} // namespace hw

namespace esi {
class ESIDialect;
} // namespace esi

} // namespace circt

namespace dynamatic {

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "dynamatic/Conversion/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_PASSDETAILS_H
