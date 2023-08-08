//===- HandshakeOptimizeBitwidths.h - Optimize channel widths ---*- C++ -*-===//
//
// This file declares the --handshake-optimize-bitwidths pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEOPTIMIZEBITWIDTHS
#define GEN_PASS_DEF_HANDSHAKEOPTIMIZEBITWIDTHS
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeOptimizeBitwidths();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEOPTIMIZEBITWIDTHS_H
