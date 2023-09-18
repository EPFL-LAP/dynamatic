//===- ScfToCf.h - Lower scf ops to unstructured control flow ---*- C++ -*-===//
//
// This file declares the --lower-scf-to-cf pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SCFTOCF_H
#define DYNAMATIC_TRANSFORMS_SCFTOCF_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_SCFTOCF
#define GEN_PASS_DEF_SCFTOCF
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerScfToCf();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SCFTOCF_H
