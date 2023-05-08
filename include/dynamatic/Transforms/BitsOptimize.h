//===- BitsOptimize.h - Optimize bits widths --------------------*- C++ -*-===//
//
// This file declares the --optimize-bits pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BITSOPTIMIZE_H
#define DYNAMATIC_TRANSFORMS_BITSOPTIMIZE_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createBitsOptimizationPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BITSOPTIMIZE_H