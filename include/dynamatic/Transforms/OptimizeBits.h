//===- BitsOptimize.h - Optimize bits widths --------------------*- C++ -*-===//
//
// This file declares the --optimize-bits pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_OPTIMIZEBITS_H
#define DYNAMATIC_TRANSFORMS_OPTIMIZEBITS_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createOptimizeBitsPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_OPTIMIZEBITS_H