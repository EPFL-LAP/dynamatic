//===- InitCstWidth.h - Reduce the constant bits width ----------*- C++ -*-===//
//
// This file declares the --init-cstwidth pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INITCSTWIDTH_H
#define DYNAMATIC_TRANSFORMS_INITCSTWIDTH_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitCstWidthPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H