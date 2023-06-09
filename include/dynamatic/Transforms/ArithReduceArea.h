//===- ArithReduceArea.h - Reduce area of arith operations ------*- C++ -*-===//
//
// This file declares the --arith-reduce-area pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_ARITHREDUCEAREA_H
#define DYNAMATIC_TRANSFORMS_ARITHREDUCEAREA_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createArithReduceArea();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_ARITHREDUCEAREA_H
