//===- ArithReduceStrength.h - Reduce strength of arith ops -----*- C++ -*-===//
//
// This file declares the --arith-reduce-strength pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H
#define DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createArithReduceStrength();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_ARITHREDUCESTRENGTH_H
