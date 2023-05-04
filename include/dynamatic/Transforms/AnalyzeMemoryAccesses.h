//===- AnalyzeMemoryAccesses.h - Analyze memory accesses --------*- C++ -*-===//
//
// This file declares the --analyze-memory-accesses pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_ANALYZEMEMORYACCESSES_H
#define DYNAMATIC_TRANSFORMS_ANALYZEMEMORYACCESSES_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAnalyzeMemoryAccesses();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_ANALYZEMEMORYACCESSES_H
