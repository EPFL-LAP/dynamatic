//===- AnalyzeMemoryAccesses.h - Analyze memory accesses --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
