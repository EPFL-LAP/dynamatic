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

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_ANALYZEMEMORYACCESSES
#define GEN_PASS_DEF_ANALYZEMEMORYACCESSES
#include "dynamatic/Transforms/Passes.h.inc"
std::unique_ptr<dynamatic::DynamaticPass<false>> createAnalyzeMemoryAccesses();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_ANALYZEMEMORYACCESSES_H
