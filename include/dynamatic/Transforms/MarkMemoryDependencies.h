//===- MarkMemoryDependencies.h - Mark mem. deps. in the IR -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --mark-memory-dependencies pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_MARK_MEMORY_DEPENDENCIES
#define DYNAMATIC_TRANSFORMS_MARK_MEMORY_DEPENDENCIES

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_MARKMEMORYDEPENDENCIES
#define GEN_PASS_DEF_MARKMEMORYDEPENDENCIES
#include "dynamatic/Transforms/Passes.h.inc"
std::unique_ptr<dynamatic::DynamaticPass> createMarkMemoryDependencies();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_MARK_MEMORY_DEPENDENCIES
