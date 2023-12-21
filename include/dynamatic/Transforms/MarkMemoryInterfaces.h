//===- MarkMemoryInterfaces.h - Mark memory interfaces ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --mark-memory-interfaces pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_MARK_MEMORY_INTERFACES
#define DYNAMATIC_TRANSFORMS_MARK_MEMORY_INTERFACES

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_MARKMEMORYINTERFACES
#define GEN_PASS_DEF_MARKMEMORYINTERFACES
#include "dynamatic/Transforms/Passes.h.inc"
std::unique_ptr<dynamatic::DynamaticPass> createMarkMemoryInterfaces();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_MARK_MEMORY_INTERFACES
