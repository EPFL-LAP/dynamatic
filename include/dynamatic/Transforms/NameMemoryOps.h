//===- NameMemoryOps.h - Give a unique name to all memory ops ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --name-memory-ops pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H
#define DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createNameMemoryOps();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H
