//===- LLVMToControlFlow.h - Convert LLVM to CF  ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_LLVMTOCONTROLFLOW_H
#define DYNAMATIC_CONVERSION_LLVMTOCONTROLFLOW_H

#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

namespace dynamatic {

#define GEN_PASS_DECL_LLVMTOCONTROLFLOW
#define GEN_PASS_DEF_LLVMTOCONTROLFLOW
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createLLVMToControlFlowPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_LLVMTOCONTROLFLOW_H