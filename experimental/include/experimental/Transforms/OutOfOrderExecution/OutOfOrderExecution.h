//===- OutOfOrderExecution.h - Enable Out-of-Order Execution in Dataflow
// Circuits
//-*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --out-of-order-execution pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_OUTOFORDEREXECUTION_OUTOFORDEREXECUTION_H
#define EXPERIMENTAL_TRANSFORMS_OUTOFORDEREXECUTION_OUTOFORDEREXECUTION_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace outoforder {

std::unique_ptr<dynamatic::DynamaticPass> createOutOfOrderExecution();

#define GEN_PASS_DECL_OUTOFORDEREXECUTION
#define GEN_PASS_DEF_OUTOFORDEREXECUTION
#include "experimental/Transforms/Passes.h.inc"

} // namespace outoforder
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_OUTOFORDEREXECUTION_OUTOFORDEREXECUTION_H