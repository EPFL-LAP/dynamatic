//===- ConvertMuxToMerge.h - Conversion from mux to merge  -*--- C++
//-*-----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_CONVERTMUXTOMERGE_H
#define DYNAMATIC_TRANSFORMS_CONVERTMUXTOMERGE_H
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

std::unique_ptr<dynamatic::DynamaticPass> createConvertMuxToMerge();

#define GEN_PASS_DECL_CONVERTMUCTOMERGE
#define GEN_PASS_DEF_CONVERTMUXTOMERGE
#include "experimental/Transforms/Passes.h.inc"

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_CONVERTMUXTOMERGE_H
