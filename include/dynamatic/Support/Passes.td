//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper classes to create passes for Dynamatic.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_PASSES_TD
#define DYNAMATIC_SUPPORT_PASSES_TD

include "mlir/Pass/PassBase.td"

class DynamaticPass<string passArg, string requireNames = "false">
  : PassBase<passArg, "::dynamatic::DynamaticPass<" # requireNames # ">">;

#endif // DYNAMATIC_SUPPORT_PASSES_TD