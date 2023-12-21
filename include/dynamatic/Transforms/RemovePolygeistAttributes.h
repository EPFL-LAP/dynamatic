//===- RemovePolygeistAttributes.h - Remove useless attrs -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --remove-polygeist-attributes pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_REMOVEPOLYGEISTATTRIBUTES_H
#define DYNAMATIC_TRANSFORMS_REMOVEPOLYGEISTATTRIBUTES_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_REMOVEPOLYGEISTATTRIBUTES
#define GEN_PASS_DEF_REMOVEPOLYGEISTATTRIBUTES
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createRemovePolygeistAttributes();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_REMOVEPOLYGEISTATTRIBUTES_H