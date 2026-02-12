//===- crush.h - Credit-Based Sharing -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Credit-based sharing pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_CREDIT_BASED_SHARING_H
#define EXPERIMENTAL_TRANSFORMS_CREDIT_BASED_SHARING_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_CREDITBASEDSHARING
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_CREDIT_BASED_SHARING_H
