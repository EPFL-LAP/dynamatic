//===- PushConstants.h - Push constants in using blocks ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --push-constants pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H
#define DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<dynamatic::DynamaticPass> createPushConstantsPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H
