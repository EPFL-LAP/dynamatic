//===- SwitchingEstimation.h - Switching Estimation -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Switching Estimation pass
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_SWITCHING_ESTIMATION_H
#define EXPERIMENTAL_TRANSFORMS_SWITCHING_ESTIMATION_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {
namespace switching {

#define GEN_PASS_DECL_SWITCHINGESTIMATION
#define GEN_PASS_DEF_SWITCHINGESTIMATION
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> 
createSwitchingEstimation(StringRef dataTrace = "",
                          StringRef bbList = "",
                          StringRef frequencies = "",
                          StringRef timingModels = "");

} // namespace switching
} // namespace experimental
} // namespace dynamatic 

#endif // EXPERIMENTAL_TRANSFORMS_SWITCHING_ESTIMATION_H
