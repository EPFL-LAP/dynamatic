//===- InitAllPasses.h - FTD passes registration --------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all ftd
// passes defined in the Dynamatic.
//
//===----------------------------------------------------------------------===//

#ifndef FTD_INITALLPASSES_H
#define FTD_INITALLPASSES_H

#include "experimental/ftd/Transforms/Passes.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

inline void registerAllPasses() {
  registerHandshakeCombineSteeringLogic();
}

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // FTD_INITALLPASSES_H
