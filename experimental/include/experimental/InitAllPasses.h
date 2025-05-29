//===- InitAllPasses.h - Experimental passes registration --------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all experimental
// passes defined in the Dynamatic.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INITALLPASSES_H
#define EXPERIMENTAL_INITALLPASSES_H

#include "experimental/Analysis/Passes.h"
#include "experimental/Transforms/Passes.h"

namespace dynamatic {
namespace experimental {

inline void registerAllPasses() {
  registerAnalysisPasses();
  dynamatic::experimental::registerPasses();
}

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INITALLPASSES_H
