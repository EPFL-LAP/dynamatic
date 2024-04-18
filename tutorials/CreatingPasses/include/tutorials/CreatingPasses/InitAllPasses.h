//===- InitAllPasses.h - Tutorials passes registration -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes defined
// in the Dynamatic tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_INITALLPASSES_H
#define TUTORIALS_INITALLPASSES_H

#include "tutorials/CreatingPasses/Transforms/Passes.h"

namespace dynamatic {
namespace tutorials {

inline void registerAllPasses() {
  dynamatic::tutorials::CreatingPasses::registerPasses();
}

} // namespace tutorials
} // namespace dynamatic

#endif // TUTORIALS_INITALLPASSES_H
