//===- InitAllPasses.h - Dynamatic passes registration -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes that
// Dynamatic users may care about.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_INITALLPASSES_H
#define DYNAMATIC_INITALLPASSES_H

#include "circt/Dialect/ESI/ESIPasses.h"
#include "dynamatic/Conversion/Passes.h"
#include "dynamatic/Transforms/Passes.h"

namespace dynamatic {

inline void registerAllPasses() {
  // Passes defined in Dynamatic
  registerConversionPasses();
  dynamatic::registerPasses();

  // Passes defined in CIRCT
  circt::esi::registerPasses();
}

} // namespace dynamatic

#endif // DYNAMATIC_INITALLPASSES_H
