//===- InitAllPasses.h - Tutorials passes registration -----------*- C++-*-===//
//
// This file defines a helper to trigger the registration of all experimental
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INITALLPASSES_H
#define EXPERIMENTAL_INITALLPASSES_H

#include "experimental/Conversion/Passes.h"

namespace dynamatic {
namespace experimental {

inline void registerAllPasses() { registerConversionPasses(); }

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INITALLPASSES_H
