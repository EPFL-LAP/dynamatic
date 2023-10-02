//===- InitAllPasses.h - Experimental passes registration --------*- C++-*-===//
//
// This file defines a helper to trigger the registration of all experimental
// passes defined in the Dynamatic.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INITALLPASSES_H
#define EXPERIMENTAL_INITALLPASSES_H

#include "experimental/Transforms/Passes.h"

namespace dynamatic {
namespace experimental {

inline void registerAllPasses() { dynamatic::experimental::registerPasses(); }

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INITALLPASSES_H
