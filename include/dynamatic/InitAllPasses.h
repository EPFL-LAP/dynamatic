//===- InitAllPasses.h - Dynamatic passes registration -----------*- C++-*-===//
//
// This file defines a helper to trigger the registration of all passes that
// Dynamatic users may care about.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_INITALLPASSES_H
#define DYNAMATIC_INITALLPASSES_H

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Transforms/Passes.h"
#include "dynamatic/Conversion/Passes.h"
#include "dynamatic/Dialect/Handshake/HandshakePasses.h"

namespace dynamatic {

inline void registerAllPasses() {
  // Passes defined in Dynamatic
  registerConversionPasses();
  dynamatic::registerPasses();

  // Passes defined in CIRCT
  circt::registerStandardToHandshake();
  circt::registerFlattenMemRef();
  circt::registerFlattenMemRefCalls();
  handshake::registerPasses();
}

} // namespace dynamatic

#endif // DYNAMATIC_INITALLPASSES_H
