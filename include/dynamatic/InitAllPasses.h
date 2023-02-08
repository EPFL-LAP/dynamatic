//===- InitAllPasses.h - Dynamatic Global Pass Registration ---------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_INITALLPASSES_H
#define DYNAMATIC_INITALLPASSES_H

#include "circt/InitAllPasses.h"
#include "dynamatic/Conversion/Passes.h"

namespace dynamatic {

inline void registerAllPasses() {
  registerConversionPasses();
  circt::registerAllPasses();
}

} // namespace dynamatic

#endif // DYNAMATIC_INITALLPASSES_H
