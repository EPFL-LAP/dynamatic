//===- BackwardUpdate.h ---------*- C++ -*-===//
//
// This file declares functions for backward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H
#define DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace backward {

void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::result_range vecResults)>> 
                     &mapOpNameWidth);

}
#endif // DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H