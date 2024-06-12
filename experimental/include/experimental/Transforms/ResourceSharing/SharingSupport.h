//===- SharingSupport.h - Resource Sharing Utilities-----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H
#define EXPERIMENTAL_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H

#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include <set>
#include <vector>

using namespace dynamatic::buffer;

namespace dynamatic {
namespace experimental {
namespace sharing {

// for a CFC, find the list of SCCs
// here SCCs are encoded as a component id per each op
std::map<Operation *, size_t>
getSccsInCfc(const std::set<Operation *> &cfUnits,
             const std::set<Channel *> &cfChannels);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H