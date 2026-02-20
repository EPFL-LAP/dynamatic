#pragma once

#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include <set>
#include <vector>

using namespace dynamatic::buffer;

namespace dynamatic {
// for a CFC, find the list of SCCs
// here SCCs are encoded as a component id per each op
std::map<Operation *, size_t>
getSccsInCfc(const std::set<Operation *> &cfUnits,
             const std::set<Channel *> &cfChannels);
} // namespace dynamatic
