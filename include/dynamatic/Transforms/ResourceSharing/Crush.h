#pragma once
#include "dynamatic/Support/LLVM.h"
#include <string>

namespace dynamatic {
#define GEN_PASS_DECL_CREDITBASEDSHARING
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
