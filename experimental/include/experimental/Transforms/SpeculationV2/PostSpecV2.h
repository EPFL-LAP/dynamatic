#ifndef DYNAMATIC_TRANSFORMS_POST_SPEC_V2_H
#define DYNAMATIC_TRANSFORMS_POST_SPEC_V2_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

#define GEN_PASS_DECL_POSTSPECV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_POST_SPEC_V2_H
