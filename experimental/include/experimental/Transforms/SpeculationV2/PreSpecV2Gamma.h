#ifndef DYNAMATIC_TRANSFORMS_PRE_SPEC_V2_GAMMA_H
#define DYNAMATIC_TRANSFORMS_PRE_SPEC_V2_GAMMA_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

#define GEN_PASS_DECL_PRESPECV2GAMMA
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PRE_SPEC_V2_GAMMA_H
