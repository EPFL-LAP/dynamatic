#ifndef DYNAMATIC_TRANSFORMS_SPEC_V1_POST_ADAPTOR_H
#define DYNAMATIC_TRANSFORMS_SPEC_V1_POST_ADAPTOR_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

#define GEN_PASS_DECL_SPECV1POSTADAPTOR
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPEC_V1_POST_ADAPTOR_H
