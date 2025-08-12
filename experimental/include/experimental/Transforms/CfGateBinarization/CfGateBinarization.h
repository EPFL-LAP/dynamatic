#ifndef DYNAMATIC_TRANSFORMS_CF_GATE_BINARIZATION_H
#define DYNAMATIC_TRANSFORMS_CF_GATE_BINARIZATION_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_CFGATEBINARIZATION
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_CF_GATE_BINARIZATION_H
