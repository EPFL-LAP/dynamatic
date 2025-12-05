#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKE_CUT_BBS_BY_BUFFERS_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKE_CUT_BBS_BY_BUFFERS_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

#define GEN_PASS_DECL_HANDSHAKECUTBBSBYBUFFERS
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKE_CUT_BBS_BY_BUFFERS_H
