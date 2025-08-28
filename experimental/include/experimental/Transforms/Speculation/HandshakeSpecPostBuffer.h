#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKE_SPEC_POST_BUFFER_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKE_SPEC_POST_BUFFER_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {
namespace experimental {
namespace speculation {

#define GEN_PASS_DECL_HANDSHAKESPECPOSTBUFFER
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKE_SPEC_POST_BUFFER_H
