/// Classical C-style header guard
#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKERIGIDIFICATION_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKERIGIDIFICATION_H

/// Include some basic headers
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace dynamatic;

namespace dynamatic {
namespace experimental {
namespace rigidification {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeRigidification();

#define GEN_PASS_DECL_HANDSHAKERIGIDIFICATION
#define GEN_PASS_DEF_HANDSHAKERIGIDIFICATION
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace rigidification
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKERIGIDIFICATION_H