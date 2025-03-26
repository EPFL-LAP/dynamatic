/// Classical C-style header guard
#ifndef DYNAMATIC_TRANSFORMS_HWRIGIDIFICATION_H
#define DYNAMATIC_TRANSFORMS_HWRIGIDIFICATION_H

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

std::unique_ptr<dynamatic::DynamaticPass> createHWRigidificationPass();

#define GEN_PASS_DECL_HWRIGIDIFICATION
#define GEN_PASS_DEF_HWRIGIDIFICATION
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace rigidification
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HWRIGIDIFICATION_H