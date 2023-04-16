//===- Passes.h - Transformation passes registration ------------*- C++ -*-===//
//
// This file contains declarations to register transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_H
#define DYNAMATIC_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/AnalyzeMemoryAccesses.h"
#include "dynamatic/Transforms/HandshakeInferBasicBlocks.h"
#include "dynamatic/Transforms/HandshakePrepareForLegacy.h"
#include "dynamatic/Transforms/NameMemoryOps.h"
#include "dynamatic/Transforms/PushConstants.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PASSES_H
