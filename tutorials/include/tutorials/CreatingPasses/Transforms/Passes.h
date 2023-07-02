//===- Passes.h - Transformation passes registration ------------*- C++ -*-===//
//
// This file contains declarations to register transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_CREATINGAPASSES_TRANSFORMS_PASSES_H
#define TUTORIALS_CREATINGAPASSES_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "tutorials/CreatingPasses/Transforms/GreedySimplifyMergeLike.h"
#include "tutorials/CreatingPasses/Transforms/SimplifyMergeLike.h"

namespace dynamatic {
namespace tutorials {

namespace CreatingPasses {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "tutorials/CreatingPasses/Transforms/Passes.h.inc"

} // namespace CreatingPasses
} // namespace tutorials
} // namespace dynamatic

#endif // TUTORIALS_CREATINGAPASSES_TRANSFORMS_PASSES_H