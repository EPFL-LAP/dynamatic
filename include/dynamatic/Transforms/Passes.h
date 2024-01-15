//===- Passes.h - Transformation passes registration ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations to register transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_H
#define DYNAMATIC_TRANSFORMS_PASSES_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/ArithReduceStrength.h"
#include "dynamatic/Transforms/BackAnnotate.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakeSetBufferingProperties.h"
#include "dynamatic/Transforms/FlattenMemRefRowMajor.h"
#include "dynamatic/Transforms/ForceMemoryInterface.h"
#include "dynamatic/Transforms/HandshakeCanonicalize.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeInferBasicBlocks.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "dynamatic/Transforms/HandshakeMinimizeCstWidth.h"
#include "dynamatic/Transforms/HandshakeMinimizeLSQUsage.h"
#include "dynamatic/Transforms/HandshakeOptimizeBitwidths.h"
#include "dynamatic/Transforms/HandshakePrepareForLegacy.h"
#include "dynamatic/Transforms/MarkMemoryDependencies.h"
#include "dynamatic/Transforms/MarkMemoryInterfaces.h"
#include "dynamatic/Transforms/OperationNames.h"
#include "dynamatic/Transforms/PushConstants.h"
#include "dynamatic/Transforms/RemovePolygeistAttributes.h"
#include "dynamatic/Transforms/ScfRotateForLoops.h"
#include "dynamatic/Transforms/ScfSimpleIfToSelect.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PASSES_H