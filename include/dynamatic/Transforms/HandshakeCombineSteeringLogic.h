//===- HandshakeCombineSteeringLogic.h - RCombines multiple Branches (and
// Merges as well as Muxes) that are having the same input but feeding different
// outputs
//----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-combine-steering-logic pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKECOMBINESTEERINGLOGIC
#define GEN_PASS_DEF_HANDSHAKECOMBINESTEERINGLOGIC
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> combineSteeringLogic();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H
