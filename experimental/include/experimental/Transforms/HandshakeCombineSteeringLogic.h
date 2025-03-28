//===- HandshakeCombineSteeringLogic.h - Simplify FTD  ----*- C++ -*-------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass which simplify the resulting FTD circuit by
// merging units which have the smae inputs and the same outputs.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H
#include "dynamatic/Support/DynamaticPass.h"
namespace dynamatic {
namespace experimental {
namespace ftd {

std::unique_ptr<dynamatic::DynamaticPass> combineSteeringLogic();

#define GEN_PASS_DECL_HANDSHAKECOMBINESTEERINGLOGIC
#define GEN_PASS_DEF_HANDSHAKECOMBINESTEERINGLOGIC
#include "experimental/Transforms/Passes.h.inc"

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKECOMBINESTEERINGLOGIC_H
