//===- HandshakeReshapeChannels.h - Reshape channels' signals ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-reshape-channels pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKERESHAPE_CHANNELS_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKERESHAPE_CHANNELS_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKERESHAPECHANNELS
#define GEN_PASS_DEF_HANDSHAKERESHAPECHANNELS
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeReshapeChannels();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKERESHAPE_CHANNELS_H