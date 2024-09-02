//===- HandshakeAnalyzeLSQUsage.h - LSQ flow analysis -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-analyze-lsq-usage pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEANALYZELSQUSAGE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEANALYZELSQUSAGE_H

#include "dynamatic/Support/DynamaticPass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEANALYZELSQUSAGE
#define GEN_PASS_DEF_HANDSHAKEANALYZELSQUSAGE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeAnalyzeLSQUsage();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEANALYZELSQUSAGE_H
