//===- HandshakeSpeculation.h - Speculation units placement -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-speculation pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace dynamatic {
namespace speculation {

#define GEN_PASS_DECL_HANDSHAKESPECULATION
#define GEN_PASS_DEF_HANDSHAKESPECULATION
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSpeculation(const std::string &jsonPath = "");

} // namespace speculation
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H
