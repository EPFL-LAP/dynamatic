//===- HandshakeRigidification.h - Rigidification units placement -----*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-rigidification pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_RIGIDIFICATION_PASS_H
#define DYNAMATIC_TRANSFORMS_RIGIDIFICATION_PASS_H

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

std::unique_ptr<dynamatic::DynamaticPass>
createRigidification(const std::string &jsonPath = "");

#define GEN_PASS_DECL_HANDSHAKERIGIDIFICATION
#define GEN_PASS_DEF_HANDSHAKERIGIDIFICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace rigidification
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_RIGIDIFICATION_PASS_H
