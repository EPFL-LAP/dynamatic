//===- HandshakeSizeLSQs.h - Sizes the LSQs ---------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-size-lsqs pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SIZE_LSQS_H
#define DYNAMATIC_TRANSFORMS_SIZE_LSQS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {
namespace lsqsizing {
#define GEN_PASS_DECL_HANDSHAKESIZELSQS
#define GEN_PASS_DEF_HANDSHAKESIZELSQS
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSizeLSQs(StringRef timingModels = "", StringRef collisions = "",
                        double targetCP = 10.0);

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SIZE_LSQS_H
