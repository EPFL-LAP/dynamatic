//===- SpeculationFir.h - Hardcoded Speculative units placement for FIR -----*-
// C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --speculation-fir pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_FIR_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_FIR_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace speculation {

#define GEN_PASS_DECL_SPECULATIONFIR
#define GEN_PASS_DEF_SPECULATIONFIR
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> placeSpeculativeUnitsFir();

} // namespace speculation
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_FIR_H
