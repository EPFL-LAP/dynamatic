//===- HandshakeMinimizeCstWidth.h - Min. constants bitwidth ----*- C++ -*-===//
//
// This file declares the --handshake-minimize-cst-width pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEMINIMIZECSTWIDTH
#define GEN_PASS_DEF_HANDSHAKEMINIMIZECSTWIDTH
#include "dynamatic/Transforms/Passes.h.inc"

/// Computes the minimum required bitwidth needed to store the provided integer.
unsigned computeRequiredBitwidth(APInt val);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeMinimizeCstWidth();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H