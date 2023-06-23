//===- HandshakeMinimizeCstWidth.h - Min. constants bitwidth ----*- C++ -*-===//
//
// This file declares the --handshake-minimize-cst-width pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeMinimizeCstWidth();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMINIMIZECSTWIDTH_H