//===- HandshakePrepareForLegacy.h - Prepare for legacy flow ----*- C++ -*-===//
//
// This file declares the --handshake-prepare-for-legacy pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePrepareForLegacy();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEPREPAREFORLEGACY_H
