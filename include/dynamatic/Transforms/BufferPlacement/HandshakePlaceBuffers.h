//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --handshake-place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG = false,
                                std::string stdLevelInfo = "",
                                std::string timefile = "",
                                double targetCP = 4.0, int timeLimit = 180,
                                bool setCustom = true);

} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
