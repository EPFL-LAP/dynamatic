//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "dynamatic/Transforms/UtilsForExtractMG.h"
// #include "dynamatic/Transforms/UtilsForMILPSolver.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(std::string stdLevelInfo="");

} // namespace dynamatic


#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
