//===- ScfRotateForLoops.h - Rotate for loops into do-while's ---*- C++ -*-===//
//
// This file declares the --scf-rotate-for-loops pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SCFROTATEFORLOOPS_H
#define DYNAMATIC_TRANSFORMS_SCFROTATEFORLOOPS_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createScfRotateForLoops();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SCFFORLOOPROTATION_H
