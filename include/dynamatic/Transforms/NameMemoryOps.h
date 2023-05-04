//===- NameMemoryOps.h - Give a unique name to all memory ops ---*- C++ -*-===//
//
// This file declares the --name-memory-ops pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H
#define DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createNameMemoryOps();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_NAMEMEMORYOPS_H
