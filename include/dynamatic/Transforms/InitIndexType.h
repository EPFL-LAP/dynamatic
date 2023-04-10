//===- InitIndexType.h - Transform the Index Type to IntegerType with system bit width  ---------*- C++ -*-===//
//
// This file declares the --init-indextype pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H
#define DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H

#include "dynamatic/Support/LLVM.h"


namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitIndexTypePass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H
