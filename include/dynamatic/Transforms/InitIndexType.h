//===- InitIndexType.h - Transform IndexType to IntegerType -----*- C++ -*-===//
//
// This file declares the --init-indextype pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H
#define DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitIndexTypePass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_INITINDEXTYPE_H