//===- InitIndexType.h - Transform IndexType to IntegerType -----*- C++ -*-===//
//
// This file declares the --init-indextype pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_INITINDTYPE_H
#define DYNAMATIC_TRANSFORMS_INITINDTYPE_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitIndTypePass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_INITINDTYPE_H