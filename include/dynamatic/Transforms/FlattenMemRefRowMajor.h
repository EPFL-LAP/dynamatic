//===- FlattenMemRefRowMajor.h - Flatten memory accesses --------*- C++ -*-===//
//
// This file declares the --flatten-memref-row-major pass, which is almost
// identical to CIRCT's --flatten-memref pass but uses row-major indexing for
// converting multidimensional load and store operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H
#define DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createFlattenMemRefRowMajorPass();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_FLATTENMEMREFROWMAJOR_H
