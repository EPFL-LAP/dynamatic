//===- ExportDOT.h - Export handshake to DOT pass ---------------*- C++ -*-===//
//
// This file declares the --export-dot pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_EXPORTDOT_H
#define DYNAMATIC_CONVERSION_EXPORTDOT_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportDOTPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_EXPORTDOT_H
