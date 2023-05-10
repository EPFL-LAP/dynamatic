//===- ExportCFG.h - Export CFG of std-level function -----------*- C++ -*-===//
//
// This file declares the --export-cfg pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_EXPORTCFG_H
#define DYNAMATIC_CONVERSION_EXPORTCFG_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportCFGPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_EXPORTCFG_H
