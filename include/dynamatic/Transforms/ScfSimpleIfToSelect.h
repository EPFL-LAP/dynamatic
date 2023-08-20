//===- ScfSimpleIfToSelect.h - Transform if's into select's -----*- C++ -*-===//
//
// This file declares the --scf-simple-if-to-select pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H
#define DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createScfSimpleIfToSelect();

#define GEN_PASS_DECL_SCFSIMPLEIFTOSELECT
#define GEN_PASS_DEF_SCFSIMPLEIFTOSELECT
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SCFSIMPLEIFTOSELECT_H
