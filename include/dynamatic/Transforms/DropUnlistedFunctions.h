#ifndef DYNAMATIC_TRANSFORM_DROPUNLISTEDFUNCTIONS_H
#define DYNAMATIC_TRANSFORM_DROPUNLISTEDFUNCTIONS_H

#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Attributes.h"

namespace dynamatic {

#define GEN_PASS_DECL_DROPUNLISTEDFUNCTIONS
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORM_DROPUNLISTEDFUNCTIONS_H