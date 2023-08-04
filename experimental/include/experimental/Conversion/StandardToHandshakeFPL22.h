// TODO

#ifndef EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H
#define EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

#define GEN_PASS_DECL_STANDARDTOHANDSHAKEFPL22
#define GEN_PASS_DEF_STANDARDTOHANDSHAKEFPL22
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createStandardToHandshakeFPL22Pass();

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_STANDARDTOHANDSHAKEFPL22_H