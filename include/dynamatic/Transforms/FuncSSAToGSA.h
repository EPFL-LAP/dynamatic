/// Classical C-style header guard
#ifndef DYNAMATIC_TRANSFORMS_FUNCSSATOGSA
#define DYNAMATIC_TRANSFORMS_FUNCSSATOGSA

/// Include some basic headers

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_FUNCSSATOGSA
#define GEN_PASS_DEF_FUNCSSATOGSA
#include "dynamatic/Transforms/Passes.h.inc"

struct SSAPhi {
  Block *owner_block;
  SmallVector<Operation *, 4> producer_operations;
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createFuncSSAToGSA();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKESSATOGSA