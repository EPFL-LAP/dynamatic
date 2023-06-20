//===- SimplifyMergeLike.h - Simplifies merge-like operations ---*- C++ -*-===//
//
// This file declares the --tutorial-handshake-simplify-merge-like pass.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_CREATINGPASSES_TRANSFORMS_SIMPLIFYMERGELIKE_H
#define TUTORIALS_CREATINGPASSES_TRANSFORMS_SIMPLIFYMERGELIKE_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace tutorials {

#define GEN_PASS_DEF_SIMPLIFYMERGELIKE
#include "tutorials/CreatingPasses/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createSimplifyMergeLikePass();

} // namespace tutorials
} // namespace dynamatic

#endif // TUTORIALS_CREATINGPASSES_TRANSFORMS_SIMPLIFYMERGELIKE_H