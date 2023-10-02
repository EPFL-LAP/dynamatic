//===- GreedySimplifyMergeLike.h - Simplifies merge-like ops ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --tutorial-handshake-greedy-simplify-merge-like pass.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_CREATINGPASSES_TRANSFORMS_GREEDYSIMPLIFYMERGELIKE_H
#define TUTORIALS_CREATINGPASSES_TRANSFORMS_GREEDYSIMPLIFYMERGELIKE_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace tutorials {

#define GEN_PASS_DECL_GREEDYSIMPLIFYMERGELIKE
#define GEN_PASS_DEF_GREEDYSIMPLIFYMERGELIKE
#include "tutorials/CreatingPasses/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createGreedySimplifyMergeLikePass();

} // namespace tutorials
} // namespace dynamatic

#endif // TUTORIALS_CREATINGPASSES_TRANSFORMS_GREEDYSIMPLIFYMERGELIKE_H