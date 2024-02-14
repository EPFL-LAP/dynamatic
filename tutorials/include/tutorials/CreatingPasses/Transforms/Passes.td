//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition for all transformation passes in this
// tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_CREATINGPASSES_TRANSFORMS_PASSES_TD
#define TUTORIALS_CREATINGPASSES_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SimplifyMergeLike : Pass< "tutorial-handshake-simplify-merge-like", 
                              "mlir::ModuleOp"> {
  let summary = "Simplifies merge-like operations in Handshake functions.";
  let description = [{
    The pass performs two simple transformation steps sequentially in each
    Handshake function present in the input MLIR module. First, it bypasses and
    removes all merge operations (dynamatic::handshake::MergeOp) with a single
    operand from the IR, since they serve no purpose. Second, it downgrades all
    control merge operations (dynamatic::handshake::ControlMergeOp) whose index
    result is unused into simpler merges with the same operands.
  }];
  let constructor = "dynamatic::tutorials::createSimplifyMergeLikePass()";
}

def GreedySimplifyMergeLike : Pass< 
  "tutorial-handshake-greedy-simplify-merge-like", "mlir::ModuleOp"> {
  let summary = "Greedy merge-like operation simplification.";
  let description = [{
    The pass performs the same two transformation steps as the SimplifyMergeLike
    pass, but uses a greedy pattern rewriter to drive the transformation. In
    particular, this enables control merges with a single input and with an
    unused index result to be completely removed from the IR thanks to the
    successive application of both rewrite patterns; first, the control merge
    downgrading pattern, and then, the merge removal pattern on the merge
    operation resulting from the first pattern.  
  }];
  let constructor = "dynamatic::tutorials::createGreedySimplifyMergeLikePass()";
}

#endif // TUTORIALS_CREATINGPASSES_TRANSFORMS_PASSES_TD