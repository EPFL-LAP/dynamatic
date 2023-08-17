//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains the definition for all transformation passes in this
// tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef TUTORIALS_MYCREATINGPASSES_TRANSFORMS_PASSES_TD
#define TUTORIALS_MYCREATINGPASSES_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MySimplifyMergeLike : Pass< "tutorial-handshake-my-simplify-merge-like", 
                                "mlir::ModuleOp"> {
  let summary = "Simplifies merge-like operations in Handshake functions.";
  let description = [{
    The pass performs two simple transformation steps sequentially in each
    Handshake function present in the input MLIR module. First, it bypasses and
    removes all merge operations (circt::handshake::MergeOp) with a single
    operand from the IR, since they serve no purpose. Second, it downgrades all
    control merge operations (circt::handshake::ControlMergeOp) whose index
    result is unused into simpler merges with the same operands.
  }];
  let constructor = "dynamatic::tutorials::createMySimplifyMergeLikePass()";
}

#endif // TUTORIALS_MYCREATINGPASSES_TRANSFORMS_PASSES_TD