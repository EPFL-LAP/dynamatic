//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
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
  let summary = "Simplifies merge-like operations in handshake functions.";
  let description = [{
    This is a longer and
    possibly multilined description of the pass.
  }];
  let constructor = "dynamatic::tutorials::createSimplifyMergeLikePass()";
}

#endif // TUTORIALS_CREATINGPASSES_TRANSFORMS_PASSES_TD