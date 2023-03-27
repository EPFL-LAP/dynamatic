//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains definitions for transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_TD
#define DYNAMATIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def HandshakeInferBasicBlocks : Pass<"handshake-infer-basic-blocks", "mlir::ModuleOp"> {
  let summary = "Try to infer the basic block of untagged operations.";
  let description = [{
    This pass goes through all operations inside handshake::FuncOp's that do not
    have a 'bb' attribute and tries to infer its value based on the operation's
    dataflow predecessors. When the pass's heuristic manages to infer the value,
    the 'bb' attribute is added to the operation. This pass never fails, even
    when it does not manage to infer the basic block of all operations.
  }];
  let constructor = "dynamatic::createHandshakeInferBasicBlocksPass()";
}

def HandshakePrepareForLegacy : Pass<"handshake-prepare-for-legacy", "mlir::ModuleOp"> {
  let summary = "Prepare the IR for usage in legacy Dynamatic.";
  let description = [{
    This pass transforms the circuit to make it follow conventions set by legacy
    Dynamatic. It should be ran before DOT export in order to generate a .dot
    file that will be compatible with legacy passes. At the moment, this pass
    only turns unconditional branches (which legacy Dynamatic never generates)
    into conditional branches with a constant condition input and a sinked false
    output.  
  }];
  let constructor = "dynamatic::createHandshakePrepareForLegacy()";
}

def PushConstants : Pass<"push-constants", "mlir::ModuleOp"> {
  let summary = "Push constant definitions to blocks using them.";
  let description = [{
    This pass moves arith::ConstantOp operations to the block(s) using them,
    effectively removing cross-block SSA dominance relationships between these
    operations and their user(s) within each function. This pass will duplicate
    constant definitions if they have users in multiple blocks. It will also
    delete the original operation if it is not used within its defining block.
  }];
  let constructor = "dynamatic::createPushConstantsPass()";
}

def HandshakeBitsOptimize : Pass<"optimize-bits" , "mlir::ModuleOp"> {
  let summary = "Optimize bits width that can be reduced.";
  let description = [{

  }];
  let constructor = "dynamatic::createBitsOptimizationPass()";
}

#endif // DYNAMATIC_TRANSFORMS_PASSES_TD