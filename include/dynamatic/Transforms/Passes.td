//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains definitions for transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_TD
#define DYNAMATIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def HandshakeToDot : Pass<"handshake-to-dot", "mlir::ModuleOp"> {
  let summary = "Generate .dot graph of a handshake function.";
  let description = [{
    This pass analyses a handshake::FuncOp operation and produces a .dot graph
    of its structure. If multiple functions are present in the IR, the top level
    function will be printed, and called functions will be subgraphs within the
    main graph.
  }];
  let constructor = "dynamatic::createHandshakeToDotPass()";
}

def HandshakeInferBasicBlocks : Pass<"infer-basic-blocks", "mlir::ModuleOp"> {
  let summary = "Tries to infer the basic block of untagged operations.";
  let description = [{
    This pass goes through all operations inside handshake::FuncOps that do not
    have a 'bb' attribute and tries to infer its value based on the operation's
    dataflow predecessors. When the pass's heuristic manages to infer the value,
    the 'bb' attribute is added to the operation. This pass never fails, even
    when it does not manage to infer the basic block of all operations.
  }];
  let constructor = "dynamatic::createHandshakeInferBasicBlocksPass()";
}

def ArithPushConstants : Pass<"push-constants", "mlir::ModuleOp"> {
  let summary = "Push constant definitions to blocks that use them.";
  let description = [{
    This pass moves arith::ConstantOp operations to the block(s) using them,
    effectively removing cross-block SSA dominance relationships between these
    operations and their user(s) within each function. This pass will duplicate
    constant definitions if they have users in multiple blocks. It will also
    delete the original operation if it not used within its defining block. 
  }];
  let constructor = "dynamatic::createArithPushConstants()";
}

#endif // DYNAMATIC_TRANSFORMS_PASSES_TD