//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains definitions for transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_TD
#define DYNAMATIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// Handshake passes
//===----------------------------------------------------------------------===//

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
    file that will be compatible with legacy passes. Concretely, this pass (1)
    turns unconditional branches (which legacy Dynamatic never generates) into
    conditional branches with a constant condition input and a sinked false
    output and (2) downgraded cmerges to simple merges when possible.  
  }];
  let constructor = "dynamatic::createHandshakePrepareForLegacy()";
}

//===----------------------------------------------------------------------===//
// Other passes
//===----------------------------------------------------------------------===//

def AnalyzeMemoryAccesses : Pass<"analyze-memory-accesses", "mlir::ModuleOp"> {
  let summary = "Determine memory dependencies between different accesses.";
  let description = [{
    Analyzes memory accesses in each function in a module and identifies all
    dependencies between them. Stores information about these dependencies in a
    MemDependenceArrayAttr attribute attached to each operation that is the
    source of at least one dependence. The attribute belongs to the Handshake
    dialect as it only becomes useful during the std-to-handshake lowering pass.

    The pass attempts to find dependencies between all AffineLoadOp's,
    AffineStoreOp's, memref::LoadOp's and memref::StoreOp's. It uses MLIR's
    polyhedral analysis infrastructure to identify complex dependencies between
    pairs of affine accesses, and uses a pessimistic approach for pairs of
    accesses that involve at least one non-affine access.

    The pass expects to see unique names attached to each memory access (and
    stored in a MemAccessNameAttr) to allow operations to reference one another
    in the attributes holding dependence information. As such, running the
    NameMemoryOpsPass (--name-memory-ops) is a prerequisite to running this
    pass, which will fail if a memory access is unnamed. The pass will also fail
    if MLIR's polyhedral analysis infrastructure fails to check for a dependence
    between two affine accesses.
  }];
  let constructor = "dynamatic::createAnalyzeMemoryAccesses()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
}

def NameMemoryOps : Pass<"name-memory-ops", "mlir::ModuleOp"> {
  let summary = "Give a unique name to each memory operation.";
  let description = [{
    Sets a MemAccessNameAttr attribute on each AffineLoadOp, AffineStoreOp,
    memref::LoadOp, and memref::StoreOp instance in the module that holds a
    unique name (within the context of the parent func::FuncOp) for the
    operation. These names may be used later to express dependencies between
    specific memory accesses (for example, in the AnalyzeMemoryAccessesPass).
    The attribute belongs to the Handshake dialect (and is a simple wrapper
    around a StringAttr) as it only becomes useful during the std-to-handshake
    lowering pass.

    The pass fails if one of the previously mentioned memory operation already
    has one such attribute.
  }];
  let constructor = "dynamatic::createNameMemoryOps()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
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
    This pass goes through all operations inside handshake levels, and 
    optimize the bit width through the forward (from first operation to last), 
    and backward(from last operation to first) loop process. The loop continues 
    until no more optimization can be done.
  }];
  let constructor = "dynamatic::createBitsOptimizationPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def HandshakeInitIndexType : Pass<"init-indextype", "mlir::ModuleOp"> {
  let summary = "Initialize the index type of the module.";
  let description = [{
    This pass change all the index type within operands and result operands 
    to integer type with platform dependent bit width.
  }];
  let constructor = "dynamatic::createInitIndexTypePass()";
}

def HandshakeInitCstWidth : Pass<"init-cstwidth", "mlir::ModuleOp"> {
  let summary = "Initialize the constant bit witth of the module.";
  let description = [{
    This pass rewrites constant operation with the minimum required bit width 
    according to the value of the constant. To ensure the consistency with the
    user of the constant operation, the pass inserts a extension operation if necessary. 
  }];
  let constructor = "dynamatic::createInitCstWidthPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];

}

#endif // DYNAMATIC_TRANSFORMS_PASSES_TD