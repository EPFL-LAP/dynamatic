//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains definitions for transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_TD
#define DYNAMATIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// Arith passes
//===----------------------------------------------------------------------===//

def ArithReduceStrength : Pass<"arith-reduce-strength", "mlir::ModuleOp"> {
  let summary = "Strength reduction of arithmetic operations";
  let description = [{
    Performs strength reduction on arithmetic operations in the context of
    dataflow circuits implemented on FPGAs. The goal of the pass is to increase
    circuit performance and reduce the area corresponding to arithmetic
    operations present in the circuit.
  }];
  let constructor = "dynamatic::createArithReduceStrength()";
  let options =
    [Option<"maxAdderDepthMul", "max-adder-depth-mul", "unsigned", "3",
            "The maximum number of adders that can be chained one after the "
            "other when replacing a multiplication with a tree of adders. "
            "Above this number, we consider that there would be no desirable "
            "performance/area advantage to replace the multiplication.">];
}

//===----------------------------------------------------------------------===//
// SCF passes
//===----------------------------------------------------------------------===//

def ScfForLoopRotation : Pass<"scf-rotate-for-loops", "mlir::ModuleOp"> {
  let summary = "Transforms eligible for loops into do-while's";
  let description = [{
    Rotates for loops, semantically transforming them into do-while's, whenever
    it can be determined that they execute at least once. The resulting loop at
    the std level usually has one less basic block than the original for loop
    would have had, which ultimately reduces the area of the final circuit.
  }];
  let constructor = "dynamatic::createScfRotateForLoops()";
}

def ScfSimpleIfToSelect : Pass<"scf-simple-if-to-select", "mlir::ModuleOp"> {
  let summary = "Transforms if's with simple bodies into select's.";
  let description = [{
    Transforms scf::IfOp operations with one result and a specific internal
    struture into an equivalent combination of a single arith::SelectOp and
    (potentially) an arithmetic operation present in at least one of its
    branches.
  
    The pass transforms scf::IfOp operations with the following properties.
    (1) The operation must have exactly one integer-like result.
    (2) Each branch must be made up of a single yield or of a supported
        two-operands arithmetic operation whose single result is yielded. If
        both branches have an arithmetic operation, they must be of the same
        type.
    (3) Both branches, whatever their content (constrained by the point above),
        must use an identical value. This could either be the yielded value or,
        if applicable, the value used as one of the operands of the arithmetic
        operation.
  }];
  let constructor = "dynamatic::createScfSimpleIfToSelect()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// Handshake passes
//===----------------------------------------------------------------------===//

def HandshakeCanonicalize : Pass<"handshake-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize Handshake operations.";
  let description = [{
    Canonicalizes and folds all Handshake operations in the module using a
    greedy pattern rewriter. The pass preserves the circuit's materialization
    status i.e., it inserts forks and sinks as needed to compensate for changes
    made to the IR.
  }];
  let constructor = "dynamatic::createHandshakeCanonicalize()";
}

def HandshakeInferBasicBlocks : Pass<"handshake-infer-basic-blocks", "mlir::ModuleOp"> {
  let summary = "Try to infer the basic block of untagged operations.";
  let description = [{
    Attempts to infer the logical basic block of all operations inside
    handshake::FuncOp's that are not attached to a logical basic block. The
    inference logic is conservative and relies on the operation's successors and
    predecessors to determine the operation's basic block. When the pass's
    heuristics manage to infer an operation's block, the pass sets the
    `dynamatic::BB_ATTR` on the operation. This pass does not fail, even when it
    does not manage to infer the basic block of all operations.
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

def HandshakeOptimizeBitwidths : Pass<"handshake-optimize-bitwidths",
                                      "mlir::ModuleOp"> {
  let summary = "Optimize bitwidths of all channels.";
  let description = [{
    Reduces the bitwidths of channels wherever possible. Uses a standard
    forward-backward iterative process to identify opportunities for bitwidth
    reduction, considering the specific semantics of all operations present at
    the Handhsake level.
  }];
  let constructor = "dynamatic::createHandshakeOptimizeBitwidths()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options = [Option<"legacy", "legacy", "bool", "false",
    "When legacy mode is enabled, disables bitwidth optimization of memory "
    "address channels since they sometime make the legacy dot2vhdl tool "
    "fail.">];
}

def HandshakeOptimizeBits : Pass<"optimize-bits" , "mlir::ModuleOp"> {
  let summary = "Optimize bits width that can be reduced.";
  let description = [{
    This pass goes through all operations inside handshake levels, and optimize the bit width through the forward (from first operation to last), and backward(from last operation to first) loop process. 
    The loop continues until no more optimization can be done.
  }];
  let constructor = "dynamatic::createOptimizeBitsPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def HandshakeConcretizeIndexType : Pass<"handshake-concretize-index-type", "mlir::ModuleOp"> {
  let summary = "Concretizes the IndexType to an IntegerType of machine-specific width.";
  let description = [{
    Replaces any IndexType throughout the Handhsake-level IR with an IntegerType
    of machine-specific width. In particular, all SSA values of type IndexType
    have their type changed to IntegerType.
  }];
  let constructor = "dynamatic::createHandshakeConcretizeIndexType()";
  let options =
    [Option<"width", "width", "unsigned", "64",
            "The explicit bitwidth with which to concretize IndexType's. "
            "Defaults to 64 since most machines have a 64-bit architecture. If "
            "this value is such that a Handshake constant with an IndexType "
            "result cannot fit its value in the new explicit bitwidth anymore, "
            "the pass will fail.">];
}

def HandshakeMinimizeCstWidth : Pass<"handshake-minimize-cst-width", "mlir::ModuleOp"> {
  let summary = "Minimizes the bitwidth of all Handshake constants.";
  let description = [{
    Rewrites constant operations with the minimum required bitwidth to support
    the constants' values. The pass inserts extension operations as needed to
    ensure consistency with users of constant operations. The pass also pays
    attention to not create duplicated constants indirectly due to the
    minimization process.
  }];
  let constructor = "dynamatic::createHandshakeMinimizeCstWidth()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options =
    [Option<"optNegatives", "opt-negatives", "bool", "false",
            "If true, allows bitwidth optimization of negative values.">];
}

def HandshakeSetBufferingProperties :
  Pass<"handshake-set-buffering-properties", "mlir::ModuleOp"> {
  let summary = "Attach buffering properties to specifc channels in the IR";
  let description = [{
    Adds buffering properties to specific channels in the IR. Which channels are
    annotated and how is determined by the pass's parameter. This pass is
    especially useful to run prior to running our smart buffer placement pass,
    as the latter reads these properties and ensures that the buffer placement
    it yields honors them.
  }];
  let constructor =
    "dynamatic::buffer::createHandshakeSetBufferingProperties()"; 
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
  let options =
    [Option<"version", "version", "std::string", "\"fpga20\"",
            "Set of buffering properties to apply. Options are: fpga20 "
            "(default).">];
}

def HandshakePlaceBuffers : Pass<"handshake-place-buffers", "mlir::ModuleOp"> {
  let summary = "Insert buffers in dataflow graphs.";
  let description = [{
    The pass analyzes the circuit's channel prerequisites. Based the channel
    prerequisite, the pass optimize the circuit performance by inserting buffers.
  }];
  let constructor = "dynamatic::buffer::createHandshakePlaceBuffersPass()";
  let options = [
    Option<"firstCFDFC", "firstCFDFC", "bool", "false",
    "if true, only extract the first CFDFC from the input file">,
    Option<"stdLevelInfo", "std-level-info", "std::string", "",
    "if provided, extract CFDFC from standard level simluation">,
    Option<"timefile", "time-file", "std::string", "",
    "required, file provide the time and buffer prerequisite information ">,
    Option<"targetCP", "target-period", "double", "4.0",
    "target clock period for the CFDFC">,
    Option<"dumpLogs", "dump-logs", "bool", "false",
    "If true, dump debugging information (identified CFDFCs, MILP "
    "placement decisions, solution summaries) into a directory named "
    "'buffer-placement' at the binary's location.">];
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

def FlattenMemRefRowMajor : Pass<"flatten-memref-row-major", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs (row-major order)";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories using row-major indexing. This pass is almost
    copied as is from CIRCT's FlattenMemRef, but uses row-major indexing instead
    of column-major indexing when converting multidimensional load/store
    operations.}];
  let constructor = "dynamatic::createFlattenMemRefRowMajorPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
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

#endif // DYNAMATIC_TRANSFORMS_PASSES_TD
