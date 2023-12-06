//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PASSES_TD
#define DYNAMATIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "dynamatic/Support/Passes.td"


//===----------------------------------------------------------------------===//
// Arith passes
//===----------------------------------------------------------------------===//

def ArithReduceStrength : DynamaticPass<"arith-reduce-strength"> {
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

def ScfForLoopRotation : DynamaticPass<"scf-rotate-for-loops"> {
  let summary = "Transforms eligible for loops into do-while's";
  let description = [{
    Rotates for loops, semantically transforming them into do-while's, whenever
    it can be determined that they execute at least once. The resulting loop at
    the std level usually has one less basic block than the original for loop
    would have had, which ultimately reduces the area of the final circuit.
  }];
  let constructor = "dynamatic::createScfRotateForLoops()";
}

def ScfSimpleIfToSelect : DynamaticPass<"scf-simple-if-to-select",
                                        ["mlir::arith::ArithDialect"]> {
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
}

//===----------------------------------------------------------------------===//
// Handshake passes
//===----------------------------------------------------------------------===//

def HandshakeCanonicalize : DynamaticPass<"handshake-canonicalize"> {
  let summary = "Canonicalize Handshake operations.";
  let description = [{
    Canonicalizes and folds all Handshake operations in the module using a
    greedy pattern rewriter. The pass preserves the circuit's materialization
    status i.e., it inserts forks and sinks as needed to compensate for changes
    made to the IR.
  }];
  let constructor = "dynamatic::createHandshakeCanonicalize()";
}

def HandshakeInferBasicBlocks : DynamaticPass<"handshake-infer-basic-blocks"> {
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

def HandshakePrepareForLegacy : DynamaticPass<"handshake-prepare-for-legacy"> {
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

def HandshakeOptimizeBitwidths : DynamaticPass< "handshake-optimize-bitwidths", 
                                                ["mlir::arith::ArithDialect"]> {
  let summary = "Optimize bitwidths of all channels.";
  let description = [{
    Reduces the bitwidths of channels wherever possible. Uses a standard
    forward-backward iterative process to identify opportunities for bitwidth
    reduction, considering the specific semantics of all operations present at
    the Handhsake level.
  }];
  let constructor = "dynamatic::createHandshakeOptimizeBitwidths()";
  let options = [Option<"legacy", "legacy", "bool", "false",
    "When legacy mode is enabled, disables bitwidth optimization of memory "
    "address channels since they sometime make the legacy dot2vhdl tool "
    "fail.">];
}

def HandshakeConcretizeIndexType : DynamaticPass<"handshake-concretize-index-type"> {
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

def HandshakeMinimizeCstWidth : DynamaticPass<"handshake-minimize-cst-width",
                                              ["mlir::arith::ArithDialect"]> {
  let summary = "Minimizes the bitwidth of all Handshake constants.";
  let description = [{
    Rewrites constant operations with the minimum required bitwidth to support
    the constants' values. The pass inserts extension operations as needed to
    ensure consistency with users of constant operations. The pass also pays
    attention to not create duplicated constants indirectly due to the
    minimization process.
  }];
  let constructor = "dynamatic::createHandshakeMinimizeCstWidth()";
  let options =
    [Option<"optNegatives", "opt-negatives", "bool", "false",
            "If true, allows bitwidth optimization of negative values.">];
}

def HandshakeSetBufferingProperties :
  DynamaticPass<"handshake-set-buffering-properties"> {
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
  let options =
    [Option<"version", "version", "std::string", "\"fpga20\"",
            "Set of buffering properties to apply. Options are: fpga20 "
            "(default).">];
}

def HandshakePlaceBuffers : DynamaticPass<"handshake-place-buffers"> {
  let summary = 
    "Places buffers in dataflow circuits for correctness and performance.";
  let description = [{
    Places buffers inside Handshake functions to ensure functional correctness
    and meet performance targets dictated by the chosen buffer placement
    algorithm. The currently available algorithms all rely on solving Mixed
    Integer Linear Programs (MILPs) to extract CFDFCs from circuits and decide
    where it makes sense to place buffers. To do so, we use the Gurobi C++ API
    which needs to be setup separately from Dynamatic (and is under a different,
    non-open-source licence).
  }];
  let constructor = "dynamatic::buffer::createHandshakePlaceBuffers()";
  let options = [
    Option<"algorithm", "algorithm", "std::string", "\"fpga20\"",
    "Algorithm to use for buffer placement. Choices are: 'on-merges' (default, "
    "does not require Gurobi) 'fpga20', 'fpga20-legacy' (same as fpga-20 but "
    "interprets the MILP's results as legacy Dynamatic would, placing at most "
    "a single buffer type on any given channel), 'fpl22'.">,
    Option<"frequencies", "frequencies", "std::string", "",
    "Path to CSV-formatted file containing estimated transition frequencies "
    "between basic blocks in the kernel.">,
    Option<"timingModels", "timing-models", "std::string", "",
    "Path to JSON-formatted file containing timing models for dataflow "
    "components.">,
    Option<"firstCFDFC", "first-cfdfc", "bool", "false",
    "If true, only extract the first CFDFC from the input file">,
    Option<"targetCP", "target-period", "double", "4.0",
    "Target clock period for the buffer placement CFDFC">,
    Option<"timeout", "timeout", "unsigned", "180",
    "Timeout (in seconds) for the buffer placement MILP (0 for no timeout)">,
    Option<"dumpLogs", "dump-logs", "bool", "false",
    "If true, dump debugging information (identified CFDFCs, MILP "
    "placement decisions, solution summaries) into a directory named "
    "'buffer-placement' at the binary's location.">];
}

//===----------------------------------------------------------------------===//
// Other passes
//===----------------------------------------------------------------------===//

def BackAnnotate : DynamaticPass<"back-annotate"> {
  let summary = "Back-annotates IR from JSON-formatted attributes";
  let description = [{
    Sets supported attributes on operations and operands based on
    back-annotation information contained in an input JSON file. Operations
    need to have a unique name (`circt::handshake::NameAttr` attribute) to be
    able to be referenced by the back-annotation file. Fails if the
    back-annotation file is badly formatted or if a referenced operation/operand
    cannot be found in the IR.
  }];
  let constructor = "dynamatic::createBackAnnotate()";
    let options =
    [Option<"filepath", "filepath", "std::string", "\"\"",
            "Path to JSON-formatted back-annotation file">];
}

def FlattenMemRefRowMajor : DynamaticPass<"flatten-memref-row-major",
                                          ["mlir::memref::MemRefDialect"]> {
  let summary = "Flatten memrefs (row-major order)";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories using row-major indexing. This pass is almost
    copied as is from CIRCT's FlattenMemRef, but uses row-major indexing instead
    of column-major indexing when converting multidimensional load/store
    operations.}];
  let constructor = "dynamatic::createFlattenMemRefRowMajorPass()";
}

def ForceMemoryInterface : DynamaticPass<"force-memory-interface"> {
  let summary = "Forces the use of a single type of memory interface";
  let description = [{
    Forces the type of memory interface that will be placed for all memory
    operations (from the affine or memref dialects) when lowering to Handshake.
    Internally, adds/modifies the `circt::handshake::MemoryInterfaceAttr`
    attribute on all memory operations in the IR. Exactly one of the two pass
    options should be provided at the same time.
  }];
  let constructor = "dynamatic::createForceMemoryInterface()";
  let options = [
    Option<"forceLSQ", "force-lsq", "bool", "false",
      "Whether to force the placement of LSQs in the circuit after Handshake "
      "lowering (off by default, mutually exclusice with force-mc).">,
    Option<"forceMC", "force-mc", "bool", "false",
      "Whether to force the placement of MCs in the circuit after Handshake "
      "lowering (off by default, mutually exclusice with force-lsq).">
  ];
}

def MarkMemoryDependencies : DynamaticPass<"mark-memory-dependencies"> {
  let summary = "Mark all memory dependencies in the IR.";
  let description = [{
    Analyzes memory accesses in each function in a module and identifies all
    dependencies between them. Stores information about these dependencies in a
    `circt::handshake::MemDependenceArrayAttr` attribute attached to each
    operation that is the source of at least one dependency.

    The pass attempts to find dependencies between all `affine::AffineLoadOp`,
    `affine::AffineStoreOp`, `memref::LoadOp` and `memref::StoreOp` operations.
    It uses MLIR's polyhedral analysis infrastructure to identify complex
    dependencies between pairs of affine accesses, and uses a pessimistic
    approach for pairs of accesses that involve at least one non-affine access.
    While this pass can be ran at any point before conversion to Handshake, it
    makes the most sense to run it as early as possible since affine analysis 
    relies on very high-level information.

    The pass will fail if MLIR's polyhedral analysis infrastructure fails to
    check for a dependence between two affine accesses.
  }];
  let constructor = "dynamatic::createMarkMemoryDependencies()";
}

def MarkMemoryInterfaces : DynamaticPass<"mark-memory-interfaces"> {
  let summary = "Mark memory accesses with the interface they should connect to.";
  let description = [{
    Annotates all memory accesses in the IR with a
    `circt::handshake::MemInterfaceAttr` attribute indicating the kind of
    memory interface they should eventually connect to. It uses memory
    dependencies derived during polyhedral analysis to determine the type of
    interface necessary for each memory port. As such this pass should typically
    be ran after the `--mark-memory-dependencies` pass.

    Internally, it uses `circt::handshake::MemDependenceArrayAttr` attributes
    attached to memory operations to determine whether any memory access depends
    on any other. If no such attributes are present it is assumed that there are
    no memory dependencies.  
  }];
  let constructor = "dynamatic::createMarkMemoryInterfaces()";
}

def NameAllOperations : Pass<"name-all-operations", "mlir::ModuleOp"> {
  let summary = "Gives a unique name to each operation in the IR.";
  let description = [{
    Uses our name analysis infrastructure to give a unique name to each
    operation in the IR, which is attached as an attribute to each operation.
  }];
  let constructor = "dynamatic::createNameAllOperations()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
}

def PushConstants : DynamaticPass<"push-constants"> {
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

def RemoveOperationNames : Pass<"remove-operation-names", "mlir::ModuleOp"> {
  let summary = "Deletes the unique name of all operations in the IR.";
  let description = [{
    Opposite to the `NameAllOperations` pass in that it removes any attribute
    containing a unique name for an operation. This is useful, for example, for
    FileCheck unit tests, for which we do not care that operation names are
    checked by the verifier (i.e., this pass may be ran after the pass under
    test to remove all names).
  }];
  let constructor = "dynamatic::createRemoveOperationNames()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
}

def RemovePolygeistAttributes : DynamaticPass<"remove-polygeist-attributes"> {
  let summary = "Removes useless module and function attributes set by Polygeist.";
  let description = [{
    Polygeist sets a couple attributes belonging to the `dlti` and `llvm`
    dialects on the top-level MLIR module and on functions that we do not care
    about. This simply removes them (avoid having to pass the
    --allow-unregistered-dialect flag to all optimizer invokations).
  }];
  let constructor = "dynamatic::createRemovePolygeistAttributes()";
}

#endif // DYNAMATIC_TRANSFORMS_PASSES_TD
