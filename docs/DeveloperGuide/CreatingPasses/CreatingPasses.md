# Creating Dynamatic Compiler Passes

This tutorial will walk you through the creation of a simple transformation pass for Dynamatic that simplifies merge-like operations in *Handshake*-level IR. We'll look at the process of declaring a pass in TableGen format, creating a header file for the pass that includes the auto-generated pass declaration code, and implementing the transformation as part of an `mlir::OperationPass`. Then, we'll look at how to use a greedy pattern rewriter to make our pass easier to write and able to optimize the IR in more situations.
 
This tutorial assumes basic knowledge of C++, MLIR, and of the theory behind dataflow circuits. For a basic introduction to MLIR and its related jargon, see [the MLIR primer](../MLIRPrimer.md). The full (runnable!) source code for this tutorial is located in `tutorials/include/tutorials/CreatingPasses` (headers) as well as in `tutorials/lib/CreatingPasses` (sources), and is built alongside the rest of the project by default.

This tutorial is divided in the following chapters:

- [Chapter #1](1.SimplifyingMergeLikeOps.md) | Description of what we want to achieve with the transformation pass: simlifying merge-like operations in the IR.
- [Chapter #2](2.WritingASimplePass.md) | Writing an initial version of the pass that transforms the IR in (almost!) the intended way.
- [Chapter #3](2.GreedyPatternRewriting.md) | Improving the pass design and fixing our previous issue using a `GreedyPatternRewriterDriver`.