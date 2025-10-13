# C Frontend

This document describes Dynamatic's C frontend.

Sections:

1. [Design](#design): The principle of the C frontend, and how to interact with it.
2. [LLVM to MLIR translation](#LLVM to MLIR translation)
3. [Memory dependency analysis](#Memory dependency analysis)

## Design 

### What Does a C Frontend Do?

Dynamatic's C frontend has three main objectives:
- Converting (a well-defined subset of) C to the standard MLIR dialects (see below).
- Performing common optimizations, most importantly the control-flow graph optimization (CFG).
- Performing memory analysis (which load depends on which store?).

The output of Dynamatic's C frontend is an MLIR IR built written in standard MLIR dialect, namely:
- ControlFlow dialect for representing control flow operations.
- Arith dialect for representing basic math operations.
- Math dialect for advanced floating point math operations.
- MemRef for representing arrays and memory accesses.

### Reused Components from LLVM project

Since Dynamatic depends on the `llvm-project`, it can reuse many components:
- `clang` can translate the input C code to LLVM IR.
- `opt` provides a variety of transformations (e.g., cfg optimization, code motion, strength reduction, etc) and analyses (e.g., pointer alias analysis).
- `polly` provides polyhedral analysis (we use this to analysis the index accessing pattern).

### Components in Dynamatic: 

- **LLVM IR to MLIR Standard Dialects**. As of Oct. 2025, there is no working implementation in the LLVM project that can translate LLVM IR to MLIR. Therefore, Dynamatic utilizes a conversion tool for translating the LLVM IR directly to MLIR standard dialects.
- **Memory dependency annotation**. LLVM/Polly's memory dependency analysis cannot be readily imported into Dynamatic's toolchain. We implement a memory dependency analysis tool to generate them.

## LLVM to MLIR translation

## Memory dependency analysis
