# C Frontend

This document describes Dynamatic's C frontend (under `tools/translate-llvm-to-std`).

Sections:

1. [Design](#design): The principle of the C frontend, and how to interact with it.
2. [LLVM to MLIR translation](#LLVM to MLIR translation)
3. [Memory dependency analysis](#Memory dependency analysis)

## Design 

### What Does a C Frontend Do?

Dynamatic's C frontend has three main objectives:
- Converting (a well-defined subset of) C to the standard MLIR dialects (see below).
- Performing generic IR optimizations.
- Performing memory analysis (which load depends on which store?).

The output of Dynamatic's C frontend is an MLIR IR written in standard MLIR dialect, namely:
- ControlFlow dialect for representing control flow operations.
- Arith dialect for representing basic math operations.
- Math dialect for advanced floating-point math operations.
- MemRef for representing arrays and memory accesses.

### Reused Components from the LLVM project

Since Dynamatic depends on the `llvm-project`, it can reuse many components:
- `clang` can translate the input C code to LLVM IR.
- `opt` provides a variety of transformations (e.g., cfg optimization, code motion, strength reduction, etc) and analyses (e.g., pointer alias analysis).
- `polly` provides polyhedral analysis (we use this to analyze the index accessing pattern).

#### Notable Optimizations that We Need from the LLVM Project

- `mem2reg`: Suppresses allocas (allocate memory on the heap) into regs.
- `instcombine`: Local DAG-to-DAG rewriting. Notably, this canonicalizes a chain of GEPs.
- `loop-rotate`: transform loops to do-while loops as much as possible.
- `simplifycfg`, `loopsimplify`: reducing the number of BBs (fewer branches).
- `consthoist`: Moving constants around.
- `licm`: Applies loop-invariant code motion to make the loops simplier.

### Components in Dynamatic: 

- **LLVM IR to MLIR Standard Dialects**. As of Oct. 2025, there is no working implementation in the LLVM project that can translate LLVM IR to MLIR. Therefore, Dynamatic utilizes a conversion tool for translating the LLVM IR directly to MLIR standard dialects.
- **Memory dependency annotation**. LLVM/Polly's memory dependency analysis cannot be directly imported into Dynamatic's toolchain. We implement a memory dependency analysis tool to generate them.

> [!NOTE]
> **Design choice**. Notably, MLIR internally has a translation tool for converting LLVM IR to the LLVM dialect. So one alternative is to build an MLIR dialect translation pass to convert the LLVM dialect to the standard dialect.
> 
We decided not to use this approach because the LLVM IR-to-LLVM dialect translation does not preserve custom LLVM metadata. This would drop the memory analysis annotation (marked as llvm metadata nodes).

## LLVM to MLIR translation

The translation between LLVM IR and the standard dialects (especially the subset that is supported in Dynamatic) is mostly straightforward. 

> [!NOTE]
> Key differences between LLVM IR and MLIR:
> - LLVM uses void ptrs for array inputs (both for fixed-size arrays `int arr[10][20]` and arrays with unbounded length `int * arr`). While in standard dialect, we use memref types `memref<10 * 20 * i32>` for referencing an array.
> - LLVM does not represent constants as operations, while in MILR, constants must be "materialized" as explicit constant operations.
> - The MemRef dialect does not have a special GEP operation for the array index calculation (e.g., `a[0][1]`); instead, it has a high-level syntax like `%result = memref.load [%memrefValue] %dim0, %dim1`. Therefore, GEPs are replaced by a direct connection between indices to the loads/stores. 
> - In LLVM, global values can be referenced by GEPs, but in MLIR memref dialect, global values can only be referenced via `get_global` op via the `sym_name` symbol attached to the global op.

### Type Conversion for Function Arguments

Since LLVM discards the argument types of the array arguments, `translate-llvm-to-std` analyzes the AST of the original input C code to recover the dimension(s) and the types.

### LLVM to Std Translation Algorithm

During translation, Dynamatic maintains the following mapping between the LLVM
IR and MLIR:
- **Basic blocks**: `llvm::BasicBlock *` to `mlir::Block`. We translate operations block by block. 
- **Values or block arguments**: `llvm::Value*` to `mlir::Value`. Input values are necessary to build MLIR arguments. This mapping identifies the input values of an MLIR operation. Notably, we do not need to keep track of the mapping between LLVM instructions and MLIR operations.
- **Get element pointer**: `llvm::Value *` (the gep instruction in LLVM) to `SmallVector<mlir::Value>` (the indices in MLIR). This mapping keeps track of the index operands of loads/stores in MLIR.
- **Global values**: `llvm::Value *` to `memref::GlobalOp`. This mapping is needed in addition to the normal value mappings because MLIR does not reference the global ops via `mlir::Value`.

Dynamatic performs these translation steps for the LLVM module:

- Create a corresponding `arith::ConstantOp` for each constant input of each `llvm::Instruction *` in LLVM IR.
- Create an MLIR function for each LLVM function.

For each LLVM function, Dynamatic performs the following translation:

- Create an MLIR block for every basic block in LLVM. Remember the BB mappings. For every block argument in LLVM, it creates the corresponding block argument in MLIR (for each array function argument, the original C code is used to recover the correct memref type). Remember the value mappings.
- Create a memref global operation for each global variable in LLVM.
- Create an operation in LLVM for each MLIR operation (*with exception*).

> [!IMPORTANT]
> The `instCombine` pass must be applied before the conversion to eliminate a
> chain of GEPs.

## Memory dependency analysis

TODO
