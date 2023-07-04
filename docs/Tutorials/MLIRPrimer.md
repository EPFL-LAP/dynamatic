# An MLIR Primer

This tutorial will introduce you to MLIR and its core constructs. It is intended as a short and very incomplete yet pragmatic first look into the framework for newcomers, and will provide you with valuable "day-0" information that you're likely to need as soon as you start developing in Dynamatic++. At many points, this tutorial will reference the official and definitely more complete [MLIR documentation](https://mlir.llvm.org/docs/), which you are invited to look up whenever you require more in-depth information about a particular concept. While this document is useful to get an initial idea of how MLIR works and of how to manipulate its data-structures, we strongly recommend the reader to follow a "learn by doing" philosophy. Reading documentation, especially of complex frameworks like MLIR, will only get you so far. Practice is the path toward actual understanding and mastering in the long run.

## Table of contents

- [High-level structure](#high-level-structure) | What are the core data-structures used throughout MLIR?
- [Traversing the IR](#traversing-the-ir) | How does one traverse the IR?
- [Values](#values) | What are values?
- [Operations](#operations) | What are operations and how does one manipulate them?
- [Regions](#regions) | What are regions?
- [Blocks](#blocks) | What are blocks?
- [Attributes](#attributes) | What are attributes and how does one manipulate them?
- [Dialects](#dialects) | What are MLIR dialects?
## [High-level structure](https://mlir.llvm.org/docs/LangRef/#high-level-structure)

The [MLIR language reference](https://mlir.llvm.org/docs/LangRef) has a very good description of the high-level data-structures that are core to the framework.

> MLIR is fundamentally based on a graph-like data structure of nodes, called `Operation`s, and edges, called `Value`s. Each `Value` is the result of exactly one `Operation` or `BlockArgument`, and has a `Value` `Type` defined by the type system. Operations are contained in `Block`s and `Block`s are contained in `Region`s. `Operation`s are also ordered within their containing block and `Block`s are ordered in their containing region, although this order may or may not be semantically meaningful in a given kind of region). Operations may also contain regions, enabling hierarchical structures to be represented.

All of these data-structures can be manipulated in C++ using their respective types (which are typesetted in the above paragraph). In addition, they can all be printed to a text file (by convention, a file with the `.mlir` extension) and parsed back to their in-memory representation at any point.

To summarize, every MLIR file (`*.mlir`) is recursively nested. It starts with a top-level operation (often, an `mlir::ModuleOp`) which may contain nested regions, each of which may contain an ordered list of nested blocks, each of which may contain an ordered list of nested operations, after which the hierarchy repeats.

## [Traversing the IR](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-ir-nesting)

### From top to bottom

Thanks to MLIR's recursively nested structure, it is very easy to traverse the entire IR recursively. Consider the following C++ function which finds and recursively traverses all operations nested within a provided operation.

```cpp
void traverseIRFromOperation(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks())
      for (mlir::Operation &nestedOp : block.getOperations())
        traverseIRFromOperation(&nestedOp);
}
```

MLIR also exposes the [`walk` method](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers) on the `Operation`, `Region`, and `block` types. `walk` takes as single argument a callback method that will be invoked recursively for all operations recursively nested under the receiving entity.

```cpp
void traverseAllOperationsInBlock(mlir::Block &block) {
  block.walk([&](mlir::Operation *op) {
    // Do something on the traversed operation here
  });
}
```

### From bottom to top

One may also get the parent entities of a given operation/region/block.

```cpp
// Let op be an Operation*
mlir::Operation* op = ...;

// All of the following functions may return a nullptr in case the receiving
// entity is currently unattached to a parent block/region/op or is a top-level
// operation

// Get the parent block the operation immediately belongs to
mlir::Block *parentBlock = op->getBlock();
// Get the parent region the operation immediately belongs to
mlir::Region *parentRegion = op->getParentRegion();
// Get the parent operation the operation immediately belongs to
mlir::Operation *parentOp = op->getParentOp();

// Get the parent region the block immediately belongs to
mlir::Region *blockParentRegion = parentBlock->getParent();
assert(parentRegion == blockParentRegion);
// Get the parent operation the block immediately belongs to
mlir::Operation *blockParentOp = parentBlock->getParentOp();
assert(parentOp == blockParentOp);

// Get the parent operation the region immediately belongs to
mlir::Operation *regionParentOp = parentRegion->getParentOp();
assert(parentOp == regionParentOp);
```


## Values



## [Operations](https://mlir.llvm.org/docs/LangRef/#operations)

In MLIR, everything is about operations. Operations are like "opaque functions" to MLIR; they may represent some abstraction (e.g., a function, with a `mlir::func::FuncOp` operation) or perform some computation (e.g., an integer addition, with a `mlir::arith::AddIOp`). There is no fixed set of operations; users may define their own operations with custon semantics and use them at the same time as MLIR-defined operations. Operations:
- are identified by a unique string
- can take 0 or more operands
- can return 0 or more results 
- can have [attributes](TODO-link-me!) (i.e., constant data stored in a dictionary)
- can have 0 or more successors (i.e., other operations that take in as operands at least one of the operation's results)

### C++ snippets

The snippet below shows you how to get an operation's information from C++.

```cpp
// Let op be an operation
mlir::Operation* op = ...;

// Get the unique string identifying the type of operation
StringRef name = op->getName().getStringRef();

// Get all operands of the operation
mlir::OperandRange allOperands = op->getOperands();
// Get the number of operands of the operation
size_t numOperands = op->getNumOperands();
// Get the first operand of the operation (will crash if 0 >= op->getNumOperands())
mlir::Value firstOperand = op->getOperand(0);

// Get all results of the operation
mlir::ResultRange allResults = op->getResults();
// Get the number of results of the operation
size_t numResults = op->getNumResults();
// Get the first result of the operation (will crash if 0 >= op->getNumResults())
mlir::OpResult firstResult = op->getResult(0);

// Get all attributes of the operation
mlir::DictionaryAttr allAttributes = op->getAttrDictionary();
// Try to get an attribute of the operation with name "attr-name"
mlir::Attribute someAttr = op->getAttr("attr-name");
if (someAttr)
  llvm::outs() << "Attribute attr-name exists\n";
else
  llvm::outs() << "Attribute attr-name does not exist\n";
// Try to get an integer attribute of the operation with name "attr-name" 
mlir::IntegerAttr someIntAttr = op->getAttrOfType<IntegerAttr>("attr-name");
if (someAttr)
  llvm::outs() << "Integer attribute attr-name exists\n";
else
  llvm::outs() << "Integer attribute attr-name does not exist\n";
```

<!-- One may also use the `getOps<OpTy>` method to only iterate over operations of a specific type (the `OpTy` type). The following function recursively traverses all integer additions (`mlir::arith::AddIOp`) nested within a provided operation.

```cpp
void traverseIntAddFromOperation(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks())
      for (mlir::arith::AddIOp addOp : block.getOps<mlir::arith::AddIOp>())
        traverseIntAddFromOperation(addOp);
}
```

As before, it's also possible to only iterate over a specific type of operation by providing an explicit operation type in the callback.

```cpp
void traverseAllIntAddInBlock(mlir::Block &block) {
  block.walk([&](mlir::arith::AddIOp addOp) {
    // Do something on the traversed operation here
  });
}
``` -->

## Regions

## Blocks

## Attributes

## Dialects

<!-- ## Printing to stdout/stderr

## Printing in DEBUG mode

## Getting value users

## Getting value uses -->


<!-- When iterating over a list of operations, it is also possible to only consider a specific type of operations. The following function identifies all integer additions ([`mlir::arith::AddIOp`](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddi-arithaddiop)) within a provided block. 

```cpp
void traverseAllAdditions(Block& block) {
  for (mlir::Operation &op : block.getOperations()) {
    if (!mlir::isa<mlir::arith::AddIOp>(&op))
      // Skip the loop body if the operation is not an integer addition
      continue;
    // Convert our generic operation to an AddIOp
    // Crashes if op is not actually of this type
    mlir::arith::AddIOp addOp = mlir::cast<mlir::arith::AddIOp>(&op); 
  }
}
```

Equivalently, one can use the `dyn_cast` MLIR function to slightly simplify the above code.

```cpp
void traverseAllAdditions(Block& block) {
  for (mlir::Operation &op : block.getOperations()) {
    // Attempts to convert our generic operation to an AddIOp
    // Returns a nullptr if op is not actually of this type 
    mlir::arith::AddIOp addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&op);
    if (!addOp)
      // Skip the loop body if the operation is not an integer addition
      continue 
  }
}
```

We can also combine variable assignation the and `nullptr` check inside a single `if` statement to simplify the code further.

```cpp
void traverseAllAdditions(Block& block) {
  for (mlir::Operation &op : block.getOperations()) {
    if (mlir::arith::AddIOp addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&op)) {
      // If we reach this point we know op is an integer addition 
    }
  }
}
``` -->
