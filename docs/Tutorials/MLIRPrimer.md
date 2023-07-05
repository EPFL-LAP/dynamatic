# An MLIR Primer

This tutorial will introduce you to MLIR and its core constructs. It is intended as a short and very incomplete yet pragmatic first look into the framework for newcomers, and will provide you with valuable "day-0" information that you're likely to need as soon as you start developing in Dynamatic++. At many points, this tutorial will reference the official and definitely more complete [MLIR documentation](https://mlir.llvm.org/docs/), which you are invited to look up whenever you require more in-depth information about a particular concept. While this document is useful to get an initial idea of how MLIR works and of how to manipulate its data-structures, we strongly recommend the reader to follow a "learn by doing" philosophy. Reading documentation, especially of complex frameworks like MLIR, will only get you so far. Practice is the path toward actual understanding and mastering in the long run.

## Table of contents

- [High-level structure](#high-level-structure) | What are the core data-structures used throughout MLIR?
- [Traversing the IR](#traversing-the-ir) | How does one traverse the recursive IR top-to-bottom and bottom-to-top?
- [Values](#values) | What are values and how are they used by operations?
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
      for (mlir::Operation &nestedOp : block.getOperations()) {
        llvm::outs() << "Traversing operation " << op << "\n";
        traverseIRFromOperation(&nestedOp);
      }
}
```

MLIR also exposes the [`walk` method](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers) on the `Operation`, `Region`, and `block` types. `walk` takes as single argument a callback method that will be invoked recursively for all operations recursively nested under the receiving entity.

```cpp
void traverseAllOperationsInBlock(mlir::Block &block) {
  block.walk([&](mlir::Operation *op) {
    llvm::outs() << "Traversing operation " << op << "\n";
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

Values are the edges of the graph-like structure that MLIR models. Their corresponding C++ type is `mlir::Value`. All values are typed using either a built-in type or a custom user-defined type (the type of a value is itself a C++ type called `Type`), which may change at runtime but is subject to verification constraints imposed by the context in which the value is used. Values are either produced by [operations](#operations) as operation results (`mlir::OpResult`, which is a subtype of `mlir::Value`) or are defined by blocks as part of their block arguments (`mlir::BlockArgument`, also a subtype of `mlir::Value`). They are consumed by [operations](#operations) as operation operands. A value may have 0 or more uses, but should have exactly one producer (an operation or a block).

The following C++ snippet shows how to identify the type and producer of a value and prints the index of the producer's operation result/block argument that the value corresponds to.

```cpp
// Let value be a Value
mlir::Value value = ...;

// Get the value's type and check whether it is an integer type
mlir::Type valueType = value.getType();
if (mlir::isa<mlir::IntegerType>(valueType))
  llvm::outs() << "Value has an integer type\n";
else
  llvm::outs() << "Value does not have an integer type\n";

// Get the value's producer (either a block, if getDefiningOp returns a nullptr,
// or an operation)
if (mlir::Operation *definingOp = value.getDefiningOp()) {
  // Value is a result of its defining operation and can safely be casted as such
  mlir::OpResult valueRes = cast<mlir::OpResult>(value);
  // Find the index of the defining operation result that corresponds to the value 
  llvm::outs() << "Value is result number" << valueRes.getResultNumber(); << "\n";
} else {
  // Value is a block argument and can safely be casted as such
  mlir::BlockArgument valueArg = cast<mlir::BlockArgument>(value);
  // Find the index of the block argument that corresponds to the value 
  llvm::outs() << "Value is result number" << valueArg.getArgNumber() << "\n";
}
```

The following C++ snippet shows how to iterate through all the operations that use a particular value as operand. Note that the number of uses may be equal *or larger* than the number of users because a single user may use the same value multiple times (but at least once) in its operands.

```cpp
// Let value be a Value
mlir::Value value = ...;

// Iterate over all uses of the value (i.e., over operation operands that equal
// the value)
for (mlir::OpOperand &use : val.getUses()) {
  // Get the owner of this particular use 
  mlir::Operation *useOwner = use.getOwner();
  llvm::outs() << "Value is used as operand number " 
               << use.getOperandNumber() << " of operation "
               << useOwner << "\n";
}

// Iterate over all users of the value
for (mlir::Operation *user : val.getUsers())
  llvm::outs() << "Value is used as an operand of operation " << user << "\n";
```

## [Operations](https://mlir.llvm.org/docs/LangRef/#operations)

In MLIR, everything is about operations. Operations are like "opaque functions" to MLIR; they may represent some abstraction (e.g., a function, with a `mlir::func::FuncOp` operation) or perform some computation (e.g., an integer addition, with a `mlir::arith::AddIOp`). There is no fixed set of operations; users may define their own operations with custon semantics and use them at the same time as MLIR-defined operations. Operations:
- are identified by a unique string
- can take 0 or more operands
- can return 0 or more results 
- can have [attributes](#attributes) (i.e., constant data stored in a dictionary)

The C++ snippet below shows how to get an operation's information from C++.

```cpp
// Let op be an Operation*
mlir::Operation* op = ...;

// Get the unique string identifying the type of operation
mlir::StringRef name = op->getName().getStringRef();

// Get all operands of the operation
mlir::OperandRange allOperands = op->getOperands();
// Get the number of operands of the operation
size_t numOperands = op->getNumOperands();
// Get the first operand of the operation (will fail if 0 >= op->getNumOperands())
mlir::Value firstOperand = op->getOperand(0);

// Get all results of the operation
mlir::ResultRange allResults = op->getResults();
// Get the number of results of the operation
size_t numResults = op->getNumResults();
// Get the first result of the operation (will fail if 0 >= op->getNumResults())
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

### [Op vs Operation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)

As we saw above, you can manipulate any operation in MLIR using the "opaque" `Operation` type (usually, you do so through an `Operation*`) which provides a generic API into an operation instance. However, there exists another type, `Op`, whose derived classes model a specific type of operation (e.g., an integer addition with a `mlir::arith::AddIOp`). From the [official documentation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations):

> `Op` derived classes act as smart pointer wrapper around a `Operation*`, provide operation-specific accessor methods, and type-safe properties of operations. [...] A side effect of this design is that we always pass around `Op` derived classes “by-value”, instead of by reference or pointer.

Whenever you want to manipulate an operation of a specific type, you should do so through its actual type that derives from `Op`. Fortunately, it is easy to identify the actual type of an `Operation*` using MLIR's casting infrastructure. The following snippet shows a few different methods to check whether an opaque `Operation*` is actually an integer addition (`mlir::arith::AddIOp`). 

```cpp
// Let op be an Operation*
mlir::Operation* op = ...;

// Method 1: isa followed by cast
if (mlir::isa<mlir::arith::AddIOp>(op)) {
  // We now op is actually an integer addition, so we can safely cast it
  // (mlir::cast fails if the operation is not of the indicated type)
  mlir::arith::AddIOp addOp = mlir::cast<mlir::arith::AddIOp>(op); 
  llvm::outs() << "op is an integer addition!\n";
}

// Method 2: dyn_cast followed by nullptr check
// dyn_cast returns a valid pointer if the operation is of the indicated type
// and returns nullptr otherwise
mlir::arith::AddIOp addOp = mlir::dyn_cast<mlir::arith::AddIOp>(op)
if (addOp) {
  llvm::outs() << "op is an integer addition!\n";
}

// Method 3: simultaneous dyn_cast and nullptr check
// Using the following syntax, we can simultaneously assign addOp and check if
// it is a nullptr  
if (mlir::arith::AddIOp addOp = mlir::dyn_cast<mlir::arith::AddIOp>(op)) {
  llvm::outs() << "op is an integer addition!\n";
}
```

Once you have a specific derived class of `Op` on hand, you can access methods that are specific to the operation type in question. For example, for all operation operands, MLIR will automatically generate an accessor method with the name `get<operand name in CamelCase>`. For example, `mlir::arith::AddIOp` has two operands named `lhs` and `rhs` that represent, respectively, the left-hand-side and right-hand-side of the addition. It is possible to get these operands using their name instead of their index with the following code.

```cpp
// Let addOp be an integer Operation
mlir::arith::AddIOp addOp = ...;

// Get first operand (lhs)
mlir::Value firstOperand = addOp->getOperand(0);
mlir::Value lhs = addO.getLhs();
assert(firstOperand == lhs);

// Get second operand (rhs)
mlir::Value secondOperand = addOp->getOperand(1);
mlir::Value rhs = addO.getRhs();
assert(secondOperand == rhs);
```

## Regions

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

## Blocks

## Attributes

## Dialects
