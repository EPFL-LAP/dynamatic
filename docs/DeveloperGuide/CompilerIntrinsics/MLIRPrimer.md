# An MLIR Primer

This tutorial introduces you to MLIR and its core constructs. This document is
useful to get an initial idea of how MLIR works and how to manipulate its
data-structures.

## High-Level Structure

MLIR has graph-like data structure: every IR contains nodes, called
`Operation`s, and edges, called `Value`s. Each `Value` is the result of exactly
one `Operation` or `BlockArgument`, and has a `Value` `Type` defined by the
type system. An operation has a list of operand values and a list of result
values.

Operations are contained in `Block`s, and `Block`s are contained in `Region`s.
`Operation`s are ordered within their containing block and `Block`s are ordered
in their containing region. Operations may also contain regions, enabling
hierarchical structures to be represented.

All these data structures can be manipulated in C++ using their respective
types. In addition, they can be printed to a text file (by convention, a file
with the `.mlir` extension) and parsed back to their in-memory representation
at any point. For example, the following is a textual representation of an MLIR
file:

```mlir
module {
  func.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c999_i32 = arith.constant 999 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1000_i32 = arith.constant 1000 : i32
    cf.br ^bb1(%c0_i32, %c0_i32 : i32, i32)
  ^bb1(%0: i32, %1: i32):  // 2 preds: ^bb0, ^bb1
    %2 = arith.extui %0 : i32 to i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = memref.load %arg1[%3] {handshake.name = "load0"} : memref<1000xi32>
    %5 = arith.subi %c999_i32, %0 : i32
    %6 = arith.extui %5 : i32 to i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = memref.load %arg0[%7] {handshake.name = "load1"} : memref<1000xi32>
    %9 = arith.muli %4, %8 : i32
    %10 = arith.addi %1, %9 : i32
    %11 = arith.addi %0, %c1_i32 : i32
    %12 = arith.cmpi ult, %11, %c1000_i32 : i32
    cf.cond_br %12, ^bb1(%11, %10 : i32, i32), ^bb2(%10 : i32)
  ^bb2(%13: i32):  // pred: ^bb1
    return %13 : i32
  }
}
```

The file above describes a MLIR module operation (`module`), which contains a
function operation (`func.func` with name `fir`) that represents an MLIR
function. The prefix `func.*` indicates the specific dialect that the op
belongs to (we will discuss the concept of dialect later: they are just a way
to organize the operations). This function has an internal region with 3
blocks: `^bb0` (omitted in the text), `^bb1`, and `^bb2`. And each block
contains a list of operations.

For example, the following operation has two operand values (`%1` and `%9`) and
produces one result values `%10`.
```
%10 = arith.addi %1, %9 : i32
```

To summarize, every MLIR file (`*.mlir`) is recursively nested. It starts with
a top-level operation (often, an `mlir::ModuleOp`) which may contain nested
regions, each of which may contain an ordered list of nested blocks, each of
which may contain an ordered list of nested operations, after which the
hierarchy repeats.

So far, we have seen how MLIR represents operations, values, and hierarchies.
In the following, we will discuss how to analyze them using the C++ API of MLIR

For more details on the MLIR language, check out this
[documentation](https://mlir.llvm.org/docs/LangRef/#high-level-structure).

### MLIR Dialect

Dialects: MLIR manages extensibility using *dialects*, which provide a logical
grouping of Ops, attributes and types under a unique namespace. Dialects
themselves do not introduce any new semantics but serve as a logical grouping
mechanism that provides common Op functionality (e.g., constant folding
behavior for all ops in the dialect).

The dialect namespace appears as a dot-separated prefix in the opcode.

For example, the following operation is an `addi` operation in the `arith` dialect.
```mlir
%10 = arith.addi %1, %9 : i32
```

For example, the following operation is a `func` operation in the `func` dialect.
```mlir
func.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>) -> i32 {
    // ignore the details inside the function. 
}
```

Dynamatic uses many dialects:
- Dynamatic uses the built-in MLIR dialects (i.e., available from the upstream
  MLIR repo): *ControlFlow*, *arith*, *math*, and *memref* dialects to model a
  software IR.
- Dynamatic uses the *Handshake* dialect to model dataflow circuits.

In the next section, we will see how MLIR facilitates the definition of custom
dialects and operations. For this tutorial, let's focus on where to find
important definitions we can use to write MLIR analysis and transformation
passes.

## Reviewing the Definitions of the MLIR Operations and Dialects

Defining an intermediate representation in C++ is very complex; one has to
define a complex class hierarchy for operations, values, attributes,
interfaces, and so on. For each specific class, there are also a lot of details
to implement, for example, we need to define different constructors, the
printer, and the parser.

MLIR greatly simplifies this process: it uses the Tablegen format to define IR,
which will be automatically translated into C++ files that can be
included in the regular C++ files. The tablegen definitions are usually stored
in files named with a suffix ".td".

### What is the Tablegen Format?

The Tablegen format is a domain-specific language used in the MLIR framework to
define custom IR operations and types. The MLIR infrastructure translates these
files into C++ files that can be included in the C++ source tree. 

For example, the following describes a Tablegen definition of the BranchOp used
in Dynamatic (some details omitted).

```
def BranchOp : Handshake_Op<"br", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "branch operation";
  let arguments = (ins HandshakeType:$operand);
  let results = (outs HandshakeType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` custom<HandshakeType>(type($result))
  }];

  // Extra declarations that will be inserted into the generated C++ files.
  // They are implemented in lib/Dialect/Handshake/HandshakeOps.cpp
  let extraClassDeclaration = [{
    // Utility methods for getting the result channel when the condition is
    // true/false
    // Example:
    // if (auto branchOp = llvm::dyn_cast<dynamatic::handshake::BranchOp>(op)) {
    //   Value trueResultChannel = branchOp.getTrueResult();
    // }
    mlir::Value getTrueResult();
    mlir::Value getFalseResult();
  }]
}
```

MLIR will generate:
- Declaration of the data structures: typically included in the CPP headers.
- Some default implementations of various methods for creating, analyzing, and
  manipulating these data structures: typically included in the CPP source files.

> [!NOTE]
> Tablegen is a very concise format, which makes it very easy to update the IR
> definition. Therefore, to make sure that the IR definition and its
> documentation do not go out of sync, it is very common to directly document
> how each IR operation works in these tablegen files.
> 
> As you can see in the Tablegen definition of the BranchOp above, it contains
> example of how to use the class methods.

## Traversing the IR Using the C++ API

This section presents different examples of how to use the C++ API to traverse
a hierarchical IR. We describe the IR traversal in two directions:
- From top to bottom: we recursively visit the operations in lower levels. 
- From bottom to top: we start from a lower leve and traverse the parent
  hierarchies.

### From Top to Bottom

Consider the following C++ function which finds and recursively
traverses all operations nested within a provided operation.

```cpp
void traverseIRFromOperation(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &nestedOp : block.getOperations()) {
        llvm::outs() << "Traversing operation " << op << "\n";
        traverseIRFromOperation(&nestedOp);
      }
    }
  }
}
```

MLIR also exposes the [`walk`
method](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers)
on the `Operation`, `Region`, and `block` types. `walk` takes as single
argument a callback method that will be invoked recursively for all operations
recursively nested under the receiving entity.

```cpp
// Let block be a Block&
mlir::Block &block = ...;

// Walk all operations nested in the block
block.walk([&](mlir::Operation *op) {
  llvm::outs() << "Traversing operation " << op << "\n";
});
```

### From Bottom to Top

One may also get the parent entities of a given operation, region, or block.

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

For more details on IR traversal, check out this [documentation](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-ir-nesting).

## Analyzing MLIR Values

Values are the edges of the graph-like structure that MLIR models. They
correspond to the `mlir::Value` C++ type. All values are typed using either a
built-in type or a custom user-defined type (the type of a value is itself a
C++ type called `Type`).

Values are either produced by [operations](#operations) as operation results
(`mlir::OpResult`, which is a subtype of `mlir::Value`) or are defined by
[blocks](#blocks) as part of their block arguments (`mlir::BlockArgument`, also
a subtype of `mlir::Value`). They are consumed by [operations](#operations) as
operation operands. A value may have 0 or more uses, but should have exactly
one producer (an operation or a block).

For example, consider the following MLIR snippet:

```mlir
%result0 = cfx.phi %operand0, %operand1 { bbID = 0 } : (i32, i32) -> i32
%result1 = arith.addi %result0, %operand2 { bbID = 0 } : (i32, i32) -> i32
```
in this example, the value `%result0` produced by operation `cfx.phi` is used
by operation `arith.addi`.

The following C++ snippet shows how to identify the type and producer of a
value and prints the index of the producer's operation result/block argument
that the value corresponds to.

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

The following C++ snippet shows how to iterate through all the operations that
use a particular value as operand. Note that the number of uses may be equal
*or larger* than the number of users because a single user may use the same
value multiple times (but at least once) in its operands.

```cpp
// Let value be a Value
mlir::Value value = ...;

// Iterate over all uses of the value (i.e., over operation operands that equal
// the value)
for (mlir::OpOperand &use : value.getUses()) {
  // Get the owner of this particular use 
  mlir::Operation *useOwner = use.getOwner();
  llvm::outs() << "Value is used as operand number " 
               << use.getOperandNumber() << " of operation "
               << useOwner << "\n";
}

// Iterate over all users of the value
for (mlir::Operation *user : value.getUsers())
  llvm::outs() << "Value is used as an operand of operation " << user << "\n";
```

## Operations

In MLIR, everything is about operations. Operations are like "opaque functions"
to MLIR; they may represent some abstraction (e.g., a function, with a
`mlir::func::FuncOp` operation) or perform some computation (e.g., an integer
addition, with a `mlir::arith::AddIOp`). There is no fixed set of operations;
users may define their own operations with custom semantics and use them at the
same time as MLIR-defined operations. Operations:

- are identified by a unique string
- can take 0 or more operands
- can return 0 or more results
- can have [attributes](#attributes) (i.e., constant data stored in a dictionary)

The C++ snippet below shows how to get an operation's information from C++.

```cpp
// Let op be an Operation*
mlir::Operation* op = ...;

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

### Manipulating the Operation Based on its Type

This part explains how we can analyze and manipulate MLIR operation of a specific type.


As we saw above, you can manipulate any operation in MLIR using the `Operation`
type (often through an `Operation*`) which provides a generic API into an
operation instance. In practice, we want to perform analysis and manipulate the
IR based on the specific type of that IR. This is extremely easy in MLIR.

For example, consider that we have an operation:
```c++
mlir::Operation op = ...;
```
MLIR has a very convinent way of checking if the op is of a specific type.

```c++
if (llvm::isa<mlir::arith:AddIOp>(op)) {
    // Do something if the op is an "AddIOp" from the arith MLIR dialect
}
```

MLIR also has a very convinent way of transforming (casting) that operation to that type.

```c++
if (auto addOp = llvm::dyn_cast<mlir::arith:AddIOp>(op)) {
    // Do something if the op is an "AddIOp" from the arith MLIR dialect
    Value lhsVal = addOp.getLhs();
    ...
}
```

How does MLIR enable all of these? The following describes this mechanism.

#### How Does MLIR Enable this Abstraction? Op vs. Operation

There exists another type, `Op`, whose derived classes model a specific type of
operation (e.g., an integer addition with a `mlir::arith::AddIOp`). 

`Op` derived classes act as smart pointer wrapper around a `Operation*`,
provide operation-specific accessor methods, and type-safe properties of
operations. A side effect of this design is that we always pass around `Op`
derived classes “by-value”, instead of by reference or pointer.

If you need to manipulate an operation of a specific type, you should do
so through its actual type that derives from `Op`. Fortunately, it is easy to
identify the actual type of an `Operation*` using MLIR's casting
infrastructure. The following snippet shows a few different methods to check
whether an opaque `Operation*` is actually an integer addition
(`mlir::arith::AddIOp`).

```cpp
// Let op be an Operation*
mlir::Operation* op = ...;

// Method 1: isa followed by cast
if (llvm::isa<mlir::arith::AddIOp>(op)) {
  // We now op is actually an integer addition, so we can safely cast it
  // (`llvm::cast` fails if the operation is not of the indicated type)
  mlir::arith::AddIOp addOp = mlir::cast<mlir::arith::AddIOp>(op); 
  llvm::outs() << "op is an integer addition!\n";
}

// Method 2: dyn_cast followed by nullptr check
// `llvm::dyn_cast` returns a valid pointer if the operation is of the indicated
// type and returns nullptr otherwise
auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op)
if (addOp) {
  llvm::outs() << "op is an integer addition!\n";
}

// Method 3: simultaneous dyn_cast and nullptr check
// Using the following syntax, we can simultaneously assign addOp and check if
// it is a nullptr  
if (mlir::arith::AddIOp addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
  llvm::outs() << "op is an integer addition!\n";
}
```

> [!NOTE]
> `llvm::dyn_cast`, `llvm::cast`, and `llvm::isa` are some of the dynamic
> runtime type information (RTTI) features provided by LLVM. They are used
> extensively to analyze MLIR operations. Please check out [LLVM's RTTI
> documentation](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates).


Once you have a specific derived class of `Op` on hand, you can access methods
that are specific to the operation type in question. For example, for all
operation operands, MLIR will automatically generate an accessor method with
the name `get<operand name in CamelCase>`. For example, `mlir::arith::AddIOp`
has two operands named `lhs` and `rhs` that represent, respectively, the
left-hand-side and right-hand-side of the addition. It is possible to get these
operands using their name instead of their index with the following code.

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

When iterating over the operations inside a region or block, it's possible to
only iterate over operations of a specific type using the `getOps<OpTy>`
method.

```cpp
// Let region be a Region&
mlir::Region &region = ...;

// Iterate over all integer additions inside the region's blocks
for (mlir::arith::AddIOp addOp : region.getOps<mlir::arith::AddIOp>())
  llvm::outs() << "Found an integer operation!\n";

// Equivalently, we can first iterate over blocks, then operations
for (Block &block : region.getBlocks())
  for (mlir::arith::AddIOp addOp : block.getOps<mlir::arith::AddIOp>())
    llvm::outs() << "Found an integer operation!\n";

// Equivalently, without using getOps<OpTy>
for (Block &block : region.getBlocks())
  for (Operation* op : block.getOperations())
    if (mlir::arith::AddIOp addOp = mlir::dyn_cast<mlir::arith::AddIOp>(op))
      llvm::outs() << "Found an integer operation!\n";
```

The `walk` method similarly allows one to specify a type of operation to
recursively iterate on inside the callback's signature.

```cpp
// Let block be a Block&
mlir::Block &block = ...;

// Walk all integer additions nested in the block
block.walk([&](mlir::arith::AddIOp op) {
  llvm::outs() << "Found an integer operation!\n";
});

// Equivalently, without using the operation type in the callback's signature 
block.walk([&](Operation *op) {
  if (mlir::isa<mlir::arith::AddIOp>(op))
    llvm::outs() << "Found an integer operation!\n";
});
```

For more details on how to analyze and transform an MLIR operation, check out [this documentation](https://mlir.llvm.org/docs/LangRef/#operations).

## Regions


A region is an ordered list of MLIR blocks. The semantics within a region is
not imposed by the IR. Instead, the containing operation defines the semantics
of the regions it contains. MLIR currently defines two kinds of regions: SSACFG
regions, which describe control flow between blocks, and Graph regions, which
do not require control flow between blocks.

The first block in a region, called the *entry block*, is special; its
arguments also serve as the region's arguments. The source of these arguments
is defined by the semantics of the parent operation. When control flow enters a
region, it always begins in the *entry block*. Regions may also produce a list
of values when control flow leaves the region. Again, the parent operation
defines the relation between the region results and its own results. All values
defined within a region are not visible from outside the region (they are
[encapsulated](https://mlir.llvm.org/docs/LangRef/#value-scoping)). However, by
default, a region can reference values defined outside of itself if these
values would have been usable by the region's parent operation operands.  

### SSA Regions

A function body (i.e., the region inside a `mlir::func::FuncOp` operation) is
an example of an SSACFG region, where each block represents a control-free
sequence of operations that executes sequentially. The last operation of each
block, called the *terminator operation* (see the [next sextion](#blocks)),
identifies where control flow goes next; either to another block, called a
*successor block* in this context, inside the function body (in the case of a
*branch*-like operation) or back to the parent operation (in the case of a
*return*-like operation).

### Graph Regions

Graph regions, on the other hand, can only contain a single basic block and are
appropriate to represent concurrent semantics without control flow. This makes
them the perfect representation for dataflow circuits which have no notion of
sequential execution. In particular (from the [language
reference](https://mlir.llvm.org/docs/LangRef/#graph-regions))

All values defined in the graph region as results of operations are in scope
within the region and can be accessed by any other operation in the region. In
graph regions, the order of operations within a block and the order of blocks
in a region is not semantically meaningful and non-terminator operations may be
freely reordered.

## [Blocks](https://mlir.llvm.org/docs/LangRef/#blocks)

A block is an ordered list of MLIR operations. The last operation in a block
must be a terminator operation, unless it is the single block of a region whose
parent operation has the `NoTerminator` trait (`mlir::ModuleOp` is such an
operation).

As mentioned in the [prior section on MLIR values](#values), blocks may have
block arguments. 

Blocks in MLIR take a list of block arguments, notated in a function-like way.
Block arguments are bound to values specified by the semantics of individual
operations. Block arguments of the entry block of a region are also arguments
to the region and the values bound to these arguments are determined by the
semantics of the parent operation. Block arguments of other blocks are
determined by the semantics of terminator operations (e.g., branch-like
operations) which have the block as a successor.

In SSACFG regions, these block arguments often implicitly represent the passage
of control-flow dependent values. They remove the need for [*PHI*
nodes](https://en.wikipedia.org/wiki/Static_single-assignment_form#Converting_to_SSA)
that many other SSA IRs employ (like LLVM IR).

For more details on MLIR region and block, check out this [documentation](https://mlir.llvm.org/docs/LangRef/#blocks).

## Attributes

MLIR attributes are used to attach data/information to operations that
cannot be expressed using a value operand. Additionally, attributes allow us to
propagate meta-information about operations down the lowering pipeline. This is
useful whenever, for example, some analysis can only be performed at a "high IR
level" but its results only become relevant at a "low IR level". In these
situations, the analysis's results would be attached to relevant operations
using attributes, and these attributes would then be propagated through
lowering passes until the IR reaches the level where the information must be
acted upon.

For more details on the MLIR attributes, check out this [documentation](https://mlir.llvm.org/docs/LangRef/#attributes).


## Dialects

The *Handshake* dialect, defined in the `dynamatic::handshake` namespace, is
core to Dynamatic. *Handshake* allows us to represent dataflow circuits inside
[graph regions](#regions). Throughout the repository, whenever we mention
"*Handshake*-level IR", we are referring to an IR that contains *Handshake*
operations (i.e., dataflow components), which together make up a dataflow
circuit.

For more details on the MLIR dialects, check out this [documentation](https://mlir.llvm.org/docs/LangRef/#dialects).

## Printing to the Console

### Printing to stdout and stderr

LLVM/MLIR has wrappers around the standard program output streams that you
should use whenever you would like something displayed on the console. These
are `llvm::outs()` (for stdout) and `llvm::errs()` (for stderr), see their
usage below.

```cpp
// Let op be an Operation*
Operation *op = ...;

// Print to standard output (stdout)
llvm::outs() << "This will be printed on stdout!\n";

// Print to standard error (stderr)
llvm::errs() << "This will be printed on stderr!\n"
             << "As with std::cout and std::cerr, entities to print can be "
             << "piped using the '<<' C++ operator as long as they are "
             << "convertible to std::string, like the integer " << 10
             << " or an MLIR operation " << op << "\n";
```

> [!CAUTION]
> Dynamatic's optimizer prints the IR resulting from running all the passes it
> was asked to run to standard output. As a consequence **you should never
> explicitly print anything to stdout yourself**, as it will mix up with the IR
> text serialization. Instead, all error messages should go to stderr.

### Printing Information Related to an Operation

You will regularly want to print a message to stdout/stderr and attach it to a
specific operation that it relates to. While you could just use `llvm::outs()`
or `llvm::errs()` and pipe the operation in question after the message (as
shown above), MLIR has very convenient methods that allow you to achieve the
same task more elegantly in code and with automatic output formatting; the
operation instance will be (pretty-)printed with your custom message next to
it.

```cpp
// Let op be an Operation*
Operation *op = ...;

// Report an error on the operation
op->emitError() << "My error message";
// Report a warning on the operation
op->emitWarning() << "My warning message";
// Report a remark on the operation
op->emitRemark() << "My remark message";
```
