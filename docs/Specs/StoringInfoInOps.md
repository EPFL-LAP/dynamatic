
# Storing Information in Operations

It is often useful to store additional information on an operation. One example is if we wish to parameterize the generation of their RTL. This requires storing the parameter value on the operation itself, as well as passing it to the relevant parts of the code-base to ensure correct generation.


# Storing the Information

## How to Store the Information

Additional information about an operation should be stored as an operation-specific attribute. The [operation definition specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) (ODS) covers how MLIR uses tablegen files to declaratively define operations, including attributes. 
The [specific section](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) of the ODS relevant to attributes is on operation arguments. Specifically, attributes are operation arguments which are "compile-time known constant values".

Arguments are specified in an operation's tablegen entry like so:

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
  <property>:$<property-name>,
);
```

This is not order-specific: all types of arguments can be specified in any order, and attributes are identified by the fact they are preceded by an attribute constraint.

## Why Do We Store it Like This?

Operation-specific attributes give us enough flexibility to store many types of information, generate convenient named getter functions, and allow easy constraints and verification of values.

When used properly, they communicate clearly what information an operation must contain as well as what information it could contain. Rules for what kind of values are allowed are also easily to declaratively rpovide.

While functions exist to remove operation-specific attributes from the operation, each operation is automatically verified at the end of each pass: if a required attribute has been removed, compilation will fail.

## What Types of Information to Store

It is important to remember that an attribute could be altered by other sections of the code-base- it is therefore dangerous to place dependant data in an attribute. Given value A dependant on value B, there are 2 problematic scenarios: 

1. Value B is updated by another pass later on, but value A is not, which leaves stale information in the IR.
2. A later pass does not realize that value A is dependant on value B, and illegally changes value A directly, and then stores impossible information in the IR.

To avoid these problems, ideally information stored in the IR should be **decisions**: any information that is downstream of a set of decisions should be calculated on-demand, based on the real values of its causal variables. 

## Good Examples
#### FIFO-Depth for a Save-Commit

The save-commit operation (the details of what this operation does are not relevant to this document) has an internal FIFO.

The depth of this FIFO is specified using a required, operation-specific attribute, which is declared in its arguments:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L1199-L1201 

The first two arguments are constrained by type-constraints, while the 3rd is constrained by an attribute constraint.

This results in the following C++ to create a save-commit operation:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Transforms/Speculation/HandshakeSpeculation.cpp#L384-L387 

It is important to note that since fifoDepth is a required attribute, fifoDepth **must** be passed to the builder in order to create a save-commit operation. The [builder methods](https://mlir.llvm.org/docs/DefiningDialects/Operations/#builder-methods) section of the MLIR Dialect documentation explains well how different C++ builder functions are generated from an operation's tablegen declaration. 

The following getter function is also generated, for accessing the attribute's value:

```c++
saveCommitOp.getFifoDepth()
```

This named getter is generated automatically by declaring the attribute in the tablegen file.

#### SharingWrapperOp for Crush

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L1351-L1356

This declares 4 attributes for the `SharingWrapperOp`: an array of integer credits, and 3 integers with different minimum values. This shows one of the strengths of operation-specific attributes- the ability to declaratively specify constrained attributes.


The C++ to add a `SharingWrapperOp` then looks like this:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Transforms/ResourceSharing/Crush.cpp#L634-L655

to better explain the builder, here is a commented version:
```c++
handshake::SharingWrapperOp wrapperOp =
    builder.create<handshake::SharingWrapperOp>(
        sharedOp->getLoc(), 
        /*aggregate result type arg=*/sharingWrapperOutputTypes, 
        /*dataOperands=*/dataOperands,
        /*sharedOpResult=*/sharedOp->getResult(0), 
        /*credits=*/llvm::ArrayRef<int64_t>(credits),
        /*numSharedOperations=*/credits.size(), 
        /*numSharedOperands=*/sharedOp->getNumOperands(),
        /*latency=*/(unsigned)round(latency)
        );
```

Helpful getter functions are also generated for each of its attributes.

There is an interesting redudancy to note in the attributes of the SharingWrapperOp- `numSharedOperations` **must** be equal both to the size of `credits`, and is also determistic based on its number of inputs. In best practice, values like this which can be calculated should not be stored as attributes.

## Example to Avoid



#### BufferOp
As of time of writing, the buffer operation stores its RTL parameter attributes in a dictionary called "hw.parameters". 

Since this dictionary is not verified, a BufferOp may not even have the attributes present at all when the operation is passed to the backend.

Its arguments do not contain any attribute information:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L226

But it defines additional hardcoded strings to use as dictionary keys using:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L233-L271

To try and replicate some of the behaviour of dedicated attributes, a custom builder is declared in tablegen:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L229-L232

and then implemented separately in C++:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/lib/Dialect/Handshake/HandshakeOps.cpp#L198-L221

Due to this custom builder, the C++ to add a new operation looks as if it had used dedicated attributes:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Transforms/HandshakePlaceBuffersCustom.cpp#L102-L103

This builder successfully enforces the presence of buffer type at construction, but does not prevent later code from removing this attribute from the hw.parameters dictionary (in the above c++ accessed through RTL_PARAMETERS_ATTR_NAME). 

There are also no named getters generated, and therefore these attributes must be accessed very awkwardly through the dictionary attribute:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Support/SubjectGraph.cpp#L825-L830

# Non-Operation Specific Information
Sometimes information  may exist across many different operations. 

If we followed the above rules, adding the new attribute to many operations would result in code duplication. 

Instead, we recommend to add the attribute (using wrapper methods) to an interface, and to add that interface to the relevant operations.

## How to define interface attributes

As interfaces cannot store information, attributes cannot be directly added to the interface. The attribute must therefore be stored on the operation itself, using the [setAttr](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#ae5f0d4c61e6e57f360188b1b7ff982f6) and [getAttrOfType](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a8ec626dafc87ed8fb20d8323017dec72) functions.

Since the getAttr function relies on hardcoded strings to access the attribute, these calls should be wrapped in hand-written getters and setters. Since [getAttrOfType](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a04107e2edf0122ca87f0550148e347a7) can return null, the getter is always safe to call, but must be checked to see if the attribute was actually returned.

```tablegen
InterfaceMethod<[{
  Sets a StringAttr named "myAttr".
}], "void", "setMyAttr", (ins "::llvm::StringRef":$value), [{
  Operation *op = $_op.getOperation();
  op->setAttr("myAttr",
                StringAttr::get(op->getContext(), value));
}]>,

InterfaceMethod<[{
    Gets the StringAttr named "myAttr". 
  }], "StringAttr", "getMyAttr", (ins), [{
    Operation *op = $_op.getOperation();
    return op->getAttrOfType<StringAttr>("myAttr"))      
  }]>
```

Interface attributes are not verified, and so it is not guaranteed that an attribute will remain present: even if your code (at time of writing) guarantees that the attribute is always present, this may be changed by later changes to the codebase. You must always check to see if the getter succeeded at returning the attribute before using it.


## Good Example

### InternalDelay for Arithmetic Ops

Rather than returning null attributes, if the attribute is not present, a default value of "0.0" is returned.

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeInterfaces.td#L221-L246

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeArithOps.td#L21-L23

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Transforms/BufferPlacement/HandshakePlaceBuffers.cpp#L166-L175

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L693-L697 

<!-- # RTL Entity Sharing

Operations in the Handshake IR are checked for uniqueness due to the desire for shared RTL entities: if there are two 32-bit floating point multipliers in the circuit, the RTL defining what is a 32-bit floating point multiplier should be present only once. 

However, if an operation has a parameter which affects RTL generation, it also affects the "uniqueness" of the operation.

Currently, uniqueness is identified using a dictionary attribute called "hw.parameters". Previous documentation has specified that code anywhere in the compilation flow could instantiate hw.parameters and place data inside of it. However, we no longer consider this is good practice, and dedicated attributes should be used instead. We hope to eventually depreciate support for this, and if so, data placed inside hw.parameters before the handshake to hardware pass would be ignored.

When an operation uses dedicated attributes, it must still eventually pass its data into hw.parameters. This is done (currently) in 

https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L511-L521

which contains a case-statement for each operation, to allow each operation to add its own operation-specific information to hw.parameters.

The save-commit operation does so like this:
https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L680-L682

# Passing an Attribute to the Backend

If the attribute has been added to "hw.parameters" to support RTL uniqueness evaluation, then the value of that attribute is accessible by its key in the backend JSONs (rtl-config-vhdl.json, rtl-config-verliog.json, etc.)

In the operation's entry in the JSON, the attribute should also be listed in the operation's parameters list.

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/data/rtl-config-vhdl-beta.json#L212-L220

# Future Changes to this Process

The case statement in ModuleDiscriminator is an unsustainable solution. In the future, we intend to use operation interfaces, allowing operations to internally specify what RTL parameters they have. 

This is also important for our single-source-of-truth philosophy, which requires that each tablegen entry should entirely define an MLIR operation. The case statement in ModuleDiscriminator is an example of a distributed operation definition, which violates this principle. -->