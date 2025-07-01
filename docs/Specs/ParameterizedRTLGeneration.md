
# Parameterized RTL Generation


For some operations, we may wish to **parameterize** the generation of their RTL. This requires storing the parameter value on the operation itself, as well as passing it to the relevant parts of the code-base to ensure correct generation.

# Storing the parameter

## How to store the parameter

A parameter which affects an operation's RTL should be stored as an operation-specific attribute. The [operation definition specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) (ODS) covers how MLIR uses tablegen files to declaratively define operations, including attributes. 
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

## Why do we store it like this?

Operation-specific attributes give us enough flexibility to store all information that is needed, generate convenient named getter functions, and allow easy constraints and verification of the parameter values.

When used properly, they ensure that all operations have a valid set of RTL parameters, and that all required parameters are in fact present.

### Good Examples
#### FIFO-Depth for a Save-Commit

The save-commit operation (the details of what this operation does are not relevant to this document) has an internal FIFO.

The depth of this FIFO is specified using a required, operation-specific attribute, which is declared in its arguments:

```tablegen
let arguments = (ins HandshakeType:$dataIn,
                      ChannelType:$ctrl,
                      UI32Attr:$fifoDepth);
```

The first two arguments are constrained by type-constraints, while the 3rd is constrained by an attribute constraint.

This results in the following C++ to create a save-commit operation:
```c++
SpecSaveCommitOp newOp = builder.create<SpecSaveCommitOp>(
    dstOp->getLoc(), /*resultType=*/srcOpResult.getType(),
    /*dataIn=*/srcOpResult, /*ctrl=*/ctrlSignal,
    /*fifoDepth=*/fifoDepth);
```

It is important to note that since fifoDepth is a required attribute, fifoDepth **must** be passed to the builder in order to create a save-commit operation. The [builder methods](https://mlir.llvm.org/docs/DefiningDialects/Operations/#builder-methods) section of the MLIR Dialect documentation explains well how different C++ builder functions are generated from an operation's tablegen declaration. 

The following getter function is also generated, for accessing the attribute's value:

```c++
saveCommitOp.getFifoDepth()
```

This named getter is generated automatically by declaring the attribute in the tablegen file.

#### SharingWrapperOp for Crush

```tablegen
let arguments = (ins Variadic<ChannelType> : $dataOperands,
  ChannelType : $sharedOpResult,
  DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$credits,
  ConfinedAttr<I32Attr, [IntMinValue<2>]>:$numSharedOperations,
  ConfinedAttr<I32Attr, [IntMinValue<1>]>:$numSharedOperands,
  ConfinedAttr<I32Attr, [IntMinValue<1>]>:$latency);
```

This declares 4 attributes for the `SharingWrapperOp`: an array of integer credits, and 3 integers with different minimum values. This shows one of the strengths of operation-specific attributes- the ability to declaratively specify constrained attributes.


The C++ to add a `SharingWrapperOp` then looks like this:
```c++
// Determining the number of credits of each operation that share the
// unit based on the maximum achievable occupancy in critical CFCs.
llvm::SmallVector<int64_t> credits;
for (Operation *op : group) {
  double occupancy = opOccupancy[op];
  // The number of credits must be an integer. It is incremented by 1 to
  // hide the latency of returning a credit, and accounts for token
  // staying in the output buffers due to the effect of sharing.
  credits.push_back(1 + std::ceil(occupancy));
}

assert(sharingWrapperOutputTypes.size() ==
            sharedOp->getNumOperands() + group.size() &&
        "The sharing wrapper has an incorrect number of output ports.");

builder.setInsertionPoint(*group.begin());
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

and also generates helpful getter functions for each of its attributes.

There is an interesting redudancy to note in the attributes of the SharingWrapperOp- `numSharedOperations` **must** be equal both to the size of `credits`, and is also determistic based on its number of inputs (as seen in the assertion). In best practice, values like this which can be calculated should not be stored as attributes.

### Example to Avoid



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

This builder enforces the presence of buffer type at construction, but does not prevent later code from removing this attribute from the hw.parameters dictionary (in the above c++ accessed through RTL_PARAMETERS_ATTR_NAME).

There are also no named getters generated, and therefore these attributes must be accessed very awkwardly through the dictionary attribute:

```c++
// Get the Buffer type and data width from the operation attributes
auto params =
    bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
auto bufferTypeNamed =
    params.getNamed(handshake::BufferOp::BUFFER_TYPE_ATTR_NAME);
auto bufferTypeAttr = dyn_cast<StringAttr>(bufferTypeNamed->getValue());
```

# RTL Entity Sharing

Operations in the Handshake IR are checked for uniqueness due to the desire for shared RTL entities i.e. if there are two 32-bit floating point multipliers in the circuit, the RTL defining what is a 32-bit floating point multiplier should be present only once. 

However, if an operation has a parameter which affects RTL generation, it also affects the "uniqueness" of the operation.

Currently, uniqueness is identified using a dictionary attribute called "hw.parameters". Previous documentation has specified that code anywhere in the compilation flow could instantiate hw.parameters and place data inside of it- this is not good practice, and dedicated attributes should be used instead. In future, data placed inside hw.parameters before the handshake to hardware pass will be ignored, and so code that currently does this should be updated.

When an operation uses dedicated attributes, it must still eventually pass its data into hw.parameters. This is done (currently) in 


https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L511-L521

which contains a case-statement for each operation, to allow each operation to add its own operation-specific information to hw.parameters.

The save-commit operation does so like this:
https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L680-L682

# Passing an Attribute to the Backend

If the attribute has been added to "hw.parameters" to allow RTL entity sharing, then the value of that attribute is accessible by its key in the backend JSONs (rtl-config-vhdl.json, rtl-config-verliog.json, etc.)

In the operation's entry in the JSON, the attribute should also be listed in the operation's parameters list.

```json
{
  "name": "handshake.spec_save_commit",
  "parameters": [
    {
      "name": "FIFO_DEPTH",
      "type": "unsigned"
    }
  ],
  "generator": "python $DYNAMATIC/experimental/tools/unit-generators/vhdl/vhdl-unit-generator.py -n $MODULE_NAME -o $OUTPUT_DIR/$MODULE_NAME.vhd -t spec_save_commit -p fifo_depth=$FIFO_DEPTH bitwidth=$BITWIDTH extra_signals=$EXTRA_SIGNALS"
}
```

# Future Changes to this Process

The case statement in ModuleDiscriminator is an unsustainable solution. Instead, operation interfaces should be used to allow operations to internally specify what RTL parameters they have. 

This is also important for our single-source-of-truth philosophy, that each tablegen entry should entirely define an MLIR operation.