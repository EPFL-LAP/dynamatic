
# Parameterized RTL Generation


For some operations, we may wish to **parameterize** the generation of their RTL. This requires storing the parameter value on the the operation itself, as well as passing it to the relevant parts of the code-base to ensure correct generation.

# Storing the parameter

## How to store the parameter

A parameter which affects an operations RTL should be stored as an operation-specific attribute. The [operation definition specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) (ODS) covers how MLIR uses tablegen files to declaratively define operations, including attributes. 
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

Operation-specific attributes give us enough flexibility to store all information that is needed, generate convienient named getter functions, and allow easy constraints and verifications of the parameter values.

When used properly, they ensure that all operations have a valid set of RTL parameters, and that all required parameters are in fact present.

#### Good Examples
##### FIFO-Depth for a Save-Commit

The save-commit operation (the details of what this operation does are not relevant to this document) has an internal FIFO.

The depth of this FIFO is specified using a required, operation-specific attribute, which is declared in its arguments:

```tablegen
let arguments = (ins HandshakeType:$dataIn,
                      ChannelType:$ctrl,
                      UI32Attr:$fifoDepth);
```

The first two arguments are type-constraints, while the 3rd is an attribute constraint.

This results in following C++ to create a save-commit operation:
```c++
SpecSaveCommitOp newOp = builder.create<SpecSaveCommitOp>(
    dstOp->getLoc(), /*resultType=*/srcOpResult.getType(),
    /*dataIn=*/srcOpResult, /*ctrl=*/ctrlSignal,
    /*fifoDepth=*/fifoDepth);
```

and the following getter function to access the attribute's value:

```c++
saveCommitOp.getFifoDepth()
```

This named getter is generated automatically by declaring the attribute in the tablegen file.

##### SharingWrapperOp for Crush

```tablegen
let arguments = (ins Variadic<ChannelType> : $dataOperands,
  ChannelType : $sharedOpResult,
  DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$credits,
  ConfinedAttr<I32Attr, [IntMinValue<2>]>:$numSharedOperations,
  ConfinedAttr<I32Attr, [IntMinValue<1>]>:$numSharedOperands,
  ConfinedAttr<I32Attr, [IntMinValue<1>]>:$latency);
```

This declares 4 attributes for the SharingWrapperOp: an array of integer credits, which defaults to empty, and 3 integers with different minimum values. This shows one of the strengths of operation-specific attributes- the ability to declaritively specific default values and constraints.


The c++ to add a SharingWrapperOp then looks like so:
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
        sharedOp->getLoc(), sharingWrapperOutputTypes, dataOperands,
        sharedOp->getResult(0), llvm::ArrayRef<int64_t>(credits),
        credits.size(), sharedOp->getNumOperands(),
        (unsigned)round(latency));
```

and allows the helpful getter functions for each of its attributes.

#### Example to Avoid



##### BufferOp
As of time of writing, the buffer operation stores its RTL parameter attributes in a dictionary called "hw.parameters". 

Since this dictionary is not verified, a BufferOp may not even have the attributes present at all when the operation is passed to the backend.

Its arguments looks like so:
```tablegen
let arguments = (ins HandshakeType:$operand);
```

And it defines additional hardcoded strings to use as dictionary keys using:

```tablegen
let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral NUM_SLOTS_ATTR_NAME = "NUM_SLOTS",
                                           TIMING_ATTR_NAME = "TIMING",
                                           BUFFER_TYPE_ATTR_NAME = "BUFFER_TYPE";
    
    /// ONE_SLOT_BREAK_DV: This buffer breaks the D and V signal paths.
    /// Previously known as a slot of OEHB (Opaque Elastic Half-Buffer),
    /// it introduces one cycle of latency on the D and V paths.
    /// It does not break the R signal path and adds no latency on R.
    static constexpr ::llvm::StringLiteral ONE_SLOT_BREAK_DV = "ONE_SLOT_BREAK_DV";
    
    /// ONE_SLOT_BREAK_R: This buffer breaks the R signal path.
    /// Previously known as a slot of TEHB (Transparent Elastic Half-Buffer),
    /// it introduces one cycle of latency on the R path.
    /// It does not break the D and V signal path and adds no latency on D and V.
    static constexpr ::llvm::StringLiteral ONE_SLOT_BREAK_R = "ONE_SLOT_BREAK_R";
    
    /// FIFO_BREAK_NONE: Previously known as a 'tfifo' (Transparent FIFO),
    /// this is a FIFO_BREAK_DV with a bypass, adding no latency to any signal paths.
    /// Its only purpose is to hold tokens.
    static constexpr ::llvm::StringLiteral FIFO_BREAK_NONE = "FIFO_BREAK_NONE";
    
    /// FIFO_BREAK_DV: This buffer breaks the D and V paths.
    /// It has multiple slots but, unlike a chain of ONE_SLOT_BREAK_DV buffers,
    /// its structure cannot be split. It introduces one cycle of latency on the
    /// D and V paths regardless of the number of slots and has no latency on R.
    /// It was previously called an 'elastic_fifo_inner'.
    static constexpr ::llvm::StringLiteral FIFO_BREAK_DV = "FIFO_BREAK_DV";
    
    /// ONE_SLOT_BREAK_DVR: Each slot of this buffer breaks the D, V, and R signal
    /// paths and introduces one cycle of latency on all of them.
    static constexpr ::llvm::StringLiteral ONE_SLOT_BREAK_DVR = "ONE_SLOT_BREAK_DVR";
    
    /// SHIFT_REG_BREAK_DV: Breaks D and V paths. Multiple slots share the same
    /// handshake control signals, causing them to accept or stall inputs
    /// simultaneously.
    static constexpr ::llvm::StringLiteral SHIFT_REG_BREAK_DV = "SHIFT_REG_BREAK_DV";
    
  }];
```

To try and replicate some of the behaviour of dedicated attributes, a custom builder is declared in tablegen:

```tablegen
let builders = [OpBuilder<
  (ins "Value":$operand, "const ::dynamatic::handshake::TimingInfo &":$timing,
        "std::optional<unsigned>":$numSlots, "StringRef":$bufferType)>
];
```

and then implemented separately in C++:
```c++
void BufferOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Value operand, const TimingInfo &timing,
                     std::optional<unsigned> numSlots, StringRef bufferType) {
  odsState.addOperands(operand);
  odsState.addTypes(operand.getType());

  // Create attribute dictionary
  SmallVector<NamedAttribute> attributes;
  MLIRContext *ctx = odsState.getContext();
  attributes.emplace_back(StringAttr::get(ctx, TIMING_ATTR_NAME),
                          TimingAttr::get(ctx, timing));
  if (numSlots) {
    attributes.emplace_back(
        StringAttr::get(ctx, NUM_SLOTS_ATTR_NAME),
        IntegerAttr::get(IntegerType::get(ctx, 32, IntegerType::Unsigned),
                         *numSlots));
  }

  attributes.emplace_back(StringAttr::get(ctx, BUFFER_TYPE_ATTR_NAME),
                          StringAttr::get(ctx, bufferType));

  odsState.addAttribute(RTL_PARAMETERS_ATTR_NAME,
                        DictionaryAttr::get(ctx, attributes));
}
```

Due to this custom builder, the C++ to add a new operation looks as if it had used dedicated attributes:

```c++
auto bufOp = builder.create<handshake::BufferOp>(channel.getLoc(), channel,
                                                  timing, slots, bufferType);
```

This builder enforces the presence of buffer type at construction, but does not prevent later code from removing this attribute from the hw.parameters dictionary (in the above c++ accessesed through RTL_PARAMETERS_ATTR_NAME).

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

https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L511C1-L511C58

which contains a case-statement for each operation, to allow each operation to add its own operation-specific attributes.

The save-commit operation does so like this:
https://github.com/EPFL-LAP/dynamatic/blob/0f29d6f1f8d8277ae003f3eb9b40319a5dca61df/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L680C1-L682C9

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