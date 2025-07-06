
# Storing Information in Operations

It is often useful to store additional information on an operation. One example is if we wish to parameterize the generation of their RTL. This requires storing the parameter value on the operation itself, as well as passing it to the relevant parts of the code-base to ensure correct generation.

Additional information about an operation should be stored as an operation-specific attribute. There are two main ways to define an attribute:
- In Operation TableGen: declared as part of the operation's definition (at compile-time)
- In Interface Attribute: set dynamically in a pass by an interface (at runtime)
  
The rest of this document explores both approaches in detail, highlighting their respective advantages and trade-offs. Below is an outline of the document structure:

- [Operation TableGen Approach](#operation-tablegen-approach)
  - [How to Store the Information](#how-to-store-the-information)
  - [Why Do We Store it Like This?](#why-do-we-store-it-like-this)
  - [What Types of Information to Store](#what-types-of-information-to-store)
  - [Maintaining Backwards Compatibility](#maintaining-backwards-compatibility)
  - [Attribute Constraint Wrappers](#optional-and-default-valued-attributes)
    - [OptionalAttr](#optionalattr)
    - [DefaultValuedAttr](#defaultvaluedattr)
  - [Good Examples](#good-examples)
    - [FIFO-Depth for a Save-Commit](#fifo-depth-for-a-save-commit)
    - [SharingWrapperOp for Crush](#sharingwrapperop-for-crush)
  - [Example to Avoid](#example-to-avoid)
    - [BufferOp](#bufferop)

- [Interface Attribute Approach](#interface-attribute-approach)
  - [How to Define Interface Attributes](#how-to-define-interface-attributes)
  - [Good Example](#good-example)
    - [InternalDelay for Arithmetic Ops](#internaldelay-for-arithmetic-ops)

- [Pros and Cons Summary](#pros-and-cons-summary)

- [Appendix: How to Calculate Dependant Information On Demand](#how-to-calculate-dependant-information-on-demand)


# Operation TableGen Approach

## How to Store the Information

The [operation definition specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) (ODS) covers how MLIR uses tablegen files to declaratively define operations, including attributes. 
The [specific section](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) of the ODS relevant to attributes is on operation arguments. 

Arguments are specified in an operation's tablegen entry like so:

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>
);
```
Operands and attributes are both listed in the `ins` section. The constraint type determines which is which:
- Type constraints define operands (runtime values).
- Attribute constraints define attributes (compile-time constants).

For example, in the integer comparator from Dynamatic:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeArithOps.td#L225-L226

Here, `$lhs` and `$rhs` are operands constrained to a custom type constraint (`ChannelType`), while `$predicate` is an attribute using a custom enum-based attribute constraint (`Handshake_CmpIPredicateAttr`).

Argument order is flexible; operands and attributes can be mixed, as long as their constraint types clarify their roles.

## Why Do We Store It Like This?

Operation-specific attributes give us enough flexibility to store many types of information, generate convenient named getter functions, and allow easy constraints and verification of values.

When used properly, they communicate clearly what information an operation **must** contain (as well as what information it **could** contain). Rules for what kind of values are allowed are also easy to declaratively provide.

While functions exist to remove operation-specific attributes from the operation, each operation is automatically verified at the end of each pass: if a required attribute has been removed, compilation will fail.

## What Types of Information to Store

It is important to remember that an attribute could be altered by other sections of the code-base- it is therefore dangerous to place dependant data in an attribute. Consider a scenario where a pass assigns two attributes to an operation: `mode` and `latency`, where the `latency` value depends on the chosen `mode`. This setup can lead to two common problems:
1. A later pass updates `mode` but forgets to update `latency`, leaving inconsistent or outdated information in the IR
2. Another pass modifies `latency` without realizing it's derived from `mode`, potentially storing an invalid or contradictory combination.

To avoid these issues, the IR should ideally store **decisions**, not derived data. In this case, the IR should only store `mode`—the actual decision—and compute `latency` on demand when needed. If both values must be persisted, they should be combined into a single, coherent attribute (e.g., a struct or enum) that captures their relationship explicitly.

Any data that depends on other values should be derived, not stored. This ensures that updates to the underlying decision automatically propagate, and future passes can't accidentally introduce inconsistencies.

How to calculate dependant information on-demand (i.e., `latency` in the example) is discussed in detail [below](#how-to-calculate-dependant-information-on-demand), to avoid breaking the logical flow.

## Maintaining Backwards Compatibility

When an attribute is added to an operation, it alters the builders of that operation. Builders are used to add an operation to the IR. If a new attribute is added to an operation, any pre-existing code that adds that operation to the IR will break.

There are 2 typical situations.

1. This pre-existing code **should** break, as the attribute is now required to be specified, and you must alter the code to provide the attribute.
2. The attribute is not relevant to the pre-existing code, and you should provide backwards compatibility.

In order to provide backward compatibility, it is recommended to add a custom builder that provides a default value for the attribute:

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```
In this example, we define an attribute `$attr` with the constraint type `F32Attr`.  We also add a custom builder that takes a float input with a default value of 0.5 (`ins CArg<"float", "0.5f">:$val`). The builder assigns  `val` to the `$attr` attribute. For more details on custom builders, see the [MLIR doc](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods). Then, MLIR generates the following C++ code from the tablegen definition:
```c++
/// Header file.
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val = 0.5f);
};

/// Source file.
MyOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
            float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```
which will maintain backwards compatibility with any builders that do not provide the attribute.

A general principle is to **avoid modifying core passes and builders**—especially when those changes would impact widely used or shared code—unless they represent fundamental structural or philosophical improvements. Instead, it's preferable to add custom solutions tailored to specific needs or passes, keeping the core infrastructure stable and broadly applicable.

## 

MLIR offers two attribute constraints that stack on top of normal ones: [OptionalAttr](https://mlir.llvm.org/docs/DefiningDialects/Operations/#optional-attributes) and [DefaultValuedAttr](https://mlir.llvm.org/docs/DefiningDialects/Operations/#attributes-with-default-values). These wrappers modify how attributes behave during parsing, printing, and code generation.

Here’s an example of using `OptionalAttr` in TableGen:

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins
    OptionalAttr<F32Attr>:$optionalAttr
  );
```
In this case, `$optionalAttr` is an attribute of type `F32Attr` wrapped with the `OptionalAttr` constraint.

While they appear helpful at first glance, neither works exactly as you might expect.

### OptionalAttr

An attribute marked with `OptionalAttr` means that verification will not fail if the attribute is not present. Hence, the attribute is no longer required for the operation to be valid.

It does not mean that the attribute is an optional argument when adding the operation to the IR. By default, some value **must** still be provided for the attribute when adding the operation to the IR. The builder of the operation still requires the attribute as one of its inputs, which would break backwards compatibility (look [here](#maintaining-backwards-compatibility) for more info on backward compatibility).

As described above, backwards compatibility should be maintained using custom builders. 

I believe (but am not 100% sure) that providing a builder which does not take the optional attribute, and does not add it to the state, provides the required backwards compatibility:

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins
    F32Attr:$requiredAttr,
    OptionalAttr<F32Attr>:$optionalAttr
  );

  let builders = [
    OpBuilder<(ins "float":$val), [{
      $_state.addAttribute("requiredAttr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```
(If you are unfamiliar with tablegen builders, more info can be found [here](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods).)

Otherwise, it can be explicitly set using:
```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins
    F32Attr:$requiredAttr,
    OptionalAttr<F32Attr>:$optionalAttr
  );

  let builders = [
    OpBuilder<(ins "float":$val), [{
      build($_builder, $_state, val, std::nullopt);
    }]>
  ];
}
```



### DefaultValuedAttr

The use of `DefaultValueAttr` affects primarily IR printing and parsing: attributes set to their default values are omitted in the .mlir file for more concise IR.

Rarely, a `DefaultValueAttr` does result in an automatically generated custom builder, which allows us to add an operation to the IR without explicitly providing an attribute value. However, due to implementation reasons of C++, this usually does not happen.

Therefore, the use of `DefaultValueAttr` does not typically affect the previously described need for custom builders when maintaining backward compatibility.


## Good Examples
#### FIFO-Depth for a Save-Commit

The save-commit operation (the details of what this operation does are not relevant to this document) has an internal FIFO.

The depth of this FIFO is specified using a required, operation-specific attribute, which is declared in its arguments:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L1199-L1201 

The first two arguments are constrained by type constraints (operands), while the 3rd is constrained by an attribute constraint (attribute).

This results in the following C++ code to create a save-commit operation:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Transforms/Speculation/HandshakeSpeculation.cpp#L384-L387 

It is important to note that since fifoDepth is a required attribute, fifoDepth **must** be passed to the builder in order to create a save-commit operation. The [builder methods](https://mlir.llvm.org/docs/DefiningDialects/Operations/#builder-methods) section of the MLIR Dialect documentation explains well how different C++ builder functions are generated from an operation's tablegen declaration. 

The following getter and setter functions are also generated for accessing the attribute's value:

```c++
saveCommitOp.getFifoDepth()
saveCommitOp.setFifoDepth(int fifoDepth)
```

These functions are generated automatically by declaring the attribute in the tablegen file.

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

Helpful getter and setter functions are also generated for each of its attributes.

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

# 

Instead of defining attributes statically in the Operation TableGen, MLIR allows attributes to be added dynamically at runtime using C++ API functions:
- [setAttr](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#ae5f0d4c61e6e57f360188b1b7ff982f6): assigns a value to an attribute.
- [getAttrOfType](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a8ec626dafc87ed8fb20d8323017dec72): retrieves an attribute of a specific type.

Example:
```
Operation *op;
op->setAttr("mode", StringAttr::get(op->getContext(), "EXEC"));
StringAttr op->getAttrOfType<StringAttr>("mode");
```
In this example, an attribute named `mode` of type `StringAttr` with value `EXEC` is attached to an operation using the `setAttr` function. The attribute is later retrieved with the `getAttrOfType`.

This dynamic approach is simple and flexible—it doesn’t require modifying TableGen definitions or maintaining custom builders. However, it introduces fragility through hardcoded attribute names (e.g.,`mode` in the example), which can lead to duplication and errors.

To address this, we recommend using [interfaces](https://mlir.llvm.org/docs/Interfaces/). Interfaces in MLIR allow operations, types, or attributes to expose structured, reusable behavior through custom methods. This enables:
- Avoiding hardcoded strings,
- Encapsulating attribute logic,
- Sharing attribute handling across multiple operations.

Hence, interfaces offer a more maintainable and scalable solution when attribute behavior becomes more complex or widely reused.

## How to Define Interface Attribute

The following TableGen snippet defines a simple interface for setting and getting a `StringAttr` named `myAttr`:
```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let methods = [
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
        if ( auto attr = op->getAttrOfType<StringAttr>("myAttr") )
          return attr;
        return StringAttr::get(op->getContext, "NULL");      
      }]>
  ]
}
```
This interface defines:
- `setMyAttr`: assigns a string attribute called `myAttr` using `setAttr`.
- `getMyAttr`: retrieves the `myAttr` attribute, returning `NULL` if it's not set.

To enable an operation to use this interface, list it in the operation definition:
```tablegen
def MyOp : Op<"my_op", [MyInterface]> {
  let arguments = ...
```
Now, any `MyOp` operation supports `getMyAttr()` and `setMyAttr(...)` methods. Here's how you would safely use the interface in C++:
```
if (auto myInterface = llvm::dyn_cast<MyInterface>(op)) {
  myInterface.setMyAttr("EXEC");
}
```
This dynamically checks whether the operation implements `MyInterface` before calling the setter method.

As mentioned before, using interfaces to manage attributes in MLIR avoids hardcoding attribute names and centralizes logic, making code more maintainable and reusable. It also allows you to extend operations without modifying their core definitions or builders, which is useful for adding dynamic behavior across multiple ops.

However, attributes added via interfaces are not verified by MLIR, so their presence isn't guaranteed. This makes the approach more flexible but also riskier—getters must always check for attribute existence, and changes elsewhere in the codebase can silently break assumptions.


## Good Example

### InternalDelay for Arithmetic Ops

This is an example of an interface to store the attribute that saves internal delay information in Dynamatic.

This is the code structure of the interfaces in TableGen:

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeInterfaces.td#L221-L246

The two functions in the interface set and retrieve the attribute. Rather than returning null attributes, if the attribute is not present, a default value of "0.0" is returned.

Then, all Handshake_Arith_Op operations can use this interface since we added it in the corresponding TableGen:

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeArithOps.td#L21-L23

The interface can be used to set the value of the attribute (in Buffer Placement) or to retrieve the value of the attribute (in the HandshakeToHw conversion pass):

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Transforms/BufferPlacement/HandshakePlaceBuffers.cpp#L166-L175

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L693-L697 

As shown in this example, this interface can be used across different passes and at different stages of the flow.

# Pros and Cons Summary

The following is a table that sums up the pros and cons of both approaches

TableGen-Based Attribute:

| Pros | Cons |
|------|------|
| Declarative | Adding optional/default values may require custom builders |
| Built-in verification | May break existing code if attribute definitions change |
| Good IR printing/parsing support | Less flexible at runtime |
| Clear schema for operation authors |  |

Interface-Based Attribute:

| Pros | Cons |
|------|------|
| Highly flexible and runtime configurable | No built-in attribute verification |
| Allows adding logic without touching op definitions | Getters must always check for attribute presence |
| Logic can be reused across ops via interfaces | Harder to trace and debug if misused |
| Avoids hardcoding attribute names when wrapped in interface |  |


# Appendix: How to Calculate Dependant Information On Demand

In this section, we show how we can extract information on demand from decisions made in previous passes.

We take the example case of specifying an arithmetic operation's implementation (decision), which in turn defines its internal delay (information).

We first represent the implementation as a StringAttr, add verification logic (in the `verify` function), and a custom builder. 
```
  let arguments = (ins StrAttr:$implementation);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral IMPL1 = "IMPL1";
    static constexpr ::llvm::StringLiteral IMPL2 = "IMPL2";
    static constexpr ::llvm::StringLiteral IMPL3 = "IMPL3";

    static LogicalResult verify(Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("implementation");
      if (!attr)
        return op->emitError("requires 'implementation' attribute");
      auto val = attr.getValue();
      if (val != IMPL1 && val != IMPL2 && val != IMPL3)
        return op->emitError() << "'implementation' must be '" << IMPL1
                              << "', '" << IMPL3 << "', or '" << IMPL3 << "'";
      return success();
    }
  }];

  let builders = [
    // Uses hardcoded string to set 'implementation' 
    // since TableGen builders can't reference C++ constants
    OpBuilder<(ins), [{
      build($_builder, $_state, /*implementation=*/"IMPL1");
    }]>
  ];
```
In this way, we ensure that a decision respects the required assumptions. For instance, in the example, the attribute has to exist, and it must be one of three types (`IMPL1`, `IMPL2`, or `IMPL3`). Additionally, the custom builder allows to respect backward compatibility.

Then, the function to calculate internal delay could look something like this:
```
int getInternalDelay() {
  auto val = getImplementation().getValue();
  if (val == IMPL1) return 1;
  if (val == IMPL2) return 2;
  if (val == IMPL3) return 3;
  llvm_unreachable("Invalid implementation string");
}
```
The information (delay) can be extracted from the decision (implementation).

If this function should be callable across multiple operations, it could come from an interface:
```
def InternalDelayInterface : OpInterface<"InternalDelayInterface"> {
  let methods = [
    InterfaceMethod<[
      "Returns the internal delay (as int) based on the implementation string."
    ], "int", "getInternalDelay", (ins)>
  ];
}
```
This would ensure a safe execution across different passes and operations.

The final operation would look something like:
```
def AddOp : Op<"mydialect.add", [InternalDelayInterface]> {
  let summary = "Add operation with implementation mode";
  let arguments = (ins StrAttr:$implementation);
  let results = (outs);

  let builders = [
    // Uses hardcoded string to set 'implementation' 
    // since TableGen builders can't reference C++ constants
    OpBuilder<(ins), [{
      build($_builder, $_state, /*implementation=*/"IMPL1");
    }]>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr ::llvm::StringLiteral IMPL1 = "IMPL1";
    static constexpr ::llvm::StringLiteral IMPL2 = "IMPL2";
    static constexpr ::llvm::StringLiteral IMPL3 = "IMPL3";

    int getInternalDelay() {
      auto val = getImplementation().getValue();
      if (val == IMPL1) return 1;
      if (val == IMPL2) return 2;
      if (val == IMPL3) return 3;
      llvm_unreachable("Invalid implementation string");
    }

    static LogicalResult verify(Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("implementation");
      if (!attr)
        return op->emitError("requires 'implementation' attribute");
      auto val = attr.getValue();
      if (val != IMPL1 && val != IMPL2 && val != IMPL3)
        return op->emitError() << "'implementation' must be '" << IMPL1
                               << "', '" << IMPL2 << "', or '" << IMPL3 << "'";
      return success();
    }
  }];
}
```
which combines the previous codes.

Finally, to set the attribute in C++ would use the following code:

```
addOp.setImplementation(AddOp::IMPL1);
```
which would be a safe assignment that would not break backward compatibility.

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
