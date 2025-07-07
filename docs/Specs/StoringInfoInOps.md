
# Storing Information in Operations

It is often useful to store additional information on an operation. One example is if we wish to parameterize the generation of its RTL. This requires storing the parameter value on the operation itself, as well as passing it to the relevant parts of the codebase to ensure correct generation. 

- [What are Attributes](#what-are-attributes)
- [Attributes in Tablegen](#attributes-in-tablegen)
  - [How to Declare an Attribute in Tablegen](#how-to-declare-an-attribute-in-tablegen)
  - [Maintaining Backwards Compatibility](#maintaining-backwards-compatibility)
  - [Attribute Constraint Wrappers](#attribute-constraint-wrappers)
    - [OptionalAttr](#optionalattr)
    - [DefaultValuedAttr](#defaultvaluedattr)
  - [Good Examples](#good-examples)
    - [FIFO-Depth for a Save-Commit](#fifo-depth-for-a-save-commit)
    - [SharingWrapperOp for Crush](#sharingwrapperop-for-crush)
- [Manually Abstracting and Verifying Attributes](#manually-abstracting-and-verifying-attributes)
  - [Abstracting](#abstracting)
  - [Verifying](#verifying)
  - [Mixed Example](#mixed-example)
    - [BufferOp](#bufferop)
- [Multi-Operation Attributes](#multi-operation-attributes)
  - [Interfaces](#interfaces)
    - [Good Example](#good-example)
      - [InternalDelay for Arithmetic Ops](#internaldelay-for-arithmetic-ops)
  - [Free Functions](#free-functions)
    - [Medium Example](#medium-example)
      - [Basic Block Number](#basic-block-number)
- [Pros and Cons Summary](#pros-and-cons-summary)
- [Appendix: How to Calculate Dependent Information On Demand](#appendix-how-to-calculate-dependent-information-on-demand)

# What are Attributes?

Additional information about an operation should be stored using attributes. Each operation in MLIR has an attribute dictionary, which stores compile-time constants in name-value pairs. However, due to the fragility of name-value pairs, this underlying structure should be both **abstracted** and **strictly verified**.

MLIR offers us an automated flow to perform this abstraction and verification, by declaring the attribute in an operation's tablegen definition. However, if the generated code is not exactly what you require, it is also possible to implement manually.

Additionally, if an attribute should be present on many units, the automated flow can be cumbersome to add to many operations. We discuss this case [at the end](#multi-operation-attributes), and provide two possible solutions.

## Why Do We Store It Like This?

Intermediate representations must be serializable, which means to convert them to a text representation and store them in a file. Therefore the complex C++ data structures representing operations, and their connections, must be serializable. Since attributes are a feature of MLIR, this serialization (and de-serializaton) of our extra information is handled automatically.

If your information does not need to persist between passes, you may be better off using a custom temporary data structure. However, any information that should remain attached to an operation for the rest of compilation should be an attribute.


## What Types of Information to Store

It is important to remember that an attribute could be altered by other sections of the codebase- it is therefore dangerous to place dependent data in an attribute. Consider a scenario where a pass assigns two attributes to an operation: `mode` and `latency`, where the `latency` value depends on the chosen `mode`. This setup can lead to two common problems:
1. A later pass updates `mode` but forgets to update `latency`, leaving inconsistent or outdated information in the IR
2. Another pass modifies `latency` without realizing it's derived from `mode`, potentially storing an invalid or contradictory combination.

To avoid these issues, the IR should ideally store **decisions**, not derived data. In this case, the IR should only store `mode`—the actual decision—and compute `latency` on demand when needed.

Any data that depends on other values should be derived, not stored. This ensures that updates to the underlying decision automatically propagate, and future passes can't accidentally introduce inconsistencies.

How to calculate dependent information on-demand (i.e., `latency` in the example) is discussed in detail [below](#how-to-calculate-dependent-information-on-demand), to avoid breaking the logical flow.


# Attributes in Tablegen

If you declare an attribute using tablegen, the abstraction and verification of the name-value pair is handled automatically. 

## How to Declare an Attribute in Tablegen

The [operation definition specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) (ODS) covers how MLIR uses tablegen files to declaratively define operations, including attributes. 
The [specific section](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) of the ODS relevant to attributes is on operation arguments. 

Arguments can be specified in an operation's tablegen entry like so:

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

Argument order is flexible; operands and attributes can be mixed, as their constraint types specify their roles.

## Maintaining Backwards Compatibility

When an attribute is added to an operation, it alters the default builder of that operation. Builders are used to add an operation to the IR. If a new attribute is added to an operation, any pre-existing code that adds that operation to the IR will break.

There are 2 typical situations.

1. This pre-existing code **should** break, as the attribute is now required to be specified, and you must alter the code to provide the attribute.
2. The attribute is not relevant to the pre-existing code, and you should provide backwards compatibility.

To choose between the two, you must consider allocation of responsibility. Is your attribute relevant to the pre-existing code? If not, it should not be responsible for choosing the value of the attribute. 

This includes setting the value to null: if your attribute is not relevant to the pre-existing code, that code should remain unaware of the attribute.

In order to implement option 2, and provide backward compatibility, it is recommended to add a custom builder that provides a default value for the attribute:

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

## Attribute Constraint Wrappers

MLIR offers two attribute constraints that stack on top of normal ones: [OptionalAttr](https://mlir.llvm.org/docs/DefiningDialects/Operations/#optional-attributes) and [DefaultValuedAttr](https://mlir.llvm.org/docs/DefiningDialects/Operations/#attributes-with-default-values). These wrappers modify how attributes behave during parsing, printing, and code generation.

Here’s an example of using `OptionalAttr` in TableGen:

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins
    OptionalAttr<F32Attr>:$optionalAttr
  );
}
```
In this case, `$optionalAttr` is an attribute of type `F32Attr` wrapped with the `OptionalAttr` constraint.

While they appear helpful at first glance, neither works exactly as you might expect.

### OptionalAttr

An attribute marked with `OptionalAttr` means that verification will not fail if the attribute is not present. Hence, the attribute is no longer required for the operation to be valid.

It does not mean that the attribute is an optional argument when adding the operation to the IR. By default, some value **must** still be provided for the attribute when adding the operation to the IR. Like any [std::optional](https://en.cppreference.com/w/cpp/utility/optional.html), a value of [std::nullopt](https://en.cppreference.com/w/cpp/utility/optional/nullopt.html) is used to indicate that no value is present. Since the builder of the operation still requires the attribute as one of its inputs, adding an optional attribute will still break backwards compatibility (look [here](#maintaining-backwards-compatibility) for more info on backward compatibility).

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

The use of `DefaultValuedAttr` affects primarily IR printing and parsing: attributes set to their default values are omitted in the .mlir file for a more concise IR.

Rarely, a `DefaultValuedAttr` does result in an automatically generated custom builder, which allows us to add an operation to the IR without explicitly providing an attribute value. However, due to implementation reasons of C++, this usually does not happen.

Therefore, the use of `DefaultValuedAttr` does not typically affect the previously described need for custom builders when maintaining backward compatibility.


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

# Manually Abstracting and Verifying Attributes

Instead of declaratively specifying an attribute in a single line in the Operation TableGen, you could implement this directly.

The lower level functions to manage the attribute by name-value pair are:

- [setAttr](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#ae5f0d4c61e6e57f360188b1b7ff982f6): assigns a value to an attribute.
- [getAttrOfType](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html#a8ec626dafc87ed8fb20d8323017dec72): retrieves an attribute of a specific type.

Example:
```
Operation *op;
op->setAttr("mode", StringAttr::get(op->getContext(), "EXEC"));
StringAttr op->getAttrOfType<StringAttr>("mode");
```
In this example, an attribute named `mode` of type `StringAttr` with value `EXEC` is attached to an operation using the `setAttr` function. The attribute is later retrieved with the `getAttrOfType`.

This approach can be beneficial, as it avoids MLIR taking actions we do not want, such as breaking backwards compatibility of builders. However, it introduces fragility through hardcoded attribute names (e.g.,`mode` in the example).

## Abstracting

Therefore, these functions should almost always be wrapped in dedicated getters or setters, which should be declared in `extraClassDeclaration`

```tablegen
let extraClassDeclaration = [{
  void setMyAttr(StringRef value);
  StringAttr getMyAttr();
}];
```

These must then be implemented in `lib/Dialect/Handshake/HandshakeOps.cpp` (or in a similar file, if not a handshake operation) like so:

```c++
void MyOp::setMyAttr(StringRef value) {
  (*this)->setAttr("myAttr", StringAttr::get(getContext(), value));
}

StringAttr MyOp::getMyAttr() {
  return (*this)->getAttrOfType<StringAttr>("myAttr");
}
```

`getAttrOfType` is safe to call if the attribute is not present, but will return a null pointer.

## Verifying

The attribute should additionally be verified, to make sure any rules for its presence or value are respected.

You can add additional verification to an operation by adding
```tablegen
let hasVerifier = 1;
```
to its tablegen definition.

This adds a verify function to the operation, which you can add a custom definition for (again in `lib/Dialect/Handshake/HandshakeOps.cpp` or similar)

```c++
mlir::LogicalResult MyOp::verify() {
  if (!getMyAttr())
    return emitOpError("requires 'myAttr' to be set");

  return mlir::success();
}
```

This verify function will be called at the end of every pass.

## Mixed Example

#### BufferOp
As of time of writing, the buffer operation stores its RTL parameter attributes in a dictionary called "hw.parameters". This is less ideal than the above description of manual attribute handling, but in other ways this operation follows this paradigm.

Its arguments do not contain any attribute information:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L226

But it defines additional hardcoded strings to use as dictionary keys using:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L233-L271

A custom builder is declared in tablegen:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L229-L232

and then implemented separately in C++:
https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/lib/Dialect/Handshake/HandshakeOps.cpp#L198-L221

This C++ does not use dedicated getters and setters, but rather `constexpr llvm::StringLiteral`.

Due to this custom builder, the C++ to add a new operation looks as if it had used dedicated attributes:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Transforms/HandshakePlaceBuffersCustom.cpp#L102-L103


Verification is done manually, but weakly: the presence of the attributes is optional, as if the upper level hw.parameters is not present, verification passes. Direct strings are used to access the attributes, which is a very fragile approach. 

https://github.com/EPFL-LAP/dynamatic/blob/7532a8e33cd3acf4196fece75a4dc20a60b66bb3/lib/Dialect/Handshake/HandshakeOps.cpp#L429-L455

There are also no named getters written, and therefore these attributes must be accessed very awkwardly through the dictionary attribute:

https://github.com/EPFL-LAP/dynamatic/blob/66162ef6eb9cf2ee429e58f52c5e5e3c61496bdd/experimental/lib/Support/SubjectGraph.cpp#L825-L830

The above getter is also not safe, as the buffer verification does not enforce that hw.parameters is present, and could cause a null pointer exception.

# Multi-Operation Attributes

As mentioned at the beginning, if an attribute could be present on many operations, to perform the above steps per operation can become burdensome.

There are 2 alternatives:
1. Interfaces
2. Free Functions

## Interfaces

To address this, we recommend using [interfaces](https://mlir.llvm.org/docs/Interfaces/). Interfaces in MLIR allow operations to share methods, reducing code duplication. The hardcode string of the attribute name can be placed in these shared methods, reducing the fragility of name-value pairs.

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
        return op->getAttrOfType<StringAttr>("myAttr") )
      }]>
  ]
}
```
This interface defines:
- `setMyAttr`: assigns a string attribute called `myAttr` using `setAttr`.
- `getMyAttr`: retrieves the `myAttr` attribute, returning a null pointer if it is not present.

To enable an operation to use this interface, list it in the operation definition:
```tablegen
def MyOp : Op<"my_op", [MyInterface]> {
  let arguments = ...
```
Now, any `MyOp` operation supports `getMyAttr()` and `setMyAttr(...)` methods, usable in C++:
```
if (auto myInterface = llvm::dyn_cast<MyInterface>(op)) {
  myInterface.setMyAttr("foo");
}
```

However, you cannot add custom verification directly to an interface, making interface attribute methods less safe than operation attribute methods. If desired, you could call an interface verify function from the verify function of each operation that implements it, but this scales poorly. 

In this way, interfaces offer attribute abstraction, but not attribute verification.

### Good Example

#### InternalDelay for Arithmetic Ops

This is an example of an interface to store the attribute that saves internal delay information in Dynamatic.

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeInterfaces.td#L221-L246

The two functions in the interface set and retrieve the attribute. Rather than returning null attributes, if the attribute is not present, a default value of "0.0" is returned.

The interface is added to Handshake_Arith_Op, the base class for all arithmetic operations.

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/include/dynamatic/Dialect/Handshake/HandshakeArithOps.td#L21-L23

The interface can be used to set the value of the attribute (in Buffer Placement) or to retrieve the value of the attribute (in the HandshakeToHw conversion pass):

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Transforms/BufferPlacement/HandshakePlaceBuffers.cpp#L166-L175

https://github.com/EPFL-LAP/dynamatic/blob/f52cb1f922897faf1b66fb087a601064f89e11b4/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L693-L697 

As shown in this example, this interface can be used across different passes and at different stages of the flow.

## Free Functions

If the presence or absence of an attribute is independent of an operation's type, interfaces may not make sense, as the interface would need to be added to every single operation. Instead getters and setters can be implemented as free functions. Free functions are functions which are not part of any class.

Free verification functions could also be written if desired, but like with interfaces, they would need to be manually added to the verify function of each individual operation, which makes verification more-or-less impossible.

### Medium Example

#### Basic Block Number

Operations in the Handshake IR have a non-operation-specific attribute to indicate which basic block they belong to.

And free getter is provided to get the attribute:

https://github.com/EPFL-LAP/dynamatic/blob/7532a8e33cd3acf4196fece75a4dc20a60b66bb3/lib/Support/CFG.cpp#L33-L38

And a free setter is used to set the attribute.
https://github.com/EPFL-LAP/dynamatic/blob/7532a8e33cd3acf4196fece75a4dc20a60b66bb3/lib/Support/CFG.cpp#L56-L60

Interesting, a `constexpr llvm::StringLiteral` is used to store the attribute name.

https://github.com/EPFL-LAP/dynamatic/blob/7532a8e33cd3acf4196fece75a4dc20a60b66bb3/include/dynamatic/Support/CFG.h#L32-L34

This is due to inconsistant design of how the BB attribute is accessed and used, which increases the fragility of name-value pairs. While you may wish to use a `constexpr llvm::StringLiteral` even when you are stricter with your getter and setter, needing this is a sign of poor design.

# Pros and Cons Summary

## Declarative (TableGen-Based) Operation Attributes
| Aspect                  | Description                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Use Case                | Best when attributes are operation-specific.                                  |
| Abstraction             | Declared in the `let arguments` list using attribute constraints. MLIR auto-generates typed getters and setters.                 |
| Verification            | Automatic verification of attribute presence. Value is verified using the provided attribute constraint.  |
| Attribute Name Handling | Attribute name is embedded in generated getter and setter. |
| Integration             | **Breaks the default builder**, often requiring custom builders to be provided.          |
| Scalability             | Not scalable across many ops, as requires logic to be duplicated.                           |

## Manual Operation Attribute Handling
| Aspect                  | Description                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| Use Case                | Best when attributes are operation-specific. |
| Abstraction             | Implemented using explicit getter/setter methods that call `getAttrOfType` / `setAttr`.                  |
| Verification            | Must be added manually using `let hasVerifier = 1` and a custom `verify()` method.                       |
| Attribute Name Handling | Attribute name is hardcoded in getter and setter.  |
| Integration             | Does not alter the default builder, which avoids breaking backwards compatibility with existing code.             |
| Scalability             | Not scalable across many ops, as requires logic to be duplicated.                        |

## Interfaces

| Aspect                  | Description                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| Use Case                | Best suited when the same attribute and logic are shared across many related operations.                        |
| Abstraction             | Define shared getter/setter methods via `OpInterface`.      |
| Verification            | No built-in verification: if needed, it must be added manually in each op’s `verify()`, which **scales poorly**.              |
| Attribute Name Handling | Centralized in one place inside the interface getter and setter.    |
| Integration             | Only requires adding the interface to each participating op's definition. External logic handling the attribute can check for a single interface, rather than many operations. |
| Scalability             | Scales well for logic reuse, but poorly for enforcing correctness or presence of the attribute.                 |

## Free Functions

| Aspect                  | Description                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Use Case                | Best suited when attribute presence is independent of operation type and may apply to many ops.                                 |
| Abstraction             | Implement free getter/setter functions, defined outside of any operation class.                                                 |
| Verification            | Possible via helper functions, but must be manually called from each op’s `verify()`, making it **not scalable**.                |
| Attribute Name Handling | Should be centralized inside the free getter and setter. If not centralized, storing attribute name in a `constexpr llvm::StringLiteral` can reduce fragility, but usually indicates poor design. |
| Integration             | Doesn't require editing op definitions as functions are external.                                                            |
| Scalability             | Easy to reuse logic across many ops                                   |

# Appendix: How to Calculate dependent Information On Demand

In this section, we show how we can extract information on demand from decisions made in previous passes.

We take the example case of specifying an arithmetic operation's implementation (decision), which in turn defines its internal delay (information).

We first represent the implementation as a custom `EnumAttr`, to enforce a limited set of choices. Since the implementations vary per operation, this enum would be defined per arithmetic operation.
```
def Handshake_MyOpImpAttr : I64EnumAttr<
    "MyOpImp", "",
    [
      I64EnumAttrCase<"Impl0", 0, "impl1">,
      I64EnumAttrCase<"Impl1", 1>,
      I64EnumAttrCase<"Impl2", 2>,
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}
```

An `I64EnumAttrCase<symbol, intVal, strVal>` is used to specify the cases. 
1.`symbol` is the C++ used to retrieve constants, e.g. `MyOpImp::IMPL0`
2.`intVal` is the integer value of the enum case.
3.`strVal` is an optional value, which will be returned by the auto-generated `ConvertToString(MyOpImp val)` function, if you wish to use that function for something. If `strVal` is absent, `ConvertToString(MyOpImp val)` returns `symbol` instead.


We then use this attribute constraint to declaratively add an attribute to the operation in tablegen. Additionally, we add a custom builder to maintain backwards compatibility.

```
def MyOp : Op<> {
  let arguments = (ins Handshake_MyOpImpAttr:$implementation);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, /*implementation=*/MyOpImp::Impl1);
    }]>
  ];
}
```

Then, the function to calculate internal delay could look something like this:
```
int getInternalDelay() {
  switch (getImplementation().getValue()) {
    case MyOpImp::Impl1: return 1;
    case MyOpImp::Impl2: return 2;
    case MyOpImp::Impl3: return 3;
  }
  llvm_unreachable("Invalid implementation enum");
}
```
The information (delay) is extracted from the decision (implementation).

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

The final operation would look something like:
```
def Handshake_MyOpImpAttr : I64EnumAttr<
    "MyOpImp", "",
    [
      I64EnumAttrCase<"Impl0", 0, "impl1">,
      I64EnumAttrCase<"Impl1", 1>,
      I64EnumAttrCase<"Impl2", 2>,
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}

def MyOp : Op<[InternalDelayInterface]> {
  let summary = "Operation with implementation mode";
  let arguments = (ins Handshake_MyOpImpAttr:$implementation);
  let results = (outs);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, /*implementation=*/MyOpImp::Impl1);
    }]>
  ];

  let extraClassDeclaration = [{
    int getInternalDelay() {
      switch (getImplementation().getValue()) {
        case MyOpImp::Impl1: return 1;
        case MyOpImp::Impl2: return 2;
        case MyOpImp::Impl3: return 3;
      }
      llvm_unreachable("Invalid implementation enum");
    }
  }];
}
```
which combines the previous codes.

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
