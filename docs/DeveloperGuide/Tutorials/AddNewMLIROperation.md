# How to Add a New Component

This document explains how to add a new component to Dynamatic.

It does **not** cover when a new component *should* be created or how it *should* be designed. A separate guideline for that will be added.

### Summary of Steps

- Define a Handshake Op.
- Implement the logic to propagate it to the backend.
- Add the corresponding RTL implementation.

## 1. Define a Handshake Op

The first step is to define a Handshake *op*. Note that in MLIR, an op refers to a specific, concrete operation (see [Op vs Operation](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Tutorials/MLIRPrimer.md#op-vs-operation) for more details).

Handshake ops are defined using the [LLVM TableGen format](https://llvm.org/docs/TableGen/index.html), in either `include/dynamatic/Dialect/Handshake/HandshakeOps.td` or `HandshakeArithOps.td`.

The simplest way to define your op is to mimic an existing, similar one. A typical op declaration looks like this:

```
def SomethingOp : Handshake_Op<"something", [
  AllTypesMatch<["operand1", "result1", "result2"]>,
  IsIntChannel<"operand2">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
  // more traits if needed
]> {
  let summary = "summary";
  let description = [{
    Description.

    Example:

    ```mlir
    %res1, %res2 = something %op1, %op2 : !handshake.channel<i32>, !handshake.channel<i8>
    ```
  }];

  let arguments = (ins HandshakeType:$operand1,
                       ChannelType:$operand2,
                       UI32Attr:$attr1);
  let results = (outs HandshakeType:$result1,
                      HandshakeType:$result2);

  let assemblyFormat = [{
    $operand1 `,` $operand2 attr-dict
      `:` type($operand1) `,` type($operand2)
  }];
  let extraClassDeclaration = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getNumOperands() && "index too high");
      return (idx == 0) ? "operand1" : "operand2";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < getNumResults() && "index too high");
      return (idx == 0) ? "result1" : "result2";
    }
  }];
}
```

Here’s a breakdown of each part of the op definition:

- `def SomethingOp : Handshake_Op<"something", ...> {}`
   This defines a new op named `SomethingOp`, inheriting from `Handshake_Op`.
  - `SomethingOp` becomes the name of the corresponding C++ class.
  - `"something"` is the op’s *mnemonic*, which appears in the IR.
- `[AllTypesMatch<...>, ...]`
   This is a list of **traits**. Traits serve multiple purposes: categorizing ops, indicating capabilities, and enforcing constraints.
  - `AllTypesMatch<["operand1", "result1", "result2"]>`: Ensures that all listed operands/results share the same type.
  - `IsIntChannel<"operand2">`: Constrains `operand2` to have an integer type.
  - `DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>`: **Required**. Indicates that the op implements the `NamedIOInterface`, specifically the `getOperandName` and `getResultName` methods. These are used during RTL generation.

- `let summary = ...` / `let description = ...`
   These provide a short summary and a longer description of the op.

- `let arguments = ...`
   Defines the op's inputs, which can be operands, attributes, or properties.

  - `HandshakeType:$operand1`: Defines `operand1` as an operand of type `HandshakeType`.

  - `UI32Attr:$attr1`: Defines `attr1` as an attribute of type `UI32Attr`. Attributes represent op-specific data, such as comparison predicates or internal FIFO depths. For example:
  https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/include/dynamatic/Dialect/Handshake/HandshakeArithOps.td#L225
  https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L1196

- `let results = ...`
   Defines the results produced by the op.

- `let assemblyFormat = ...`
   Specifies a declarative assembly format for the op's representation.
  - Some existing ops use a custom format with `let hasCustomAssemblyFormat = 1`, but this should only be used if the declarative approach is insufficient (which is rare).
- `let extraClassDeclaration = ...`
   Declares additional C++ methods for the op.
  - You should implement `getOperandName` and `getResultName` from `NamedIOInterface` here, in this declaration block, to follow the single-source-of-truth principle.
    - These methods are necessary because operand/result names defined in TableGen are not accessible from C++; MLIR internally identifies them only by index. The names are primarily used during static code generation via [ODS (Operation Definition Specification)](https://mlir.llvm.org/docs/DefiningDialects/Operations/).
    - Some existing ops declare these methods in external C++ files, which should be avoided as it reduces traceability.

For more details, refer to the [MLIR documentation](https://mlir.llvm.org/docs/DefiningDialects/). However, in practice, reviewing existing op declarations in the Handshake or HW dialects, or even in [CIRCT](https://github.com/llvm/circt) often provides a more concrete and intuitive understanding.

### Design Guidelines

A complete guideline for designing an op will be provided in a separate document. Below are some key points to keep in mind:

- **Define operands and results clearly.** Here's an example of poor design, where the declaration gives no insight into the operands:
  https://github.com/EPFL-LAP/dynamatic/blob/13f600398f6f028adc9538ab29390973bff44503/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L1398
  Use precise and meaningful types for operands and results. Avoid using variadic operands/results for fundamentally different values. This makes the op's intent explicit and helps prevent it from being used in unintended ways that could cause incorrect behavior.
- **Use traits to enforce type constraints.** Apply appropriate type constraints directly using traits in TableGen. Avoid relying on op-specific verify methods for this purpose unless absolutely necessary.  
Below are poor examples from CMerge and Mux, for two main reasons:  
  (1) The constraints should be expressed as traits, and  
  (2) They should be written in the TableGen definition for better traceability.
  https://github.com/EPFL-LAP/dynamatic/blob/69274ea6429c40d1c469ffaf8bc36265cbef2dd3/lib/Dialect/Handshake/HandshakeOps.cpp#L302-L305
  https://github.com/EPFL-LAP/dynamatic/blob/69274ea6429c40d1c469ffaf8bc36265cbef2dd3/lib/Dialect/Handshake/HandshakeOps.cpp#L375-L377
- **Prefer declarative definitions over external C++ implementations.** Write methods in TableGen whenever possible. Only use external C++ definitions if the method becomes too long or compromises readability.
- **Use dedicated attributes instead of `hw.parameters`.** The `hw.parameters` attribute in the *Handshake* IR is a legacy mechanism for passing data directly to the backend. While some existing operations like `BufferOp` still use it in the Handshake IR, new implementations should use dedicated attributes instead, as described above. Information needed for RTL generation should be extracted later in a serialized form.
  Note: `hw.parameters` remains valid in the *HW* IR, and the legacy backend requires it.

## 2. Implement Propagation Logic to the Backend

From this point on, the steps depend on which backend you're targeting: the legacy backend or the newer *beta backend* of VHDL (used for speculation and out-of-order execution).

In this guide, we assume you're supporting both backends and outline the necessary steps for each.

> [!NOTE]
> This process is subject to change. A backend redesign is planned, which may significantly alter these steps.

### `HandshakeToHW.cpp` (Module Discriminator)

First, update the conversion pass from Handshake IR to HW IR, located in
 `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp`.

Start by registering a rewrite pattern for your op, like this:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L1786

Then, implement the corresponding rewrite pattern (**module discriminator**). Most of the infrastructure is already in place; you mainly need to define op-specific hardware parameters (`hw.parameters`) where applicable. For the **legacy backend**, you need to explicitly register type information and any additional data here for the RTL generation. For example:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L517-L521

You should also add dedicated attributes to `hw.parameters` at this stage:
https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L662-L664
https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L680-L683

For the **beta backend**, most parameter registration is handled in `RTL.cpp`. However, if you define dedicated attributes, you need to pass their values into `hw.parameters` here, as shown above. Note that even if no extraction is needed, you still have to add an empty case for the op here, as follows:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L676-L679

### `RTL.cpp` (Parameter Analysis)

Second, to support the **beta backend**, you need to update `lib/Support/RTL/RTL.cpp`, which handles RTL generation. Specifically, you'll need to add **parameter analysis** for your op, which extracts information such as bitwidths or extra signals required during RTL generation.

In most cases, if your op enforces traits like `AllTypesMatch` across all operands and results, extracting a single bitwidth or `extra_signals` is sufficient. Examples (you can **scroll** these code blocks):

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Support/RTL/RTL.cpp#L338-L350

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Support/RTL/RTL.cpp#L434-L453

> [!NOTE]
> At this stage, you're working with HW IR, not Handshake IR, so operands and results must be accessed by index, not by name.

The reason this analysis is performed here is to bypass all earlier passes and avoid any unintended transformations or side effects.

### JSON Configuration for RTL Matching

You'll need to update the appropriate JSON file to enable RTL matching for your op.

- For the **legacy backend**, we use `data/rtl-config-vhdl.json`. You need to add a new entry specifying the VHDL file and any `hw.parameters` you registered in `HandshakeToHW.cpp`, like in this example:
  https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/data/rtl-config-vhdl.json#L10-L17
- For the **beta backend**, we use `data/rtl-config-vhdl-beta.json`. This JSON file resolves compatibility with the current `export-rtl` tool. Basically, you just need to specify the generator and pass the required parameters as arguments:
  https://github.com/EPFL-LAP/dynamatic/blob/c618f58e7909a4cc9cf53e432e49f451210a8c76/data/rtl-config-vhdl-beta.json#L7-L10
  However, if you define dedicated attributes and implement a module discriminator, you should declare the parameters in the JSON, as well as specifying them as arguments, in the following way:
  https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/data/rtl-config-vhdl-beta.json#L30-L39
  https://github.com/EPFL-LAP/dynamatic/blob/1875891e577c655f374a814b7a42dd96cd59c8da/data/rtl-config-vhdl-beta.json#L211-L220
  The parameter names match those used in the `addUnsigned` or `addString` calls within each module discriminator.

- You may also need to update the JSON files for other backends, such as Verilog or SMV, depending on your use case.

## 3. Add the RTL Implementation

To complete support for your op, you need to provide an RTL implementation for the relevant backend.

- For the **legacy backend**, place your VHDL file in the `data/vhdl/` directory.

- For the **beta backend**, add a **VHDL module generator written in Python** under `experimental/tools/unit-generators/vhdl/generators/handshake/`. To implement your generator, please refer to the existing implementations in this directory for guidance.

  Your generator should define a function named `generate_<unit_name>(name, params)`, as shown in this example:

  https://github.com/EPFL-LAP/dynamatic/blob/c618f58e7909a4cc9cf53e432e49f451210a8c76/experimental/tools/unit-generators/vhdl/generators/handshake/addi.py#L5-L12

  After that, register your generator in `experimental/tools/unit-generators/vhdl/vhdl-unit-generator.py`:

  https://github.com/EPFL-LAP/dynamatic/blob/c618f58e7909a4cc9cf53e432e49f451210a8c76/experimental/tools/unit-generators/vhdl/vhdl-unit-generator.py#L39-L44

- You may also need to implement RTL for other backends, such as Verilog and SMV. Additionally, to support XLS generation, you'll need to update the `HandshakeToXls` pass accordingly.

## Other Procedures

To fully integrate your op into Dynamatic, additional steps may be required. These steps are spread throughout the codebase, but in the future, they should all be tied to the **tablegen definition** (as interfaces or other means) to maintain the single-source-of-truth principle and improve readability. The RTL propagation logic (Step 2) is also planned to be implemented as an interface through the backend redesign.

- Timing/Latency Models: To support MLIP-based buffering algorithms, register the timing and latency values in `data/components.json`. Additionally, add a case for your op in `lib/Support/TimingModels.cpp` if needed. Further modifications may be required.

- `export-dot`: To assign a color to your op in the visualized circuit, you’ll need to add a case for it in `tools/export-dot/export-dot.cpp`:

  https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/tools/export-dot/export-dot.cpp#L276-L283
