# How to Add a New Component

This document explains how to add a new component to Dynamatic.

### Summary of Steps

- Define a Handshake Op.
- Implement the logic to propagate it to the backend.
- Add the corresponding RTL implementation.

## 1. Define a Handshake Op

The first step is to define a Handshake Op. In MLIR, an Op refers to a specific, concrete operation (see [Op vs Operation](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Tutorials/MLIRPrimer.md#op-vs-operation) for more details).

Handshake Ops are defined using the [LLVM TableGen format](https://llvm.org/docs/TableGen/index.html), typically in either `include/dynamatic/Dialect/Handshake/HandshakeOps.td` or `HandshakeArithOps.td`.

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
                       UI32Attr:$fifoDepth);
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
   Defines the op's inputs, which can be operands or attributes (or properties, which are not used).

  - `HandshakeType:$operand1`: Defines `operand1` as an operand of type `HandshakeType`.

  - `UI32Attr:$fifoDepth`: Defines `fifoDepth` as an attribute of type `UI32Attr`.

- `let results = ...`
   Defines the results produced by the op.

- `let assemblyFormat = ...`
   Specifies a declarative assembly format for the op's representation.
  - Some existing ops use a custom format with `let hasCustomAssemblyFormat = 1`, but this should only be used if the declarative approach is insufficient (which is rare).
- `let extraClassDeclaration = ...`
   Declares additional C++ methods for the op.
  - You should implement `getOperandName` and `getResultName` from `NamedIOInterface` here, in this declaration block, to follow the single-source-of-truth principle.
  - These methods are necessary because operand/result names defined in TableGen are not accessible from C++; MLIR internally identifies them only by index. The names are primarily used during static code generation via [ODS (Operation Definition Specification)](https://mlir.llvm.org/docs/DefiningDialects/Operations/).

For more details, refer to the [MLIR documentation](https://mlir.llvm.org/docs/DefiningDialects/). However, in practice, reviewing existing op declarations in the Handshake or HW dialects, or even in [CIRCT](https://github.com/llvm/circt) often provides a more concrete and intuitive understanding.

## 2. Implement Propagation Logic to the Backend

From this point on, the steps depend on which backend you're targeting: the legacy backend or the newer *beta backend* (used for speculation, out-of-order execution, and SMV generation).

In this guide, we assume you're supporting both backends and outline the necessary steps for each.

> [!NOTE]
> This process is subject to change. A backend redesign is planned, which may significantly alter these steps.

### `HandshakeToHW.cpp`

First, update the conversion pass from Handshake IR to HW IR, located in
 `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp`.

Start by registering a rewrite pattern for your op, like this:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L1786

Then, implement the corresponding rewrite pattern. Most of the infrastructure is already in place; you mainly need to define op-specific hardware parameters (`hw.parameters`) where applicable. For the **legacy backend**, you need to explicitly register type information and any additional data here for the RTL generation. For example:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L517-L521

For the **beta backend**, even if your op doesn't require any `hw.parameters`, you still need to add a case for it, like in this example:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Conversion/HandshakeToHW/HandshakeToHW.cpp#L676-L679

### `RTL.cpp`

Second, to support the **beta backend**, you need to update `lib/Support/RTL/RTL.cpp`, which handles RTL generation. Specifically, you'll need to add **parameter analysis** for your op, which extracts information such as bitwidths or additional signals required during RTL generation.

In most cases, if your op enforces traits like `AllTypesMatch` across all operands and results, extracting a single bitwidth or `extra_signals` is sufficient. Examples:

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Support/RTL/RTL.cpp#L338-L350

https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/lib/Support/RTL/RTL.cpp#L434-L453

Note: At this stage, you're working with HW IR, not Handshake IR, so operands and results must be accessed by index, not by name.

The reason this analysis is performed here is to bypass all earlier passes and avoid any unintended transformations or side effects.

### JSON Configuration for RTL Matching

You'll need to update the appropriate JSON file to enable RTL matching for your op.

- For the **legacy backend**, we use `data/rtl-config-vhdl.json`. You need to add a new entry specifying the VHDL file and any `hw.parameters` you registered in `HandshakeToHW.cpp`, like in this example:
  https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/data/rtl-config-vhdl.json#L10-L17
- For the **beta backend**, we use `data/rtl-config-vhdl-beta.json`. This JSON file resolves compatibility with the current `export-rtl` tool. Simply specify the generator and pass the required parameters as arguments:
  https://github.com/EPFL-LAP/dynamatic/blob/c618f58e7909a4cc9cf53e432e49f451210a8c76/data/rtl-config-vhdl-beta.json#L7-L10

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

To fully integrate your op into Dynamatic, additional steps may be required. These steps are spread throughout the codebase, but in the future, they should all be connected to the **tablegen** definition (as interfaces or other means) to maintain the single-source-of-truth principle and improve readability. The RTL propagation logic (Step 2) is also planned to be implemented as an interface through the backend redesign.

- Timing/Latency Models: Register the timing and latency values in `data/components.json`. Additionally, add a case for your op in `lib/Support/TimingModels.cpp`. Further modifications may be required.

- `export-dot`: To assign a color to your op in the visualized circuit, you’ll need to add a case for it in `tools/export-dot/export-dot.cpp`:

  https://github.com/EPFL-LAP/dynamatic/blob/1887ba219bbbc08438301e22fbb7487e019f2dbe/tools/export-dot/export-dot.cpp#L276-L283
