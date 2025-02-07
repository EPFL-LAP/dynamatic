I think TableGen files are unique, so you might not be familiar with them. I'll cover the basics here.

## Operations

**Operations** in the Handshake IR (such as `MergeOp` or `ConstantOp`) are defined declaratively in TableGen files (`HandshakeOps.td` or `HandshakeArithOps.td`).

Each operation has **arguments**, which are categorized into operands, attributes, and properties. However, only **operands** are relevant here. Operands represent the inputs to the RTL here. For example, `ConditionalBranchOp` has two operands: one for the condition and one for the data.

https://github.com/EPFL-LAP/dynamatic/blob/32df72b2255767c843ec4f251508b5a6179901b1/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L457-L458

Some operands are **variadic**, meaning they can have a variable number of inputs. For example, the data operand of `MuxOp` is variadic.

https://github.com/EPFL-LAP/dynamatic/blob/32df72b2255767c843ec4f251508b5a6179901b1/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L362-L363

More on operation arguments: https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments

Each operation also has **results**, which represent the outputs of the RTL here. For instance, `ConditionalBranchOp` has two results, corresponding to the "true" and "false" branches.

https://github.com/EPFL-LAP/dynamatic/blob/32df72b2255767c843ec4f251508b5a6179901b1/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L459-L460

More on operation results: https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results

## Types

You may notice that operands and results are often denoted by types like `HandshakeType` or `BoolChannel`. In Handshake IR, types specify the kind of RTL port.

There are two main types:

- **`ChannelType`** – Represents a data port with **data + valid + ready** signals.
- **`ControlType`** – Represents a control port with **valid + ready** signals.

We define **`HandshakeType`** as `ChannelType`+`ControlType`.

These types are defined in `HandshakeTypes.td`.



The actual operands have concrete **instances** of these types. For example, an operand of `AddIOp` (integer addition) has a `ChannelType`, meaning its actual type will be:

- `!handshake.channel<i32>` (for 32-bit integers)
- `!handshake.channel<i8>` (for 8-bit integers)

Since `ChannelType` allows different data types, multiple type instances are possible.

`BoolChannel` is a special case that only allows `!handshake.channel<i1>`.



Some `HandshakeType` instances may include **extra signals** beyond `(data +) valid + ready`. For example:

- `!handshake.channel<i32, [spec: i1]>`
- `!handshake.control<[spec: i1, tag: i8]>`



In some cases, we need an operand **without** extra signals. To enforce this, we use **simple types**:

- `SimpleHandshake`
- `SimpleChannel`
- `SimpleControl`

These types also appear in this pull request. For example, in `MemoryControllerOp`, some operands/results are of SimpleType.

https://github.com/EPFL-LAP/dynamatic/blob/28872676a0f3438e82c064242fac517059e22fc2/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L621-L624

These ensure that the type instance does not carry any additional signals beyond the basic ones.

## Traits

**Traits** are constraints applied to operations. They serve various purposes, but here they primarily enforce rules on operand and result types.

A key part of this pull request is the **implementation of certain traits** and their **application to operations**.



For example, In `ConditionalBranchOp`, the **data operand** and **trueOut/falseOut results** must have the same type instance (e.g., `!handshake.channel<i32>`). However, simply specifying `ChannelType` for each is not enough—without additional constraints, the operation could exist with mismatched types, like:

- `dataOperand: !handshake.channel<i8>`
- `trueResult / falseResult: !handshake.channel<i32>`

To enforce type consistency, we apply the **`AllTypesMatch`** trait:

https://github.com/EPFL-LAP/dynamatic/blob/32df72b2255767c843ec4f251508b5a6179901b1/include/dynamatic/Dialect/Handshake/HandshakeOps.td#L436-L437

This ensures that all three elements share the exact same type instance.



MLIR provides `AllTypesMatch`, but we've introduced similar traits:

- **`AllDataTypesMatch`** – Ignores differences in extra signals.
- **`AllExtraSignalsMatch`** – Ensures the extra signals match, ignoring the data type (if exists).



Traits are sometimes called **multi-entity constraints** because they enforce relationships across multiple operands or results.

- In contrast, types (or type constraints) are called **single-entity constraints** to enforce properties on individual elements.

More on constraints: https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints

## More Information

The MLIR documentation can be complex, but it covers the key concepts well. You can check out the following links for more details:

https://mlir.llvm.org/docs/DefiningDialects/Operations

https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes

