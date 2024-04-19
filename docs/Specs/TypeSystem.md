# Type System

> [!NOTE]
> This is a proposed design change; it is not implemented yet.

Currently, at the Handshake IR level, all SSA values are implicitly assumed to represent dataflow channels, even when their type seems to denote a simple "raw" signal. More accurately, the `handshake::FuncOp` MLIR operation---which maps down from the original C kernel and eventually ends up as the top-level RTL module representing the kernel---provides implicit Handshake semantics to all SSA values defined within its regions.

For example, consider a trivial C kernel.

```c
int adder(int a, int b) { return a + b; }
```

At the Handshake level, the IR that Dynamatic generates for this kernel would like as follows (some details unimportant in the context of this proposal are ommited for brevity).

```mlir
handshake.func @adder(%a: i32, %b: i32, %start: none) -> i32  {
    %add_result = arith.addi %a, %b : i32
    %ret = return %add_result : i32
    end %ret : i32
}
```

Each `i32`-typed SSA value in this IR represents in fact a dataflow channel with a 32-bit data bus (which should be interpreted as an integer). Also note that control-only dataflow channel (with no data bus) are somewhat special-cased in the current type system by using the standard MLIR `NoneType` (written as `none`) in the IR. While this may be a questionnable design decision in the first place (the `i0` type, which is legal in MLIR, could be conceived as a better choice), it is not fundamentally important for this proposal.

## The problem

On one hand, implicit dataflow semantics within Handshake functions have the advantage of yielding neat-looking IRs that do not bother to deal with an explicit parametric "dataflow type" repeated everywhere. On the other hand, it also prevents us from mixing regular dataflow channels (downstream data bus, downstream valid wire, and upstream ready wire) with any other kind of signal bundle.

1. On one side, "raw" un-handshaked signals would look indistinguishable from regular dataflow channels in the IR. If a dataflow channel with a 32-bit data bus is represented using `i32`, then no existing type can represent a 32-bit data bus without the valid/ready signal bundle. Raw signals could be useful, for example, for any kind of partial circuit rigidification, where some channels that provably do not need handshake semantics could drop their valid/ready bundle and only be represented as a single data bus.
2. On the other side, adding extra signals to some dataflow channels that may need to cary additional information around is also impossible modulo addition of a new parametric type. For example, speculation bits or thread tags cannot currently be modeled by this simple type system.

While MLIR attributes attached to operations whose adjacent channels are "special" (either because they drop handshake semantics or add extra signals) could potentially be a solution to the issue, we argue that it would be cumbersome to work with and error-prone for the following reasons.

1. MLIR treats custom attribute opaquely, and therefore cannot automatically verify that they make any sense in any given context. We would have to define complex verification logic ourselves and think of verifying IR sanity every time we transform it.
2. Attributes heavily clutter the IR, making it harder to look at whenever many operations possess (potentially complex) custom attributes. This hinders debuggability since it is sometimes useful to look directly at the serialized IR to understand what a pass inputs or outputs.

## Proposed solution

### New types

We argue that the only way to obtain the flexibility outlined above is to

1. make dataflow semantics explicit in Handshake functions through the introduction of custom IR types, and
2. use MLIR's flexible and customizable type system to automatically check for IR sanity at all times.

We propose to add two new types to the IR to enable us to reliably model our use cases inside Handshake-level IR.

- A *non-parametric* type to model control-only dataflow channels which lowers to a bundle made up of a downstream valid wire and upstream ready wire. This `handshake::ControlType` would serialize to `control` inside the IR.
- A *parametric* type to model dataflow channels with an arbitrary data type and optional extra signals. In their most basic form, SSA values of this type would be a composition of an arbitrary "raw-typed" SSA value (e.g., `i32`) and of a `control`-typed SSA value. It follows that values of this type, in their basic form, would lower to a bundle made up of a downstream data bus of a specific bitwidth plus what the `control`-typed SSA value lowered to (valid and ready wires). Optionally, this type could also hold extra "raw-typed" signals (e.g., speculation bits, thread tags) that would lower to downstream or upstream buses of corresponding widths. This `handshake::ChannelType` would serialize to `channel<data-type, {optional-extra-types}>` inside the IR.

### New operations

Re-considering our initial simple example, it seems that the proposed changes would make the IR look identical modulo cosmetic type changes.

```mlir
handshake.func @adder(%a: channel<i32>, %b: channel<i32>, %start: control) -> channel<i32>  {
    %add_result = arith.addi %a, %b : channel<i32>
    %ret = return %add_result : channel<i32>
    end %ret : channel<i32>
}
```

However, this in fact would be rejected by MLIR. The problem is that the standard MLIR operation representing the addition (`arith.addi`) expects operands of a raw integer-like type, as opposed to some custom data-type it does not know (i.e., `channel<i32>`). This in fact may have been one of the motivations behind the implicit dataflow semantic design assumption in Handshake; all operations from the standard `arith` and `math` dialects expect raw integer or floating-point types (depending on the specific operation) and cannot consequently accept custom types like the one we are proposing here.

In a way, an additional unstated assumption in our current Handshake-level IR is that an add operation (`arith.addi`) is not in fact *just* an addition, it is a join of both operands' control ports *in parallel to* an addition of the operands' data port (the same is true for all other mathematical operations). In the spirit of making dataflow semantics explicit within the IR, it therefore would make sense to start treating an add operation as *just* an addition, and make the joining logic explicit in the IR when it happens. We would need to introduce a couple new Handshake operations to make this work.

- An unpacking operation (`handshake::UnpackOp`) which takes as operand a channel-typed SSA value and breaks it down into its two individual components; a raw data-typed SSA value and a control-typed SSA value.
- A corresponding packing operation (`handshake::PackOp`) which takes two operands, a raw data-typed SSA value and a control-typed SSA value, and produces a single channel-typed SSA value as result.
- An explicit join operation (`handshake::JoinOp`) which takes an arbitray number of control-typed SSA values as operands and produces a single control-typed SSA value which results from the join of all the operands (join operations with a single operand could be canonicalized away since they would act as no-op). This operation actually already exists in Handshake.

With those new operations available, a correct IR for our simple adder would be much more formal and explicit, at the cost of some added operations.

```mlir
handshake.func @adder(%a: channel<i32>, %b: channel<i32>, %start: control) -> channel<i32>  {
    // Split a and b into their data/control components
    %a_data, %a_control = handshake.unpack %a : i32, control
    %b_data, %b_control = handshake.unpack %b : i32, control
    
    // Perform the addition between a's and b's data port and the join
    // between their respective control ports separately
    %add_data = arith.addi %a_data, %b_data : i32
    %add_control = handshake.join %a_control, %b_control : control

    // Recombine the addition's result with the join's result and return 
    %add = handshake.pack %add_data, %add_control : channel<i32>
    %ret = return %add : channel<i32>
    end %ret : channel<i32>
}
```

## Discussion

There is no question that the proposed redesign would yield IRs that are longer and visually more complex than what our current Handshake implementation produces. However, we argue that, in addition to enabling arbitrarily complex mixing of signal bundles as part of the IR's formalized type system, it would be easier in many cases to reason about the IR systematically from compiler passes, even in cases where our circuit would only be made up of "standard" (data, valid, and ready) dataflow channels.
