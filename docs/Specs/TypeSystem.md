# Type System

> [!NOTE]
> This is a proposed design change; it is not implemented yet.

Currently, at the Handshake IR level, all SSA values are implicitly assumed to represent dataflow channels, even when their type seems to denote a simple "raw" signal. More accurately, the `handshake::FuncOp` MLIR operation---which maps down from the original C kernel and eventually ends up as the top-level RTL module representing the kernel---provides implicit Handshake semantics to all SSA values defined within its regions.

For example, consider a trivial C kernel.

```c
int adder(int a, int b) { return a + b; }
```

At the Handshake level, the IR that Dynamatic generates for this kernel would like as follows (some details unimportant in the context of this proposal are omitted for brevity).

```mlir
handshake.func @adder(%a: i32, %b: i32, %start: none) -> i32  {
    %add = arith.addi %a, %b : i32
    %ret = handshake.return %add : i32
    handshake.end %ret : i32
}
```

Each `i32`-typed SSA value in this IR represents in fact a dataflow channel with a 32-bit data bus (which should be interpreted as an integer). Also note that control-only dataflow channel (with no data bus) are somewhat special-cased in the current type system by using the standard MLIR `NoneType` (written as `none`) in the IR. While this may be a questionnable design decision in the first place (the `i0` type, which is legal in MLIR, could be conceived as a better choice), it is not fundamentally important for this proposal.

## The problem

On one hand, implicit dataflow semantics within Handshake functions have the advantage of yielding neat-looking IRs that do not bother to deal with an explicit parametric "dataflow type" repeated everywhere. On the other hand, it also prevents us from mixing regular dataflow channels (downstream data bus, downstream valid wire, and upstream ready wire) with any other kind of signal bundle.

1. On one side, "raw" un-handshaked signals would look indistinguishable from regular dataflow channels in the IR. If a dataflow channel with a 32-bit data bus is represented using `i32`, then no existing type can represent a 32-bit data bus without the valid/ready signal bundle. Raw signals could be useful, for example, for any kind of partial circuit rigidification, where some channels that provably do not need handshake semantics could drop their valid/ready bundle and only be represented as a single data bus.
2. On the other side, adding extra signals to some dataflow channels that may need to carry additional information around is also impossible modulo addition of a new parametric type. For example, speculation bits or thread tags cannot currently be modeled by this simple type system.

While MLIR attributes attached to operations whose adjacent channels are "special" (either because they drop handshake semantics or add extra signals) could potentially be a solution to the issue, we argue that it would be cumbersome to work with and error-prone for the following reasons.

1. MLIR treats custom attribute opaquely, and therefore cannot automatically verify that they make any sense in any given context. We would have to define complex verification logic ourselves and think of verifying IR sanity every time we transform it.
2. Attributes heavily clutter the IR, making it harder to look at whenever many operations possess (potentially complex) custom attributes. This hinders debuggability since it is sometimes useful to look directly at the serialized IR to understand what a pass inputs or outputs.

## Proposed solution

### New types

We argue that the only way to obtain the flexibility outlined above is to

1. make dataflow semantics explicit in Handshake functions through the introduction of custom IR types, and
2. use MLIR's flexible and customizable type system to automatically check for IR sanity at all times.

We propose to add two new types to the IR to enable us to reliably model our use cases inside Handshake-level IR.

- A *nonparametric* type to model control-only tokens which lowers to a bundle made up of a downstream valid wire and upstream ready wire. This `handshake::ControlType` type would serialize to `control` inside the IR.
- A *parametric* type to model dataflow channels with an arbitrary data type and optional extra signals. In their most basic form, SSA values of this type would be a composition of an arbitrary "raw-typed" SSA value (e.g., `i32`) and of a `control`-typed SSA value. It follows that values of this type, in their basic form, would lower to a bundle made up of a downstream data bus of a specific bitwidth plus what the `control`-typed SSA value lowered to (valid and ready wires). Optionally, this type could also hold extra "raw-typed" signals (e.g., speculation bits, thread tags) that would lower to downstream or upstream buses of corresponding widths. This `handshake::ChannelType` type would serialize to `channel<data-type, {optional-extra-types}>` inside the IR.

Considering again our initial simple example, it seems that the proposed changes would make the IR look identical modulo cosmetic type changes.

```mlir
handshake.func @adder(%a: channel<i32>, %b: channel<i32>, %start: control) -> channel<i32>  {
    %add_result = arith.addi %a, %b : channel<i32>
    %ret = handshake.return %add_result : channel<i32>
    handshake.end %ret : channel<i32>
}
```

However, this in fact would be rejected by MLIR. The problem is that the standard MLIR operation representing the addition (`arith.addi`) expects operands of a raw integer-like type, as opposed to some custom data-type it does not know (i.e., `channel<i32>`). This in fact may have been one of the motivations behind the implicit dataflow semantic design assumption in Handshake; all operations from the standard `arith` and `math` dialects expect raw integer or floating-point types (depending on the specific operation) and cannot consequently accept custom types like the one we are proposing here. We will therefore need to redefine the standard arithmetic and mathematical operations within Handshake to support our custom data types. The IR would look identical as above except for the name of the dialect prefixing `addi`.

```mlir
handshake.func @adder(%a: channel<i32>, %b: channel<i32>, %start: control) -> channel<i32>  {
    %add_result = handshake.addi %a, %b : channel<i32>
    %ret = handshake.return %add_result : channel<i32>
    handshake.end %ret : channel<i32>
}
```

### New operations

Occasionaly, we will want to unbundle channel-typed SSA values into their individual signals and later recombine the individual components into a single channel-typed SSA value. We propose to introduce two new operations to fulfill this requirement.

- An unbundling operation (`handshake::UnbundleOp`) which generally breaks down its channel-typed SSA operand into its individual components, which it produces as separate SSA results.
- A converse bundling operation (`handshake::BundleOp`) which generally combines multiple raw-typed SSA operands and combines them into a single channel-typed SSA value which it produces as a single SSA result.

We include a simple example below (see the [next subsection](#extra-signal-handling) for more complex use cases).

```mlir
// Breaking down a simple 32-bit dataflow channel into its individual
// control and data components, then rebundling it
%channel = ... : channel<i32>
%control, %data = handshake.unbundle %channel : control, i32
%channelAgain = handshake.bundle %control, %data : channel<i32>
```

### Extra signal handling

To support the use case where extra signals need to be carried on some dataflow channel (e.g., speculation bits, thread tags), the `handshake::ChannelType` needs to be flexible enough to model an arbitrary number of extra raw data-types (in addition to the "regular" data-type). In order to prepare for future use cases, each extra signal should also be characterized by its direction, either downstream or upstream. Extra signals may also optionally declare unique names to refer themselves by, allowing client code to more easily query for a specifc signal in complex channels.  

Below are a few MLIR serialization examples for dataflow channels with extra signals.

```mlir
// A basic channel with 32-bit integer data and no extra signal
%channel = ... : channel<i32>

// -----

// A channel with 32-bit integer data and an extra unnamed 1-bit signal (e.g., a
// speculation bit) going downstream
%channel = ... : channel<i32, [i1]>

// -----

// A channel with 32-bit integer data and two extra named thread tags,
// respectively of 2-bit width and 4-bit width, both going downstream
%channel = ... : channel<i32, [tag1: i2, tag2: i4]>

// -----

// A channel with 32-bit integer data and an extra 1-bit signal going upstream,
// as indicated by the "(U)"; extra signals are by default downstream (most
// common use case) so they get no such annotation
%channel = ... : channel<i32, [otherReady: (U) i1]>
```

The unbundling and bundling operations would also unbundle and bundle, respectively, all the extra signals together with the raw data bus and control-only token.

```mlir
// Multiple thread tags example from above
%channel = ... : channel<i32, [tag1: i2, tag2: i4]>

// Unbundle into control-only token and all individual signals
%control, %data, %tag1, %tag2 = handshake.unbundle %channel : control, i32, i2, i4

// Bundle to get back the original channel
%bundled = handshake.bundle %control, %data [%tag1, %tag2] : channel<i32, [tag1: i2, tag2: i4]>

// -----

// Upstream extra signal example from above
%channel = ... : channel<i32, [otherReady: (U) i1]>

// Unbundle into control-only token and raw data; note that, because the extra
// signal is going upstream, it is an input of the unbundling operation instead
// of an output 
%control, %data = handshake.unbundle %channel, %otherReady : control, i32

// Bundle to get back the original channel; note that, because the extra signal
// is going upstream, it is an output of the bundling operation instead of an
// input
%bundled, %otherReady = handshake.bundle %control, %data : channel<i32, [otherReady: (U) i1]>

// -----

// Control-typed values can be further unbundled into their individual signals 
%control = ... : control
%valid = handshake.unbundle %control, %ready : i1
%controlAgain, %ready = handshake.bundle %valid : control, i1
```

Most operations accepting channel-typed SSA operands will likely not care for these extra signals and will follow some sort of simple forwarding behavior for them. It is likely that pairs of specific Handshake operations will care to add/remove certain types of extra signals between their operands and results. For example, in the speculation use case, the specific operation marking the beginning of a speculative region would take care of adding an extra 1-bit signal to its operand's specific channel-type. Conversely, the special operation marking the end of the speculative region would take care of removing the extra 1-bit signal from its operand's specific channel-type.

Going further, if multiple regions requiring extra signals were ever nested within each other, it is likely that adding/removing extra signals in a stack-like fashion would suffice to achieve correct behavior. However, if that is insufficient and extra signals were not necessarily removed at the same rate or in the exact reverse order in which they were added, then the unique extra signal names could serve as identifiers for the specific signals that a signal-removing unit should care about removing.

## Discussion

In this section we try to alleviate potential concerns with the proposed change and discuss the latter's impact on other parts of Dynamatic.

### Type checking

Using MLIR's type system to model the exact nature of each channel in our circuits makes us benefit from MLIR's existing type management and verification infrastructure. We will be able to cleanly define and check for custom type checking rules on each operation type, ensuring that the relationships between operand and result types always makes sense; all the while permitting our operations to handle an infinite number of variations of our parametric types.

For example, the integer addition operation (`handshake.addi`) would check that its two operands and result have the same type. Furthermore, this type would only be required to be a channel with a non-zero-width integer type.

```mlir
// Valid
%addOprd1, %addOprd2 = ... : channel<i32>
%addResult = handshake.addi %addOprd1, %addOprd2 : channel<i32>

// -----

// Invalid, data type has 0 width
%addOprd1, %addOprd2 = ... : channel<i0>
%addResult = handshake.addi %addOprd1, %addOprd2 : channel<i0>
```

### IR complexity

Despite the added complexity introduced by our parametric channel type, the representation of core dataflow components (e.g., merges and branches) would remain structurally identical beyond cosmetic type name changes.

```mlir
// Current implementation
%mergeOprd1 = ... : none
%mergeOprd2 = ... : none
%mergeResult, %index = handshake.control_merge %mergeOprd1, %mergeOprd2 : none, i1

%muxOprd1 = ... : i32
%muxOprd2 = ... : i32
%muxResult = handshake.mux %index [%muxOprd1, %muxOprd2] : i32

// -----

// With proposed changes 
%mergeOprd1 = ... : control
%mergeOprd2 = ... : control
%mergeResult, %index = handshake.control_merge %mergeOprd1, %mergeOprd2 : control, channel<i1>

%muxOprd1 = ... : channel<i32>
%muxOprd2 = ... : channel<i32>
%muxResult = handshake.mux %index [%muxOprd1, %muxOprd2] : channel<i1>, channel<i32>

// -----

// No extra operations when extra signals are present 
%mergeOprd1 = ... : control
%mergeOprd2 = ... : control
%mergeResult, %index = handshake.control_merge %mergeOprd1, %mergeOprd2 : control, channel<i1>

%muxOprd1 = ... : channel<i32, [i2, i4]>
%muxOprd2 = ... : channel<i32, [i2, i4]>
%muxResult = handshake.mux %index [%muxOprd1, %muxOprd2] : channel<i1>, channel<i32, [i2, i4]>
```

### Backend changes

The support for "nonstandard" channels in the IR means that we have to match this support in our RTL backend. Indeed, most current RTL components take the data bus's bitwidth as an RTL parameter. This is no longer sufficient when dataflow channels can carry extra downstream or upstream signals, which must somehow be encoded in the RTL parameters of numerous core dataflow components (e.g., all merge-like and branch-like components). Complex channels will need to become encodable as RTL parameters for the underlying RTL implementations to be concretized correctly. It is basically a given that generic RTL implementations which we largely rely on today will not be sufficient, and that the design change will require us moving to RTL generators for most core dataflow components. Alternatively, we could use a form of [signal composition](#signal-compositon) (see below) to narrow down the amount of channel types our components have to support.

### Signal compositon

In some instances, it may be useful to compose all of a channel's signals going in the same direction (downstream or upstream) together around operations that do not care about the actual content of their operands' data buses (e.g., all data operands of merge-like and branch-like operations). This would allow us to expose to certain operations "regular" dataflow channels without extra signals; their *exposed data buses* would in fact be constituted of the *actual data buses* plus all extra downstream signals. Just before lowering to HW and then RTL (after applying all Handshake-level transformations and optimizations to the IR), we could run a signal-composition pass that would apply this transformation around specific dataflow components in order to make our backend's life easier.

Considering again the last example with extra signals from the [IR complexity](#ir-complexity) subsection above, we could make our current generic mux implementation work with the new type system without modifications to the RTL.

```mlir
%index = ... : channel<i1>
%muxOprd1 = ... : channel<i32, [i2, i4]>
%muxOprd2 = ... : channel<i32, [i2, i4]>

// Our current generic RTL mux implementation does not work because of the extra
// signals attached to the data operands' channels
%muxResult = handshake.mux %index [%muxOprd1, %muxOprd2] : channel<i1>, channel<i32, [i2, i4]>

// -----

// Same inputs as before 
%index = ... : channel<i1>
%muxOprd1 = ... : channel<i32, [i2, i4]>
%muxOprd2 = ... : channel<i32, [i2, i4]>

// Compose data operands's extra signals with the data bus
%muxComposedOprd1 = handshake.compose %muxOprd1 : channel<i32, [i2, i4]> -> channel<i38> 
%muxComposedOprd2 = handshake.compose %muxOprd2 : channel<i32, [i2, i4]> -> channel<i38> 

// Our current generic RTL mux implementation would work out-of-the-box!
%muxComposedResult = handshake.mux %index [%muxComposedOprd1, %muxComposedOprd2] : channel<i1>, channel<i38>

// Most likely some operation down-the-line actually cares about the isolated
// extra signals, so undo handshake.compose's effect on the mux result 
%muxResult = handshake.decompose %muxComposedResult : channel<i38> -> channel<i32, [i2, i4]>
```

The RTL implementations of the `handshake.compose` and `handshake.decompose` signals would be trivial and offload complexity from the dataflow components themselves, making the latter's RTL implementations simpler and area smaller.

A similar yet slightly different composition behavior could help us simplify the RTL implementation of arithmetic operations---which would usually forward all extra signals between their operands and results---as well. In cases where it makes sense, we could compose all of the operands' and results' downstream extra signals into a single one that is still separate from the data signal, which arithmetic operations actually use. We could then design a (couple of) generic implementation(s) for these arithmetic operations that would work for all channel types, removing the need for a generator.

```mlir
%addOprd1 = ... : channel<i32, [i2, i4, (U) i4, (U) i8]>
%addOprd2 = ... : channel<i32, [i2, i4, (U) i4, (U) i8]>

// Given the variability in the extra signals, this operation would require an
// RTL generator
%addResult = handshake.addi %addOprd1, %addOprd2 : channel<i32, [i2, i4, (U) i4, (U) i8]>

// -----

// Same inputs as before 
%addOprd1 = ... : channel<i32, [i2, i4, (U) i4, (U) i8]>
%addOprd2 = ... : channel<i32, [i2, i4, (U) i4, (U) i8]>

// Compose all extra signals going in the same direction into a single one
%addComposedOprd1 = handshake.compose %addOprd1 : channel<i32, [i2, i4, (U) i4, (U) i8]> 
                                                  -> channel<i32, [i6, (U) i12]> 
%addComposedOprd2 = handshake.compose %addOprd2 : channel<i32, [i2, i4, (U) i4, (U) i8]>
                                                  -> channel<i32, [i6, (U) i12]> 

// We could design a generic version of the adder that accepts a single
// downstream extra signal and a single upstream data signal
%addComposedResult = handshake.addi %addComposedOprd2, %addComposedOprd2 : channel<i32, [i6, (U) i12]>

// Decompose back into the original type
%addResult = handshake.decompose %addComposedResult : channel<i32, [i6, (U) i12]>
                                                      -> channel<i32, [i2, i4, (U) i4, (U) i8]>
```
