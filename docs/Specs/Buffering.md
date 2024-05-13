# Buffering

> [!NOTE]
> This is a proposed design change; it is not implemented yet.

Currently, Dyanamtic represents dataflow circuit buffers using a pair of MLIR operations that are part of the Handshake dialect.

- `handshake::OEHBOp` for **O**paque **E**lastic **H**alf **B**uffers.
- `handshake::TEHBOp` for **T**ransparent **E**lastic **H**alf **B**uffers.

Both of these operations have similar interfaces, with a single operand and result for their single input and output dataflow channel, respectively, and a strictly positive integer attribute denoting the number of slots the buffer has (i.e., the maximum number of dataflow tokens it can hold concurrently at any given time).

## The problem

While workable, our current buffer representation in MLIR exposes an issue in how we model dataflow buffers in general. Indeed, at the MLIR level it is unclear what the cycle latency is between every input/output port pair of a buffer. By port we mean RTL-level port (and not SSA value), so a single wire or bus; in practice, the data bus, the valid wire, or the ready wire. This lack in representation restricts the kind of buffers we can place in our dataflow circuits without IR changes and carries unclear assumptions on what path(s) the opaque and transparent half buffers are exactly supposed to cut.

## Proposed solution

Having latency information encoded in the IR itself is generally undesirable for other operations. That information is usually exclusively present in external timing models that can be loaded into memory when necessary, for example during buffer placement. However, since the sole purpose of buffers is to cut paths, latency information characterizes them as much as, for example, the number of output ports characterizes a dataflow fork. In the same way that we want to provide the number of fork results when inserting one in the IR, when placing a buffer on a dataflow channel we want to be able to specify which paths are cut, and by how many cycles.

### Operation representation

To address the current issue, we propose to replace the two existing buffer operations (`handshake::OEHBOp` and `handshake::TEHBOp`) with a generic single-operand and single-result buffer operation (`handshake::BufferOp`) that takes as additional input latency information associating each cut path with the number of cycles the buffer delays the flow of data by, thus filling the current implementation's representation gap. A new MLIR attribute `handshake::LatencyInfo` would represent that information conveniently as a list of input-to-output port associations with their associated latency. For convenience, all unspecified paths would be assumed to have 0 latency.

The textual representation of the `handshake::BufferOp` operation would look as follows.

```mlir
%dataOut = handshake.buffer [<number of slots>] %dataIn [<latency information>] : <channel-type>
```

Here `%dataIn` is the buffer's operand SSA value (the input dataflow channel) and, conversely,`%dataOut` is the buffer's result SSA value (the output dataflow channel).

For example, a 1-slot buffer on a 32-bit dataflow channel which induces a one-cycle latency on the data to data and valid to valid paths would be serialized to the following.

```mlir
%dataOut = handshake.buffer [1] %dataIn [D -> D: 1, V -> V: 1] : channel<i32>
```

Paths between identically-typed ports (data, valid, or ready) could even be shortened for brevity, yielding a shorter textual representation than the previous one.

```mlir
%dataOut = handshake.buffer [1] %dataIn [D: 1, V: 1] : channel<i32>
```

### Extra characterization

Using Dynamatic's new backend and its support for adding RTL parameters to operations at any point in the compilation flow (see [relevant section in backend documentation](Backend.md#identifying-necessary-modules), in particular the note at the end), it would also be possible to differentiate between multiple types of buffers with identical latency characteristics through user-defined "extra RTL parameters". For example, to differentiate between two different implementations (say, `A` and `B`) of a buffer that adds a one-cycle latency on the data and valid paths, one could define and add an `IMPLEMENTATION` RTL parameter on relevant buffers at the Handshake IR level. With proper support in [RTL configuration files](Backend.md#rtl-configuration), the backend would then be able to instantiate the appropriate buffer implementation for each `handshake::BufferOp` operation in the IR.

```mlir
// One-slot data/valid-cutting buffer, implementation "A" 
%dataOut1 = handshake.buffer [1] %dataIn1 [D: 1, V: 1] {hw.parameters = {IMPLEMENTATION = "A"}} : channel<i32>

// One-slot data/valid-cutting buffer, implementation "B" 
%dataOut2 = handshake.buffer [1] %dataIn2 [D: 1, V: 1] {hw.parameters = {IMPLEMENTATION = "B"}} : channel<i32>
```

This effectively allows arbitrary complexity in modeling buffers in the IR and eventually emit them to RTL.

### Interaction with buffer placement pass

Currently, the buffer placement pass assumes that it can only place the two types of elastic buffers we can represent using Handshake, OEHBs and TEHBs, and incorporates assumptions on the latency each of them induce on specific paths. Adding a new type of buffer currently requires changing the implementation of the pass itself (though this would be a small change, the pass being implemented in a relatively generic way). The proposed redesign would remove these assumptions and make the pass completely general in the nature of buffers it can model inside the MILP and eventually decide to place on our circuits' channels. There are two general ways in which to approach MILP-based buffer placement with our new generic buffer representation while ensuring that (1) all combinational loops are cut by at least one buffer and that (2) all combinational paths will be able to meet the target clock period.

1. *Path-based approach*, in which we express the MILP independently from the set of buffer types available. The MILP's results encode which path(s) it decides to cut on each channel, if any. It is then up to the user to honor the MILP's results by placing buffers on appropriate channels depending on the set of buffer types available. This is basically what we currently do.
2. *Buffer-based approach*, in which the set of available buffer types is encoded in the MILP itself. The MILP's results directly encode which specific buffer type(s) it decides to place on each channel, if any. This has the advantage of incorporating the full timing characteristics (including combinational delays) of each placable buffer type in the MILP's calculations, yielding placements more likely to meet the target clock period. The MILP, however, is likely to be more complex and as such harder to solve than the path-based one when many buffer types are available.  

### RTL generation

The expressiveness of the proposed MLIR buffer operation would need to be met by our RTL backend. While not all combinations of path latencies necessarily make sense, we are likely to need one or more buffer generator that are able to generate buffer implementations on-demand that meet certain latency characteristics. As a starting point, we could still restrict ourselves to the generic RTL implementations we already have for OEHBs and TEHBs. While this would initially prohibit us from placing different types of buffers in our circuit, future-proofing the design now would improve the overall code architecure of Dynamatic (especially, of the buffer placement pass) and enable us to easily introduce new types of buffers in the future.
