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

To address the current issue, we propose to replace the two existing buffer operations (`handshake::OEHBOp` and `handshake::TEHBOp`) with a single buffer operation (`handshake::BufferOp`) that takes as additional input latency information associating each cut path with the number of cycles the buffer delays the flow of data by, thus filling the current implementation's representation gap. A new MLIR attribute `handshake::LatencyInfo` would represent that information conveniently as a list of input-to-output port associations with their associated latency. For convenience, all unspecified paths would be assumed to have 0 latency. An instance of the `handshake::BufferOp` operation with no latencies explicitly specified would therefore behave as a no-op and could be canonicalized during compilation.

The textual representation of the `handshake::BufferOp` operation would look as follows.

```mlir
%dataOut = handshake.buffer [<number of slots>] %dataIn [<latency information>] : <channel-type>
```

Here `%dataIn` is the buffer's operand SSA value (the input dataflow channel) and, conversely,`%dataOut` is the buffer's result SSA value (the output dataflow channel).

For example, a 1-slot buffer on a 32-bit dataflow channel which induces a one-cycle latency on the data to data and valid to valid paths would be serialized to the following text.

```mlir
%dataOut = handshake.buffer [1] %dataIn [D -> D: 1, V -> V: 1] : channel<i32>
```

Paths between identically-typed ports (data, valid, or ready) could even be shortened for brevity, yielding a shorter textual representation than the previous one.

```mlir
%dataOut = handshake.buffer [1] %dataIn [D: 1, V: 1] : channel<i32>
```

### Interaction with buffer placement pass

Currently, the buffer placement pass assumes that it can only place the two types of elastic buffers we can represent using Handshake, OEHBs and TEHBs, and incorporates assumptions on the latency each of them induce on specific paths. Adding a new type of buffer currently requires changing the implementation of the pass itself (though this would be a small change, the pass being implemented in a relatively generic way).

The proposed redesign would remove these assumptions and make the pass completely general in the nature of buffers it can model inside the MILP and eventually decide to place on our circuits' channels. In terms of workflow, a user would now need to inform the buffer placement pass of the set of buffer types (characterized by their `handshake::LatencyInfo` attribute) that it is allowed to place in the circuit. The pass would then retrieve complete timing models for each of the requested buffers by using a list of [RTL configuration files](RTLConfiguration.md) (the same ones that the backend uses later for RTL emission). Each timing model, in addition to holding the same path latencies contained in the `handshake::LatencyInfo` attribute it was queried from, would provide all relevant combinational delays within the buffer (including on paths with 0 latency). Finally, each of these buffer model would be accurately transcribed inside the MILP, allowing the pass to place our specific buffers so that (1) all combinational loops are cut by at least one buffer and (2) that all combinational paths will be able to meet the target clock period.  

### RTL generation

The expressiveness of the proposed MLIR buffer operation would need to be met by our RTL backend. While not all combinations of path latencies necessarily make sense, we are likely to need one or more buffer generator that are able to generate buffer implementations on-demand that meet certain latency characteristics. As a starting point, we could still restrict ourselves to the generic RTL implementations we already have for OEHBs and TEHBs. While this would initially prohibit us from placing different types of buffers in our circuit, future-proofing the design now would improve the overall code architecure of Dynamatic (especially, of the buffer placement pass) and enable us to easily introduce new types of buffers in the future.
