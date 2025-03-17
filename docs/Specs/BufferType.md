# Buffering: BufferType

This is a design update. Previously, the MLIR represents dataflow circuit buffers with a single-operand, single-result buffer operation (`handshake::BufferOp`). The buffer operation is characterized by a number of slots and a timing attribute that specifies cycle latencies on various signal paths. For example, the textual representation of the `handshake::BufferOp` operation of a 1-slot buffer on a 32-bit dataflow channel which induces a one-cycle latency on the data to data and valid to valid paths would be serialized to the following.

```mlir
%dataOut = handshake.buffer %dataIn {hw.parameters = {TIMING = {D: 1, V: 1}, NUM_SLOTS = 1 : ui32}} : channel
```

Here `%dataIn` is the buffer's operand SSA value (the input dataflow channel) and, conversely,`%dataOut` is the buffer's result SSA value (the output dataflow channel).

## Problem

With the RTL backend now supporting a variety of buffer HDL modules, different modules or their combinations may share the same timing attribute yet exhibit different throughput and area characteristics. This increases the exploration space for the MILP, and the timing attribute alone is no longer sufficiently general.

## Proposed Solution

In addition to the timing attribute to characterize `BufferOp`, we directly represent the buffer type. Each buffer type corresponds to a specific RTL backend HDL module. A `BufferOp` now includes a type field. The previous example is now converted to:

```mlir
%dataOut = handshake.buffer %dataIn {hw.parameters = {BUFFER_TYPE = “DV”, TIMING = {D: 1, V: 1}, NUM_SLOTS = 1 : ui32}} : channel
```

## Supported Buffer Types

- ONE_SLOT_BREAK_DV: This buffer breaks the D and V signal paths. Previously known as a slot of OEHB (Opaque Elastic Half-Buffer), it introduces one cycle of latency on the D and V paths. It does not break the R signal path and adds no latency on R.

- ONE_SLOT_BREAK_R: This buffer breaks the R signal path. Previously known as a slot of TEHB (Transparent Elastic Half-Buffer), it introduces one cycle of latency on the R path. It does not break the D and V signal path and adds no latency on D and V.

- ONE_SLOT_BREAK_DVR: Each slot of this buffer breaks the D, V, and R signal paths and introduces one cycle of latency on all three.

- FIFO_BREAK_DV: This buffer breaks the D and V paths. It has multiple slots but, unlike a chain of ONE_SLOT_BREAK_DV buffers, its structure cannot be split. It introduces one cycle of latency on the D and V paths regardless of the number of slots and has no latency on R. It was previously called an 'elastic_fifo_inner'.

- FIFO_BREAK_NONE: Previously known as a 'tfifo' (Transparent FIFO), this is a FIFO_BREAK_DV with a bypass, adding no latency to any signal paths. Its only purpose is to hold tokens.

- SHIFT_REG_BREAK_DV (Not Implemented): This buffer breaks the D and V paths. It has multiple slots that share a single handshake control unit, so all slots stall together or accept inputs simultaneously. This design introduces the same latency as a chain of ONE_SLOT_BREAK_DV buffers with the same slot number. However, when the initiation interval is greater than one, its token capacity is lower than that of ONE_SLOT_BREAK_DV buffers. Its main advantage is a significantly lower area cost when slot number is high.

> [!NOTE]
> All six buffer types can be used together in a channel to handle various needs. For the first three types, you can chain multiple modules if you need more slots. The last three types allow multiple slots within their module parameters, so they need not be chained in a channel.

## Map MILP Result to Buffer Types

Originally, the MILP result of `FPGA20Buffers` and `FPL22Buffers` produced timing attributes `opaque` and `transparent`, which were then mapped via a JSON file to corresponding buffer HDL modules. In the new approach, the MILP result is mapped to a set of buffer types with specified slot numbers, and each buffer type directly corresponds to an HDL module. In other words, the conversion still produces the same inputs and outputs as before—the only change is that the intermediate representation is now expressed as explicit buffer types rather than timing attributes. Since each buffer type corresponds one-to-one with an HDL module, the Buffer Placement Pass and RTL backend require no additional smart conversion.

The mapping is as follows:

```
1. For Opaque Buffers:
When numslot = 1, map to ONE_SLOT_BREAK_DV.
When numslot = 2, map to ONE_SLOT_BREAK_DV + ONE_SLOT_BREAK_R.
When numslot > 2, map to (numslot - 1) * FIFO_BREAK_DV + ONE_SLOT_BREAK_R.

2. For Transparent Buffers:
When numslot = 1, map to ONE_SLOT_BREAK_R.
When numslot > 1, map to numslot * FIFO_BREAK_NONE.

3. The previous steps result in the same buffer HDL modules as using timing attributes. This step optimizes area usage without affecting functionality:
If the number of ONE_SLOT_BREAK_R exceeds 1, convert its additional slots into equivalent FIFO_BREAK_NONE slots. 
Then, if both ONE_SLOT_BREAK_DV/FIFO_BREAK_DV and FIFO_BREAK_NONE are present, convert all FIFO_BREAK_NONE slots into equivalent FIFO_BREAK_DV slots.
```