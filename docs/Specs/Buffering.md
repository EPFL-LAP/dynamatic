# Buffering

This is a design update. Previously, the MLIR represents dataflow circuit buffers with a single-operand, single-result buffer operation (`handshake::BufferOp`). The buffer operation is characterized by a number of slots and a timing attribute that specifies cycle latencies on various signal paths. For example, the textual representation of the `handshake::BufferOp` operation of a 1-slot buffer on a 32-bit dataflow channel which induces a one-cycle latency on the data to data and valid to valid paths would be serialized to the following.

```mlir
%dataOut = handshake.buffer %dataIn {hw.parameters = {TIMING = {D: 1, V: 1}, NUM_SLOTS = 1 : ui32}} : channel
```

Here `%dataIn` is the buffer's operand SSA value (the input dataflow channel) and, conversely,`%dataOut` is the buffer's result SSA value (the output dataflow channel).

## Problem

With the RTL backend now supporting a variety of buffer HDL modules, different modules or their combinations may share the same timing attribute yet exhibit different throughput and area characteristics. This increases the exploration space for the MILP, and the timing attribute alone is no longer sufficiently general.

## Proposed Solution

We abandon the use of the timing attribute to characterize `BufferOp` and instead directly represent the buffer type. Each buffer type corresponds to a specific RTL backend HDL module. A `BufferOp` now includes a type field. The previous example is now converted to:

```mlir
%dataOut = handshake.buffer %dataIn {hw.parameters = {BUFFER_TYPE = “DV”, NUM_SLOTS = 1 : ui32}} : channel
```

## Supported Buffer Types

- DV buffer. A DV buffer cuts DV signal paths. It consists of a chain of OEHBs (Opaque Elastic Half-Buffers). Each slot in a DV buffer is an OEHB, which introduces one cycle of latency on DV paths. Therefore, the latency on DV paths equals its number of slots. A DV buffer does not cut the R signal path and introduces no latency on R.

- R buffer. An R buffer cuts the R signal path. It consists of a chain of TEHBs (Transparent Elastic Half-Buffers). Each slot introduces one cycle of latency on the R path. The latency on R equals its number of slots, while DV paths remain unaffected.

- DVR buffer. A DVR buffer cuts both DV and R signal paths. It consists of a chain of DVR slots. Each slot introduces one cycle of latency on both DV and R paths, so the latency on both equals its number of slots.

- DVE (DV Elastic) buffer. A DVE buffer cuts DV paths. Unlike a DV buffer, which is composed of concatenated OEHB slots, a DVE has an unsplittable structure. It introduces one cycle of latency on DV paths regardless of the number of slots and has no latency on R.

- T (Transparent) Buffer. A T buffer is a DVE equipped with a bypass, introducing no latency on any signal paths. Its sole function is to contain tokens.

## MILP Remapping

The MILP result is remapped by decomposing the original `BufferOp` (previously expressed with the timing attribute) into a combination of `BufferOp`s of different types, each with its own slot number. This combination covers all cases previously expressed by the timing attribute and provides greater diversity in throughput and area characteristics. Since each buffer type corresponds one-to-one with an RTL HDL module, the Buffer Placement Pass and RTL backend require no additional smart conversion.

1. For `FPGA20Buffers`, the remap is as follows:

For Opaque Slots:
When numslot = 1, map to a 1-slot DV buffer.
When numslot = 2, map to a 1-slot DV buffer plus a 1-slot R buffer.
When numslot > 2, map to (numslot - 1) DVE buffers plus a 1-slot R buffer.

For Transparent Slots:
When numslot = 1, map to a 1-slot R buffer.
When numslot > 1, map to a numslot-slot T buffer.

2. For `FPL22Buffers`, the remap is as follows:

Based on the FPGA20 mapping, if the R slot count exceeds 1, convert the additional slots beyond 1 into T buffers.
Then, if both DVE and T buffers are present, convert the T buffers into DVE buffers.