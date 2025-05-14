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

| Type name              | Legacy name        |  Latency                    | Timing                    |
| ---------------------- | ------------------ | --------------------------- | ------------------------- |
| `ONE_SLOT_BREAK_DV`    | OEHB               | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |
| `ONE_SLOT_BREAK_R`     | TEHB               | Data: 0, Valid: 0, Ready: 1 | Break: R; Bypass: D, V    |
| `ONE_SLOT_BREAK_DVR`   | N/A                | Data: 1, Valid: 1, Ready: 1 | Break: D, V, R            |
| `FIFO_BREAK_DV`        | elastic_fifo_inner | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |
| `FIFO_BREAK_NONE`      | TFIFO              | Data: 0, Valid: 0, Ready: 0 | Bypass: D, V, R           |
| `SHIFT_REG_BREAK_DV`   | N/A                | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |

> [!NOTE]
> `SHIFT_REG_BREAK_DV` is currently not implemented.
> All six buffer types can be used together in a channel to handle various needs. For the first three types, you can chain multiple modules if you need more slots. The last three types allow multiple slots within their module parameters, so they need not be chained in a channel.
> An assertion is placed in the BufferOp builder to ensure that if the buffer type is ONE_SLOT, then num_slots == 1.
## Map MILP Result to Buffer Types

This section describes how the MILP result is mapped to buffer placement decisions. This mapping logic is specific to each buffer placement algorithm (e.g., FPGA20 and FPL22 have their separate mapping logic).

For `FPGA20Buffers`,

```
1. If breaking DVR:
When numslot = 1, map to ONE_SLOT_BREAK_DV + ONE_SLOT_BREAK_R;
When numslot = 2, map to ONE_SLOT_BREAK_DV + ONE_SLOT_BREAK_R;
When numslot > 2, map to ONE_SLOT_BREAK_DV + (numslot - 2) * FIFO_BREAK_NONE + ONE_SLOT_BREAK_R.

2. If breaking none:
Map to numslot * FIFO_BREAK_NONE.
```

For `FPL22Buffers`,

```
1. If breaking DV & R:
When numslot = 1, map to ONE_SLOT_BREAK_DV + ONE_SLOT_BREAK_R;
When numslot = 2, map to ONE_SLOT_BREAK_DV + ONE_SLOT_BREAK_R;
When numslot > 2, map to ONE_SLOT_BREAK_DV + (numslot - 2) * 
                            FIFO_BREAK_NONE + ONE_SLOT_BREAK_R.

2. If only breaking DV:
When numslot = 1, map to ONE_SLOT_BREAK_DV;
When numslot > 1, map to ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.

3. If only breaking R:
Map to ONE_SLOT_BREAK_R + numslot * FIFO_BREAK_NONE.

4. If breaking none:
Map to numslot * FIFO_BREAK_NONE.
```