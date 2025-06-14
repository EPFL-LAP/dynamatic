# Buffering

## Overview

This document describes the current buffer placement infrastructure in Dynamatic.

Dynamatic represents dataflow circuit buffers using the `handshake::BufferOp` operation in the MLIR Handshake dialect. This operation has a single operand and a single result, representing the buffer’s input and output ends.

The document provides:
- A description of the `handshake::BufferOp` operation and its key attributes
- An overview of available buffer types
- Mapping strategies from MILP results to buffer types
- Additional buffering heuristics (also referenced in code comments)
- Clarification of RTL backend behavior

It serves as a unified reference for buffer-related logic in Dynamatic.

## Buffer Operation Representation

The `handshake::BufferOp` operation takes several attributes that characterize the buffer:

1. `BUFFER_TYPE`: Specifies the type of buffer implementation to use
2. `TIMING`: A timing attribute that specifies cycle latencies on various signal paths
3. `NUM_SLOTS`: A strictly positive integer denoting the number of slots the buffer has (i.e., the maximum number of dataflow tokens it can hold concurrently)


In its textual representation, the `handshake::BufferOp` operation appears as follows:

```mlir
%dataOut = handshake.buffer %dataIn {hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
```

Here `%dataIn` is the buffer's operand SSA value (the input dataflow channel) and `%dataOut` is the buffer's result SSA value (the output dataflow channel).

## Timing Information

The `TIMING` attribute specifies how many cycles of latency the buffer introduces on each handshake signal: data (D), valid (V), and ready (R).

- `D: 1` means 1-cycle latency on the data path  
- `R: 0` means no latency on the ready path  

## Buffer Types

Each buffer type corresponds to a specific RTL backend HDL module with different timing, throughput and area characteristics. The `Legacy name` refers to the name previously used in the source code or HDL module before the standardized buffer type naming was introduced.

| Type name              | Legacy name        | Latency                     | Timing                    |
| ---------------------- | ------------------ | --------------------------- | ------------------------- |
| `ONE_SLOT_BREAK_DV`    | OEHB               | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |
| `ONE_SLOT_BREAK_R`     | TEHB               | Data: 0, Valid: 0, Ready: 1 | Break: R; Bypass: D, V    |
| `ONE_SLOT_BREAK_DVR`   | N/A                | Data: 1, Valid: 1, Ready: 1 | Break: D, V, R            |
| `FIFO_BREAK_DV`        | elastic_fifo_inner | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |
| `FIFO_BREAK_NONE`      | TFIFO              | Data: 0, Valid: 0, Ready: 0 | Bypass: D, V, R           |
| `SHIFT_REG_BREAK_DV`   | N/A                | Data: 1, Valid: 1, Ready: 0 | Break: D, V; Bypass: R    |

> Different from `ONE_SLOT_BREAK_DV`, the slots in `SHIFT_REG_BREAK_DV` share a single handshake control and thus accept or stall inputs together.
> All six buffer types can be used together in a channel to handle various needs. For the first three types, multiple modules can be chained to provide more slots. The last three types allow multiple slots within their module parameters, so they need not be chained in a channel.
> An assertion is placed in the BufferOp builder to ensure that if the buffer type is `ONE_SLOT`, then `NUM_SLOTS` == 1.

## Mapping MILP Results to Buffer Types

In MILP-based buffer placement (Mixed Integer Linear Programming), such as those used in the [FPGA20](https://doi.org/10.1145/3477053) and [FPL22](https://doi.org/10.1109/FPL57034.2022.00063) algorithms, the optimization model determines:

- Which signal paths (D, V, R) are broken by the buffer on each channel
- The number of buffer slots (`numslot`) for the buffer on each channel

The MILP does **not** model or select buffer types directly. Instead, buffer types are assigned afterward based on the MILP results, using mapping logic specific to each buffer placement algorithm:

### FPGA20 Buffers

```
1. If breaking DV:
   Map to ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.

2. If breaking none:
   Map to numslot * FIFO_BREAK_NONE.
```

### FPL22 Buffers

```
1. If breaking DV & R:
   When numslot = 1, map to ONE_SLOT_BREAK_DVR;
   When numslot > 1, map to ONE_SLOT_BREAK_DV + (numslot - 2) * FIFO_BREAK_NONE + ONE_SLOT_BREAK_R.

2. If only breaking DV:
   Map to ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.

3. If only breaking R:
   Map to ONE_SLOT_BREAK_R + (numslot - 1) * FIFO_BREAK_NONE.

4. If breaking none:
   Map to numslot * FIFO_BREAK_NONE.
```

## Additional Buffering Heuristics

In addition to the MILP formulation and its buffer type mapping logic, Dynamatic applies a number of additional buffering heuristics, either encoded as extra constraints within the MILP or applied during buffer placement, to ensure correctness and improve circuit performance.

The following rules are currently implemented:

### Buffering before LSQ Memory Ops to Mitigate Latency Asymmetry

In the current dataflow circuit, we observe the following structure:  

![Image](https://github.com/user-attachments/assets/ec087895-1786-4f62-9b71-fab425dd7f79)

`Store` issues memory writes and sends a token to the `LSQ` after argument dispatch.  
`LSQ` uses group-based allocation, triggered by `CMerge`, to dynamically schedule memory accesses.

The problem is that, the `Store` can only forward its token to `LSQ` **one cycle after** the `CMerge`-side token triggers allocation. Since the store path lacks a buffer, this creates an **asymmetric latency** across the two sides. As a result, back pressure from the store side propagates upstream and **causes II += 1** in some benchmarks.

Currently, our buffer placement algorithm does **not** account for the group allocation latency and the dependency of `Store` on that allocation.  

The same latency asymmetry applies to `Load` operations, which also depend on LSQ group allocation.

To mitigate this issue, a minimum slot number is enforced at the input of `Store` and `Load` operations connected to LSQs. This serves as a temporary workaround until a better solution is developed.

### Breaking Ready Paths after Merge-like Operations (FPGA20)

In the FPGA20 buffer placement algorithm, buffers only break the data and valid paths. To prevent combinational cycles on ready paths, ready-breaking buffers are inserted after merge-like operations (e.g., `Mux`, `Merge`) if the output channel is part of a cycle.

### Buffering after Merge Ops to Prevent Token Reordering

For any `MergeOp` with multiple inputs, at least one slot is required on each output if the output channel is part of a cycle. This prevents token reordering and ensures correct circuit behavior.

The following example illustrates the issue:

![Image](https://github.com/user-attachments/assets/0faa9cf0-36cb-46f7-b71a-469ac4e1ff30)

In this figure: 

- the token enters the loop through the left input of the merge
- there is no buffer before the merge and the first eager fork

Suppose the first eager fork is backpressured by one of its outputs, but not backpressured by the output that circulates the token back to the right input of the merge. Then, there is a risk that the fork duplicates the token and passes it to the right input of the merge while there is still an incoming token to the left input of the merge. And merge might reorder these two tokens.

But if we always make sure that there is a buffer in between the merge and the first eagerfork below it, there is no such problem.

![Image](https://github.com/user-attachments/assets/b0929e49-7ae4-42ab-a9a8-7081ccbd66f9)

### Unbufferizable Channels

- Memory reference arguments are not real edges in the graph and are excluded from buffering.
- Ports of memory interface operations are also unbufferizable.

These channels are skipped during buffer placement.

### Buffering on LSQ Control Paths

- Fork outputs leading to other group allocations of the same LSQ must have a buffer that breaks data/valid paths.
- Other fork outputs must have a buffer that does not break data/valid paths.

See [this paper](https://doi.org/10.1145/3174243.3174264) for background.

## RTL Generation

The RTL backend selects buffer implementations based on the `BUFFER_TYPE` attribute in each `handshake::BufferOp`. This determines the HDL module to instantiate. The `NUM_SLOTS` attribute is passed as a generic parameter.

The backend does not use `TIMING` when generating RTL. Latency information is kept in the IR for buffer placement only.

This design simplifies support for new buffer types: adding a new module and registering it in the JSON file is sufficient.
