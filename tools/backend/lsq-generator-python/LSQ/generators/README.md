# `vhdl_gen.generators` — Folder Overview

This folder contains all code‑generation helpers responsible for emitting
parameterized VHDL RTL used by the Load/Store Queue (LSQ) for the spatial computing and
its supporting structures.


## Quick Map

| Module               | Role in the pipeline                                    | 
| ------               | --------------------                                    |
| `dispatchers.py`     | Generates Port-to-Queue/Queue-To-Port Dispatchers | 
| `group_allocator.py` | Generates Group Allocator | 
| `lsq.py`             | Top‑level generator that produces the **complete LSQ RTL** and plugs in dispatchers + allocator |

## Modules
- `dispatchers.py`
    - `PortToQueueDispatcher`: Generates the `entity` and `architecture` for port-to-queue dispatchers (load/store address & store data ports)
    - `QueueToPortDispatcher`: Generates the `entity` and `architecture` for queue-to-port dispatchers (load data & store-back ports).
    - `PortToQueueDispatcherInst`: Produces the VHDL `port map` string to instantiate a Port-to-Queue dispatcher, connecting top-level signal (reset, clock, entries, and ports) to the dispatcher entity.
    - `QueueToPortDispatcherInst`: Produces the VHDL `port map` string to instantiate a Queue-to-Port dispatcher, connecting top-level signal (reset, clock, entries, and ports) to the dispatcher entity.


- `group_allocator.py`
    - `GroupAllocator`: Generates the `entity` and `architecture` for the group allocator, managing handshake between load and store groups based on free entry counts and load-store ordering.
    - `GroupAllocatorInst`: Produces the VHDL `port map` string to instantiate the Group Allocator, connecting top-level signal to the group allocator entity.

- `lsq.py`
    - `LSQ`: Top-Level generator that emits a complete LSQ VHDL design by instantiating the GroupAllocator and various dispatchers, connecting them to memory interfaces, load/store queues, and optional features (pipelining, master interface).