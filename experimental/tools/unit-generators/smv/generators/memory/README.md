# Memory units

In Dynamatic memory is accessed through two different types of units:
- Memory Controllers (MCs)
- Load-Store Queues (LSQs)

These units control when memory accesses start, end, and is which order they should be executed. More information about memory can be found in [here](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Specs/CircuitInterface.md).


## Memory controllers

Memory controllers are the interface units between the external memory and the Load/Store units. Each Load and Store unit is connected to a MC through a dedicated port. The MC is responsible for arbitrating which unit is served first, prioritizing:
- Loads with a valid address channel and a ready data channel
- Stores with valid address and data channels.

The memory controller has also the responsibility of handling the `mem_start` and `mem_end` signals. The `mem_start` is an input signal used to indicate when the memory is ready to be accessed. The `mem_end` is an output signal that indicates that the MC won't issue any more stores.

If the memory controller doesn't have any store ports it is called storeless; if it doesn't have any load proty it is called loadless. By combining a storeless and a loadless memeory controller, we obtain a general memeory controller.
In the following diagrams we show the current VHDL/Verilog implementation of the memory controllers. The SMV implementration is based on these diagramas and it has been proved to be equivalent using Yosys.

Memory controller storeless:
![Image](https://github.com/user-attachments/assets/09766580-2037-4f32-a33b-f64a58f217d8)

Memory controller loadless:
![Image](https://github.com/user-attachments/assets/68eff081-3adc-4fdc-89e7-789c4ebbc2f6)