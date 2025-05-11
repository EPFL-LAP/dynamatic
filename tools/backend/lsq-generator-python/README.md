### LSQ generator
This Python-based LSQ generator generates the LSQ design outlined in Hailin Wang's master thesis.

### Configuration parameters

- `ldOrder`: Defines the order matrix for each group, the same as the previous `loadOffsets` parameter with a new format.
- `ldPortIdx`: Specifies the access port index for each load operation within a group.
- `stPortIdx`: Specifies the access port index for each store operation within a group.
- `indexWidth`: Number of bits for the ID in the memory interfaces.
- `numLdChannels`: Indicates the number of load channels at the memory interface (fixed to 1 in this design).
- `numStChannels`: Indicates the number of store channels at the memory interface (fixed to 1 in this design).
- `stResp`: Enables or disables the store response channel in the store access port.
- `groupMulti`: Whether multiple groups are allowed to request an allocation at the same cycle.
- `pipe0En`: Enables or disables the insertion of pipeline register 0 in the LSQ.
- `pipe1En`: Enables or disables the insertion of pipeline register 1 in the LSQ.
- `pipeCompEn`: Enables or disables the insertion of the `pipeComp` pipeline register in the LSQ.
- `headLagEn`: Determines whether the head pointer of the load queue updates one cycle later than the valid bits of its entries.


### Sampele usage

```
usage: lsq-generator.py [-h] [--output-dir OUTPUT_PATH] --config-file CONFIG_FILES
```

### Sample json configuration file (Example: Histogram)

```
{
  "addrWidth":10,
  "bufferDepth":0,
  "dataWidth":32,
  "fifoDepth":16,
  "fifoDepth_L":16,
  "fifoDepth_S":16,
  "groupMulti":0,
  "headLagEn":0,
  "indexWidth":4,
  "ldOrder":[[0]],
  "ldPortIdx":[[0]],
  "loadOffsets":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  "loadPorts":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  "master":true,
  "name":"handshake_lsq_lsq1",
  "numBBs":1,
  "numLdChannels":1,
  "numLoadPorts":1,
  "numLoads":[1],
  "numStChannels":1,
  "numStorePorts":1,
  "numStores":[1],
  "pipe0En":0,
  "pipe1En":0,
  "pipeCompEn":0,
  "stPortIdx":[[0]],
  "stResp":0,
  "storeOffsets":[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  "storePorts":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
  }
```

### Generated Files

- `<lsq_name>.vhd` : A wrapper module that instantiates the core LSQ logic and integrates the required components for memory port interfaces. The new design assumes AXI interfaces.
- `<lsq_name>_core.vhd`: Contains the core LSQ logic, which is derived from Hailin Wang's master thesis. Minor modifications have been made to the code for integration purposes, but the core logic remains unchanged.

---
### Revert to chisel LSQ generator
Configuration parameters needed for both chisel and Python based LSQ-generator coexist in the JSON file. If you want to use the chisel based LSQ generator, please change the corresponding location in `$DYNAMATIC/data/rtl-config-vhdl.json` to:

```
{
    "name": "handshake.lsq",
    "generator": "java -jar -Xmx7G \"$DYNAMATIC/bin/generators/lsq-generator.jar\" --target-dir \"$OUTPUT_DIR\" --spec-file \"$OUTPUT_DIR/$MODULE_NAME.json\" > /dev/null",
    "use-json-config": "$OUTPUT_DIR/$MODULE_NAME.json",
    "hdl": "verilog",
    "io-kind": "flat",
    "io-map": [{ "clk": "clock" }, { "rst": "reset" }, { "*": "io_*" }],
    "io-signals": {
      "data": "_bits"
    }
  },
```



## Directory & File Descriptions

- **lsq-generator.py**  
  Runs the tool.

- **vhdl_gen/**
  - **\_\_init__.py**  
    Re-exports a curated list of public API symbols (e.g. `main`, `generate`, `Logic`, `LSQ`).

  - **cli.py**  
    Parses command-line arguments (with `argparse`), converts them into a `Configs` instance, and calls the core generator.

  - **codegen.py**  
    Implements the `codeGen(config: Configs)` function.

  - **configs.py**  
    Defines the `Configs` class.

  - **context.py**  
    Defines the `VHDLContext` class. It substitutes the previous `global` VHDL context variables.

  - **utils.py**  
    - Defines `VHDLLogicType`, `VHDLLogicVecType`, `VHDLLogicTypeArray`, `VHDLLogicVecTypeArray`, `OpTab`.
    - `IntToBits`, `Zero`, `GetValue`, `MaskLess`, `isPow2`, `log2Ceil` helper functions.
    - Classes and functions need to be relocated into other files later.

  - **signals.py**  
    Defines the four signal classes:  `Logic`, `LogicVec`, `LogicArray`, `LogicVecArray`.

  - **vhdlgen/operators/**  
    Low-level functions that generate VHDL snippets:  
    - `assign.py`: `Op`  
    - `arithmetic.py`: `WrapAdd`, `WrapAddConst`, `WrapSub`
    - `conversions.py`: `VecToArray`, `BitsToOH`, `BitsToOHSub1`, `OHToBits`
    - `masking.py`: `CyclicPriorityMasking`  
    - `mux.py`: `Mux1H`, `Mux1HROM`, `MuxIndex`, `MuxLookUp`
    - `reduction.py`: `ReduceLogicVec`, `ReduceLogicArray`, `ReduceLogicVecArray`, `Reduce`
    - `shifts.py`: `RotateLogicVec`, `RotateLogicArray`, `RotateLogicVecArray`, `CyclicLeftShift`

  - **vhdlgen/generators/**  
    High-level modules that build complete entities/architectures:  
    - `dispatchers.py` : `PortToQueueDispatcher`, `QueueToPortDispatcher`, `PortToQueueDispatcherInit`, `QueueToPortDispatcherInit`
    - `group_allocator.py` : `GroupAllocator`, `GroupAllocatorInit`
    - `lsq.py` : `LSQ`

 
