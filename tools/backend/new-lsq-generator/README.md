### LSQ Generator
This Python-based LSQ generator generates the LSQ design outlined in Hailin Wang's master thesis.

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
Both the old and new configuration parameters coexist in the JSON file. If you want to use the chisel based LSQ generator, please change the corresponding location in `$DYNAMATIC/data/rtl-config-vhdl.json` to:

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