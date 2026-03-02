# Blif Importer


The BLIF Importer is a binary that reads a BLIF file and generates a corresponding Synth dialect circuit inside an `hw.module`. It serves as a standalone entry point for importing externally synthesized logic directly into Dynamatic's IR.

This code interprets each BLIF model as an AIG-style circuit composed of 1-bit registers and AND-with-inverter gates, and re-expresses it directly in the Synth dialect while preserving the instance’s interface.

---

## Code Structure

The binary `import-blif` can be called as follows:
```bash
./bin/export-blif <output-mlir-file> <intput-blif-file>
```

where `output-mlir-file` is the file that will contain the Synth circuit to be imported and `input-blif-file` is the file containing imported BLIF.

The code core function is `importBlifCircuit(moduleOp, loc, blifFilePath)` which imports the blif specified in `blifFilePath` in the module operation `moduleOp`.

This function executes the following steps:

1. Extract the module name, input and output port names of the module from the BLIF file using the `getBlifModuleHeader` function.
2. Create an hw module operation `hw.HWModuleOp` with input and output ports corresponding to the one of the model described in the BLIF file.
3. Create all synth operations inside the `hw.HWModuleOp` using the function `generateSynthCircuitFromBlif`.
4. Attach all the outputs of the synth circuit to the terminator of the `hw.HWModuleOp`.

---

## Support Functions

In this subsection of the doc, we highlight the key support functions.


### Generate Synth Operations

The core function to generate synth operations is `generateSynthCircuitFromBlif`. It parses the BLIF body line by line and emits the corresponding Synth operations. It maintains two maps throughout:

- `nodeValuesMap` which maps a BLIF node name to its Synth Value. Pre-populated with all input port signals before parsing begins.
- `tmpValuesMap` which maps a BLIF node name to a temporary `hw.constant 0` placeholder, created when a node is referenced before being defined. Placeholders are replaced and erased once the real value is available.

It handles two types of units defined in the BLIF:

1. `.latch` from which it parses input node, output node, and oprtional fields (`latchType`, `controlName`, `initVal`). The optional fields follow the BLIF syntax `[type control] [init]` and are decoded based on the number of tokens. A `synth.latch` of type `i1` is created and registered via `updateOutputSynthSignalMapping` in order to record its output value.
2. `.names` from which it parses the input and output node names and the truth-table function. Depending on the number of node names, the behaviour is different:
  - 1 node name: it is a constant and an `hw.constant` is generated
  - 2 node names: it is a wire or an inverter and it is handled by `createSynthWire`
  - 3+ node names: it is a logic gate and it is handled by `createSynthLogicGate`

After parsing, all the outputs of the synth circuit are appended in the `synthOutputs` variable.


### Retrieve the Value of the Input of a BLIF Node

The function that retrieves the input of a synth node is `getInputMappingSynthSignal`. It resolves a BLIF node name to a Synth Value applying the following steps:

1. Return the value from `nodeValuesMap` if already defined.
2. Return the existing placeholder from `tmpValuesMap` if one exists.
3. Create a new `hw.constant 0` placeholder, store it in `tmpValuesMap`, and return it.

### Save the Value of the Output of a BLIF Node

The function that saves the MLIR Value corresponding to the output of a BLIF node is `updateOutputSynthSignalMapping`. It registers  a newly created Synth value as the definitive signal for a given node name. If a temporary placeholder exists in `tmpValuesMap` for that name, all its uses are replaced with the new value, its defining `hw.constant` op is erased, and it is removed from `tmpValuesMap`. The new value is then recorded in `nodeValuesMap`.

### Create a Wire

The function that creates a wire in the Synth circuit is `createSynthWire`. It handles two-node `.names` blocks. Compares the input and output bits of the truth-table function string (format: "X Y"):

- If equal, the output node is aliased directly to the input value via `updateOutputSynthSignalMapping` and no new op is created.
- If different, a `synth.aig.and_inv` is created with the input inverted and `ANDed` with a constant 1, effectively modelling a NOT gate.

### Create a Logic Gate

The function that creates a logic gate in the Synth circuit is `createSynthLogicGate`. It handles multi-input `.names` blocks. Currently asserts that exactly 2 inputs are present, reflecting that the code operates on already-binarized AIG-form BLIF. The truth-table function string (format: "XY Z") determines the inversion flags for each input. A `synth.aig.and_inv` is created with the appropriate inversion flags. If the output bit is `0`, a second `synth.aig.and_inv` ANDing with constant 1 and inverting the first input is appended to negate the result.


