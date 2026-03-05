# BLIF Exporter

The **BLIF Exporter** is an binary that serializes a Synth dialect circuit contained inside `hw.module` operations into a BLIF file. It serves as the inverse of the BLIF Importer, allowing Dynamatic's IR to be exported back to a standard BLIF representation for use with external synthesis tools.

---

## Code Structure

The binary `export-blif` can be called as follows:
```bash
./bin/export-blif <input-mlir-file> <output-blif-file>
```

where `input-mlir-file` is the file that contains the Synth circuit to be exported and `output-blif-file` is the file containing the exported BLIF.

The core functionality is the following:

- It iterates over all `hw.HWModuleOp` operations in the module and calls `exportBlifCircuit` on each of them to generate a BLIF module.

The core function `exportBlifCircuit` executes the following steps:

1. It writes the `.model`, `.inputs`, and `.outputs` lines, and collects all port names into the `inputPorts` and `outputPorts` vectors using the `generateBlifHeader` function.
2. It iterates over the ops inside the module body and emit the corresponding BLIF statements using the `generateBlifCircuitFromSynth` function.
3. Writes the `.end` terminator to close the BLIF model.

---

## Support Functions

In this subsection of the doc, we highlight the key support functions.

### Generate BLIF Header

The core function that writes the header section of the BLIF file for a given `hw.HWModuleOp` is `generateBlifHeader`. It:

1. Writes the `.model` line using the `hw.module` name.
2. Iterates over the module's port list and writes all input ports on the `.inputs` line. It populates the vector `inputPorts` which contains the same list.
3. Iterates over the module's port list and writes all output ports on the `.outputs` line. It populates the vector `outputPorts` which contains the same list.

### Generate BLIF Functionality 

The core function that translates the Synth operations inside an `hw.HWModuleOp` into BLIF statements is `generateBlifCircuitFromSynth`. It iterates over all ops in the module body and handles three operation types:

- **`synth.latch`**: the function emits a `.latch` statement with the format `.latch <input> <output> [type control] [init]`. The input and output operand names are resolved via `getValueName`. Optional fields are emitted only when present: if a control signal exists, the latch type (defaulting to `"re"` if unset) and control signal name are written; if an init value is set, it is appended.

- **`synth.aig.and_inv`**: the function emits a `.names` statement listing both input operand names and the output result name, followed by a truth-table row. Each input position in the row is `"0"` if that input is inverted and `"1"` otherwise.

- **`hw.constant`**: the function emits a single-node `.names` statement (`.names <output>`) followed by either `1` or `0` on the next line, corresponding to the constant's value. Only 1-bit constants are supported.

- **`hw.output`**: after all ops are processed, the function checks whether any output port is directly connected to an input port (i.e., a pass-through with no Synth op in between). For each such case, a two-node `.names` wire statement is emitted (`1 1`), connecting the input port name to the output port name. At the end, the function asserts that every output port has been accounted for.

Any other operation type is reported as unsupported via an error message.


### Value Naming in BLIF

Value names in the BLIF output are resolved by the internal `getValueName` lambda, which applies the following priority:

1. If the value is a block argument (input port), return its name from `inputPorts` using the argument index.
2. If the value is used as an operand of `hw.output` (output port), return its name from `outputPorts` using the operand index offset by the number of input ports.
3. Otherwise, print the value using MLIR's `AsmState` (e.g., `%4`), strip the leading `%`, and prepend `n` to produce a valid BLIF node name (e.g., `n4`).

This function ensure uniqueness and consistency of names inside a BLIF module.
