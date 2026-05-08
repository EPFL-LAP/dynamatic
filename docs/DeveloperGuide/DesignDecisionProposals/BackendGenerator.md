# BackendGenerator

## Overview

`BackendGenerator` is a C++ class responsible for generating backend representations of Handshake operations. It currently supports two output formats:

- **Verilog**
- **BLIF**

The high-level overview of the code is the following:
1. It queryies operation parameters through the Handshake RTL interface.
2. It selects the output directory depending on the operation.
3. It calls the generator for each backend.
4. It records the produced outputs.

---

## Supported Backends

### Verilog Backend

- Generates RTL using a generator command defined in the JSON config.
- Produces one or more `.v` files.
- No synthesis or optimization is performed.

### BLIF Backend

- Calls on the Verilog backend.
- Synthesizes the generated Verilog using **Yosys**.
- Optimizes the result using **ABC**.
- Produces a final `.blif` file.

---

## Inputs

`BackendGenerator` requires the following inputs:

### Operation

- An `mlir::Operation*` implementing `handshake::RTLAttrInterface`
- The operation must provide its parameters via `getRTLParameters()`

### Backend Selection

- A `Backend` enum specifying the target format:
  - `Backend::Verilog`
  - `Backend::BLIF`

### Backend Parameters

- `rtlConfigPath`: Path to the JSON RTL configuration file
- `dynamaticRoot`: Root path of the Dynamatic repository (used in substitutions)
- `outputBaseDir`: Base directory for generated files


---

## Generation Flow

### Preparation Steps

For all backends:

1. **Parameter Extraction**  
   Parameters are retrieved from the operation via  
   `handshake::RTLAttrInterface::getRTLParameters()`.

2. **Output Directory Construction**  
   The output directory structure is the following:
   `<baseDir>/<operation>/<param1>/<param2>/...` where `baseDir` is the base directory specified as input to the backend generator, `operation` is the name of the operation, `param1`,`param2` and so on are the values of the parameter for the operation.

3. **JSON Config Lookup**  
The file specified by `rtlConfigPath` is parsed to locate a matching entry:
- Match is based on `"name"` == operation name
- Optional `"parameters"` constraints must also match

4. **Generator Command Execution**  
The `"generator"` field is expanded using parameter substitution:
- `$PARAM_NAME` -> actual value
- Special variables that are set by the backend generator code are the following:
  - `MODULE_NAME`
  - `OUTPUT_DIR`
  - `DYNAMATIC`
  - `BITWIDTH`
  - `EXTRA_SIGNALS`

---

### Verilog Backend Pipeline

1. Execute the generator command.
2. Expect output:
  `<outputDir>/<moduleName>.v`
3. Collect:
- Generated Verilog file
- Additional dependency files from JSON (`generic` entries)

---

### BLIF Backend Pipeline

The BLIF backend reuses the Verilog backend, then applies synthesis:

1. **Verilog Generation**  
Internally invokes the Verilog backend.

2. **Yosys Synthesis**  
A script `run_yosys.sh` is generated and executed. It performs:
- Verilog parsing
- Hierarchy resolution
- Lowering and optimization
- BLIF emission

3. **ABC Optimization**  
A script `run_abc.sh` is generated and executed. It applies:
```
strash
6x (rewrite -> balance -> refactor -> balance)
```

For the BLIF backend, the directory additionally contains:
- Intermediate BLIF file from Yosys
- Final optimized BLIF file
- Generated Yosys and ABC scripts

---

## Parameter Handling

Parameters are extracted from the operation using the MLIR function `handshake::RTLAttrInterface::getRTLParameters()`. and stored in a map:
`std::map<std::string, std::string>`

---

## External Dependencies

### Verilog Backend
- No external tools required beyond the generator command.

### BLIF Backend
- **Yosys**
- **ABC**

Both these tools are automatically installed when using the flags `--enable-abc` and `--enable-yosys` when building Dynamatic with the `build.sh` script. If there is a specific choice of the versions of both tools, please include the binaries of both tools in the `PATH` variable.

Paths to these tools must be provided through the backend parameters.
