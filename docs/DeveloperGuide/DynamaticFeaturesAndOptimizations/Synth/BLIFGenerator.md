# BLIFGenerator

## Overview

`BLIFGenerator` is a C++ class that synthesises a BLIF file on demand for a given Handshake component by invoking **Yosys** (synthesis) followed by **ABC** (logic optimisation). It replicates the logic of `tools/blif-generator/blif_generator.py` and is called automatically by `BLIFFileManager` when a required BLIF file is not yet present on disk.

The generator requires `DYNAMATIC_ENABLE_ABC` and `DYNAMATIC_ENABLE_YOSYS` to be enabled at build time (see [CMake configuration](#cmake-configuration)).

---

## Synthesis Pipeline

For each component, generation proceeds in the following steps:

1. **JSON config lookup**: The component name is looked up in `data/rtl-config-verilog.json` to obtain the Verilog source file(s) and the names of the parameterisable generics.
2. **Verilog generation**: If the JSON entry has a `generator` command instead of a static `generic` file, the generator is executed first to produce the Verilog.
3. **Dependency collection**: All Verilog files listed under `dependencies` are collected recursively.
4. **Yosys synthesis**: A `run_yosys.sh` script is written to the output directory and executed. It uses `chparam` to set the generic values, then synthesises to BLIF.
5. **ABC optimisation**: A `run_abc.sh` script applies a multi-iteration rewrite/refactor sequence (`strash` + 6x (`rewrite`; `balance`; `refactor`; `balance`)) and writes the optimised BLIF.
6. **Blackbox post-processing**: For components whose internal logic should remain abstract, `.names` / `.latch` lines are stripped from the output BLIF.

---

## Output Directory Layout

Generated files are placed under `<blifDirPath>/<component>/<param1>/<param2>/`:

```
<blifDirPath>/
  <component>/
    <param1>/
      <param2>/
        <component>_<param1>_<param2>_yosys.blif   <- raw Yosys output
        <component>.blif                            <- final ABC-optimised output
        run_yosys.sh                                <- synthesis script
        run_abc.sh                                  <- optimisation script
```

The path follows the same convention used by `BLIFFileManager::combineBlifFilePath`.

---

## JSON Config Lookup

The generator reads `data/rtl-config-verilog.json` (located two levels above `blifDirPath`) to find the entry for the requested component. The lookup priority for matching an entry is:

1. `module-name` field (exact match)
2. `name` field with the `"handshake."` prefix stripped
3. Basename of the `generic` file path (for support modules without a `name`)

---

## Parameter Mapping

The JSON `parameters` array may contain a mix of fixed and range (generic) parameters. Only **range parameters** contribute to the Yosys `chparam` command and to the BLIF path. A parameter is a range parameter when:
- Its `type` is `"unsigned"` or `"dataflow"`, **and**
- It has no `eq` or `data-eq` fixed-value constraint, **and**
- Its `generic` field is not explicitly `false`.

The `paramValues` passed to `generate()` are **right-aligned** against the ordered list of range parameter names. This means a single value always maps to the innermost (last) range parameter, which is correct for components like `tfifo` where the BLIF path encodes only `DATA_TYPE` but the JSON also declares `NUM_SLOTS` as a range parameter.

---

## Key Functions

### Constructor
```cpp
BLIFGenerator(const std::string &blifDirPath,
              const std::string &yosysExecutable,
              const std::string &abcExecutable);
```
- `blifDirPath`: absolute path to the BLIF file tree (e.g. `<repo>/data/blif`). The Dynamatic root and the JSON config path are derived from this.
- `yosysExecutable` / `abcExecutable`: paths to the Yosys and ABC binaries.

---

### `generate`
```cpp
bool generate(const std::string &component,
              const std::vector<std::string> &paramValues);
```
Runs the full synthesis pipeline for the given component and parameter values. Returns `true` if `<component>.blif` exists in the output directory after generation.

`component` must match the JSON `module-name` (or the stripped `name`) of the target entry, e.g. `"addi"`, `"fork_dataless"`, `"oehb"`.

---

## CMake Configuration

`BLIFGenerator` is only active when both `DYNAMATIC_ENABLE_ABC=ON` and `DYNAMATIC_ENABLE_YOSYS=ON` are passed to CMake (e.g. via `./build.sh --enable-abc --enable-yosys`). These options:

1. Build ABC and Yosys as CMake `FetchContent` dependencies.
2. Set `DYNAMATIC_ABC_EXECUTABLE` and `DYNAMATIC_YOSYS_EXECUTABLE` CMake variables to the built binary paths.

In `BLIFFileManager`, the call to `BLIFGenerator` is guarded by:
```cpp
#if defined(DYNAMATIC_YOSYS_EXECUTABLE) && defined(DYNAMATIC_ABC_EXECUTABLE)
```
If neither macro is defined, a missing BLIF file causes an assertion failure with a message directing the user to enable those options.
