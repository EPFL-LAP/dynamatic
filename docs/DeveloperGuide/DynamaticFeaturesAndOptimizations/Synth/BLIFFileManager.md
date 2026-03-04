# BLIFFileManager

## Overview

`BLIFFileManager` is a utility class in the **Dynamatic** project that resolves `.blif` file paths describing the functionality of Handshake IR operations. Given a base directory and an operation, it constructs the expected file path and validates that the file exists.

---

## File Path Convention

BLIF files are organized by module type and parameters using this structure:

```
<blifDirPath>/<moduleType><extraSuffix>/<param1>/<param2>/.../<moduleType><extraSuffix>.blif
```

**Example:** A `fork` operation with size `2` and data width `32` would resolve to:
```
<blifDirPath>/fork_type/2/32/fork_type.blif
```

---

## Key Functions

### `combineBlifFilePath`
```cpp
std::string combineBlifFilePath(std::string moduleType,
                                std::vector<std::string> paramValues,
                                std::string extraSuffix = "");
```
Builds the full file path from the module type, a list of parameter values, and an optional suffix.

---

### `getBlifFilePathForHandshakeOp`
```cpp
std::string getBlifFilePathForHandshakeOp(Operation *op);
```
Dispatches on the operation type to extract the relevant parameters and construct the BLIF file path. Returns an empty string for unsupported operations.

If the resolved path does not exist on disk, an error is emitted and an assertion fails.

---

## Supported Operations & Parameters

| Operation(s) | Parameters Used |
|---|---|
| `AddI`, `AndI`, `CmpI`, `OrI`, `ShLI`, `ShRSI`, `ShRUI`, `SubI`, `XOrI`, `MulI`, `DivSI`, `DivUI`, `Select`, `SIToFP`, `FPToSI`, `ExtF`, `TruncF` | Data width of first operand |
| `Constant` | Data width of result |
| `Branch`, `Sink` | Data width (or `_dataless` suffix if 0) |
| `Buffer` | Buffer type + data width (or `_dataless` if 0) |
| `ConditionalBranch` | Data width of data operand (or `_dataless` if 0) |
| `ControlMerge` | Number of inputs + index type width (dataless only; data variant unsupported) |
| `ExtSI`, `ExtUI`, `TruncI` | Input width + output width |
| `Fork`, `LazyFork` | Number of outputs + data width (or `_dataless` if 0) |
| `Mux` | Number of inputs + data width + select width |
| `Merge` | Number of inputs + data width |
| `Load`, `Store` | Data width + address width |
| `Source` | No parameters |

---

## Error Handling

- If the resolved BLIF file path does not exist, an error is printed to `stderr` identifying the operation by its unique name and the expected path.
- An assertion then halts execution to prompt investigation of the path construction logic.
- Operations not matched by the type switch silently return an empty string (no file lookup is attempted).
