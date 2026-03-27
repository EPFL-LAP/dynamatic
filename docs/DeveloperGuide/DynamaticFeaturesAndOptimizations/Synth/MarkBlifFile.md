# HandshakeMarkBLIFImpl Pass

## Overview

`HandshakeMarkBLIFImpl` is a pass that annotates every Handshake operation in a module with the path to its corresponding `.blif` implementation file. These annotations are later consumed by the **Handshake** to **Synth** conversion pass.

The pass uses [`BLIFFileManager`](./BLIFFileManager.md) to resolve file paths and attaches them to operations via the `BLIFImplInterface`.

---

## Pass Behaviour

1. Validates that the `blifDirPath` option is non-empty; fails the pass otherwise.
2. Instantiates a `BLIFFileManager` with the provided directory path.
3. Walks all operations in the module, skipping `handshake::FuncOp`.
4. For each operation, resolves the BLIF file path using `BLIFFileManager::getBlifFilePathForHandshakeOp`.
5. Annotates the operation by calling `BLIFImplInterface::setBLIFImpl` with the resolved path.

---

## Options

| Option | Type | Description |
|---|---|---|
| `blifDirPath` | `string` | Base directory containing all `.blif` files. Must be non-empty. |

---

## BLIFImplInterface


An MLIR `OpInterface` that allows any Handshake operation to carry a reference to its BLIF implementation. 

The interface stores the BLIF path as a `StringAttr` under the attribute name `"blif_impl"`.

### Methods

#### `setBLIFImpl(StringRef value)`
Sets the `"blif_impl"` attribute on the operation to the given string value.

#### `getBLIFImpl() → StringAttr`
Returns the `"blif_impl"` attribute. If the attribute is not set, returns an empty `StringAttr`.

---

## Example Attribute

After the pass runs, an annotated operation will carry an attribute such as:

```
blif_impl = "/path/to/blif/fork_type/2/32/fork_type.blif"
```
