# Handshake to Synth Conversion Pass

The **Handshake to Synth** pass (`--handshake-to-synth`) converts a circuit expressed in the Handshake dialect into an equivalent circuit expressed in the HW and Synth dialects. The output is a hierarchical, gate-level netlist.


The pass requires that every Handshake operation has been annotated beforehand with the path of the BLIF file containing its gate-level implementation. This is done by a separate pass (`--mark-handshake-blif-impl`). Please refer to [this doc](MarkBlifFile.md) for more information.

---

## Invocation

The pass is registered as `--handshake-to-synth` and operates on a `builtin.module` containing exactly one non-external `handshake.func`. It can be applied via `dynamatic-opt`:

```bash
dynamatic-opt --mark-handshake-blif-impl="blif-dir=<path>" \
              --handshake-to-synth \
              input.mlir -o output.mlir
```

---

## Overview

The pass executes in two sequential steps, each building directly on the output of the previous one.

**Step 1: Unbundle** converts every Handshake operation into an `hw.module` definition and an `hw.instance` call site inside the top-level `hw.module`. Handshake's bundled channel types (`!handshake.channel<T>`, `!handshake.control<>`) are split into flat `i1` data, valid, and ready signals. Ready signals have their direction flipped relative to data and valid. Multi-bit data channels are split into individual `i1` ports. Clock and reset ports are appended to every `hw.module`. Temporary backedge placeholders (`UnrealizedConversionCast` ops) bridge forward references during conversion and are eliminated by the end of this step.

**Step 2: Populate** replaces the `synth.subckt` placeholder body that Step 1 inserted into each `hw.module` with the real gate-level netlist imported from the corresponding BLIF file.

---

## Pre-check

Before Step 1 begins, `runDynamaticPass()` walks every operation inside the `handshake.func` and asserts that it implements `BLIFImplInterface` and carries a non-empty BLIF file path. Any operation that fails this check signals a pass failure with an error message directing the user to run `--mark-handshake-blif-impl` first.

---

## Step 1: Unbundle Handshake types

#### Conversion flow

`unbundleHandshakeChannels()` runs the following sequence:

1. Locates the single non-external `handshake.func` in the module.
2. Calls `unbundlePorts()` and `buildPortInfoFromHandshakeUnitPorts()` to compute the top-level `hw.module`'s port list, then creates it with `hw::HWModuleOp`.
3. Initializes the `BackedgeBuilder` and records `clk` and `rst` from the new module's block arguments (the last two arguments).
4. Walks the top-level module's block arguments in order, calling `saveUnbundledValues()` to save in  `unbundledValuesMap` the map between the hw module block argument `Value`s with the original handshake function ones.
5. Iterates over every operation in the function body (skipping `handshake.end`) and calls `convertHandshakeOp()` for each.
6. Calls `convertHandshakeFunc()` to wire the terminator.

`convertHandshakeOp()` looks up or creates the `hw.module` definition for the op via `createHWModuleHandshakeOp()`, then collects input operands by calling `getUnbundledValues()` for each input port (inserting `pendingValuesMap` placeholders for values that have not been created yet), appends `clk` and `rst`, and creates an `hw.instance`. Output results are saved to `unbundledValuesMap` via `saveUnbundledValues()`, grouping multi-bit data ports by handshake signal before saving.

`convertHandshakeFunc()` collects the flat output values for the top module's terminator by calling `getUnbundledValues()` for each `handshake.end` operand (data and valid) and each function argument's ready signal, sets them on the `hw.output` terminator, verifies that `pendingValuesMap` is empty, then erases the `handshake.end` and `handshake.func`.

### Key data structures

`PortKind` is a `std::variant` of three tag structs: `DataPortInfo` (carrying `bitIndex` and `totalBits`), `ValidPortInfo`, and `ReadyPortInfo`. It is used throughout Step 1 to distinguish which component of a handshake channel a flat port corresponds to.

`HandshakeUnitPort` represents a single flat port produced by unbundling. It carries the port name, direction, the original handshake `Value` it corresponds to, and a `PortKind`.

`UnbundledHandshakeChannel` holds the resolved flat `Value`s for a single handshake signal, with named fields `dataBits`, `valid`, and `ready`. It provides `setValues(PortKind, SmallVector<Value>)` and `getValues(PortKind)` methods that dispatch on the variant to access the correct field, and an `empty()` predicate used to detect fully resolved entries.


### Type unbundling rules

| Handshake type | Flat components |
|---|---|
| `!handshake.channel<T>` | `T` (data), `i1` (valid), `i1` (ready) |
| `!handshake.control<>` | `i1` (valid), `i1` (ready) |
| `memref<NxT>` | `T` (data), `i1` (valid), `i1` (ready) |
| any other type | the type itself (data pass-through) |

### Port naming

Port names are derived from `NamedIOInterface`. For `handshake.func` the names come directly from the function's argument and result name attributes. For all other ops, names are obtained from `NamedIOInterface::getOperandName` / `getResultName`, then legalized by `legalizeBlifPortNames()`, which converts the `root_N` index pattern to `root[N]` array notation (back-patching `root` to `root[0]` when index 1 is first encountered). Signal-kind suffixes are then appended: `_valid` and `_ready` are inserted before the first `[` for array names, or appended at the end otherwise. Data port names for multi-bit channels use the same formatting rule with a numeric index.

### Signal tracking and placeholder resolution

`HandshakeUnbundler` maintains two maps keyed by handshake `Value`:

`unbundledValuesMap` holds the definitive flat `Value`s for each handshake signal, stored as `UnbundledHandshakeChannel` entries.

`pendingValuesMap` holds temporary backedge placeholders (`UnrealizedConversionCast` ops created via `BackedgeBuilder`) for signals whose producing op has not yet been converted. When `saveUnbundledValues()` is later called with the real values, it calls `replaceAllUsesWith()` on each placeholder to redirect uses, then clears the entry. The `BackedgeBuilder` (owned as `std::unique_ptr<BackedgeBuilder>`) erases the now-unused placeholder ops in its destructor.

---

## Step 2: Populate hw modules with BLIF netlists

### Code structure

Step 2 is implemented by `populateHWModule()` and the public entry point `populateAllHWModules()`.

`populateHWModule()` resolves the `hw.module` definition referenced by a given `hw.instance`, looks up its BLIF path in `opToBlifPathMap`, imports the netlist with `importBlifCircuit()`, then swaps the original module out of the symbol table and inserts the imported one under the same name. A `DenseSet<StringRef>` of already-populated module names prevents the same module from being imported more than once when multiple instances reference it.

`populateAllHWModules()` locates the top-level `hw.module` by name, collects all `hw.instance` ops it contains via a walk, and calls `populateHWModule()` for each.

### BLIF path lookup

The BLIF path for each module is stored in the global `opToBlifPathMap` (`DenseMap<Operation*, string>`), which is written during Step 1 in `createHWModuleHandshakeOp()`. An empty path means the module's body should not be replaced.

---

## Global shared state

`opToBlifPathMap` (`DenseMap<Operation*, string>`) maps each `hw.module` operation to the path of the BLIF file that should replace its placeholder body. It is written in Step 1 and consumed in Step 2.

`CLK_PORT` and `RST_PORT` are string constants used consistently by Step 1 when building port info and when identifying clock and reset ports during instance operand collection.