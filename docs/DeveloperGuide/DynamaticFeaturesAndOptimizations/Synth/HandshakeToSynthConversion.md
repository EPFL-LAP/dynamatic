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

The pass executes in three sequential steps, each building directly on the output of the previous one.

**Step 1: Unbundle** converts every Handshake operation into an `hw.module` definition and an `hw.instance` call site. Handshake's bundled channel types (`!handshake.channel<T>`, `!handshake.control<>`) are split into flat `i1`/`iN` data, valid, and ready signals. Temporary `UnrealizedConversionCast` ops bridge the bundled to flat boundary during conversion and are eliminated at the end of this step.

**Step 2: Rewrite signals** corrects the direction of ready signals. In the Handshake protocol, ready travels opposite to data and valid; after Step 1 all signals share the same direction. Step 2 flips every ready port (input becomes output and vice versa), splits any remaining multi-bit ports into individual `i1` ports, and adds `clk` and `rst` ports to every `hw.module`.

**Step 3: Populate** replaces the `synth.subckt` placeholder body that Step 1 inserted into each `hw.module` with the real gate-level netlist imported from the corresponding BLIF file.

---

## Pre-check

Before Step 1 begins, `runDynamaticPass()` walks every operation inside the `handshake.func` and asserts that it implements `BLIFImplInterface` and carries a non-empty BLIF file path. Any operation that fails this check signals a pass failure with an error message directing the user to run `--mark-handshake-blif-impl` first.

---

## Step 1: Unbundle Handshake types

### Code structure

Step 1 is organised into four layers that depend strictly top-to-bottom. Each layer calls only layers above it.

**Layer 1: TypeUnbundler** contains functions that split a Handshake `Type` into a list of `(SignalKind, Type)` component pairs. `SignalKind` is one of `DATA_SIGNAL`, `VALID_SIGNAL`, or `READY_SIGNAL`. No IR is constructed here.

**Layer 2: PortInfoBuilder** derives flat port names from a Handshake operation's `NamedIOInterface` and zips them with the types from Layer 1 to produce an `hw::ModulePortInfo`. String-formatting utilities (`insertSuffixBeforeBracket`, `formatIndexedPortName`, `elaboratePortName`) are private to this layer. The public entry point is `buildPortInfo(op)`, which returns a complete `hw::ModulePortInfo` in one call.

**Layer 3: ConversionPatterns** contains the `TypeConverter`, cast helpers, a synth placeholder installer, the two conversion functions (`convertOpToHWModule`, `convertFuncOpToHWModule`), and their `OpConversionPattern` wrappers. It uses Layers 1 and 2.

**Layer 4: Orchestrator** is the single entry point `unbundleAllHandshakeTypes()`, which runs the three conversion phases sequentially and calls nothing else directly.

### Type unbundling rules

| Handshake type | Flat components |
|---|---|
| `!handshake.channel<T>` | `T` (data), `i1` (valid), `i1` (ready) |
| `!handshake.control<>` | `i1` (valid), `i1` (ready) |
| `memref<NxT>` | `T` (data), `i1` (valid), `i1` (ready) |
| any other type | the type itself (data pass-through) |

### Port naming

Port names are derived from `NamedIOInterface`. The interface emits names in the form `"name_N"` (e.g. `"in_0"`, `"in_1"`). `elaboratePortName()` converts these to the BLIF array notation `"name[N]"`. A special back-patching rule applies: when `"name_1"` is encountered, the previously registered `"name"` entry is retroactively renamed to `"name[0]"`. Signal-kind suffixes are then appended before the first `[` to produce names like `"in_valid[0]"` and `"in_ready[0]"`.

### Conversion phases

The orchestrator runs three phases in strict order:

**Phase 1a** converts all inner Handshake operations (every op except `handshake.func` and `handshake.end`) using `ConvertAnyHandshakeOp`, a single `OpInterfaceConversionPattern<BLIFImplInterface>` that fires for every op implementing `BLIFImplInterface`. `FuncOp` and `EndOp` are kept legal during this phase so that they can host the not-yet-converted inner ops. Each converted op becomes an `hw.module` definition with a `synth.subckt` placeholder body, plus an `hw.instance` that replaces the original op. The BLIF file path is transferred from the op's `BLIFImplInterface` attribute to `opToBlifPathMap`.

**Phase 1b** converts `handshake.func` using `ConvertFuncToHWMod`. The function body is inlined into a new `hw.module`, with `UnrealizedConversionCast` ops bridging the flat module arguments back to the bundled types the body expects. The `handshake.end` terminator is replaced by `hw.output` with flat operands.

**Phase 1c** eliminates all `UnrealizedConversionCast` pairs. The expected pattern is `value -> cast1 (bundled->flat) -> cast2 (flat->bundled) -> user`. Both casts are erased and each use of `cast2`'s result is replaced with the corresponding operand of `cast1`. The phase asserts that no casts remain afterwards.

---

## Step 2: Rewrite signal directions

### Code structure

Step 2 is implemented by three sub-components declared in `HandshakeToSynth.h`, plus the public entry point `SignalRewriter::rewriteAllSignals()`.

**Sub-component A: PortLayout** is a pure data layer with no IR construction. `computePortLayout(oldMod)` analyses the old module's port list and returns a `PortLayout` containing a `SmallVector<RewrittenPort>` and the index slots reserved for `clk` and `rst`. `buildModulePortInfo(layout, ctx)` converts a `PortLayout` into an `hw::ModulePortInfo` ready for module creation.

**Sub-component B: SignalTracker** owns the two maps that track how old-module `Value`s correspond to new-module `Value`s as modules are rewritten one at a time. It is used in three phases: `recordInputs()` seeds the tracker from a freshly created module's block arguments; `resolve()` returns the new `Value`(s) for an old signal, inserting `hw.constant(0)` placeholders when the producing instance has not been rewritten yet; `commit()` records definitive result `Value`s from a newly created `hw.instance` and patches any outstanding placeholders.

**Sub-component C: ModuleRewriter** orchestrates A and B through three focused methods. `rewriteModule()` calls `computePortLayout`, creates the new `hw.module`, seeds the tracker, then dispatches either the leaf path (re-install a `synth.subckt` placeholder) or the instance path (call `rewriteInstance()` for every old `hw.instance`), and finally calls `wireTerminator()`. `rewriteInstance()` classifies each port of the old module as one of four cases (non-ready input, ready-output-becomes-input, ready-input-becomes-output, non-ready output), builds the new operand list via `resolve()`, creates the new `hw.instance`, then calls `commit()` for every output. `wireTerminator()` collects resolved output `Value`s from the tracker in port-index order and sets them on the new module's `hw.output`.

### Direction rewrite rules

| Old port | New port | Rationale |
|---|---|---|
| ready input | output | ready travels toward the producer |
| ready output | input | ready arrives from the consumer |
| non-ready input | input | data/valid direction unchanged |
| non-ready output | output | data/valid direction unchanged |

All ports are additionally split to `i1` regardless of their original width. `clk` and `rst` are appended as the last two inputs of every rewritten module.

### Placeholder forwarding

Because modules are rewritten in dependency order (a module is rewritten before any module that instantiates it), most `resolve()` calls find an already-committed mapping. The placeholder mechanism handles the remaining cases: when an instance references a signal whose producing instance has not yet been rewritten, a `hw.constant(0)` placeholder is inserted and recorded. When `commit()` is later called for the producing instance it calls `replaceAllUsesWith` on each placeholder and erases the constant op.

---

## Step 3: Populate hw modules with BLIF netlists

### Code structure

Step 3 is implemented by `BlifPopulator` (declared in `HandshakeToSynth.h`) and the free function `populateAllHWModules()`.

`BlifPopulator` owns two pieces of state: a `SymbolTable` built once at construction (so that repeated symbol lookups during the walk are efficient), and a `DenseSet<StringAttr>` of already-populated module names that prevents the same module from being imported more than once when multiple instances reference it.

`populate(inst)` resolves the `hw.module` definition referenced by `inst`, looks up its BLIF path in `opToBlifPathMap`, imports the netlist with `importBlifCircuit()`, then swaps the original module out of the symbol table and inserts the imported one under the same name. The original name is preserved so all existing `hw.instance` references remain valid without any update.

`populateAllHWModules(modOp, topModuleName)` locates the top-level `hw.module`, collects all `hw.instance` ops it contains, constructs a single `BlifPopulator`, and calls `populate()` for each instance.

### BLIF path lookup

The BLIF path for each module is stored in the global `opToBlifPathMap` (`DenseMap<Operation*, string>`) which is written during Step 1 (`convertOpToHWModule`) and transferred to each rewritten module during Step 2 (`rewriteModule`). An empty path means the module's body is already expressed in `hw`/`synth` ops and should not be replaced.

---

## Global shared state

Two pieces of pass-wide state are declared in `HandshakeToSynth.h` and shared across all three steps:

`opToBlifPathMap` (`DenseMap<Operation*, string>`) maps each `hw.module` operation to the path of the BLIF file that should replace its placeholder body. It is written in Step 1 and transferred (with key update) in Step 2 when modules are renamed.

`clockSignal` and `resetSignal` are string constants (`"clk"` and `"rst"`) used consistently by both Step 1 (when building port info) and Step 2 (when adding ports to rewritten modules).

