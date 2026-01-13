# Handshake to Synth


The role of this pass is to transform a Handshake-level representation of a dataflow circuit into a Synth-level representation expressed through HW modules and synth operations. Conceptually, it sits between the Handshake optimization pipeline and a future synthesis backend that will further refine synth operations into a final RTL- or gate-oriented netlist.

There are three main sections in this document.

1. Main pass and usage: Overall structure and rationale of the pass.
2. Unbundling conversion: Lowering Handshake channel types to flat HW ports and `synth.subckt`.
3. Ready inversion: Fixing the direction of ready signals to follow the standard handshake protocol and unbundling multi-bit data signals into multiple single bit signals.


The pass is called [`HandshakeToSynthPass`](HandshakeToSynth.cpp).

---

## Main pass and usage

At a high level, the *HandshakeToSynth* pass performs the following transformations on a module containing a **single non-external** `handshake.func`:

- Converts all Handshake-typed values (channels, control, memory) into flat HW-level ports by unbundling them into `{data, valid, ready}` signals.
- Lowers each Handshake operation (including the function itself) to an `hw.module`/`hw.instance` plus an internal `synth.subckt` representing its behavior (except from the top handshake function).
- Rewrites all generated HW modules to enforce the standard handshake convention where ready signals flow in the opposite direction from data and valid, and propagates this convention recursively through module instances.

The pass operates on:

- A single non-external `handshake.func` per module (enforced by the pass).
- A graph of Handshake operations inside the function, with bundled channel types.

and produces:

- A pure HW/Synth module hierarchy:
  - The original `handshake.func` is replaced by a top-level `hw.module`.
  - Each Handshake operation becomes an `hw.instance` of an `hw.module` that contains a `synth.subckt`.
- No remaining values of Handshake types and no remaining Handshake operations or functions.


The overall pass is implemented as:

- `class HandshakeToSynthPass : public dynamatic::impl::HandshakeToSynthBase<HandshakeToSynthPass>`.

Its `runDynamaticPass()` method:

- Retrieves the `mlir::ModuleOp` and `MLIRContext`.
- Ensures that there is at most one non-external `handshake.func` in the module and that if none is found, the pass is a no-op.
- Runs Phase 1 – Unbundling by calling `unbundleAllHandshakeTypes(modOp, ctx)`.
- Runs Phase 2 – Ready inversion by instantiating a `ReadySignalInverter` and calling `invertAllReadySignals(modOp)`.
- Leaves a future Phase 3 – further refinement of `synth.subckt` into other synth operations (registers, combinational logic, etc.) as a TODO.

In a typical flow, this pass is run after all Handshake-level optimizations and buffer insertion, and before a dedicated synth backend that will interpret or further lower the generated synth operations.

---

## Unbundling conversion

The first major phase of the pass is *unbundling*, which removes all Handshake-specific bundled types from the IR by converting them to multiple scalar ports using MLIR’s conversion infrastructure. Intuitively, this phase works by temporarily inserting *casts* that bridge between the old, bundled Handshake world and the new, flat HW/Synth world, and then erasing those casts once everything has been rewritten.

From a top-down perspective, unbundling proceeds in three steps:

1. Convert all non-function Handshake ops to `hw.module`/`hw.instance` pairs.
2. Convert the `handshake.func` itself to an `hw.module`.
3. Remove all temporary `unrealized_conversion_cast` ops that were used as glue.

These steps are orchestrated by:

- `LogicalResult unbundleAllHandshakeTypes(mlir::ModuleOp modOp, MLIRContext *ctx)`.

This function implements Phase 1 of the pass and is split into the three substeps described below.

---

### Casts: high-level intuition

Unbundling fundamentally changes the *shape* of values: a single Handshake channel value becomes three scalar signals (data, valid, ready) at the HW level. Directly replacing all uses in one shot would be confusing, so the pass introduces `unrealized_conversion_cast` ops as temporary adapters between the old and new representations.

Conceptually, the pass maintains the following invariant during conversion:

- Anywhere a still-bundled value must talk to a newly-unbundled region (or vice versa), an `unrealized_conversion_cast` is inserted to “pack” or “unpack” channels.

Later, once every Handshake op and the function itself have been converted and no part of the IR *needs* bundled values anymore, these casts are removed by wiring original producers and consumers directly. This lets the pass use MLIR’s pattern-based conversion cleanly without having to perform a fragile global rewrite.

Two helper utilities implement this logic:

- `UnrealizedConversionCastOp createCastBundledToUnbundled(Value input, Location loc, PatternRewriter &rewriter)`:
  - Given one bundled value, produces a multi-result cast whose results are all unbundled components of that value (e.g., data/valid/ready).
- `SmallVector<UnrealizedConversionCastOp> createCastResultsUnbundledOp(TypeRange originalResultTypes, SmallVector<Value> unbundledValues, Location loc, PatternRewriter &rewriter)`:
  - Given a flat list of unbundled values (e.g., the results of an `hw.instance`), groups them back into casts that each reconstruct one original bundled result type.
  - Asserts that the number of unbundled components for each original result matches `unbundleType(resultType).size()`.

These casts are introduced systematically when:

- Wiring HW instances into a still-bundled Handshake region.
- Inlining a Handshake function body into an HW module and reconnecting arguments/results.

---

### Step 1 – Convert Handshake ops to HW modules

This step converts all Handshake operations except the function and `handshake.end` terminator into `hw.module` and `hw.instance` pairs. It uses MLIR’s dialect conversion to rewrite ops while the cast logic keeps the IR temporarily well-typed across representation boundaries.

The pass sets up a `RewritePatternSet` with:

- `ChannelUnbundlingTypeConverter` as the type converter, which expands bundled types into flat signal lists.
- `ConvertToHWMod<T>` patterns instantiated for all supported Handshake operations.

A `ConversionTarget` marks:

- `synth::SynthDialect` and `hw::HWDialect` as legal.
- `UnrealizedConversionCastOp` as legal (the glue type).
- The Handshake dialect as illegal, except that `handshake.func` and `handshake.end` remain temporarily legal for Step 2.

The driver then calls:

- `applyPartialConversion(modOp, target, std::move(patterns))`.

If this conversion fails, the unbundling phase aborts.

Each `ConvertToHWMod<T>` pattern replaces the original Handshake op by:
  - Creating (or reusing) an `hw.module` with flat ports that reflect the op’s bundled interface.
  - Instantiating a `synth.subckt` inside that module.
  - Inserting an `hw.instance` at the call site where:
    - Operands from the Handshake region are *unpacked* via bundled &rarr; unbundled casts as needed.
    - Results from the `hw.instance` are *repacked* into bundled values via unbundled &rarr; bundled casts so that surrounding Handshake ops can still see their original types.

---

### Step 2 – Convert the Handshake function

The second substep lowers the remaining `handshake.func` and its `handshake.end` terminator to an `hw.module` and its `hw.output`. At this stage, the function body mostly contains the `hw.instance`/cast network produced in Step 1.

The pass constructs a new `RewritePatternSet` containing `ConvertFuncToHWMod` and a `ConversionTarget` that:

- Marks `synth::SynthDialect` and `hw::HWDialect` as legal.
- Marks `UnrealizedConversionCastOp` as legal.
- Marks the Handshake dialect as fully illegal (including `handshake.func` and `handshake.end`).

It then calls `applyPartialConversion` again.

At a high level, `ConvertFuncToHWMod`:

- Creates (or looks up) a top-level `hw.module` whose ports are the unbundled version of the function’s arguments and results.
- Inserts grouped unbundled &rarr; bundled casts for module arguments, so the inlined function body sees the same bundled argument types it had originally.
- Inlines the `handshake.func` body into the HW module, remapping arguments to the cast results.
- Replaces the `handshake.end` with:
  - Bundled &rarr; unbundled casts for each operand.
  - A final `hw.output` using those unbundled values.
- Erases the original `handshake.func`.

From the cast perspective:

- Module arguments are *packed* into bundled values before inlining so that the existing body does not need to be rewritten.
- Function results are *unpacked* at the end so that the top `hw.module` exposes flat ports only.

At the end of Step 2:

- All Handshake operations and functions have been removed and replaced by HW/Synth operations.
- The only remaining artifacts of this conversion are the `UnrealizedConversionCastOp`s that still separate some bundled/unbundled regions internally.

---

### Step 3 – Cleanup of casts

The third substep removes all temporary casts by calling:

- `LogicalResult removeUnrealizedConversionCasts(mlir::ModuleOp modOp)`.

This cleanup relies on the invariant that casts appear in *simple*, paired chains:

- `producer  &rarr;  cast1  &rarr;  cast2  &rarr;  consumer`.

It tries to find the following cast pattern:

operation1 &rarr; cast1 &rarr; cast2 &rarr; operation2

in order to remove cast1 and cast2 and directly connect operation1 and operation2.

The removal algorithm:

- Walks all `UnrealizedConversionCastOp` in the module.
- For each `cast1`:
  - Skips it if its input is another cast (`cast2`) and asserts that there is no third cast in the chain.
  - Verifies that all uses of its results are `UnrealizedConversionCastOp`s; otherwise the pattern is considered too complex and fails.
- For each matched pair `(cast1, cast2)`:
  - Asserts that `cast1`’s number of operands matches `cast2`’s number of results.
  - Replaces every result of `cast2` with the corresponding operand of `cast1`, effectively bypassing the casts.
  - Schedules both casts for erasure.
- Erases all scheduled casts and checks that no `UnrealizedConversionCastOp` remain.

After Step 3, all temporary adapters are gone, and the IR consists purely of HW and Synth operations on flat ports—completing the unbundling phase.

---

### Handshake types and unbundling (details)

Handshake-level IR uses rich types to represent channels, control, and memory interfaces, whereas HW modules operate on flat ports. To bridge that gap, the pass introduces a dedicated type converter and helper utilities derived from MLIR’s `TypeConverter` so they can be plugged directly into pattern-based conversion.

#### Type-level unbundling

The central helper function is:

- `SmallVector<std::pair<SignalKind, Type>> unbundleType(Type type)`.

It maps high-level types to lists of `(SignalKind, Type)` pairs:

- For `handshake::ChannelType`:
  - `DATA_SIGNAL`  &rarr;  payload type (e.g., `i32`).
  - `VALID_SIGNAL`  &rarr;  `i1`.
  - `READY_SIGNAL`  &rarr;  `i1`.
- For `handshake::ControlType`:
  - `VALID_SIGNAL`  &rarr;  `i1`.
  - `READY_SIGNAL`  &rarr;  `i1`.
- For `MemRefType`:
  - `DATA_SIGNAL`  &rarr;  element type.
  - `VALID_SIGNAL`  &rarr;  `i1`.
  - `READY_SIGNAL`  &rarr;  `i1`.
- For all other types, the default is a single `DATA_SIGNAL`.

This logic is wrapped into:

- `ChannelUnbundlingTypeConverter : public TypeConverter`.

which expands any incoming type to its unbundled component types by calling `unbundleType`.

#### Operation port unbundling

At the operation level, unbundling is handled by:

- `void unbundleOpPorts(Operation *op, SmallVector<...> &inputPorts, SmallVector<...> &outputPorts)`.

This function:

- Extracts and unbundles the ports of:
  - Handshake functions (`handshake.func`) by visiting the function type’s arguments and results.
  - Non-function Handshake operations by visiting operands and results directly.
- Produces, for each original port, a vector of `(SignalKind, Type)` describing the corresponding flat signals.

These descriptors are then converted into HW module port information via:

- `void getHWModulePortInfo(Operation *op, SmallVector<hw::PortInfo> &inputs, SmallVector<hw::PortInfo> &outputs)`.

This function:

- Calls `unbundleOpPorts` on the operation.
- Assigns deterministic names based on the original port index and signal kind, e.g.:
  - `in_data_0`, `in_valid_0`, `in_ready_0` for input port 0.
  - `out_data_0`, `out_valid_0`, `out_ready_0` for output port 0.
- Produces `hw::PortInfo` lists suitable for constructing `hw::HWModuleOp`.

---

### Conversion of Handshake ops to HW modules (details)

The actual lowering from Handshake operations to HW is driven by a family of conversion patterns and helpers.

#### Per-operation conversion

The core helper is:

- `hw::HWModuleOp convertOpToHWModule(Operation *op, ConversionPatternRewriter &rewriter)`.

It performs:

1. **Module interface synthesis.**  
   - Computes HW input/output ports using `getHWModulePortInfo` on the Handshake operation.
2. **Module creation / lookup.**  
   - Gets the parent `mlir::ModuleOp` and its `SymbolTable`.
   - Computes a unique module name for the Handshake operation using `getUniqueName`.
   - Looks up an existing `hw::HWModuleOp` with that name, or creates one at module scope if none exists.
3. **Behavior instantiation.**  
   - Calls `instantiateSynthOpInHWModule` on the `hw::HWModuleOp` to create a `synth.subckt` representing the converted operation.
4. **Instance insertion.**  
   - Replaces the original Handshake operation with an `hw.instance` of the module:
     - For each operand:
       - If the defining operation belongs to the Handshake dialect or is a block argument, a bundled &rarr; unbundled cast is created and all its results are used as instance operands.
       - Otherwise, the operand is assumed already unbundled and used directly.
     - For each group of instance results, an unbundled &rarr; bundled cast is created via `createCastResultsUnbundledOp` and its single bundled result replaces the original result.
   - Asserts that the number of casted results matches the original operation’s result count.


Each generated HW module (representing a dataflow unit) is therefore a wrapper around a `synth.subckt` that describes the original Handshake operation’s behavior.

#### Generic conversion patterns

To drive conversion over the whole function, the pass defines:

- `template <typename T> class ConvertToHWMod : public OpConversionPattern<T>`.

with pattern callback:

- `LogicalResult ConvertToHWMod<T>::matchAndRewrite(T op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const`.

The pattern calls `convertOpToHWModule(op, rewriter)` and fails if no module could be created.

The pass instantiates this template for all Handshake operations. All are converted in a uniform way by the same template.

#### Function-level conversion (details)

The Handshake function itself (`handshake.func`) is converted by:

- `hw::HWModuleOp convertFuncOpToHWModule(handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter)`.

It:

- Computes the HW module interface by unbundling the function’s arguments/results via `getHWModulePortInfo`.
- Ensures a unique top-level `hw::HWModuleOp` exists for the function.
- Creates grouped unbundled &rarr; bundled casts for each module argument so that the inlined body still sees the same bundled arguments as before.
- Inlines the original Handshake body into the HW module, remapping arguments to cast results.
- Finds the unique `handshake.end`, inserts bundled &rarr; unbundled casts for each operand, and replaces it with an `hw.output` using all unbundled values.
- Deletes any old empty `hw.output` and erases the original `handshake.func`.

The pattern `class ConvertFuncToHWMod : public OpConversionPattern<handshake::FuncOp>` simply delegates to this helper.


---

## Ready signal inversion

After unbundling, the generated HW modules use a raw mapping of Handshake channels to `{data, valid, ready}` signals. All hw modules represent ready in the same direction as data/valid, while standard handshake protocols require ready to travel against data/valid. The second major phase of the pass fixes this. Additionally, **all data multi-bit signals are split into multiple single-bit signals**.

This phase is implemented in the `ReadySignalInverter` helper class and its methods.

Unlike the previous phase, this one does not rely on MLIR’s pattern-rewrite infrastructure. Because it introduces no type changes and does not modify operations, the full conversion machinery is unnecessary; instead, the phase is implemented using a simpler, ad-hoc approach.

This phase is also intentionally kept separate from the earlier one. The existing cast logic assumes a clear boundary, with unbundled signals on one side and bundled signals on the other. Extending that logic here would break this assumption: the ready signal flows in the opposite direction of the data and valid signals, eliminating the clean separation and significantly complicating the implementation.


The global entry point for ready inversion is:

- `LogicalResult ReadySignalInverter::invertAllReadySignals(mlir::ModuleOp modOp)`.

It:

- Builds a `SymbolTable` for the module.
- Collects all `hw::HWModuleOp`s into `oldHWModules` keyed by name.
- For each `(name, module)` pair in `oldHWModules`, calls `invertReadySignalHWModule` to create the rewritten module and populate `newHWModules`.
- Erases all old modules.
- Renames new modules back to their original names and updates all `hw.instance` operations so that they reference the restored module names rather than the temporary rewritten names.

At the end of this phase, the HW module hierarchy is structurally the same as after unbundling, but all ready signals follow the standard handshake direction.



### Goals and assumptions

The ready inversion phase:

- Iterates over all HW modules.
- Produces new HW modules with ports reorganized so that:
  - Data and valid remain in their original directions.
  - Ready signals are swapped from input to output or vice versa, to follow the opposite direction convention.
- Rewrites `hw.instance` operations to use the new modules and reconnects all signals consistently.
- Propagates changes recursively through the module hierarchy.

Two maps track signal equivalences during rewriting:

- `DenseMap<Value, Value> oldModuleSignalToNewModuleSignalMap`: maps original signals to their new counterparts.
- `DenseMap<Value, Value> oldModuleSignalToTempValueMap`: maps original signals to temporary placeholder constants when the final signal is not yet available.

It assumes that each Handshake channel connects exactly one producer to exactly one consumer; otherwise, the signal-mapping logic based on `DenseMap<Value, Value>` may not be valid.

### Per-module rewriting

The core of the inversion is:

- `void ReadySignalInverter::invertReadySignalHWModule(hw::HWModuleOp oldMod, ModuleOp parent, SymbolTable &symTable, DenseMap<StringRef, hw::HWModuleOp> &newHWModules, DenseMap<StringRef, hw::HWModuleOp> &oldHWModules)`.

It proceeds in four steps.

#### Step 1 – Already processed check

If a module name already exists in `newHWModules`, the function returns immediately.

#### Step 2 – New module interface and creation

The function:

- Scans the old module’s `PortList` to construct new input and output port lists with inverted directions for ready signals.
- For each port `p`:
  - If `p` is an input and its name contains `ready`, it becomes a ready output in the new module.
  - If `p` is an output and its name contains `ready`, it becomes a ready input in the new module.
  - Non-ready inputs and outputs keep their direction but are re-mapped with appropriate index tracking.

While doing so, it builds:

- `SmallVector<std::pair<unsigned, Value>> newModuleOutputIdxToOldSignal`:
  - For each new output index, records the old signal (either a module argument or terminator operand) that semantically corresponds to it.
- `SmallVector<std::pair<unsigned, Value>> newModuleInputIdxToOldSignal`:
  - Similarly for new inputs.

It then:

- Creates a new `hw::HWModuleOp` after `oldMod` with name `<oldName>_rewritten` via `getNewModuleName` and ports `newInputs`, `newOutputs`.
- Registers the new module in `newHWModules` under the original name.
- For each new input index, maps its argument to the corresponding old signal in `oldModuleSignalToNewModuleSignalMap`.

#### Step 3 – Body rewriting

The body of the old module is processed to handle instances and non-instance modules.

- The pass first scans the body:
  - If it finds any `hw.instance`, `hasHwInstances` is set to true.
- If instances exist:
  - Each `hw.instance` is rewritten by calling `invertReadySignalHWInstance`.
- If no instances exist:
  - The module is expected to contain only `hw.output` and `synth.subckt` operations.
  - Any deviation is treated as an error and causes an assertion.
  - In this pure glue module case, a new `synth.subckt` is instantiated in the new module via `instantiateSynthOpInHWModule` to maintain the same connectivity under the new interface.

#### Step 4 – New terminator wiring

Finally, the new module’s terminator (`hw.output`) is updated.

- For each `(newOutputIdx, oldSignal)` pair in `newModuleOutputIdxToOldSignal`, the builder:
  - Looks up the mapped new signal from `oldModuleSignalToNewModuleSignalMap`.
  - Asserts that the mapping exists.
  - Collects the mapped values in order of new output index.
- The terminator’s operands are replaced with this new operand list.

### Instance rewriting

Instances are rewritten by:

- `void ReadySignalInverter::invertReadySignalHWInstance(hw::InstanceOp oldInst, ModuleOp parent, SymbolTable &symTable, DenseMap<StringRef, hw::HWModuleOp> &newHWModules, DenseMap<StringRef, hw::HWModuleOp> &oldHWModules)`.

This function:

- Ensures the instance’s callee module has been rewritten by checking `newHWModules` and calling `invertReadySignalHWModule` if needed.
- Determines the rewritten callee module (`newMod`), the original callee module (`oldMod`), the top-level module containing the instance, and its corresponding rewritten module (`newTopMod`) where new operations will be inserted.
- Prepares three lists:
  - `nonReadyOutputs`: `(outputName, oldResultIdx)` pairs for outputs that are not ready and whose uses must be re-routed.
  - `oldReadyInputs`: `(inputName, oldValue)` pairs for ready inputs that will become ready outputs in the new module.
  - `newOperands`: operands for the new `hw.instance`.
- Iterates over `oldMod`’s ports and classifies each:
  - For ready inputs in the old module:
    - The corresponding signal in the new module is an output; the old input value is stored in `oldReadyInputs`.
  - For ready outputs in the old module:
    - These become inputs in the new module.
    - `getInputSignalMappingValue` is called on the old output to find or create the matching new input value, which is added to `newOperands`.
  - For non-ready inputs:
    - `getInputSignalMappingValue` is called similarly and added to `newOperands`.
  - For non-ready outputs:
    - They are recorded in `nonReadyOutputs` for later output mapping.
- Creates a new instance of `newMod` in the rewritten top module with the original instance name and `newOperands` as operands.
- Updates signal mappings:
  - For each `(outputName, oldResultIdx)` in `nonReadyOutputs`, `updateOutputSignalMappingValue` binds the old non-ready result to the corresponding new result and replaces any temporary placeholders.
  - For each `(inputName, oldReadyInput)` in `oldReadyInputs`, `updateOutputSignalMappingValue` maps the old ready input to the new ready output of the rewritten module.

Helpers:

- `Value ReadySignalInverter::getInputSignalMappingValue(Value oldInputSignal, OpBuilder &builder, Location loc)`:
  - Returns a mapped or temporary value for a given old input, creating a `hw.constant` placeholder if necessary and storing it in `oldModuleSignalToTempValueMap`.
- `void ReadySignalInverter::updateOutputSignalMappingValue(Value oldResult, StringRef outputName, hw::HWModuleOp newMod, hw::InstanceOp newInst)`:
  - Finds the output index of `outputName` in `newMod`, obtains the result from `newInst`, updates `oldModuleSignalToNewModuleSignalMap`, replaces any temporary uses, and erases the temporary constant op if present.

