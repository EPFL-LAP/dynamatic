# Handshake to Synth


The role of this pass is to transform a Handshake-level representation of a dataflow circuit into a Synth-level representation expressed through HW modules and synth operations. Conceptually, it sits between the Handshake optimization pipeline and a future synthesis backend that will further refine synth operations into a final RTL- or gate-oriented netlist.

There are three main sections in this document.

1. Main pass and usage: Overall structure and rationale of the pass.
2. Unbundling conversion: Lowering Handshake channel types to flat HW ports and `synth.subckt`.
3. Signal rewriting: Inverting the direction of ready signals to follow the standard handshake protocol, unbundling multi-bit data signals into multiple single bit signals and adding reset and clock signal to each module and connecting them to the top function ones.
4. AIG construction: Convert all hw instances into AIG nodes.


The pass is called [`HandshakeToSynthPass`](ConversionHandshakeToSynth.md).

---

## Main pass and usage

At a high level, the *HandshakeToSynth* pass performs the following transformations on a module containing a **single non-external** `handshake.func`:

- Mark each handshake operation with the blif path which describes it AIG description. This information is propagated through each step as an attribute (`blif_path`) present in each hw module. This information will be useful in the last step.
- Converts all Handshake-typed values (channels, control, memory) into flat HW-level ports by unbundling them into `{data, valid, ready}` signals.
- Lowers each Handshake operation (including the function itself) to an `hw.module`/`hw.instance` plus an internal `synth.subckt` representing its behavior (except from the top handshake function).
- Rewrites all generated HW modules to enforce the standard handshake convention where ready signals flow in the opposite direction from data and valid, and propagates this convention recursively through module instances. During this step, it also unbundles the multi-bit data signals into multiple single bit signals and adds reset and clock signals to each module.
- Rewrite all generated HW instances into synth operations (AIG nodes mainly) and connect them accordingly. The description of each hw instance is specified in the BLIF library. More information on how to generate this blif library are specified in the doc [BlifGenerator](../Buffering/MapBuf/BlifGenerator.md)

The pass operates on:

- A single non-external `handshake.func` per module (enforced by the pass).
- A graph of Handshake operations inside the function, with bundled channel types.

and produces:

- A pure HW/Synth module hierarchy:
  - The original `handshake.func` is replaced by a top-level `hw.module`.
  - Each Handshake operation becomes a set of synth operations.
- No remaining values of Handshake types and no remaining Handshake operations or functions.


The overall pass is implemented as:

- `class HandshakeToSynthPass : public dynamatic::impl::HandshakeToSynthBase<HandshakeToSynthPass>`.

Its `runDynamaticPass()` method:

- Retrieves the `mlir::ModuleOp` and `MLIRContext`.
- Ensures that there is at most one non-external `handshake.func` in the module and that if none is found, the pass is a no-op.
- Runs Phase 0 – Mark each handshake unit with the blif path that describes its AIG beaviour using the function `getBlifFilePathForHandshakeOp(op, blifDirPath)`
- Runs Phase 1 – Unbundling by calling `unbundleAllHandshakeTypes(modOp, ctx)`.
- Runs Phase 2 – Signal rewriting by instantiating a `SignalRewriter` and calling `rewriteAllSignals(modOp)`.
- Runs Phase 3 – Rewrite of hw instances into synth operations (registers, combinational logic, etc.).

In a typical flow, this pass is run after all Handshake-level optimizations and buffer insertion, and before a dedicated synth backend that will interpret or further lower the generated synth operations.

**IMPORTANT**: For now, the identification of the blif path heavily relies on the values of parameters of each handshake unit. Since there is still no unique database that correlates each handhshake unit with its parameter to instantiate its RTL, these information are hard-coded for each operation type in the `getBlifFilePathForHandshakeOp(op, blifDirPath)` function.

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

## Signal rewriting

After unbundling, all Handshake channels are expressed as `{data, valid, ready}` signals on HW modules, but the low-level HW representation still does not match the desired per-bit structure and signal-direction conventions. he second major phase of the pass rewrites the HW module hierarchy to enforce the standard handshake direction for ready signals and to normalize signal groups to the chosen bit-level representation (e.g., single-bit signals for each component), without changing the circuit’s observable behavior.


This phase is implemented in the `SignalRewriter` helper class and its methods.

Unlike the previous phase, this one does not rely on MLIR’s pattern-rewrite infrastructure. Because it introduces no type changes and does not modify operations, the full conversion machinery is unnecessary; instead, the phase is implemented using a simpler, ad-hoc approach.

This phase is also intentionally kept separate from the earlier one. The existing cast logic assumes a clear boundary, with unbundled signals on one side and bundled signals on the other. Extending that logic here would break this assumption: the ready signal flows in the opposite direction of the data and valid signals, eliminating the clean separation and significantly complicating the implementation.


At a high level, this phase:

- Iterates over all HW modules in the IR.
- Creates rewritten HW modules whose ports reflect:
  - Correct ready direction (ready opposite to data/valid).
  - Normalized bit-level structure for signals that have been unbundled.
- Rewrites all `hw.instance` operations to use these rewritten modules and reconnects their operands/results consistently.
- Propagates these changes recursively through the entire HW hierarchy.
  
At the end of this phase, the HW module hierarchy is structurally the same as after unbundling, but all ready signals follow the standard handshake direction.



### Goals and assumptions

The signal rewriting phase has three main goals:

- Enforce the standard handshake convention that ready flows in the opposite direction to data/valid on every channel.
- Normalize the representation of signals to bit-level groups (e.g., 1-bit signals for each component) where required by the unbundling and downstream synthesis flow.
- Preserve the original structural hierarchy as much as possible, changing only port directions, bit layout, and wiring.

Two maps track signal equivalences during rewriting:

- `DenseMap<Value, SmallVector<Value>> oldModuleSignalToNewModuleSignalsMap`  
  Maps an original signal (from old modules) to the vector of new signals that represent it in the rewritten hierarchy (e.g., a group of single-bit signals after unbundling).
- `DenseMap<Value, SmallVector<Value>> oldModuleSignalToTempValuesMap`  
  Maps an original signal to a vector of temporary placeholder values used when the final rewritten signals are not yet available (e.g., because the producing instance has not been rewritten yet).

It assumes that each Handshake channel connects exactly one producer to exactly one consumer; otherwise, the signal-mapping logic based on `DenseMap<Value, Value>` may not be valid.

### Per-module rewriting

- `void SignalRewriter::rewriteHWModule(hw::HWModuleOp oldMod, ModuleOp parent, SymbolTable &symTable, DenseMap<StringRef, hw::HWModuleOp> &newHWModules, DenseMap<StringRef, hw::HWModuleOp> &oldHWModules)`.

This function rewrites one module in four conceptual steps: skip already processed modules, build a rewritten interface, transform the body, and wire the new terminator.

#### Step 1 – Already processed check

If a module name already exists in `newHWModules`, the function returns immediately.

#### Step 2 – New module interface and creation

The rewriter first determines the interface of the new module.

- It scans `oldMod.getPortList()` and, for each port `p`, decides:
  - Whether the signal should keep its direction (e.g., data/valid).
  - Whether the signal must have its direction flipped (e.g., ready-like signals, recognized by name).
  - How multi-bit ports should be represented after unbundling (e.g., grouped single-bit ports), if applicable.
  - 

While scanning, it builds two association tables:

- `SmallVector<std::pair<unsigned, Value>> newModuleOutputIdxToOldSignal`  
  For each new output index, records the old signal (either a module argument or terminator operand) that it logically corresponds to.

- `SmallVector<std::pair<unsigned, Value>> newModuleInputIdxToOldSignal`  
  For each new input index, records the old signal or group of signals that feed it in the original module.


Using these port lists, `SignalRewriter` then:

- Creates a new `hw::HWModuleOp` immediately after `oldMod`, with name returned by `getRewrittenModuleName(oldMod, ctx)` (e.g., `<oldName>_rewritten`) and ports `newInputs`, `newOutputs`.
- Inserts this new module into `newHWModules` under the *original* module name, so that lookups by name see the rewritten version.
- For each `(newInputIdx, oldSignal)` in `newModuleInputIdxToOldSignal`:
  - Fetches the new module argument at index `newInputIdx`.
  - Records a mapping from the old signal to the new argument in `oldModuleSignalToNewModuleSignalsMap`.
  
This step defines how the new interface maps back to the original signals, including direction changes and bit-level expansions.

#### Step 3 – Rewrite the body

The body of the old module is then transformed to match the rewritten interface.

- The rewriter scans all operations in `oldMod`:
  - If it encounters any `hw.instance`, it sets `hasHwInstances = true`.

If `hasHwInstances` is true:

- Every `hw.instance` in `oldMod` is rewritten using `rewriteHWInstance`, which:
  - Ensures the callee module has a rewritten version (recursively calling `rewriteHWModule` if needed).
  - Creates a new instance of the rewritten module in the corresponding rewritten top module.
  - Updates the mapping structures with the new instance’s results.

If `hasHwInstances` is false:

- The module is expected to contain only:
  - `hw.output` operations, and
  - `synth.subckt` operations.
- Any additional operation is treated as an error because such modules are assumed to be pure “glue” between ports and a `synth.subckt`.
- In this case, the rewriter invokes `instantiateSynthOpInHWModule` on the new module to recreate a `synth.subckt` that reflects the original behavior under the new interface.

This step ensures that the internal logic of each module is re-expressed in terms of the new interface and the new per-bit signal representation.

#### Step 4 – Wire the new terminator

Finally, the rewriter wires the terminator of the new module.

- For each `(newOutputIdx, oldSignal)` in `newModuleOutputIdxToOldSignal`, it:
  - Looks up the group of new signals corresponding to `oldSignal` in `oldModuleSignalToNewModuleSignalsMap`, asserting that a mapping exists.
  - Accumulates these new signals in order of `newOutputIdx` and per-bit position.
- It then sets the operands of the new module’s `hw.output` terminator to this ordered list of values.

After this step, the new module is self-consistent: its ports, body, and terminator all use the updated directions and bit-level structure.

---

### Instance rewriting

Rewriting instances of HW modules is handled by:

- `void SignalRewriter::rewriteHWInstance(hw::InstanceOp oldInst, ModuleOp parent, SymbolTable &symTable, DenseMap<StringRef, hw::HWModuleOp> &newHWModules, DenseMap<StringRef, hw::HWModuleOp> &oldHWModules)`.

This function replaces an instance of an old module with an instance of the corresponding rewritten module, reconnecting operands and results according to the new interface.

#### Step 1 – Ensure the callee is rewritten

From `oldInst.getModuleName()`, the rewriter:

- Checks whether the rewritten callee already exists in `newHWModules`.
- If not, it looks up the original callee in `oldHWModules` and calls `rewriteHWModule` on it to create the rewritten version.

This guarantees that each instance always has a rewritten callee to target.

#### Step 2 – Find the rewritten context and insertion point

The rewriter identifies:

- `newMod`: the rewritten callee module from `newHWModules`.
- `oldMod`: the original callee from `oldHWModules`.
- `oldInstTopModule`: the HW module that contains `oldInst`.
- `newTopMod`: the rewritten counterpart of `oldInstTopModule` in `newHWModules` where the new instance will be created.

It sets the builder insertion point just before the terminator of `newTopMod`, so that new instances are appended at the end of its body.

#### Step 3 – Classify ports and build operands

The rewriter then constructs three lists:

- A list of outputs whose structure/direction remains stable and whose uses will later be redirected (e.g., data/valid outputs).
- A list of old signals whose direction or bit-level representation changes and that must be mapped to new outputs (e.g., ready-like signals or unbundled bits).
- A list of operands to feed the new instance (`newOperands`).

To do this, it iterates over `oldMod.getPortList()` and, for each port `p`:

- For ports whose direction must be flipped in the new module (e.g., ready-like ports detected by name):
  - Ready-like inputs in the old module become outputs in the new one; their old values are recorded for later mapping.
  - Ready-like outputs in the old module become inputs in the new one; the rewriter calls `getInputSignalMapping` on the old result to obtain or create the new input signals, then appends those to `newOperands`.

- For ports whose direction stays the same but whose bit representation may change:
  - Non-ready inputs are handled by calling `getInputSignalMapping` on the old operand and appending the returned signals to `newOperands`.
  - Non-ready outputs are recorded as “stable outputs” whose uses must later be redirected to groups of new signals.

The helper `SmallVector<Value> SignalRewriter::getInputSignalMapping(Value oldInputSignal, OpBuilder builder, Location loc)` behaves as follows:

- If `oldInputSignal` is already in `oldModuleSignalToNewModuleSignalsMap`, it returns the associated vector of new signals.
- If it appears in `oldModuleSignalToTempValuesMap`, it returns the associated temporary vector.
- Otherwise, it:
  - Creates one or more `hw.constant` ops of the appropriate type(s), typically producing single-bit zero values.
  - Stores these constants in `oldModuleSignalToTempValuesMap[oldInputSignal]`.
  - Returns the vector of constant results.

This mechanism enables instance rewriting even when the producers of some signals have not yet been rewritten.

#### Step 4 – Create the new instance and update mappings

Once `newOperands` is complete, the rewriter:

- Creates a new `hw::InstanceOp` `newInst` in `newTopMod`, using:
  - `newMod` as the callee.
  - The same instance name attribute as `oldInst`.
  - `newOperands` as operands.

It then updates the mapping structures:

- For each stable output `(outputName, oldResultIdx)`:
  - Let `oldResult = oldInst.getResult(oldResultIdx)`.
  - Invoke `updateOutputSignalMapping(oldResult, outputName, newMod, newInst)`, which:
    - Finds all indexes in `newMod` that correspond to `outputName` (e.g., multiple bits).
    - Gathers the corresponding `newInst` results into a vector.
    - Registers `oldModuleSignalToNewModuleSignalsMap[oldResult] = newResults`.
    - If `oldResult` appears in `oldModuleSignalToTempValuesMap`, checks that the temporary group has the same size as `newResults`, replaces all uses of each temporary by the corresponding new value, removes the temps’ defining ops, and erases the entry from the temp map.

- For each signal whose direction or structure changed (e.g., old ready inputs now mapped to new ready outputs), `updateOutputSignalMapping` is used similarly, but the original value may be an operand instead of a result.

After this step, any user that was wired to the old instance (or to its temporary proxies) can be redirected to the final new signals by consulting `oldModuleSignalToNewModuleSignalsMap`.

## AIG construction

The final step of the pass refines the HW-level hierarchy into a purely Synth-level representation by replacing each HW instance with a network of Synth operations derived from its BLIF description. Conceptually, this phase interprets each BLIF model as an AIG-style circuit composed of 1-bit registers and AND-with-inverter gates, and re-expresses it directly in the Synth dialect while preserving the instance’s interface.

### Goals and high-level behavior

This phase has three main goals:

- Eliminate HW instances by inlining their behavior as Synth operations, while keeping the top HW module as the structural shell.
- Interpret each BLIF model as a combination of 1-bit latches and AIG-style combinational logic and reconstruct the same structure in the Synth dialect.
- Maintain a one-to-one correspondence between BLIF inputs/outputs and the HW instance’s operands/results, so that external connectivity remains unchanged.

The top-level driver is:

- `LogicalResult convertHWInstancesToSynthOps(mlir::ModuleOp modOp, StringRef topModuleName, MLIRContext *ctx)`.

It:

- Looks up the top HW module corresponding to `topModuleName` and iterates over all `hw.instance` operations inside it.
- For each instance, calls `convertHWInstanceToSynthOps(instOp, builder)` to build the corresponding Synth circuit.
- After successful conversion of all instances, erases all non-top HW modules, leaving a representation where the top HW module’s body contains only Synth operations (and no nested HW modules).

If a given HW module has no BLIF path attribute or if conversion is not supported, the phase falls back to replacing the instance with a single `synth.subckt` that preserves its flat interface.

### Per-instance BLIF-based refinement

The per-instance conversion is handled by:

- `LogicalResult convertHWInstanceToSynthOps(hw::InstanceOp instOp, OpBuilder &builder)`.

Its behavior is as follows:

- Retrieves the callee `hw::HWModuleOp` via `instOp.getModuleName()` and checks that it carries the `blifPathAttrStr` attribute containing the BLIF file path.
- Parses the BLIF file associated with that module and builds an internal mapping between BLIF node names and Synth `Value`s using two maps:
  - `nodeValuesMap` – mapping from BLIF node name to final Synth value.
  - `tmpValuesMap` – mapping from BLIF node name to temporary placeholder constants when a node is referenced before being defined.
- Establishes a direct correspondence between BLIF `.inputs` / `.outputs` and HW instance operands/results, so that input node names map to `instOp` operands and output node names determine which Synth values will eventually replace `instOp` results.

At the end of BLIF parsing, the conversion collects Synth values for all BLIF output nodes that correspond to the instance’s results and replaces `instOp` with these values; the HW instance operation is then erased.

### Latch construction (`.latch` lines)

BLIF `.latch` lines describe sequential elements, which the pass lowers to Synth latches:

- For each `.latch` line, the parser extracts:
  - The input node name (register data input).
  - The output node name (register output).
  - Optional additional fields (e.g., reset values), which are currently either checked or ignored depending on support.
- The helper `getInputMappingSynthSignal(loc, nodeValuesMap, tmpValuesMap, inputName, builder)` is used to obtain the Synth value corresponding to the latch input node:
  - If the node was already defined, its value is returned from `nodeValuesMap`.
  - If it was only referenced previously, an existing temporary constant is retrieved from `tmpValuesMap`.
  - Otherwise, a new 1-bit `hw.constant` zero is created, recorded in `tmpValuesMap`, and returned as a placeholder.
- A `synth::LatchOp` is created with a 1-bit integer type (`i1`) and the chosen input value.
- The result of the latch is registered with the output node name via `updateOutputSynthSignalMapping(latchResult, outputName, nodeValuesMap, tmpValuesMap, builder)`, which:
  - Replaces any previous temporary values for that node by the latch’s result.
  - Erases the defining ops of those temporaries.
  - Records the latch output in `nodeValuesMap` as the final signal for that node.

This process ensures that every sequential element in the BLIF model is represented by an explicit Synth latch in the reconstructed circuit.

### Combinational AIG logic (`.names` lines)

BLIF `.names` lines describe combinational logic using a simple truth-table syntax. The pass distinguishes between three structural cases depending on the number of node names that follow `.names`.

#### Case 1 – Single-node `.names` (constants)

If the `.names` line lists only one node name, it represents a constant node:

- The associated truth table line (e.g., `1` or `0`) determines whether the constant is logical 1 or 0.
- A 1-bit `hw.constant` of the appropriate value is created at the current insertion point.
- `updateOutputSynthSignalMapping` is called to associate this constant with the node name and to remove any temporary values previously created for that node.

#### Case 2 – Two-node `.names` (wire or inverter)

If the `.names` line has exactly two node names (one input, one output), the block encodes either a direct wire or an inversion between them, which is implemented via the helper:

- `LogicalResult createSynthWire(Location loc, ArrayRef<std::string> ports, StringRef function, DenseMap<StringAttr, Value> &nodeValuesMap, DenseMap<StringAttr, Value> &tmpValuesMap, OpBuilder &builder)`.

The behavior is:

- The single input node is resolved to a Synth value using `getInputMappingSynthSignal`.
- The truth table (`function`) is inspected:
  - If it indicates that the output equals the input (no inversion), no new operation is created; `updateOutputSynthSignalMapping` simply records that the output node name maps to the same Synth value as the input.
  - If it indicates inversion, the pass creates a `synth::aigAndInverterOp` with:
    - The original input as its sole data input.
    - An implied AND with logical 1 and an inversion flag that effectively models a NOT gate.
- In both cases, `updateOutputSynthSignalMapping` registers the final output node and replaces any temporaries for that name.

This guarantees that identity and inversion edges in the BLIF graph are represented explicitly (or by aliasing) in the Synth-level AIG.

#### Case 3 – Multi-input `.names` (logic gates)

If the `.names` line has more than two node names, it represents a multi-input combinational gate described by its truth table and is implemented using:

- `LogicalResult createSynthLogicGate(Location loc, ArrayRef<std::string> ports, StringRef function, DenseMap<StringAttr, Value> &nodeValuesMap, DenseMap<StringAttr, Value> &tmpValuesMap, OpBuilder &builder)`.

This helper:

- Asserts that there is at least one input and exactly one output node name.
- Resolves all input node names to Synth values using `getInputMappingSynthSignal`.
- Interprets the BLIF truth table as specifying the conditions under which the output is 1, and constructs an AIG network using only `synth::aigAndInverterOp` operations:
  - Each product term (row of the BLIF table) is converted into a small tree of AND-with-inverter nodes, where literals may be inverted as required.
  - Multiple product terms are combined using additional AND-with-inverter nodes and constant signals to emulate OR behavior in AIG form.
- The final Synth value for the gate output node is registered via `updateOutputSynthSignalMapping`, which also resolves and eliminates any temporary placeholders created earlier for that node.

By restricting the implementation to `synth::aigAndInverterOp`, the pass ensures that **all combinational logic** is normalized to an AIG representation consisting of AND gates and literal inversion flags only.

### Temporary signal handling and final wiring

Throughout BLIF parsing, the helpers `getInputMappingSynthSignal` and `updateOutputSynthSignalMapping` ensure that forward references and late definitions are handled robustly:

- `getInputMappingSynthSignal`:
  - Returns a previously defined Synth value if the node name is in `nodeValuesMap`.
  - Returns an existing temporary constant if the node name is in `tmpValuesMap`.
  - Otherwise, creates a new 1-bit zero `hw.constant`, records it in `tmpValuesMap`, and returns it as a placeholder.
- `updateOutputSynthSignalMapping`:
  - If the node name exists in `tmpValuesMap`, replaces all uses of the temporary constant with the new result value and erases the constant operation.
  - Inserts the new result into `nodeValuesMap` as the canonical value for that node.

After all `.latch` and `.names` sections of the BLIF model have been processed, the converter:

- Gathers Synth values for all BLIF output node names that correspond to the HW instance’s outputs (using the `hwInstOutputs` mapping built from BLIF `.outputs`).
- Replaces each result of the original `hw.instance` with the corresponding Synth value.
- Erases the `hw.instance` from the IR.

At this point, the behavior previously encapsulated in the HW instance and its referenced HW module has been fully re-expressed as an inlined AIG of `synth::LatchOp` and `synth::aigAndInverterOp` operations connected directly to the top module’s ports, which completes the last step of the Handshake-to-Synth conversion.
