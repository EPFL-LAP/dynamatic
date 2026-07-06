# HW Flatten Modules Pass

The **HW Flatten Modules** pass inlines all `hw.instance` operations into their
parent `hw.module`, replacing each instance with the body of the referenced
module. After inlining, the now-redundant module definitions are erased, leaving
a single flat `hw.module` per top-level circuit.

---

## Overview

The pass operates on a `builtin.module` containing one or more `hw.module`
operations, some of which instantiate others via `hw.instance`. After the pass
runs, no `hw.instance` ops remain and every module that was instantiated has
been removed.

The pass can be invoked as follows:
```bash
./bin/dynamatic-opt --flatten-modules <input-mlir-file>
```

---

## Code Structure

The core logic lives in `FlattenModulesPass::runOnOperation()` and proceeds in
three phases:

1. **Record instantiated modules.** Before any inlining, the pass walks all
   `hw.instance` ops and records the name of every referenced module. This
   snapshot is used later to decide which modules to erase.

2. **Inline all instances.** The pass repeatedly walks the IR looking for
   `hw.instance` ops. Each time one is found, the body of the referenced
   `hw.module` is inlined at the call site using MLIR's `inlineRegion` utility,
   and the instance op is erased. The walk restarts after each mutation and
   continues until no instances remain.

3. **Erase inlined modules.** Any `hw.module` whose name was recorded in step 1
   is erased from the top-level module, as its definition is no longer needed.

---

## Support: `HWInliner`

The `HWInliner` struct subclasses MLIR's `InlinerInterface` and provides the
inlining policy used in step 2. It implements three methods:

- **`isLegalToInline(Region*, ...)`**: always returns `true`, since inlining
  `hw.module` bodies is always legal in this context.

- **`isLegalToInline(Operation*, ...)`**: always returns `true` for the same
  reason.

- **`handleTerminator`**: called when the inliner encounters the `hw.output`
  terminator of the inlined module. It wires each output value back to the
  corresponding result of the original `hw.instance` op by replacing all uses.

---

## Limitations

- The pass inlines **all** instances unconditionally. There is no heuristic to
  skip large or stateful modules.
- Only `hw.instance` ops targeting `hw.module` definitions are inlined.
  Instances targeting `hw.module.extern` or other module-like ops are silently
  skipped.