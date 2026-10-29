# Formal Properties Infrastructure

This document describes the infrastructure for supporting formal properties in Dynamatic, focusing on the design decisions, implementation strategy, and intended usage. This infrastructure is used to express circuit-level runtime properties, primarily to enable formal verification via model checking.

## Overview

The infrastructure introduces a compiler pass called `annotate-properties`, which collects formal properties information from the Handshake IR, and serializes them to a shared .json database for other tools to consume (e.g., model checkers, code generators, etc.). This infrastructure is built to express "runtime" properties, which in the context of HLS mean properties that will appear in the circuit (or in the SMV model), and will be checked only during simulation (or model checking). This infrastructure does NOT support compile-time checks. These checks should be carried out through the MLIR infrstructure.

## Properties

A property (`FormalProperty` in `experimental/include/experimental/Support/FormalProperty.h`) is a record with the following fields:
- `id`: Unique identifier, which also names the property in the SMV model.
- `kind`: What the property describes (e.g., `AbsenceOfBackpressure`, `EagerForkNotAllOutputSent`).
- `tag`: Purpose of the property: `OPT` for a property whose outcome decides an optimization, `INVAR` for an invariant that holds by construction and is assumed when checking the other properties.
- `check`: Outcome of formal verification (`true`, `false`, or `null` if unchecked).
- `expr`: The formula of the property (see below).

The formula is a `PropertyExpr` (`experimental/include/experimental/Support/PropertyExpr.h`), a container of any of the following nodes:
- `State`: the internal state of a unit (an `InternalStateNamer`, e.g., whether a buffer slot is full or a fork output is in the sent state), optionally constrained to the value of the token it holds.
- `Signal`: the valid or ready signal of an operand or result port of a unit (e.g., `fork0.outs_0_valid`). The properties that a downstream pass acts upon identify their channels through the valid signals of result ports (e.g., rigidification removes the ready signal of the channel of an `AbsenceOfBackpressure` property that holds).
- `Constant`: an integer or boolean constant.
- `UnaryExpr`: `!`, `toint`, and the CTL operators `AG`, `AF`, `AX`, `EG`, `EF`, `EX`.
- `BinaryExpr`: `->`, `<->`, `-`, `*`, and comparisons.
- `NaryExpr`: `&`, `|`, `+`, and `count`.

The `annotate-properties` pass is the only place in the code where the MLIR is analyzed to create properties: the formula must be complete, so that downstream tools only need to print it. Invariants cannot contain temporal operators, as they are assumed in every state.

Formal properties are stored in a shared JSON database, with each property entry following this schema:

```
{
  "check": null,
  "expr": {
    "op": "implies",
    "operands": [
      { "op": "signal", "signal": "valid", "unit": "fork0", "side": "result", "index": 0, "port": "outs_0" },
      { "op": "signal", "signal": "ready", "unit": "constant0", "side": "operand", "index": 0, "port": "ctrl" }
    ]
  },
  "id": 0,
  "kind": "AbsenceOfBackpressure",
  "tag": "OPT"
}
```

## Adding a New Property

To illustrate how a new property can be integrated, we take an example from the paper [Automatic Inductive Invariant Generation for Scalable Dataflow Circuit Verification](https://dynamo.ethz.ch/wp-content/uploads/sites/22/2023/10/Xu_IWLS23_Inductive_Invariants.pdf): "for any fork, the number of outputs that are in the sent state must be smaller than the total number of fork outputs".

### Write a Generator in the `annotate-properties` Pass

A generator walks the circuit and adds one property per instance with `addProperty`, building its formula from the internal states and signals of the units:

```
LogicalResult
HandshakeAnnotatePropertiesPass::annotateEagerForkNotAllOutputSent(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        // count(fork0.outs_0_sent, fork0.outs_1_sent) < 2
        std::vector<EagerForkSentNamer> sents =
            forkOp.getInternalSentStateNamers();
        PropertyExpr expr =
            BinaryExpr(NaryExpr(NaryExpr::Count, getStates(sents)),
                       BinaryExpr::Lt, Constant::integer(sents.size()));
        addProperty(EAGER_FORK_NOT_ALL_OUTPUT_SENT, FormalProperty::TAG::INVAR,
                    expr);
      }
    }
  }
  return success();
}
```

Then add the generator to the `GENERATORS` table of the pass, under the name with which it is selected in the `annotate-list` option. Nothing else needs to change: `export-rtl` prints the formula of any property, choosing between `INVAR`, `INVARSPEC`, and `CTLSPEC` depending on the tag and on whether the formula is temporal.

### Expose the Internal State in the Model

A formula can only refer to state that the SMV model of the unit exposes. If a new property refers to a state that is not yet exposed (e.g., the `sent` state of a fork output), the SMV generator of the unit must define it so that it is accessible outside of the unit, and an `InternalStateNamer` must name it.

For example the generated SMV code will look like this:

```
MODULE fork (ins_0, ins_0_valid, outs_0_ready, outs_1_ready)

  -- fork logic

  DEFINE sent_0 := ...;
  DEFINE sent_1 := ...;
```

## FAQs
### Why use JSON?

- Allows decoupling between IR-level passes and later tools.
- Easily inspectable and extensible.
- Serves as a contract between compiler passes and formal verification tools.

### Can I add properties from an IR different than Handshake?

In theory this system supports adding properties at any time in the compilation flow because the .json file is always accessible, but we strongly advise against it. Properties must be fully specified by the end of compilation, and earlier IRs may lack the necessary information to construct them correctly.

If needed, a possible approach is to perform an early annotation pass that creates partial property entries (with some fields left blank), and then complete them later in Handshake via the `annotate-properties` pass. Still, whenever possible, we suggest implementing property generation directly within Handshake to avoid inconsistencies and simplify the flow.
