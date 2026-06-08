# Synth dialect


The **Synth** dialect is a low-level synthesis-oriented dialect that provides a common in-MLIR interface for logic synthesis operations and data structures. It is intended as an intermediate representation between higher-level hardware IR (e.g., Handshake or HW) and concrete synthesis backends, capturing both logic networks (such as AIGs) and meta-operations that steer synthesis decisions.

In Dynamatic, operations of the synth dialect are always described inside an `hw.module` operation of the hw dialect.

---

## Design and role in the flow

The Synth dialect has a core responsibility:

- Support **logic network representations** such as AIG (And-Inverter Graph) and sequential elements as latches.

In other words, the Synth dialect is not tied to a particular hardware backend or HDL; instead, it serves as an abstraction layer where logic-level manipulations can be expressed, analyzed, and transformed before being committed to a specific RTL or gate-level format.

---

## Operations 

This dialect describes two core operations:

- `AndInverterOp` which describes the nodes of an AIG.
- `LatchOp` which describes the presence of a latch.

### And-Inverter Node

It represents a node in an AIG. It computes the bitwise AND of all inputs, each of which may be individually inverted. 

```mlir
%r1 = synth.aig.and_inv %a, %b : i1
%r2 = synth.aig.and_inv not %a, %b : i1
%r3 = synth.aig.and_inv not %a, not %b : i1
```

### Latch Node 

It models a storage element, directly corresponding to the BLIF `.latch` construct. Accepts three optional parameters: a `latchType` string ("fe", "re", "ah", "al", "as"), a control clock/enable signal, and an initVal encoding the initial state (0, 1, 2 = don't care, 3 = unknown).

```mlir
%q = synth.latch %d : i1
%q = synth.latch %d clock %clk init 0 : i1
```

