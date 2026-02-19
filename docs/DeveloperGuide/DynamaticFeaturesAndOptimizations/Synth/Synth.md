# Synth dialect


The **Synth** dialect is a low-level synthesis-oriented dialect that provides a common in-MLIR interface for logic synthesis operations and data structures. It is intended as an intermediate representation between higher-level hardware IR (e.g., Handshake or HW) and concrete synthesis backends, capturing both logic networks (such as AIGs and MIGs) and meta-operations that steer synthesis decisions.

In Dynamatic, the Synth dialect is the target of the [Handshake-to-Synth lowering pass](ConversionHandshakeToSynth.md): each Handshake operation is wrapped in a `synth.subckt` (or similar Synth operation) inside an `hw.module`, and later passes are free to refine these subcircuits into more detailed networks and optimization steps.

---

## Design and role in the flow

The Synth dialect has three core responsibilities:

- Provide **operations and types for logic synthesis**, i.e., operations that manipulate logic-level representations and synthesis state.
- Include **meta operations** that encode synthesis decisions (e.g., which implementation, which optimization pass, or which strategy to apply on a given subcircuit).
- Support **logic network representations** such as AIG (And-Inverter Graph) and MIG (Majority-Inverter Graph), as well as the infrastructure to represent synthesis pipelines in MLIR.

In other words, the Synth dialect is not tied to a particular hardware backend or HDL; instead, it serves as an abstraction layer where logic-level manipulations can be expressed, analyzed, and transformed before being committed to a specific RTL or gate-level format.

---

## Operations and types (conceptual overview)

This dialect focuses on three aspects:

- **Logic representations:**
  - AIG-style networks, where logic is modeled as AND gates and inverters.
  - MIG-style networks, where logic is modeled as majority nodes with optional inversion on edges.
- **Meta operations for synthesis decisions:**
  - Operations that encode choices such as:
    - Which implementation variant to use for a given functionality.
    - Which optimization passes or strategies have been applied or should be applied.
    - How to annotate subcircuits for later backend-specific handling.
- **Synthesis pipeline infrastructure:**
  - Operations that describe, parameterize, or orchestrate sequences of synthesis steps.
  - Potential hooks for external tools or generators that operate on Synth-level IR.

The Synth dialect is therefore designed as a *synthesis workbench* inside MLIR: it is where logic-level structures and decisions live, separate from higher-level behavioral IR and lower-level RTL or gate-level netlists.

---

## Relationship with Handshake and HW

In the Dynamatic flow, the Synth dialect is tightly coupled to Handshake and HW through the [Handshake-to-Synth lowering pass](ConversionHandshakeToSynth.md):

- Handshake operations are first converted to `hw.module`/`hw.instance` structures with flat ports.
- Each module’s behavior is described by synth operations inside the HW module, making Synth describe the logic function of each operation.
- Later passes can:
  - Refine `synth.subckt` into more detailed Synth operations (e.g., explicit AIG/MIG nodes).
  - Apply logic optimization and technology mapping in terms of Synth operations, with HW only providing the module/instance hierarchy and IO ports.

This separation of concerns allows Dynamatic to keep Handshake-level reasoning, module hierarchy (HW dialect), and synthesis-specific logic (Synth dialect) cleanly partitioned, while still enabling tight interaction through well-defined IR boundaries.


