# GSA Analysis
## Introduction
In Static Single Assignment (SSA) form, every variable is assigned exactly once, and ϕ (phi) functions are introduced to merge values coming from different control flow paths. While SSA is powerful, it does not explicitly encode the control-flow decisions that determine which value is actually chosen at runtime.

**Gated Single Assignment (GSA)** was introduced as an extension of SSA to make these control-flow decisions explicit. Instead of a single generic ϕ merge, GSA introduces specialized gates:
- The **μ (mu) gate** appears at loop headers. It chooses between an initial value coming from outside the loop and a value produced inside the loop. The decision is driven by the loop’s condition: if the loop is starting, the initial value is used; if the loop is iterating, the loop value is used.
- The **γ (gamma) gate** replaces a ϕ at control-flow merges. It selects between a true value and a false value depending on a condition signal (like at the end of an if–else). In hardware, this becomes a multiplexer.

For Dynamatic’s **Fast Token Delivery (FTD)** algorithm, having the program represented in GSA form is required. The MLIR cf dialect already provides ϕ-gates in SSA form, but these must be translated into their GSA equivalents. During this translation, every block argument(pottential ϕ) in the control-flow is rewritten as either a μ or a γ gate.

### Example
Consider the following control-flow graph and its corresponding `cf_dyn_transformed.mlir` code.
- bb1 and bb3 both receive arguments from multiple predecessors. Implicit ϕ-gates are therefore placed in these blocks.

- The first argument of bb1 (%0) chooses between the initial value %c0 from bb0 and the loop-carried value %8 from bb3. This corresponds to a μ function.

- The second argument of bb1 (%1) is also updated inside the loop, so it too becomes a μ function.

- The argument of bb3 (%7) comes from two mutually exclusive control-flow paths (bb1 or bb2). This corresponds to a γ function.

![CFG](./Figures/if_loop_add_CFG.png)

```
module {
  func.func @if_loop_add(%arg0: memref<1000xf32> {handshake.arg_name = "a"}, %arg1: memref<1000xf32> {handshake.arg_name = "b"}) -> f32 {
    %c0 = arith.constant {handshake.name = "constant2"} 0 : index
    %cst = arith.constant {handshake.name = "constant3"} 0.000000e+00 : f32
    cf.br ^bb1(%c0, %cst : index, f32) {handshake.name = "br0"}
  ^bb1(%0: index, %1: f32):  // 2 preds: ^bb0, ^bb3
    %cst_0 = arith.constant {handshake.name = "constant4"} 0.000000e+00 : f32
    %2 = memref.load %arg0[%0] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : memref<1000xf32>
    %3 = memref.load %arg1[%0] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : memref<1000xf32>
    %4 = arith.subf %2, %3 {handshake.name = "subf0"} : f32
    %5 = arith.cmpf oge, %4, %cst_0 {handshake.name = "cmpf0"} : f32
    cf.cond_br %5, ^bb2, ^bb3(%1 : f32) {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %6 = arith.addf %1, %4 {handshake.name = "addf0"} : f32
    cf.br ^bb3(%6 : f32) {handshake.name = "br1"}
  ^bb3(%7: f32):  // 2 preds: ^bb1, ^bb2
    %c1000 = arith.constant {handshake.name = "constant5"} 1000 : index
    %c1 = arith.constant {handshake.name = "constant6"} 1 : index
    %8 = arith.addi %0, %c1 {handshake.name = "addi0"} : index
    %9 = arith.cmpi ult, %8, %c1000 {handshake.name = "cmpi0"} : index
    cf.cond_br %9, ^bb1(%8, %7 : index, f32), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"} %7 : f32
  }
}
```
### Translation Process
The conversion from SSA to GSA is done in three main steps:

1. Identify implicit ϕ gates introduced by SSA form.

2. Convert ϕ gates into μ gates

3. Convert remaining ϕ gates into γ gates.

## Identify Implicit ϕ Gates
In the `convertSSAToGSA` function, the first step is to convert all block arguments in the IR into ϕ gates, carefully extracting information about their producers and senders. Later, these ϕ gates are transformed into either γ or μ gates. In this section, we focus on the details of this first step.

Note: If there is only one block in the region being checked, nothing needs to be done since there is no possibility of multiple assignments.

In pseudo-code, the process looks like this:
```
For each block in the region:
  For each argument of the block:
    → treat this argument as a potential ϕ.

    For each predecessor of the block:
      Identify the branch terminator that jumps into the block.
      Extract the value passed to the argument.

      If the value is a block argument and its parent block has predecessors(so its parent is not bb0):
        → this value is itself the output of another ϕ.
        Record it as a “missing phi” to be connected later.
      Else:
        → the value is a plain input and can be added directly.

      In both cases, check if the value is already recorded:
        - `isBlockArgAlreadyPresent` checks block arguments.
        - `isValueAlreadyPresent` checks plain SSA values.

      If the value is new:
        - Wrap it in a `gateInput` structure.
        - If it is a missing phi:
            * Add it to `phisToConnect` (records phis that need reconnection later).
            * Add it to `operandsMissPhi` (helps `isBlockArgAlreadyPresent` detect duplicates).
        - Add the predecessor block to the `senders` list of this gate input.
        - Add the gate input to `gateInputList` (the global list of all gate inputs).
        - Add the gate input to `operands` (the inputs of the current ϕ).
    
    After all predecessors are processed:
      If `operands` is not empty (the ϕ has at least one input):
        → create the ϕ gate and associate it with the block.
```
After all ϕ gates are created, the final step is to connect the missing inputs recorded in phisToConnect. 

## Convert ϕ Gates into μ Gates

## Convert ϕ Gates into γ Gates