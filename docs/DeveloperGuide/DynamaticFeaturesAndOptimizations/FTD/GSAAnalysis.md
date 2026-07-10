# GSA Analysis
## Introduction
In Static Single Assignment (SSA) form, every variable is assigned exactly once, and ϕ (phi) functions are introduced to merge values coming from different control flow paths. While SSA is powerful, it does not explicitly encode the control flow decisions that determine which value is actually chosen at runtime.

**Gated Single Assignment (GSA)** was introduced as an extension of SSA to make these control flow decisions explicit. Instead of a single generic ϕ merge, GSA introduces specialized gates:
- The **μ (mu) gate** appears at loop headers. It chooses between an initial value coming from outside the loop and a value produced inside the loop. The decision is driven by the loop’s condition: if the loop is starting, the initial value is used; if the loop is iterating, the loop value is used. In hardware, it is translated into a multiplexer whose select signal is driven through an INIT component. The INIT first emits an initial token, forcing the mux to select the outside-loop value for the first iteration, and then forwards the loop condition tokens for subsequent iterations.
- The **γ (gamma) gate** appears at the confluence point of an if–else structure. It selects between a true value and a false value depending on a condition signal. In hardware, this maps to a multiplexer controlled by the block’s branching condition.

This document focuses on the μ and γ gates implemented by Dynamatic’s GSA analysis.

For Dynamatic’s **Fast Token Delivery (FTD)** algorithm, having the program represented in GSA form is required. The MLIR `cf` dialect already encodes SSA behavior implicitly through block arguments: each predecessor branch passes values to the successor block arguments. These implicit ϕ gates must be translated into their GSA equivalents. During this translation, block arguments are first modeled as ϕ gates and then converted into μ or γ gates.

### Example
Consider the following control flow graph and its corresponding `cf_dyn_transformed.mlir` code.
- bb1 and bb3 both receive arguments from multiple predecessors. Implicit ϕ-gates are therefore placed in these blocks.

- The first argument of bb1 (%0) chooses between the initial value %c0 from bb0 and the loop-carried value %8 from bb3. This corresponds to a μ function.

- The second argument of bb1 (%1) is also updated inside the loop, so it too becomes a μ function.

- The argument of bb3 (%7) comes from two mutually exclusive control flow paths (bb1 or bb2). This corresponds to a γ function.

![if_loop_add_CFG](./Figures/if_loop_add_CFG.png)

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

2. Convert eligible loop-header ϕ gates into μ gates

3. Convert remaining ϕ gates into γ gates.

The following sections explain these steps in order.

## Identify Implicit ϕ Gates
In the `convertSSAToGSA` function, the first step is to convert all block arguments in the IR into ϕ gates, carefully extracting information about their producers and senders. Later, these ϕ gates are transformed into either γ or μ gates.

Before describing the conversion process, it is important to understand the data structures used to represent gates and their inputs.

### Gate and GateInput Structures

#### Gate Structure

A **Gate** represents a logical or control flow construct in the GSA form. Each gate encapsulates:

- The value it produces,

- The inputs it depends on,

- Its gate type (ϕ, γ, or μ),

- Control flow and predicate information,

- And a unique identity within the region.

Below is a simplified version of the Gate structure used in the implementation:
```
struct Gate {
  Value result;                          // Value produced by this gate
  SmallVector<GateInput *> operands;     // Inputs of this gate
  GateType gsaGateFunction;              // Type: PhiGate, GammaGate, or MuGate
  Block *conditionBlock;                 // Block driving the gate’s condition
  boolean::BoolExpression *condition;    // Boolean condition of the gate
  std::vector<std::string> cofactorList; // Condition cofactors
  Block *gateBlock;                      // Block where the gate is located
  bool muGenerated;                      // True if generated from MU expansion
  unsigned index;                        // Unique gate index
  bool isRoot;                           // True if it’s a root gate
};
```
##### Additional Details on Fields:
`conditionBlock`

This field identifies the block whose terminator provides the control predicate for the gate.

- For γ gates, it is the producer block of the condition that determines which input the γ selects.

- For μ gates, it refers to the producer of the unique loop exit condition (the condition that decides when the loop terminates).

- For ϕ gates and μ gates with multiple exits, this field is nullptr, as no single condition block drives their creation.


`condition`

A Boolean expression representing the logical condition associated with the gate.

- For γ gates, it corresponds directly to the predicate that decides which branch or input is selected.

- For μ gates, it is the negation of the loop exit condition, expressing the continuation of the loop.

- For ϕ gates, it is initialized to a constant `BoolZero()` (a neutral condition), since ϕ gates only merge data and are not controlled by predicates.

`cofactorList`

A list of string identifiers corresponding to condition cofactors.

These are useful when decomposing complex Boolean expressions into simpler components.

`gateBlock`

The block where this gate is logically placed.

In most cases, `gateBlock` coincides with the block containing the gate’s output value (`result`).

However, for γ gates generated during μ expansion, the gate is placed in the same block as its `conditionBlock`, following the rules described in [Placement rule section](#5-placement-rule).
This ensures that the γ is evaluated under the same control context as its condition.

`muGenerated`<!--TODO: check the new implementation , why mu generated is needed or not needed anymore"-->

A flag that indicates whether this gate was created during the `convertPhiToMu` phase, specifically to resolve multiple loop-input cases.

This flag later determines where the corresponding γ gates should be placed, as their `gateBlock` is adjusted according to the μ placement logic (see [Placement rule section](#5-placement-rule) for details).

`isRoot`

Specifies whether the gate is the root of its gate tree.

- All μ gates are considered roots.

- Only the base γ in a γ-tree is marked as root.

- ϕ gates do not require root tracking; the field is 
initialized as false for them.

#### GateInput Structure

A `GateInput` represents a single input to a gate.
Each input can be:

- a `Value` which is result of an operation,

- output of another `Gate`, or

- empty (if not yet connected, e.g., a missing ϕ).

It also keeps track of all the **sender blocks** (i.e., CFG predecessors that forward this input to the gate).

This distinction between producers and senders is crucial:

- The producer is where the value (or gate result) is originally defined.

- The sender is the block whose terminator forwards that value to the gate. For control dependent analysis (like γ construction), predicates must be computed with respect to the sender blocks (check "[Sender filtering for operands from the same block](#sender-filtering-for-operands-from-the-same-block)" for more information).


Structure Summary:
```
struct GateInput {
  std::variant<Value, Gate *> input;        // The actual input: IR value, gate, or empty.
  std::unordered_set<Block *> senders;      // CFG blocks forwarding this input.
};
```

---

With the `Gate` and `GateInput` structures defined, we can now outline how implicit ϕ gates are identified. This process scans each block in the region, examines its block arguments, and creates corresponding ϕ gates by linking them to their incoming values and sender blocks.

Note: If there is only one block in the region being checked, nothing needs to be done since there is no possibility of multiple assignments.

For each block argument, the analysis builds the operand list of one potential ϕ gate. During this scan:

- `gatesPerBlock` keeps the gates discovered for each block. Each new ϕ gate is inserted into the entry of the block that owns the block argument.

- `operands` contains the inputs collected for the ϕ currently being built.

- `coveredPredecessors` prevents the same predecessor block from being processed more than once for the same block argument.

- `gateInputList` stores every allocated `GateInput` so the analysis can deallocate them later.

### Missing ϕ Inputs

The input of a ϕ gate can itself be another ϕ. This happens when the input comes from a block argument of another block, excluding the entry block. In this case, the ϕ input cannot be connected immediately. Instead, it is marked as missing and the necessary information is stored. After all ϕ gates are extracted, these missing inputs are revisited and the connections are reconstructed.

Each missing ϕ input stores:

- `pi`: the temporary `GateInput` that must later point to the source ϕ.

- `blockArg`: the block argument whose corresponding ϕ should feed that input.

The missing input is recorded in two places:

- `phisToConnect`: the global list of missing ϕ inputs to reconnect after all block arguments have been scanned.

- `operandsMissPhi`: the missing ϕ inputs of the current ϕ, used to detect duplicate missing inputs while processing the same block argument.

In pseudo-code, the process is:
```
For each block in the region:
  Initialize an empty gate list in `gatesPerBlock`.

  For each argument of the block:
    Treat this argument as the result of a potential ϕ.
    Initialize empty `operands`, `operandsMissPhi`, and `coveredPredecessors`.

    For each predecessor of the block:
      Skip the predecessor if it was already processed.
      Identify the branch terminator that jumps into the block.
      Extract the value passed to the argument.

      If the value is a block argument and its parent block has predecessors (i.e., it is not the entry block):
        This value is the output of another ϕ.
        If this missing input is new:
          Create a temporary `GateInput`.
          Record it in `phisToConnect` and `operandsMissPhi`.
        Otherwise:
          Reuse the existing missing input.
      Else:
        This value is a direct SSA input.
        If this value input is new:
          Create a `GateInput` for the value.
        Otherwise:
          Reuse the existing value input.

      For the new or reused input:
        Add the predecessor block to its `senders` set.
      For a newly created input:
        Add the input to `gateInputList`.
        Add the input to `operands`.
    
    After all predecessors are processed:
      If `operands` is not empty (the ϕ has at least one input):
        Create the ϕ gate and insert it into `gatesPerBlock`.
```
After all ϕ gates are extracted, the analysis revisits the missing inputs recorded in `phisToConnect`. For each missing input, it looks up the ϕ gate associated with the recorded block argument and updates the temporary `GateInput` to point to that gate.

### `isBlockArgAlreadyPresent` and `isValueAlreadyPresent`

These two helper functions avoid recording the same ϕ input more than once.

- `isValueAlreadyPresent`

  This helper is used for direct SSA values. It checks whether the same value is already present in the current `operands` list.

  If the value is already present, the analysis reuses the existing `GateInput` and adds the current predecessor (`pred`) to its `senders` set.

- `isBlockArgAlreadyPresent`

  This helper is used for missing ϕ inputs. Since the input has not yet been connected to its source ϕ gate, the analysis compares the recorded block argument instead.

  Two missing ϕ inputs are considered the same if they refer to the same block argument: same parent block and same argument number.

  If the block argument is already present, the analysis reuses the existing temporary `GateInput` and adds the current predecessor (`pred`) to its `senders` set.

In both cases, the purpose is the same: if the same logical input reaches the ϕ through multiple CFG predecessors, the analysis keeps one `GateInput` and records all of those predecessors as senders.

## Convert ϕ Gates into μ Gates

### Identifying μ Candidates
A ϕ gate is classified as a μ gate if the following conditions hold:

1. It is inside a loop.

2. It has at least two operands.

3. It is located in the loop header.

### Grouping Initial and Loop Inputs

Once a candidate μ is identified, its operands (inputs) are divided into two groups:

- **Loop inputs:** values produced inside the same loop as the ϕ.

- **Initial inputs:** values originating from outside the loop.

#### Notice: Inputs from Nested Loops
`CFGLoopInfo::getLoopFor(block)` returns the innermost loop containing a block. Because of this, inputs from nested loops might be mistakenly recognized as “initial inputs” instead of loop inputs if only the innermost loop is checked.

*Example:* in the CFG below, an input value from block `bb3` may appear to belong to a different loop than the one containing `bb1` (the ϕ’s loop). To prevent this, the `IsBlockInLoop` function checks whether any parent loop of the input matches the ϕ’s loop.

![gemm_CFG](./Figures/gemm_CFG.png)

### Updating the ϕ into a μ

A valid μ gate must have exactly two inputs: one from outside the loop and one from inside the loop. The grouping step above ensures we can identify these two roles.

- If either group is empty → the ϕ cannot be a μ.

- If a group has exactly one member → that value becomes the corresponding μ input (loop or initial).

- If a group has multiple members → an intermediate ϕ is created in the block to merge them, and the output of this generated ϕ becomes the corresponding μ input. This extra ϕ will later be replaced by a γ (or tree of γs) during the ϕ to γ conversion phase.

The implementation does not create a new μ gate. Instead, it updates the original ϕ gate in place by changing its type to μ and replacing its operands with exactly two inputs: the initial input first, then the loop-carried input.

### Condition of the μ Gate

The μ gate outputs its initial value during the first iteration of the loop. On subsequent iterations, if the loop continues (i.e., the exit condition is false), it selects the loop-generated value. When the loop finally exits, the initial input will be used again if the loop is re-entered.

Therefore, the condition of a μ gate is defined as the **negation of the loop exit condition**.

Later, during FTD implementation, the μ gate is replaced by a MUX whose condition comes from an INIT. The INIT functionality can be implemented either as a merge between a constant and the loop’s iterating condition, or as a one-entry buffer preloaded with an initial value. In both cases, the INIT first drives the MUX to select the initial input. After that, it forwards the loop condition, so the MUX selects between the initial input and the loop-carried input based on whether the loop exits or continues.

#### Note:
The `getLoopExitCondition` function computes the overall exit condition by OR-ing the conditions of all loop exiting blocks. This function relies on `getBlockLoopExitCondition`, which computes the exit condition for a single block.

Eventually, this OR-ing is between operations that produce different token counts; therefore, it gets implemented in the **FTDConversion** pass using **Shannon's expansion**.
<!--TODO: check with aya if it is still the case-->

## Convert ϕ Gates into γ Gates

All remaining ϕ gates (i.e., those not turned into μ gates) must be converted into γ gates.
However, a single γ gate is only a **two-input multiplexer**, while a ϕ can have multiple inputs.
To handle this, we build a tree of γ gates, each driven by a simple condition.
The following steps describe the process.

### Step 1. Input Ordering
The inputs of a ϕ are sorted based on the dominance relationship between their originating basic blocks.

- If block Bi dominates block Bj, then the input from Bi is placed before Bj.

- This ordering does not affect the semantics of the ϕ (ϕ is order-less), but it simplifies later analysis.

### Step 2. Find Common Dominator

Find **the nearest common dominator** among all input blocks of the ϕ.

This block will be used as the root for path exploration in the next step.

### Step 3. Path Identification

For each input operand:

- Find all paths from the common dominator to the ϕ’s block that **pass through the operand’s block** but **avoid later operand blocks** that come later in the dominance order.

- Paths are explored through the `findAllPaths` function, which calls `dfsAllPaths` to:

  - finds all possible paths between two blocks,

  - avoids certain blocks,

  - and allows a block to be revisited only if it is both the start and end (for loop cases).

<!--TODO: add a simple example for the paths and 1.why revisit start and end, and 2.why avoid later operand blocks(example from Giacomo thesis) -->

#### Sender filtering for operands from the same block

If one block produces multiple values (operands) for the same ϕ, the DFS initially gives them identical paths.

This is the case where producer and sender must be distinguished. The producer block tells us where the value was defined, but the sender block tells us which CFG predecessor actually passed that value to the ϕ. During path search, `operand->getBlock()` returns the producer block, so two operands produced in the same block can initially receive the same candidate paths. The `senders` set then filters these paths by requiring the block immediately before the ϕ block to be one of the recorded senders for that operand.

For example, in the cf_dyn_transformed.mlir snippet below, which corresponds to the shown CFG:

```
module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    ...
    cf.br ^bb1(%arg0, %c0_i32, %true : i32, i32, i1) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i32, %2: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    ...
    cf.cond_br %4, ^bb2(%0, %1 : i32, i32), ^bb6 {handshake.name = "cond_br0"}
  ^bb2(%5: i32, %6: i32):  // pred: ^bb1
    ...
    %8 = arith.shrsi %7, %c1_i32 {handshake.name = "shrsi0"} : i32
    ...
    cf.cond_br %14, ^bb1(%8, %6, %15 : i32, i32, i1), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    ...
    cf.cond_br %16, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    ...
    cf.br ^bb1(%5, %17, %15 : i32, i32, i1) {handshake.name = "br1"}
  ^bb5:  // pred: ^bb3
    ...
    %18 = arith.addi %8, %c-1_i32 {handshake.name = "addi2"} : i32
    cf.br ^bb1(%18, %6, %15 : i32, i32, i1) {handshake.name = "br7"}
  ^bb6:  // pred: ^bb1
    return {handshake.name = "return0"} %0 : i32
  }
}

```
![iterative_sqrt_CFG](./Figures/iterative_sqrt_CFG.png)

Blocks `bb0`, `bb2`, `bb4`, and `bb5` send values to argument 0 of `bb1`.
Notice that `%5` and `%8` are both produced in `bb2`, but they are forwarded to `bb1` by different senders: `%8` is sent by `bb2`, while `%5` is sent by `bb4`.

When exploring paths, the DFS detects two possible routes from the producer `bb2` to the consumer `bb1`:
`{bb2, bb1}` and `{bb2, bb3, bb4, bb1}`.

However, the sender information disambiguates the two operands:

- The sender of `%8` is `bb2`, so only `{bb2, bb1}` is a valid path.

- The sender of `%5` is `bb4`, so only `{bb2, bb3, bb4, bb1}` is valid.

The implementation therefore filters paths by the sender block (the block immediately before the ϕ in the path).
Only paths whose sender matches the operand’s recorded sender are kept.

### Step 4. Boolean Conditions
<!--TODO: check if it can be described in a better clearer way-->
<!--TODO: it seems that condition and expression and cofactore are used instead of each other make it clear which one is which and what do they mean. in theis ection and next section (build gamma tree)-->

For each operand, compute a Boolean expression representing when that operand is chosen:

- The condition of a path = AND of all branch conditions along that path.

- The condition of an operand = OR of the conditions of all valid paths.

- The resulting Boolean expressions are minimized.

All Boolean conditions (cofactors) are collected and sorted by the index of their originating block.

### Step 5. Build the γ Tree

The `expandGammaTree` function takes a ϕ gate (with its inputs and Boolean conditions) and recursively builds a binary tree of γ gates.
Each γ is a two-input MUX driven by one simple Boolean condition.

The process works as follows:

#### 1. Pick a cofactor (condition):

The function starts from the queue of cofactors (i.e., the Boolean conditions collected and sorted in the previous step). Since they are ordered by block index, the first cofactor we take is guaranteed to be common to all input expressions (because the blocks associated with it dominate the others). This ensures that splitting on this cofactor applies consistently across all inputs.

#### 2. Split expressions by condition:

For each input expression (operand + condition):

- Restrict the Boolean expression once assuming the cofactor = `true`.

- Restrict it again assuming the cofactor = `false`.

- Add the non-zero result(s) to either `conditionsTrueExpressions` or `conditionsFalseExpressions`.

#### 3. Build γ inputs:

Now we decide what should feed the true and false inputs of the γ gate being built:

- For each condition outcome (`conditionsTrueExpressions` or `conditionsFalseExpressions`), check how many expressions it contains.

- If it contains **more than one expression**, this means multiple operands could be selected under that branch of the condition. To resolve this, we recursively call `expandGammaTree` on that subset. The resulting γ gate from the recursion becomes the input of the current γ.

- If it contains `exactly one expression`, its operand is directly assigned as the input of the current γ.

- If it contains `no expressions`, that outcome of the condition is never taken, and an empty input is created.

#### 4. Create the γ gate:

A new γ is generated:

- Its inputs are the “true” and “false” operands from the step above.

- Its condition is the cofactor currently being expanded.

- Internally, its output is temporarily set to the original ϕ’s result. If the γ is not the root, this output will later become a “true” or “false” input of another γ, and the connection is updated when that parent γ is created.

#### 5. Placement rule:

Normally, new γ gates are placed in the **same block as the original ϕ**.

However, there is one special case: when the ϕ was introduced during `convertPhiToMu` to resolve multiple loop-carried inputs. These temporary ϕs (marked with muGenerated) cannot have their γ replacements placed in the loop header.

**Why is this a problem?**

When we later run direct path analysis for control dependencies, the control signal that drives such a γ is generated inside the loop body.
If we place the γ in the loop header:

- The γ would appear before its control signal producer.

- In the direct path search from the function entry (bb0) to the γ, we would never encounter the block that generates the control signal.


**The fix:**

For γs created from these muGenerated phis, we instead place them **in the same block as their condition producer**. 
<!--TODO: check if it is still the case in code-->

### Step 6. Reconnect Uses

Once a ϕ is replaced by its γ tree:

- All gates that previously used the ϕ’s output are updated to use the root γ gate instead.

---
### Example:

To illustrate the conversion of a ϕ gate into a tree of γ gates, consider the following reduced MLIR function:
```
module {
  func.func @example(%arg0: memref<8xi32> {handshake.arg_name = "a"}) -> i32 {
    ...
    %c0_i32 = arith.constant {handshake.name = "constant2"} 0 : i32
    cf.cond_br %2, ^bb1, ^bb3(%c0_i32, %c2_i32 : i32, i32) {handshake.name = "cond_br0"}

  ^bb1:  // pred: ^bb0
    ...
    %c0_i32_0 = arith.constant {handshake.name = "constant10"} 0 : i32
    cf.cond_br %4, ^bb3(%c0_i32_0, %5 : i32, i32), ^bb2 {handshake.name = "cond_br1"}

  ^bb2:  // pred: ^bb1
    ...
    %9 = arith.select %8, %c5_i32_2, %c0_i32_3 {handshake.name = "select1"} : i32
    cf.br ^bb3(%9, %5 : i32, i32) {handshake.name = "br3"}

  ^bb3(%10: i32, %11: i32):  // 3 preds: ^bb0, ^bb1, ^bb2
    %12 = arith.addi %11, %10 {handshake.name = "addi0"} : i32
    return {handshake.name = "return0"} %12 : i32
  }
}
```
The corresponding control flow graph is shown below:

![gamma_example](./Figures/loop_multiply_CFG.png)

We will convert the ϕ in block bb3 (which merges values coming from bb0, bb1, and bb2) into an equivalent γ tree.

### Step 1. Input Ordering

The inputs from `bb0`, `bb1`, and `bb2` (namely `%c0_i32`, `%c0_i32_0`, and `%9`) are referred to as **x₀**, **x₁**, and **x₂**, respectively.  
After dominance-based ordering, we have:  
`x₀ (bb0)`, `x₁ (bb1)`, `x₂ (bb2)`.


### Step 2. Find Common Dominator

The nearest common dominator of all three input blocks is **bb0**.



### Step 3. Path Identification

For each input, all valid paths from the common dominator (`bb0`) to the ϕ’s block (`bb3`) are identified, while avoiding blocks corresponding to **later operand producers**:

- **x₀:** `{bb0, bb3}`  *(must not pass through `bb1` or `bb2`, which produce later operands)*  
- **x₁:** `{bb0, bb1, bb3}`  *(must not pass through `bb2`, the later operand producer)*  
- **x₂:** `{bb0, bb1, bb2, bb3}`  

Next, we filter paths by the **sender block** (the block immediately before the ϕ) to ensure correct operand assignment:

- `%c0_i32` (x₀) sender: `bb0` → path `{bb0, bb3}`  
- `%c0_i32_0` (x₁) sender: `bb1` → path `{bb0, bb1, bb3}`  
- `%9` (x₂) sender: `bb2` → path `{bb0, bb1, bb2, bb3}`


### Step 4. Boolean Conditions

Compute a Boolean expression for each operand:

- **x₀:** `!c0`  
- **x₁:** `c0 & c1`  
- **x₂:** `c0 & !c1`  

These expressions indicate under which conditions each input is selected.


### Step 5. Build the γ Tree

1. **Pick the first cofactor (c0):**  
   - Both x₁ and x₂ remain non-zero when `c0 = true` → more than one expression → recursive `expandGammaTree` call needed.  

2. **Second cofactor (c1) inside recursion:**  
   - x₁: non-zero for `c1 = true` → true input  
   - x₂: non-zero for `c1 = false` → false input  

   Resulting γ gate: `γ(c1, x2, x1)`  
   - Condition: c1  
   - True input: x1  
   - False input: x2  

3. **Top-level γ gate:**  `γ(c0, x0, γ(c1, x2, x1))`
   - Condition: c0  
   - True input: γ(c1, x2, x1) (from recursion)  
   - False input: x0  

This γ gate becomes the **root** of the tree.



### Step 6. Connect Uses

All uses of the original ϕ are updated to use the root γ gate instead.
