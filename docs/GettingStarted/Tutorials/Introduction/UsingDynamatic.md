# Using Dynamatic
> [!NOTE]
> Before moving forward with this section, ensure that you have installed all necessary dependencies and built Dynamatic. If not, follow the [simple build instructions](../../InstallDynamatic.md).

This section covers:
- how to use Dynamatic
- constructs to include and invalid C/C++ features (see [Kernel Code Guidelines](../../../UserGuide/KernelCodeGuideLines.md))
- Dynamatic commands and respective flags.

## Introduction to Dynamatic
> [!NOTE]  
> The virtual machine does not contain an MILP solver (Gurobi). Unfortunately, this will affect the circuits you generate as part of the exercises and you may obtain different results from what the tutorial describes.

This tutorial guides you through the
- compilation of a simple kernel function written in C into an equivalent VHDL design
- functional verification of the resulting dataflow circuit using Modelsim
- visualization of the circuit using our custom interactive dataflow visualizer. 

The tutorial assumes basic knowledge of dataflow circuits but does not require any insight into MLIR or compilers in general.

Below are some technical details about this tutorial.
- All resources are located in the repository's [tutorials/Introduction/Ch1](https://github.com/EPFL-LAP/dynamatic/tree/main/tutorials/Introduction/Ch1) folder.
- All relative paths mentionned throughout the tutorial are assumed to start at Dynamatic's top-level folder.

This tutorial is divided into the following sections:
1. [The Source Code](#the-source-code) | The C kernel function we will transform into a dataflow circuit.
2. [Using Dynamatic's Frontend](#using-dynamatics-frontend) | We use the Dynamatic frontend to compile the C function into an equivalent VHDL design, and functionally verify the latter using Modelsim.
3. [Visualizing the Resulting Dataflow Circuit](#visualize-the-resulting-datafow-circuit) | We visualize the execution of the generated dataflow circuit on test inputs
4. [Conclusion](#conclusion) | We reflect on everything we just accomplished

## The C Source Code
Below is our target C function (the kernel, in Dynamic HLS jargon) for conversion into a dataflow circuit:
```c
// The number of loop iterations
#define N 8

// The kernel under consideration
unsigned loop_multiply(int a[N]) {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * x;
  }
  return x;
}
```
This kernel: 
- multiplies a number by itself at each iteration of a loop from 0 to any number N where the corresponding element of an array equals 0.
- returns the calculated value after the loop exits. 

> [!TIP]
> This function is purposefully simple so that it corresponds to a small dataflow circuit that will be easier to visually explore later on. Dynamatic is capable of transforming much more complex functions into fast and functional dataflow circuits.

You can find the source code of this function in `tutorials/Introduction/Ch1/loop_multiply.c`. 

Observe!
- The `main` function in the file allows one to run the C kernel with user-provided arguments. 
- The `CALL_KERNEL` macro in `main`'s body calls the kernel while allowing us to automatically run code prior to and/or after the call. This is used during C/VHDL co-verification to automatically write the C function's reference output to a file for comparison with the generated VHDL design's output.
```c
int main(void) {
  in_int_t a[N];
  // Initialize a to [0, 1, 0, 1, ...]
  for (unsigned i = 0; i < N; ++i)
    a[i] = i % 2;
  CALL_KERNEL(loop_multiply, a);
  return 0;
}
```

## Using Dynamatic's Frontend
Dynamatic's frontend is built by the project in `build/bin/dynamatic`, with a symbolic link located at `bin/dynamatic`, which we will be using. In a terminal, from Dynamatic's top-level folder, run the following:
```sh
./bin/dynamatic
```
This will print the frontend's header and display a prompt where you can start inputting commands.
```
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
======================== EPFL-LAP - v2.0.0 | March 2024 ========================
================================================================================


dynamatic> # Input your command here
```
### `set-src`
Provide Dynamatic with the path to the C source code file under consideration. Ours is located at `tutorials/Introduction/Ch1/loop_multiply.c`, thus we input:
```sh
dynamatic> set-src tutorials/Introduction/Ch1/loop_multiply.c
```
> [!NOTE]
> The frontend will assume that the C function to transform has the same name as the last component of the argument to `set-src` without the file extension, here `loop_multiply`.

### `compile`
The first step towards generating the VHDL design is compilation. Here, 
- the C source goes through our MLIR frontend ([Polygeist](https://github.com/llvm/Polygeist))
- traverses a pre-defined sequence of transformation and optimization passes that ultimately yield a description of an equivalent dataflow circuit. 

That description takes the form of a human-readable and machine-parsable IR (Intermediate Representation) within the MLIR framework. It represents dataflow components using specially-defined IR instructions (in MLIR jargon, [operations](../../../DeveloperGuide/MLIRPrimer.md#operations)) that are part of the [Handshake dialect](../../../DeveloperGuide/CompilerIntrinsics/MLIRPrimer.md#dialects). 
> [!TIP]  
> A dialect is simply a collection of logically-connected IR entities like instructions, types, and attributes. 

MLIR provides standard dialects for common usecases, while allowing external tools (like Dynamatic) to define custom dialects to model domain-specific semantics. 

To compile the C function, simply input `compile`. This will call a shell script `compile.sh` (located at `tools/dynamatic/scripts/compile.sh`) in the background that will iteratively transform the IR into an optimized dataflow circuit, storing intermediate IR forms to disk at multiple points in the process.
```sh
dynamatic> set-src tutorials/Introduction/Ch1/loop_multiply.c
dynamatic> compile
```
#### Compile Flags
The compile flags are all optional and defaulted to no value.  
`--sharing` enables credit-based resource sharing  
`--buffer-algorithm` lets the compiler know which **smart buffer** placement algorithm to use. Requires Gurobi to solve MILP problems. There are two available options for this flag:  
- **fpga20**: throughput-driven buffering
- **fpl22** : throughput- and timing-driven buffering  

The default for compile is to use the minimum buffering for correctness (simple buffer placement)
| flag | function| options|
|------|---------|--------|
| --sharing | use credit-based resource shaing|None|
| --buffer-alogithm | Indicate buffer placement algorithm to use, values are 'on merges' |fpga20, fpl22|

> [!WARNING]  
> `compile` requires a MILP solver (Gurobi) for smart buffer placement. If you don't have Gurobi, abstain from using the `--buffer-algorithm` flag

You should see the following printed on the terminal after running `compile`:
```sh
...
dynamatic> compile
[INFO] Compiled source to affine
[INFO] Ran memory analysis
[INFO] Compiled affine to scf
[INFO] Compiled scf to cf
[INFO] Applied standard transformations to cf
[INFO] Applied Dynamatic transformations to cf
[INFO] Compiled cf to handshake
[INFO] Applied transformations to handshake
[INFO] Running simple buffer placement (on-merges).
[INFO] Placed simple buffers
[INFO] Canonicalized handshake
[INFO] Created loop_multiply DOT
[INFO] Converted loop_multiply DOT to PNG
[INFO] Created loop_multiply_CFG DOT
[INFO] Converted loop_multiply_CFG DOT to PNG
[INFO] Lowered to HW
[INFO] Compilation succeeded
```
After successful compilation, all results are placed in a folder named `out/comp` created next to the C source file under consideration. In this case, it is located at `tutorials/Introduction/Ch1/out/comp`. It is not necessary that you look inside this folder for this tutorial.  

> [!NOTE]  
> A DOT file and equivalent PNG of the resulting circuit is generated after compilation (`kernel_name.dot` and `kernel_name.png`) and can be visualized using a DOT file reader or image viewer without installing the interactive visualizer.

In addition to the final optimized version of the IR (in `tutorials/Introduction/Ch1/out/comp/handshake_export.mlir`), the compilation script generates an equivalent Graphviz-formatted file (`tutorials/Introduction/Ch1/out/comp/loop_multiply.dot`) which serves as input to our VHDL backend, which we call using the `write-hdl` command.

### `write-hdl`
This command converts the `.dot` file generated from compilation to the equivalent hardware description language implementation of our kernel. 
```sh
...
[INFO] Compilation succeeded

dynamatic> write-hdl
[INFO] Exported RTL (vhdl)
[INFO] HDL generation succeeded
```
> [!NOTE]
> By default, the command generates VHDL implementations. This can be changed to verilog using the `--hdl` flag with the value `verilog`

Similarly to compile, this creates a folder `out/hdl` with a `loop_multiply.vhd` file and all other `.vhd` files necessary for correct functioning of the circuit. This design can finally be co-simulated along the C function on Modelsim to verify that their behavior matches using the `simulate` command.

#### `simulate`
This command generates a testbench from the generated HDL code and feeds it inputs from the `main` function of our C code. It then runs a cosimulation of the C program and VHDL testbench to determine whether they yield the same results.
```sh
...
[INFO] HDL generation succeeded

dynamatic> simulate
[INFO] Built kernel for IO gen.
[INFO] Ran kernel for IO gen.
[INFO] Launching Modelsim simulation
[INFO] Simulation succeeded

```
The command creates a new folder `out/sim`. In this case, it is located at `tutorials/Introduction/Ch1/out/sim`. While it is not necessary that you look inside this folder for this tutorial, just know that it contains everything necessary to co-simulate the design:
- input C function
- VHDL entity values
- auto-generated testbench
- full implementation of all dataflow components, etc. 
- everything generated by the co-simulation process (output C function and VHDL entitiy values, VHDL compilation logs, full waveform).

`[INFO] Simulation succeeded` indicates that the C function and VHDL design showcased the same behavior. This just means that 
- their return values were the same after execution on kernel inputs computed in the `main` function. 
- if any arguments were pointers to memory regions, `simulate` also checked that the states of these memories are the same after the C kernel call and VHDL simulation.


That's it, you have successfully synthesized your first dataflow circuit from C code and functionally verified it using Dynamatic!

At this point, you can quit the Dynamatic frontend by inputting the exit command:
```sh
...
[INFO] Simulation succeeded

dynamatic> exit

Goodbye!
```
If you would like to re-run these commands all at once, it is possible to use the frontend in a non-interactive way by writing the sequence of commands you would like to run in a file and referencing it when launching the frontend. One such file has already been created for you at `tutorials/Introduction/Ch1/frontend-script.dyn`. You can replay this whole section by running the following from Dynamatic's top-level folder.
```
./bin/dynamatic --run tutorials/Introduction/Ch1/frontend-script.dyn
```
### `visualize`
> [!NOTE]  
> To use the visualize command, you will need to go through the [interactive dataflow visualizer](../../../UserGuide/AdvancedBuild.md#4-interactive-dataflow-circuit-visualizer) section in the Advanced Build section first.  

At the end of the last section, you used the `simulate` command to co-simulate the VHDL design obtained from the compilation flow along with the C source. This process generated a waveform file at `tutorials/Introduction/Ch1/out/sim/HLS_VERIFY/vsim.wlf` containing all state transitions that happened during simulation for all signals. After a simple pre-processing step we will be able to visualize these transitions on a graphical representation of our circuit to get more insights into how our dataflow circuit behaves.

To launch the visualizer, re-open the frontend, re-set the source with `set-src tutorials/Introduction/Ch1/loop_multiply.c`, and input the `visualize` command.
```sh
$ ./bin/dynamatic
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
==================== EPFL-LAP - <release> | <release-date> =====================
================================================================================

dynamatic> set-src tutorials/Introduction/Ch1/loop_multiply.c
dynamatic> visualize
[INFO] Generated channel changes
[INFO] Added positioning info. to DOT

dynamatic> exit

Goodbye!
```
> [!TIP]
> We do not have to re-run the previous synthesis steps because the data expected by the `visualize` command is still present on disk in the output folders generated by `compile` and `simulate`.  

`visualize` creates a folder `out/visual` next to the source file (in `tutorials/Introduction/Ch1/out/visual`) containing the data expected by the visualizer as input.

You should now see a visual representation of the dataflow circuit you just synthesized. It is basically a graph, where each node represents some kind of dataflow component and each directed edge represents a dataflow channel, which is a combination of two 1-bit signals and of an optional bus:

- A `valid` wire, going in the same direction as the edge (downstream).
- A `ready` wire, going in the opposite direction as the edge (upstream).
- An optional `data` bus of arbitrary width, going downstream. We display channels without a data bus (which we often refer to as control-only channels) as dashed.

During execution of the circuit, each combination of the valid/ready wires (a channel's dataflow state) maps to a different color. You can see this mapping by clicking the Legend button on the top-right corner of the window. You can also change the mapping by clicking each individual color box and selecting a different color. There are 4 possible dataflow states.
- `Idle` (`valid=0,ready=0`): the producer does not have a valid token to put on the channel, and the consumer is not ready to consume it. Nothing is happening, the channel is idle.
- `Accept` (`valid=0,ready=1`): the consumer is ready to consume a token, but the producer does not have a valid token to put on the channel. The channel is ready to accept a token.
- `Stall` (`valid=1,ready=0`): the producer has put a valid token on the channel, but the consumer is not ready to consume it. The token is stalled.
- `Transfer` (`valid=1,ready=1`): the producer has put a valid token on the channel which the consumer is ready to consume. The token is transferred.

The nodes each have a unique name inherited from the MLIR-formatted IR that was used to generate the input DOT file to begin with, and are grouped together based on the basic block they belong to. These are the same basic blocks used to represent control-free sequences of instructions in classical compilers. In this example, the original source code had 5 basic blocks, which are transcribed here in 5 labeled rectangular boxes.
> [!TIP]  
> Two of these basic blocks represent the start and end of the kernel before and after the loop, respectively. The other 3 hold the loop's logic. Try to identify which is which from the nature of the nodes and from their connections. Consider that the loop may have been slightly transformed by Dynamatic to optimize the resulting circuit.  

There are several interactive elements at the bottom of the window that you can play with to see data flow through the circuit.

- The horizontal bar spanning the entire window's width is a timeline. Clicking or dragging on it will let you go forward or backward in time.
- The `Play` button will iterate forward in time at a rate of one cycle per second when clicked. Cliking it again will pause the iteration.
- As their name indicates, `Prev cycle` and `Next cycle` will move backward or forward in time by one cycle, respectively.
- The `Cycle: ` textbox lets you enter a cycle number directly, which the visualizer then jumps to.
> [!TIP]  
> Observe the circuit executes using the interactive controls at the bottom of the window. On cycle 6, for example, you can see that tokens are transferred on both input channels of `muli0` in `block2`. Try to infer the multiplier's latency by looking at its output channel in the next execution cycles. Then, try to track that output token through the circuit to see where it can end up. Study the execution till you get an understanding of how tokens flow inside the loop and of how the conditional multiplication influences the latency of each loop iteration.

### Conclusion
Congratulations on reaching the end of this tutorial! You now know how to use Dynamatic to compile C kernels into functional dataflow circuits, visualize these circuits to better understand them to identify potential optimization opportunities.  
Before moving on to use Dynamatic for your custom programs, kindly refer to the [Kernel Code Guidelines](../../../UserGuide/KernelCodeGuideLines.md) guide. You can also view a more [detailed example](Examples.md) that uses some of the optional commands not mentioned in this introductory tutorial.

We are now ready for an introduction to [modiying Dynamatic](ModifyingDynamatic.md). We will identify an optimization opportunity from the previous example and write a small transformation pass in C++ to implement our desired optimization, before finally verifying its behavior using the dataflow visualizer.
