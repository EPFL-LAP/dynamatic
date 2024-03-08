# Using Dynamatic

This tutorial will walk you through the compilation of a simple kernel function written in C into an equivalent VHDL design, the functional verification of the resulting dataflow circuit using Modelsim, and the latter's visualization using our custom interactive dataflow visualizer. The tutorial assumes basic knowledge of dataflow circuits but does not require any insight into MLIR or compilers in general.

Below are some technical details about this tutorial.
- All resources are located in the repository's [`tutorials/Introduction/Ch1`](../../../tutorials/Introduction/Ch1) folder.
- All relative paths mentionned throughout the tutorial are assumed to start at Dynamatic's top-level folder.
- We assume that you have already built Dynamatic from source using the instructions in the top-level [README](../../../README.md) or that you have access to a Docker container that has a pre-built version of Dynamatic . 

This tutorial is divided into the following sections.
- [1. The source code](#the-source-code) | We take a look at the C kernel function we will transform into a dataflow circuit. 
- [2. Using Dynamatic's frontend](#using-dynamatics-frontend) | We use the Dynamatic frontend to compile the C function down to an equivalent VHDL design, and functionally verify the latter using Modelsim.
- [3. Visualizing the resulting dataflow circuit](#visualizing-the-resulting-dataflow-circuit) | We visualize the execution of the generated dataflow circuit on test inputs.
- [4. Conclusion](#conclusion) | We reflect on everything we just accomplished.

## The source code

In this tutorial, we will tranform the following C function (the *kernel*, in DHLS jargon) into a dataflow circuit.

```c
// The number of loop iterations
#define N 8

// The kernel under consideration
unsigned loop_multiply(in_int_t a[N]) {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * x;
  }
  return x;
}
```

This simple kernel multiplies a number by itself at each iteration of a simple loop from 0 to any number `N` where the corresponding element of an array equals 0. The function returns the calculated value after the loop exits. Note that this function is purposefully very simple so that it corresponds to a small dataflow circuit that will be easier to visually explore later on. Dynamatic is capable of transforming much more complex functions into fast and functional dataflow circuits.

You can find the source code of this function in [`tutorials/Introduction/Ch1/loop_multiply.c`](../../../tutorials/Introduction/Ch1/loop_multiply.c). You will notice the `main` function in the file, which allows one to run the C kernel with user-provided arguments. The `CALL_KERNEL` macro in `main`'s body, as its name indicates, calls the kernel while allowing us to automatically run code prior to and/or after the call. For example, this is used during C/VHDL co-verification to automatically write the C function's reference output to a file to later compare it with the generated VHDL design's output.

Now that we are familiar with the source code, we can move on to generating a matching dataflow circuit! 

## Using Dynamatic's frontend

We will now use Dynamatic's frontend in interactive mode to compile the `loop_multiply` kernel into a VHDL design in a couple of simple commands. Dynamatic frontend's is built by the project in `build/bin/dynamatic`, with a symbolic link located at `bin/dynamatic`, which we will be using. In a terminal, from Dynamatic's top-level folder, run the following.

```sh
./bin/dynamatic
```

This will print the frontend's header and display a prompt where you can start inputting your first command.

```
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
==================== EPFL-LAP - <release> | <release-date> =====================
================================================================================

dynamatic> # Input your command here
```

First, we must provide the frontend with the path to the C source file under consideration. As we mentionned in the previous section, ours is located at [`tutorials/Introduction/Ch1/loop_multiply.c`](../../../tutorials/Introduction/Ch1/loop_multiply.c), so input the following command into the frontend.

```sh
dynamatic> set-src tutorials/Introduction/Ch1/loop_multiply.c
```

The frontend will assume that the C function to transform has the same name as the last component of the argument to `set-src` without the file extension, here `loop_multiply`.

The first step towards generating the VHDL design is *compilation*, during which the C source goes through our MLIR frontend ([Polygeist](https://github.com/llvm/Polygeist)) and then through a pre-defined sequence of transformation and optimization passes that ultimately yield a description of an equivalent dataflow circuit. That description takes the form of a human-readable and machine-parsable IR (Intermediate Representation) within the MLIR framework. In particular, it represents dataflow components using specially-defined IR instructions (in MLIR jargon, [operations](../MLIRPrimer.md#operations)) that are part of the [*Handshake dialect*](../MLIRPrimer.md#dialects). A dialect is simply a collection of logically-connected IR entities like instructions, types, and attributes. MLIR provides so-called standard dialects for common use cases, while allowing external tools (like Dynamatic) to define their own custom dialects to model domain-specific semantics. We inherit part of the infrastructure surrounding the *Handshake* dialect from the [CIRCT project](https://github.com/llvm/circt) (a satellite project of LLVM/MLIR), but have tailored it to our specific use cases.

To compile the C function, simply input `compile`. This will call a shell script in the background that will iteratively transform the IR into an optimized dataflow circuit, storing intermediate IR forms to disk at multiple points in the process.

```
dynamatic> set-src tutorials/Introduction/Ch1/loop_multiply.c
dynamatic> compile
```

Note that the `compile` command requires that a supported MILP (Mixed Integer Linear Program) solver is available on your system to run our smart buffer placement compiler pass. Indeed, dataflow circuits need to contain buffers to ensure functional correctness and high data throughput. Our smart buffer placement algorithm expresses this optimization problem as an MILP whose results are interpreted to deduce an optimal buffer placement. If no supported MILP solver is available on your system, you can proide the `--simple-buffers` flag to `compile` to instruct it to use a trivial buffer placement algorithm that guarantees functional correctness at the cost of low throughput.   

You should see the following printed on the terminal after running `compile`.

```
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
[INFO] Built kernel for profiling
[INFO] Ran kernel for profiling
[INFO] Profiled cf-level
[INFO] Running smart buffer placement
[INFO] Placed smart buffers
[INFO] Canonicalized handshake
[INFO] Created visual DOT
[INFO] Converted visual DOT to PNG
[INFO] Created loop_multiply DOT
[INFO] Converted loop_multiply DOT to PNG
[INFO] Compilation succeeded
```

This signals that compilation succeeded. All results are placed in a folder named `out/comp` created next to the C source under consideration. In this case, it is located at `tutorials/Introduction/Ch1/out/comp`. It is not necessary that you look inside this folder for this tutorial.

In addition to the final optimized version of the IR (in `tutorials/Introduction/Ch1/out/comp/handshake_export.mlir`), the compilation script generates an equivalent Graphviz-formatted file (`tutorials/Introduction/Ch1/out/comp/loop_multiply.dot`) which serves as input to our VHDL backend, which we can now call using the `write-hdl` command.

```
...
[INFO] Compilation succeeded

dynamatic> write-hdl
[INFO] Converted DOT to VHDL
[INFO] HDL generation succeeded
```

Similarly to `compile`, this creates a folder `out/hdl` next to the C source under consideration. The command looks for the Graphviz-formatted version of our circuit (`tutorials/Introduction/Ch1/out/comp/loop_multiply.dot`) generated by the compile command and transforms it into a synthesizable VHDL design (`tutorials/Introduction/Ch1/out/hdl/loop_multiply.vhd`) corresponding to the C kernel. This design can finally be co-simulated along the C function on Modelsim to verify that their behavior matches using the `simulate` command.

```
...
[INFO] HDL generation succeeded

dynamatic> simulate
[INFO] Built kernel for IO gen.
[INFO] Ran kernel for IO gen.
[INFO] Launching Modelsim simulation
[INFO] Simulation succeeded
```

Once again, the command creates a new folder `out/sim` next to the C source under consideration. In this case, it is located at `tutorials/Introduction/Ch1/out/sim`. While it is not necessary that you look inside this folder for this tutorial, just know that it contains everything necessary to co-simulate the design (input C function and VHDL entity values, auto-generated testbench, full implementation of all dataflow components, etc.) and everything generated by the co-simulation process (output C function and VHDL entitiy values, VHDL compilation logs, full waveform). `[INFO] Simulation succeeded` indicates that the C function and VHDL design showcased the same behavior. This just means that their return values were the same after execution or simulation, respectively, on kernel inputs computed in the `main` function. If any arguments were pointers to memory regions, `simulate` would also check that the state of these memories is the name after the C kernel call and after VHDL simulation.

That's it, you have successfully synthesized your first dataflow circuit from C code and functionnaly verified it using Dynamatic! 

At this point, you can quit the Dynamatic frontend by inputting the `exit` command. 

```
...
[INFO] Simulation succeeded

dynamatic> exit

Goodbye!
```

If you would like to re-run these commands all at once, note that it is possible to use the frontend in a non-interactive way by writing the sequence of commands you would like to run in a file and referencing it when launching the frontend. One such file has already been created for you at [`tutorials/Introduction/Ch1/frontend-script.dyn`](../../../tutorials/Introduction/Ch1/frontend-script.dyn). You can replay this whole section by running the following from Dynamatic's top-level folder.

```sh
./bin/dynamatic --run tutorials/Introduction/Ch1/frontend-script.dyn
```

In the last section of this tutorial, we will take a closer look at the actual circuit that was generated by Dynamatic and visualize its execution interactively. 

## Visualizing the resulting dataflow circuit

At the end of the last section, you used the `simulate` command to co-simulate the VHDL design obtained from the compilation flow along with the C source. This process generated a waveform file at `tutorials/Introduction/Ch1/out/sim/HLS_VERIFY/vsim.wlf` containing all state transitions that happened during simulation for all signals. After a simple pre-processing step using the frontend's `visualize` command, we will be able to visualize these transitions on a graphical representation of our circuit to get more insights into how our dataflow circuit behaves.

To launch the visualizer, re-open the frontend, re-set the source with `set-src tutorials/Introduction/Ch1/loop_multiply.c`, then input the `visualize` command.

```
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

We do not have to re-run the previous synthesis steps because the data expected by the `visualize` command is still present on disk in the output folders generated by `compile` and `simulate`. `visualize` creates a folder `out/visual` next to the source file (here `tutorials/Introduction/Ch1/out/visual`) containing the data expected by the visualizer as input.

You should now see a visual representation of the dataflow circuit you just synthesized. It is basically a graph, where each node represents some kind of dataflow component and each directed edge represents a dataflow channel, which is a combination of two 1-bit signals and of an optional bus.
- A `valid` wire, going in the same direction as the edge (*downstream*).  
- A `ready` wire, going in the opposite direction as the edge (*upstream*).
- An optional `data` bus of arbitrary width, going in the same direction as the edge (*downstream*). We display channels without a `data` bus---which we often refer to as *control-only channels*---as dashed.

During execution of the circuit, each combination of the `valid`/`ready` wires---a channel's *dataflow state*---maps to a different color. You can see this mapping by clicking the `Legend` button on the top-right corner of the window. You can also change the mapping by clicking each individual color box and selecting a different color. There are 4 possible dataflow states.
- `Idle` (`valid=0,ready=0`): the producer does not have a valid token to put on the channel, and the consumer would not be ready to consume it anyway. Nothing is happening, the channel is *idle*.
- `Accept` (`valid=0,ready=1`): the consumer is ready to consume a token, but the producer does not have a valid token to put on the channel. The channel is ready to *accept* a token.
- `Stall` (`valid=1,ready=0`): the producer has put a valid token on the channel, but the consumer is not ready to consume it. The token is *stalled*.
- `Transfer` (`valid=1,ready=1`): the producer has put a valid token on the channel which the consumer is ready to consume. The token is *transferred*.

Let's now take a closer look at the nodes. They each have a unique name inherited from the MLIR-formatted IR that was used to generate the input DOT file to begin with, and are grouped together based on the basic block they belong to. These are the same basic blocks used to represent control-free sequences of instructions in classical compilers. In this example, the original source code had 5 basic blocks, which are transcribed here in 5 labeled rectangular boxes.

> [!TIP]
> Two of these basic blocks represent the start and end of the kernel before and after the loop, respectively. The other 3 hold the loop's logic. Try to identify which is which from the nature of the nodes and from their connections. Consider that the loop may have been slightly transformed by Dynamatic to optimize the resulting circuit. 

Now that we have gotten familiar with the circuit representation, it is time to see it execute! There are several interactive elements at the bottom of the window that you can play with to see data flow through the circuit.
- The horizontal bar spanning the entire window's width is a timeline. Clicking or dragging on it will let you go forward or backward in time.
- The `Play` button will iterate forward in time at a rate of one cycle per second when clicked. Cliking it again will pause the iteration.
- As their name indicates, `Prev cycle` and `Next cycle` will move backward or forward in time by one cycle, respectively.
- The `Cycle: ` textbox lets you enter a cycle number directly, which the visualizer then jumps to.

> [!TIP]
> Observe the circuit executes using the interactive controls at the bottom of the window. On cycle 6, for example, you can see that tokens are transferred on both input channels of `muli0` in `block2`. Try to infer the multiplier's latency by looking at its output channel in the next execution cycles. Then, try to track that output token through the circuit to see where it can end up. Study the execution till you get an understanding of how tokens flow inside the loop and of how the conditional multiplication influences the latency of each loop iteration.

## Conclusion

Congratulations on reaching the end of this tutorial! You now know how to use Dynamatic to compile C kernels into functional dataflow circuits, then visualize these circuits to better understand them and identify potential optimization opportunities. In the [next chapter of this tutorial](ModifyingDynamatic.md), we will identify one such opportunity and write a small transformation pass in C++ to implement our desired optimization, before finally verifying its behavior using the dataflow visualizer.
