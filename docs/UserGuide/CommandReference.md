# Command Reference
The Dynamatic shell is an interactive command line-based interface (you can launch it from Dynamatic's top level directory with `./bin/dynamatic` after building Dynamatic) that allows users to interact with Dynamatic and use the different commands available to generate dataflow circuits from C code.  

This document provides an overview of the different commands available in the Dynamatic frontend and their respective flags and options.

## Dynamatic Shell Commands
- `help`: Display list of commands.
- `set-dynamatic-path <path>`: Set the path of the root (top-level) directory of Dynamatic, so that it can locate various scripts it needs to function. This is not necessary if you run Dynamatic from said directory.
- `set-vivado-path <path>`: Set the path to the installation directory of Vivado.
- `set-polygeist-path <path>`: Sets the path to the Polygeist installation directory.
- `set-fp-units-generator <flopoco|vivado>`: Choose which floating point unit generator to use. See [this section](OptimizationsAndDirectives.md#floating-point-ips) for more information.
- `set-clock-period <clk>`: Sets the target clock period in nanoseconds.
- `set-src <source-path>`: Sets the path of the `.c` file of the kernel that you want to compile. 
- `compile [...]`: Compiles the source kernel (chosen by `set-src`) into a dataflow circuit. For more options, run `compile --help`.
> [!NOTE]  
> The `compile` command does not require Gurobi by default, but it is needed for [smart buffer placement options](OptimizationsAndDirectives.md#optimization-algorithms-in-dynamatic).  

The `--buffer-algorithm` flag allows users to use smart buffer placement algorithms notably `fpga20` and `fpl22` for throughput and timing optimizations.
- `write-hdl [--hdl <vhdl|verilog|smv>]`: Convert results from `compile` to a VHDL, Verilog or SMV file.
- `simulate`: Simulates the HDL produced by `write-hdl`. 
> [!NOTE]  
> Requires a ModelSim/Questa installation!

- `synthesize`: Synthesizes the HDL result from `write-hdl` using Vivado. 
> [!NOTE]  
> Requires a Vivado installation! 

- `visualize`: Visualizes the execution of the circuit simulated by `ModelSim`/`Questa`. 
> [!NOTE]  
> Requires Godot Engine and [the visualizer component must be built!](AdvancedBuild.md#4-interactive-dataflow-circuit-visualizer)

- `exit`: Exits the interactive Dynamatic shell.

For more information and examples on the typical usage of the commands, checkout the [using Dynamatic](../GettingStarted/Tutorials/Introduction/UsingDynamatic.md) and [example](../GettingStarted/Tutorials/Introduction/Examples.md) pages.
