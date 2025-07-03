# Dynamatic User Manual

This document serves as a high level overview of various features of Dynamatic. It is mainly intended for new students and users who are aiming to understand the key ideas behind Dynamatic without going too deeply into technical details. For installation instructions, see [the README file.](../README.md)

## Table of Contents

- [Basic Usage](#basic-usage)

- [Floating Point IPs](#floating-point-ips)

- [Buffer Placement Strategies](#buffer-placement-strategies)

- [Load-Store Queues](#load-store-queues)

- [Custom Compilation Flows](#custom-compilation-flows)

## Basic Usage

Dynamatic consists of many components, but the simplest way to use Dynamatic is using the `dynamatic` binary, which abstracts away most of the details by providing a user-friendly interactive shell. After completing the build process, just run
```
$ bin/dynamatic
```
from the root directory of the cloned Dynamatic repository.

From here, you are offered a shell with a variety of commands that you can see by typing `help`, or by referring to the list below.

### Dynamatic Shell Commands
- `help`: Display list of commands.
- `write-hdl [--hdl <VHDL|Verilog>]`: Convert results from `compile` to a VHDL or Verilog file.
- `set-vivado-path <path>`: Set the path to the installation directory of Vivado.
- `simulate`: Simulates the HDL produced by `write-hdl`. **Requires a ModelSim installation!**
- `set-fp-units-generator <flopoco|vivado>`: Choose which floating point unit generator to use. See [this section](#floating-point-ips) for more information.
- `set-clock-period <clk>`: Sets the target clock period in nanoseconds.
- `set-dynamatic-path <path>`: Set the path of the root (top-level) directory of Dynamatic, so that it can locate various scripts it needs to function. This is not necessary if you run Dynamatic from said directory.
- `set-src <source-path>`: Sets the path of the `.c` file of the kernel that you want to compile. 
- `synthesize`: Synthesizes the HDL result from `write-hdl` using Vivado. **Requires a Vivado installation!**
- `compile [...]`: Compiles the source kernel (chosen by `set-src`) into a dataflow circuit. For more options, run `compile --help`. **Does not require Gurobi by default, [but some options do!](#buffer-placement-strategies)**
- `visualize`: Visualizes the execution of the circuit simulated by `ModelSim`. **Requires Godot Engine and [the visualizer component must be built!](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/AdvancedBuild.md#interactive-dataflow-circuit-visualizer)**
- `set-polygeist-path <path>`: Sets the path to the Polygeist installation directory.
- `exit`: Exits the interactive Dynamatic shell.

An example of a usual sequence of commands is given below:
```
set-src integration-test/fir/fir.c
compile
write-hdl
simulate
```

### Reviewing HLS reports

Each command will inform you about its result (success/failure) via interactive shell output. Detailed reports are saved in a folder named `out` in the same path where the `.c` kernel source was located; i.e. if you used `integration-test/fir/fir.c`, then the reports would be in `integration-test/fir/out`.

- `out/comp`: Contains artifacts produced by the compilation. This includes all intermediate MLIR results, as well as the diagram of the dataflow circuit and the control flow graph (in DOT and PNG formats).
- `out/hdl`: Contains HDL descriptions of all components of the generated dataflow circuit.
- `out/sim`: Contains subfolders with various simulation artifacts, as well as a text file `report.txt` with the entire simulation log.

### Scripting

Running the same set of commands over and over again can get tedious, so Dynamatic has basic scripting support. You can write the sequence of commands to be executed into a file and then run them all at once using
```
$ bin/dynamatic --run=<path-to-script>
```

## Floating Point IPs

For implementing floating point operations, Dynamatic uses open-source [FloPoCo](https://flopoco.org/) components. It is possible to use proprietary Xilinx FP units from Vivado. For instructions on how to achieve this, see [this guide](Specs/FloatingPointUnits.md).

## Buffer Placement Strategies

Dynamatic automatically inserts buffers to eliminate performance bottleneck and
to achieve a particular clock frequency. This feature is **essential** to
enable for Dynamatic to achieve the best performance.

For example, the code below:

```
int fir(in_int_t di[N], in_int_t idx[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++)
    tmp += idx[i] * di[N_DEC - i];
  return tmp;
}
```

has a long latency multiplication operation, which prolongs the lifetime of
loop variables. Buffers must be sufficiently and appropriately inserted to
achieve a certain initiation interval.

The naive buffer placement algorithm in Dynamatic, `on-merges`, is used by default. Its strategy is to place buffers on the output channels of all merge-like operations. This creates perfectly valid circuits, but results in poor performance.

For better performance, two more advanced algorithms are implemented, based on the [FPGA'20](https://doi.org/10.1145/3477053) and [FPL'22](https://doi.org/10.1109/FPL57034.2022.00063) papers. They can be chosen by using `compile` in `bin/dynamatic` with the command line option `--buffer-algorithm fpga20` or `--buffer-algorithm fpl22`, respectively. **Note that these two algorithms require Gurobi to be installed and detected, otherwise they will not be available!** Installation instructions for Gurobi can be found [here](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/AdvancedBuild.md#Gurobi). A brief high-level overview of these algorithms' strategies is provided below; for more details, see the original publications linked above and [this document](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Specs/Buffering/Buffering.md).

The main idea of the FPGA'20 algorithm is to decompose the dataflow circuit into choice-free dataflow circuits (i.e. parts which don't contain any branches). The performance of these CFDFCs can be modeled using an approach based on timed Petri nets (see [here](https://www.computer.org/csdl/journal/ts/1980/05/01702760/13rRUxASuqJ) and [here](https://dspace.mit.edu/handle/1721.1/13739)). This model is formulated as a mixed-integer linear programming model, with additional constraints which allow the optimization of multiple CFDFCs. Simulation results have shown circut speedups up to 10x for most benchmarks, with some reaching even 33x. For example, the `fir` benchmark with naive buffering runs in 25.8 us, but with this algorithm, it executes in only 4.0 us, which is 6.5x faster. The downside is that the MILP solver can take a long time to complete its task, sometimes even more than an hour, and also clock period targets might not be met.

Similarly, the FPL'22 algorithm uses a MILP-based approach for modeling and optimization. The main difference is that it does not only model the circuit as single dataflow channels carrying tokens, but instead, describes individual edges carrying data, valid and ready signals, while explicitly indicating their interconnections. The dataflow units themselves are modeled with more detail; instead of nodes representing entire dataflow units, they represent distinct combinational delays of every combinational path through the dataflow units. This allows for precise computation of all combinational delays and accurate buffer placement for breaking up long combinational paths. This approach meets the clock period target much more consistently than the previous two approaches. 

## Load-Store Queues

In order to leverage the power of dataflow circuits generated by Dynamatic, a memory interface is required which would analyze data dependencies, reorder memory accesses and stall in case of data hazards. Such a component is a Load-Store Queue, specifically designed for dataflow circuits. 

The strategy for managing memory accesses is based on the concept of groups. A group is a sequence of memory accesses that cannot be interrupted by a control flow decision. Determining a correct order of accesses within a group can be done easily using static analysis and can be encoded into the LSQ at compile time. The LSQ component has as many load/store ports as there are load/store operations in the program. These ports are clustered by groups, with every port belonging to one group. Whenever a group is "activated", all load/store operations belonging to that group are allocated in the LSQ in the sequence that was determined by static analysis. Once a group has been allocated, the LSQ expects each of the corresponding ports to eventually get an access; dependencies will be resolved based on the order of entries in the LSQ.

The specifics of LSQ implementation are available in [the corresponding documentation folder.](./LSQ/) For more information on the concept itself, [see the original paper.](https://dynamo.ethz.ch/wp-content/uploads/sites/22/2022/06/JosipovicTECS17_AnOutOfOrderLoadStoreQueueForSpatialComputing.pdf)

## Custom Compilation Flows

Sometimes, for advanced usage, features provided by the `dynamatic` shell are not enough. In such case, one should invoke components such as `dynamatic-opt` (also located in the `bin` directory) directly. The default compilation flow is implemented in `tools/dynamatic/scripts/compile.sh`; you can use this as a template that you can adjust to your needs.