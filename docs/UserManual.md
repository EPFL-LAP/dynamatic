# Dynamatic User Manual

This document serves as a high level overview of various features of Dynamatic. It is mainly intended for new students and users who are aiming to understand the key ideas behind Dynamatic without going too deeply into technical details. 

## Table of Contents

- [Basic Usage](#basic-usage)

- [Floating Point IPs]()

- [Buffer Placement Strategies](#buffer-placement-strategies)

- [Load-Store Queues]()

- [Custom Compilation Flows]()

## Basic Usage

The simplest way to use Dynamatic is using the `dynamatic` binary. After completing the build process, just run
```
$ bin/dynamatic
```
from the root directory of the cloned Dynamatic repository.

From here, you are offered a shell with a variety of commands that you can see a list of by typing `help`. You will usually run the following sequence of commands:

0. `set-dynamatic-path <path>`: Used to set the path of the root directory of Dynamatic, so that it can locate various scripts it needs to function. This is not necessary if you run Dynamatic from the root directory.

1. `set-src <source-path>`: Sets the path of the `.c` file of the kernel that you want to compile. For example, you can try `set-src integration-test/fir/fir.c`.

2. `compile`: Compiles the `.c` file given in the previous step. For more information about the options for buffer placement, see [below](#buffer-placement-strategies).

3. `write-hdl`: Writes the resulting circuit into a HDL file. You can choose between VHDL and Verilog using `--hdl VHDL|Verilog`.

4. `simulate`: Runs a ModelSim simulation of the circuit. 

The result of the simulation, as well as intermediate results, can be found in a folder named `out` located in the same path as the `.c` file that was used.

Running the same set of commands over and over again can get tedious, so Dynamatic has basic scripting support. You can write the sequence of commands to be executed into a file and then run them all at once using
```
$ bin/dynamatic --run=<path-to-script>
```

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

