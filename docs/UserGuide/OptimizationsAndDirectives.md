# Optimizations And Directives
Dynamatic offers a number of options to optimize the generated RTL code to meet specific requirements. This document describes the various optimization options available as well as some directives to customize the generated RTL to specific hardware using proprietory floating point unit generators.

## Overview: What if I Want to Optimize ...
1. [Clock frequency](#1-achieving-a-specific-clock-frequency)  
2. [Area](#2-area)  
3. [Latency and throughput](#3-latency-and-throughput)  
4. [Customizing Design to Specific Hardware: Floating Point IPs](#adjusting-design-to-specific-hardware-floating-point-ips)
5. [Optimization algorithms in Dynamatic](#optimization-algorithms-in-dynamatic)
    - [Buffer placement algorithm: FPGA'20](#buffer-placement-algorithm-fpga20)
    - [Buffer placement algorithm: FPL'22](#buffer-placement-algorithm-fpl22)
    - [Area optimization: Sizing Load Store Queue depths](#area-optimization-sizing-load-store-queue-depths)
    - [Resource sharing of functional units](#resource-sharing-of-functional-units)
6. [Custom compilation flows](#custom-compilation-flows)  


### 1. Achieving a Specific Clock Frequency   
Dynamatic relies on its buffer placement algorithm to regulate the critical path in the design to achieve a specific frequency target. To achieve the desired target, set the period (`set-clock period <value_in_ns>`) and enable the buffer placement algorithm `compile --buffer-algorithm <...>...`

### 2. Area  
Circuit area can be optimized using the following compile flags
- LSQ sizing
- Credit-based resource sharing: `--sharing`
- Buffer placement :`--buffer-algorithm` with value `fpl22` 

### 3. Latency and Throughput
Latency and throughput can be improved using buffer placement with either the `fpga20` or `fpl22` values for the `--buffer-algorithm` compile flag.  

### Adjusting Design to Specific Hardware: Floating Point IPs
Dynamatic uses open-source [FloPoCo](https://flopoco.org/) components proprietory [Vivado](https://docs.amd.com/v/u/en-US/pg060-floating-point) to allow users to customize their floating point units. For instructions on how to achieve this, see [the floating point units guide](../DeveloperGuide/Specs/FloatingPointUnits.md). Floating point units can be selected using the `set-fp-units-generator <flopoco|vivado>` command as shown in the [command reference](../UserGuide/CommandReference.md).

#### Advantages of Using Vivado Over FloPoCo Floating Point IP
- Tailored for Xilinx hardware and ideal for industry level projects.
- Supports IEEE-754 single, double, and half precision floating point representation.
-  Supports NaN, infinity, denormals, exception flags, and rounding models.
- Provides plug and play floating point units.

#### Advantages of Using FloPoCo Over Vivado Floating Point IP
- Open source, hence ideal for academic research involving fine grained parameter tuning and RTL transparency
- Very good for custom floating point formats such as FP8 or "quasi-floating point".
- Users can explicitly control pipeline depth.
- Generated RTL is portable to any toolchanin unlike Vivado which is limited to Xilinx-specific resources.

## Optimization Algorithms in Dynamatic

### Throughput Optimization: Enabling Smart Buffer Placement
Dynamatic automatically inserts buffers to eliminate performance bottlenecks and achieve a particular clock frequency. This feature is **essential** to
enable for Dynamatic to achieve the best performance.

For example, the code below:

```c
int fir(in_int_t di[N], in_int_t idx[N]) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++)
    tmp += idx[i] * di[N_DEC - i];
  return tmp;
}
```

has a long latency multiplication operation, which prolongs the lifetime of loop variables. Buffers must be sufficiently and appropriately inserted to achieve a certain initiation interval.  

The naive buffer placement (default) algorithm in Dynamatic, `on-merges`, is used by default. Its strategy is to place buffers on the output channels of all merge-like operations. This creates perfectly valid circuits, but results in poor performance.

For better performance, two more advanced algorithms are implemented, based on the [FPGA'20](https://doi.org/10.1145/3477053) and [FPL'22](https://doi.org/10.1109/FPL57034.2022.00063) papers. They can be chosen by using `compile` in `bin/dynamatic` with the command line option `--buffer-algorithm fpga20` or `--buffer-algorithm fpl22`, respectively.  
> [!NOTE]  
> These two algorithms require Gurobi to be installed and detected, otherwise they will not be available!

Installation instructions for Gurobi can be found [here](../UserGuide/AdvancedBuild.md#gurobi). A brief high-level overview of these algorithms' strategies is provided below; for more details, see the original publications linked above and [this document](../DeveloperGuide/DynamaticFeaturesAndOptimizations/Buffering/Buffering.md).

#### Buffer Placement Algorithm: FPGA'20
The main idea of the `fpga20` algorithm is to decompose the dataflow circuit into choice-free dataflow circuits (i.e. parts which don't contain any branches). The performance of these CFDFCs can be modeled using an approach based on timed Petri nets (see [Performance Evaluation of Asynchronous Concurrent Systems Using Petri Nets](https://www.computer.org/csdl/journal/ts/1980/05/01702760/13rRUxASuqJ) and [Analysis of asynchronous concurrent systems by timed petri nets](https://dspace.mit.edu/handle/1721.1/13739)).  

This model is formulated as a mixed-integer linear programming model ([MILP](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/)), with additional constraints which allow the optimization of multiple CFDFCs. Simulation results have shown circut speedups up to **10x** for most benchmarks, with some reaching even 33x. For example, the `fir` benchmark with naive buffering runs in 25.8 us, but with this algorithm, it executes in only 4.0 us, which is 6.5x faster.  
> The downside is that the MILP solver can take a long time to complete its task, sometimes even more than an hour, and also clock period targets might not be met.

#### Buffer Placement Algorithm: FPL'22
The `fpl22` algorithm also uses a MILP-based approach for modeling and optimization. The main difference is that it does not only model the circuit as single dataflow channels carrying tokens, but instead, describes individual edges carrying data, valid and ready signals, while explicitly indicating their interconnections. The dataflow units themselves are modeled with more detail. Instead of nodes representing entire dataflow units, they represent distinct combinational delays of every combinational path through the dataflow units. This allows for precise computation of all combinational delays and accurate buffer placement for breaking up long combinational paths. 
> This approach meets the clock period target much more consistently than the previous two approaches. 

### Area Optimization: Sizing Load-Store Queue Depths: FPT'22 
In order to leverage the power of dataflow circuits generated by Dynamatic, a memory interface is required which would analyze data dependencies, reorder memory accesses and stall in case of data hazards. Such a component is a Load-Store Queue, specifically designed for dataflow circuits. The LSQ sizing algorithm is implemented based on [FPT'22](https://doi.org/10.1109/ICFPT56656.2022.9974425)

The strategy for managing memory accesses is based on the concept of groups. 
> [!NOTE]  
> A group is a sequence of memory accesses that cannot be interrupted by a control flow decision. 

Determining a correct order of accesses within a group can be done easily using static analysis and can be encoded into the LSQ at compile time. The LSQ component has as many load/store ports as there are load/store operations in the program. These ports are clustered by groups, with every port belonging to one group. Whenever a group is "activated", all load/store operations belonging to that group are allocated in the LSQ in the sequence that was determined by static analysis. Once a group has been allocated, the LSQ expects each of the corresponding ports to eventually get an access; dependencies will be resolved based on the order of entries in the LSQ.  

> [!WARNING]  
> A significant area improvement can be achieved by disabling the use of LSQs but this must be used cautiously.

The specifics of LSQ implementation are available in [the corresponding documentation.](../DeveloperGuide/DynamaticFeaturesAndOptimizations/LSQ/LSQ.md) For more information on the concept itself, [see the original paper.](https://dynamo.ethz.ch/wp-content/uploads/sites/22/2022/06/JosipovicTECS17_AnOutOfOrderLoadStoreQueueForSpatialComputing.pdf)  


### Resource Sharing of Functional Units: ASPLOS'25
Dynamatic uses a resource sharing strategy based on [ASPLOS'25](https://dl.acm.org/doi/10.1145/3669940.3707273). This algorithm avoids sharing-introduced deadlocks by decoupling interactions of operations in shared resources to break resource dependencies while maintaining the benefits of dynamism. It is activated using the `--sharing` compile flag as such:  
```
compile <...> --sharing
```

## Custom Compilation Flows  
Some other transformations also optimize the circuit, but they are not included in the normal compilation flow.
In such case, one should invoke components such as `dynamatic-opt` (also located in the `bin` directory) directly. The default compilation flow is implemented in `tools/dynamatic/scripts/compile.sh`; you can use this as a template that you can adjust to your needs.  

Some optimization strategies, such as speculation or fast token delivery, aren't accessible through the standard `dynamatic` interactive environment.  
These approaches often require a custom compilation flow. For example, speculation provides a Python script that enables a push-button flow execution.  

For more details, refer to the [speculation documentation](../DeveloperGuide/DynamaticFeaturesAndOptimizations/Speculation/IntegrationTests.md).
