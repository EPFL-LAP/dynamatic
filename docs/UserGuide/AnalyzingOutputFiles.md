# Analyzing Output Files
Dynamatic stores the compiled IR, generated RTL, simulation results, and useful intermediate data in the `out/` directory.
Learning about these files is essential for identifying performance bottlenecks, gaining deeper insight into the generated circuits, exporting the generated design to integrate into your existing designs, etc.  

This document provides guidance on the locations of these files and how to analyze them effectively.

### Compilation Results 
> [!NOTE]  
> Compilation results are not essential for a user but can help in debugging. This requires some knowledge of MLIR. 
- The `compile` command creates an `out/comp` directory that stores all the intermediate files as described in the [Dynamatic HLS flow](../DeveloperGuide/IntroductoryMaterial/DynamaticHLSFlow.md#dynamatics-high-level-synthesis-flow) in the developer guide.
- A file is created for every step of the compilation process, allowing the user to inspect relevant files if any unexpected behaviour results.
> [!TIP]  
> Compilation results in the creation of two PNG file, `kernel_name.png` and `kernel_name_CFG.png`, allowing the user to have an overview of the generated circuit and associated control flow graph of their kernel.  

### RTL Generation Results
The `write-hdl` command creates an `out/hdl` directory.  
`out/hdl` contains all the RTL files (adders, multipliers, muxes, etc.) needed to implement the target kernel.  
The `top level` HDL file is called `kernel_name.vhd` or `kernel_name.v` if you use VHDL or verilog respectively.

### Simulation Results
> [!IMPORTANT]
> Modelsim/Questa must be installed and added to path before running this command. See [Modelsim/Questa installation guide](AdvancedBuild.md#6-modelsimquesta-installation)  

The `simulate` command creates an `out/sim` directory. In this directory are a number of sub directories organized as shown below:
```sh
out/sim
├── C_OUT           # output from running the C program
├── C_SRC           # C source files and header files
├── HDL_OUT         # output from running the simulation of the HDL testbench
├── HDL_SRC         # HDL files and the testbench
├── HLS_VERIFY      # Modelsim/Questa files used to run simulation
├── INPUT_VECTORS   # inputs passed to the C and HDL implementations for testing
├── report.txt      # simulation report and logs
```
The `simulate` command runs a C/HDL co-simulation and prints the `SUCCESS` message when the results are the same. The comments next to each directory above give an overview of what they contain.  
> [!NOTE]  
> The `report.txt` is of special interest as it gives the user information on the simulation in both success and failure situations. If successful, the user will get information on runtime and cycle count. Otherwise, information on the cause of the failure will be reported. 

> [!TIP]
> The `vsim.wlf` file in the `HLS_VERIFY` directory contains information on simulation, the different signals and their transitions over time.  


### Visualization Results
> [!IMPORTANT]
> Dynamatic must have been build with Godot installed and the `--visual-dataflow` flag to use this feature. See [interactive visualizer setup](AdvancedBuild.md#4-interactive-dataflow-circuit-visualizer)  

The `visualize` command creates an `out/visual` directory where a LOG file is generated from the Modelsim/Questa wlf file created during simulation. he LOG file is converted to CSV and visualized using the Godot game engine, alongside the DOT file that represents the circuit structure.  

### Vivado Synthesis Results
> [!IMPORTANT]
> Vivado must be installed and sourced before running this command  

The `synthesize` command creates an `out/synth` directory where timing and resource information is logged. Users can view information on:
- clock period and timing violations
- resource utilization
- report on vivado synthesis  
The file names are intuitive and would allow users to find the information they need  