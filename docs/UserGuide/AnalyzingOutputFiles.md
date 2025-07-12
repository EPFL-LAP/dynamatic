[Documentation Table of Contents](../README.md)  
# Analyzing Output Files
Running the main commands in Dynamatic creates different directories containing relevant files that can be used by the user for various purposes. By default, Dynamatic creates output and temporary files in an `out/` directory where the C source code is located. This document describes the organization of the `out/` directory when different commands are ran. 

### Compile
After setting the source for the C file you want to target with Dynamatic, the first command you would generally run is the `compile` command.  
- It creates an `out/comp` directory that stores all the intermediate files as described in the [Dynamatic HLS flow](../DeveloperGuide/DynamaticHLSFlow.md).
- A file is created for every step of the compilation process, allowing the user to inspect relevant files if any unexpected behaviour results.
> [!TIP]
> Some knowledge of MLIR is required for accurate inspection of these files

### Write-HDL
The `write-hdl` command creates an `out/hdl` directory containing all the HDL files needed to implement the target function.  
Amongst these is a `top level` HDL file named as target function where all other HDL files in the directory are used to implement the target kernel.

### Simulate
> [!IMPORTANT]
> Modelsim/Questa must be installed and added to path before running this command. See [Modelsim/Questa installation guide](AdvancedBuild.md#6-modelsimquesta-installation)  

The `simulate` command creates an `out/sim` directory. In this directory are a number of sub directories organized as shown below:
```
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

### Visualize
> [!IMPORTANT]
> Dynamatic must have been build with Godot installed and the `--visual-dataflow` flag to use this feature. See [interactive visualizer setup](AdvancedBuild.md#4-interactive-dataflow-circuit-visualizer)  

The `visualize` command creates an `out/visual` directory where a LOG file is generated from the Modelsim/Questa wlf file created during simulation. The LOG file is converted to DOT and visualized using the Godot game engine.  
> [!NOTE]
> The `vsim.wlf` file contains information on simulation, the different signals and their transitions over time  

### Synthesize
> [!IMPORTANT]
> Vivado must be installed and sourced before running this command  

The `synthesize` command creates an `out/synth` directory where timing and resource information is logged. Users can view information on:
- clock period and timing violations
- resource utilization
- report on vivado synthesis  
The file names are intuitive and would allow users to find the information they need  