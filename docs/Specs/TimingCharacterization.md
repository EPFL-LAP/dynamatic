# Dataflow Unit Characterization Script Documentation

This document describes the structure and functionality of the Dataflow Unit Characterization script, which automates the process of characterizing hardware dataflow units. It details the conceptual approach, data handling, and implementation, following the style and clarity of the provided timing documentation.

## Overview

The script automates the extraction of VHDL entity information, testbench generation, synthesis script creation, dependency management, and parallel synthesis execution. Its primary goal is to characterize hardware units by sweeping parameter values and collecting synthesis/timing results.

## What is Unit Characterization?

Unit characterization refers to the systematic process of evaluating hardware units (e.g., VHDL modules) for various configurations. The script supports:

- **Parameter Sweeping**: Automatically varying generic parameters (e.g., bitwidth, depth) and generating the corresponding testbenches and synthesis scripts.
- **Dependency Resolution**: Ensuring all required VHDL files and dependencies are included for synthesis.
- **Parallel Synthesis**: Running multiple synthesis jobs concurrently to speed up characterization.
- **Automated Reporting**: Collecting and organizing timing and resource reports for each configuration.


## Where Characterization Data is Stored

All generated files and results are organized in a user-specified directory structure:

- **HDL Output Directory**: Contains all generated/copy VHDL files for each unit and configuration.
- **TCL Directory**: Stores synthesis scripts for each configuration.
- **Report Directory**: Contains timing and resource reports produced by the synthesis tool.
- **Log Directory**: Stores log files for each synthesis run.

Each configuration (i.e., a unique set of parameter values) is associated with its own set of files, named to reflect the parameter values used.

## Core Data Structures and Functions

The script uses several key functions and data structures to orchestrate characterization:

### Parameter Management

- **parameters_ranges**:
A dictionary mapping parameter names to lists of values to sweep. Enables exhaustive exploration of the design space.


### Entity Extraction

- **extract_generics_ports(vhdl_code, entity_name)**
Parses VHDL code to extract the list of generics (parameters) and ports for the specified entity.
    - Removes comments for robust parsing.
    - Handles multiple entity definitions in a single file.
    - Returns: `(entity_name, generics, ports)`.


### Testbench Generation

- **extract_template_top(entity_name, generics, ports, param_names)**
Produces a VHDL testbench template for the entity, with generics mapped to parameter placeholders.
    - Ensures all generics are parameterized.
    - Handles port mapping for instantiation.


### Synthesis Script Generation

- **write_tcl(top_file, top_entity_name, hdl_files, tcl_file, sdc_file, rpt_timing, ports)**
Generates a TCL script for the synthesis tool (e.g., Vivado), including:
    - Reading HDL and constraint files.
    - Synthesizing and implementing the design.
    - Generating timing reports for relevant port pairs.
- **write_sdc_constraints(sdc_file, period_ns)**
Creates an SDC constraints file specifying the clock period.


### Dependency Handling

- **get_hdl_files(unit_name, generic, generator, dependencies, hdl_out_dir, dynamatic_dir, dependency_list)**
Ensures all required VHDL files (including dependencies) are present in the output directory for synthesis.


### Synthesis Execution

- **run_synthesis(tcl_files, synth_tool, log_file)**
Runs synthesis jobs in parallel using the specified number of CPU cores.
    - Each job is executed with its own TCL script and log file.
- **run_tcl_file(args)**
Worker function for executing a single synthesis job.


### High-Level Flow

- **run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir)**
Orchestrates the full characterization process for a single unit:
    - Gathers all HDL files and dependencies.
    - Extracts entity information and generates testbench templates.
    - Sweeps all parameter combinations, generating top files and TCL scripts for each.
    - Runs synthesis and collects reports.
    - Returns a mapping from report filenames to parameter values.


## Example: Parameter Sweep and Synthesis

Suppose you want to characterize a FIFO unit with varying depths and widths. You would set up `parameters_ranges` as follows:

```python
parameters_ranges = {
    "DEPTH": [8, 16, 32],
    "WIDTH": [8, 16, 32]
}
```

The script will automatically:

- Generate all combinations (e.g., DEPTH=8, WIDTH=8; DEPTH=8, WIDTH=16; ...).
- For each combination, generate a top-level testbench, TCL script, and SDC constraints.
- Run synthesis for each configuration in parallel.
- Collect and store timing/resource reports for later analysis.


## How to Use the Script

1. **Prepare VHDL and Dependency Files**
Ensure all required VHDL files and dependency metadata are available.
2. **Configure Parameters**
Update `parameters_ranges` for the units you wish to characterize.
3. **Run Characterization**
Call `run_unit_characterization` for each unit, specifying the required directories and tool.
4. **Analyze Results**
Timing and synthesis reports are generated for each parameter combination and stored in the designated report directory.

## Notes

- The script is designed for batch automation in hardware design flows, specifically targeting VHDL and Xilinx Vivado.
- It assumes a certain structure for VHDL entities and their dependencies.
- Parallelization is controlled by the `NUM_CORES` variable.
- The script can be extended to support additional synthesis tools or more complex dependency structures.
