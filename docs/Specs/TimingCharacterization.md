# Dataflow Unit Characterization Script Documentation

This document describes how Dynamatic obtains the timing characteristics of the dataflow units. Please check out [this doc](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Specs/TimingInformation.md) if you are unfamiliar with Dynamatic's timing model. 

Dynamatic uses a [Python script](https://github.com/EPFL-LAP/dynamatic/tree/main/tools/backend/synth-characterization/main.py) to obtain the timing characterization.

**NOTE**: The script and the following documentation are tailored for the specific version of Dynamatic and the current status of the structure of the [timing information file](https://github.com/EPFL-LAP/dynamatic/blob/main/data/components-vivado.json). When generating new dataflow units, try to follow the same structure as other dataflow units (in the timing information file and in the VHDL definition). This would make it possible to extend the characterization to new dataflow units.

## What is Unit Characterization?

Unit characterization refers to the systematic process of evaluating hardware units (e.g., VHDL modules) for various configurations. The script supports:

- **Parameter Sweeping**: Automatically varying generic parameters (e.g., bitwidth, depth) and generating the corresponding testbenches and synthesis scripts.
- **Dependency Resolution**: Ensuring all required VHDL files and dependencies are included for synthesis.
- **Parallel Synthesis**: Running multiple synthesis jobs concurrently to speed up characterization.
- **Automated Reporting**: Collecting and organizing timing and resource reports for each configuration.

## How to Use the Script

1. **Prepare VHDL and Dependency Files**
Ensure all required VHDL files and dependency metadata are available.
2. **Configure Parameters**
Update `parameters_ranges` for the units you wish to characterize.
3. **Run Characterization**
Call `run_unit_characterization` for each unit, specifying the required directories and tool.
4. **Analyze Results**
Timing and synthesis reports are generated for each parameter combination and stored in the designated report directory.

### How to `Run Characterization`  

An example on how to call the script is the following one:

`python main.py --json-output out.json --dynamatic-dir /home/dynamatic/ --synth-tool "vivado-2019 vivado"`

which would save the output JSON file in `out.json` which contains timing information, it would specify the dynamatic home directory as `/home/dynamatic/` and it would call vivado using the command `vivado-2019 vivado`. An alternative call is the following one:

`python main.py --json-output out.json --dynamatic-dir /home/dynamatic/ --synth-tool "vivado-2019 vivado" --json-input struct.json`

where the only key difference is the specification of the input JSON (`struct.json`) which contains information related to RTL characteristics of each component. If unspecified, the script will look for the following file `DYNAMATIC_DIR/data/rtl-config-vhdl-vivado.json`.

## Overview

The script automates the extraction of VHDL entity information, testbench generation, synthesis script creation, dependency management, and parallel synthesis execution. Its primary goal is to characterize hardware units by sweeping parameter values and collecting synthesis/timing results.

## Where Characterization Data is Stored

All generated files and results are organized in a user-specified directory structure:

- **HDL Output Directory**: Contains all generated/copy VHDL files for each unit and configuration.
- **TCL Directory**: Stores synthesis scripts for each configuration.
- **Report Directory**: Contains timing and resource reports produced by the synthesis tool.
- **Log Directory**: Stores log files for each synthesis run.

Each configuration (i.e., a unique set of parameter values) is associated with its own set of files, named to reflect the parameter values used.

## Scripts Structure

The scripts are organized according to the following structure:
<pre lang="markdown">. 
├── hdl_manager.py # Moves HDL files from the folder containing all the HDL files to the working directory 
├── report_parser.py # Extracts delay information from synthesis reports 
├── main.py # Main script: orchestrates filtering, generation, synthesis, parsing 
├── run_synthesis.py # Runs synthesis (e.g., with Vivado), supports parallel execution 
├── unit_characterization.py # Coordinates unit-level processing: port handling, VHDL generation, exploration across all parameters 
└── utils.py # Shared helpers: common class definitions and constants </pre>

## Core Data Structures and Functions

The scripts uses several key functions and data structures to orchestrate characterization:

### Parameter Management

- **parameters_ranges**: (File `utils.py`)

    A dictionary mapping parameter names to lists of values to sweep. Enables exhaustive exploration of the design space. 


### Entity Extraction

- **extract_generics_ports(vhdl_code, entity_name)**: (File `unit_characterization.py`)
    
    Parses VHDL code to extract the list of generics (parameters) and ports for the specified entity.
    - Removes comments for robust parsing.
    - Handles multiple entity definitions in a single file.
    - Returns: `(entity_name, VhdlInterfaceInfo)`.

- **VhdlInterfaceInfo**: (File `utils.py`)

    A class that contains information related to generics and ports of a VHDL module

### Testbench Generation

- **generate_wrapper_top(entity_name, VhdlInterfaceInfo, param_names)**: (File `unit_characterization.py`)

    Produces a VHDL testbench wrapper for the entity, with generics mapped to parameter placeholders.
    - Ensures all generics are parameterized.
    - Handles port mapping for instantiation.


### Synthesis Script Generation

- **UnitCharacterization**: (File `utils.py`)

    A class that contains information related to parameters used for a characerization and the corresponding timing reports.

- **write_tcl(top_file, top_entity_name, hdl_files, tcl_file, sdc_file, rpt_timing, VhdlInterfaceInfo)**: (File `utils.py`)

    Generates a TCL script for the synthesis tool (e.g., Vivado), including:
    - Reading HDL and constraint files.
    - Synthesizing and implementing the design.
    - Generating timing reports for relevant port pairs.
- **write_sdc_constraints(sdc_file, period_ns)**: (File `run_synthesis.py`)

    Creates an SDC constraints file specifying the clock period.


### Dependency Handling

- **get_hdl_files(unit_name, generic, generator, dependencies, hdl_out_dir, dynamatic_dir, dependency_list)**: (File `hdl_manager.py`)

    Ensures all required VHDL files (including dependencies) are present in the output directory for synthesis.


### Synthesis Execution

- **run_synthesis(tcl_files, synth_tool, log_file)**: (File `run_synthesis.py`)

    Runs synthesis jobs in parallel using the specified number of CPU cores.
    - Each job is executed with its own TCL script and log file.
- **_synth_worker(args)**: (File `run_synthesis.py`)

    Worker function for executing a single synthesis job.

### Report Parsing

- **extract_rpt_data(map_unit_to_list_unit_chars, json_output)**: (File `report_parser.py`)

    Extract data from the different reports and it saves it into the `json_output` file. 
    The data `map_unit_to_list_unit_chars` contains a mapping between unit and a list of UnitCharacterization objects.
    Please look at the end of this doc to find an example of the structure of the expected report.

### High-Level Flow

- **run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir)**: (File `unit_characterization.py`)

    Orchestrates the full characterization process for a single unit:
    - Gathers all HDL files and dependencies.
    - Extracts entity information and generates testbench templates.
    - Sweeps all parameter combinations, generating top files and TCL scripts for each.
    - Runs synthesis and collects reports.
    - Returns a mapping from report filenames to parameter values.

## Using a New Synthesis Tool

For now the code has some specific information related to Vivado tool. However, adding support for a new backend should not take too long. Here it is a list of places to change to use a different backend:
- `_synth_worker` -> This function runs the synthesis tool. It assumes the tool can be called as follows: `SYNTHESIS_TOOL -mode batch -source TCL_SCRIPT`. 
- `write_tcl` -> This function writes the tcl script with tcl commands specific of Vivado. 
- `write_sdc_constraints` -> This function writes the sdc file and it is tailored for Vivado. It might also require some changes.
- `PATTERN_DELAY_INFO` -> This is a constant string used to identify the line where the report specifies the delay value. This is tailored for Vivado.
- `extract_delay` -> This function extracts the total delay of a path from the reports. This is tailored for Vivado.

These files might require some changes if the synthesis tool has different features from Vivado.

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


## Example: Expected Report Structure

The synthesis report is expected to contain this line `Data Path Delay:        DELAY_VALUEns` which is used to extract its delay.

Please refer to `Using a New Synthesis Tool` section if the lines containing ports and delays information are different in your report.

## Notes

- The script is designed for batch automation in hardware design flows, specifically targeting VHDL and Xilinx Vivado.
- It assumes a certain structure for VHDL entities and their dependencies.
- Parallelization is controlled by the `NUM_CORES` variable.
- The script can be extended to support additional synthesis tools or more complex dependency structures.
