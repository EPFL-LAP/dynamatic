# BlifGenerator

MapBuf Buffer Placement Algorithm needs AIGs (AND-Invert Graphs) of all hardware modules. To automate the AIG generation, a script is provided to convert Verilog modules into BLIF (Berkeley Logic Interchange Format) files.

This document explains how to use and extend this script.

## Requirements
- Python **3.6** or later.
- YOSYS  **0.44** or later.
- ABC  **1.01** or later.

## Running the Script
The script accepts an optional argument specifying a hardware module name. If provided, only that module’s BLIF will be generated. Otherwise, BLIF files will be created for all supported modules.

ABC and YOSYS needs to be added to 

### Generating BLIF for All Modules
```
$ python3 tools/blif-generator.py
```

### Generating BLIF for a Single Module
```
$ python3 tools/blif-generator.py (module_name)
```

Example for generating BLIF files of addi:
```
$ python3 tools/blif-generator.py handshake.addi
```

## Configuration
The script uses the JSON configuration file located at:
```
$DYNAMATIC/data/rtl-config-verilog.json
```
This file defines all module specifications including:

- Module names and paths to Verilog files
- Parameter definitions
- Dependencies between modules
- Generator commands for some modules
 
## Directory Structure
Generated BLIF files are stored under:
```
/data/blif/<module_name>/<param1>/<param2>/.../<module_name>.blif
```
Parameter subdirectories are created based on the order of definition in specified in the JSON file.

**Example:**
For mux with SIZE=2, DATA_TYPE=5, SELECT_TYPE=1:
```
/data/blif/mux/2/5/1/mux.blif
```

## BLIF Generation Flow

1) The script loads module configurations from the JSON file.

2) For each module, it retrieves the dependencies recursively to collect the Verilog files needed to synthesize the module.

3) Parameter combinations are generated based on the definitions in the JSON file.

4) For modules with generators, the generator is executed to create custom Verilog files.

5) A YOSYS script is created and executed to synthesize the module.

6) An ABC script then generates the AIG of the module.

7) Blackbox processing is applied to specific modules (addi, cmpi, subi, muli, divsi, divui).

8) Both Yosys and ABC scripts as well as intermediate files are saved for debugging.

## Key Features
### Recursive Dependency Resolution:
The script automatically automatically resolves complete dependency tree by recursively collecting the dependencies. For example, when module A depends on module B, and module B depends on module C, ```collect_dependencies_recursive()``` function ensures module C is also added as a dependency. 

### Parameter Handling
- Range-based iteration: Uses get_range_for_param() for upper bounds. For example, SIZE parameters iterate from 1-10, while DATA_TYPE parameters span 1-33, ensuring AIGs are generated for all possible parameter choices.
- Constraint support: Handles eq, data-eq, lb, data-lb constraints. If eq or data-eq are set, the iteration values retrieved from the get_range_for_param() are not used.

## Blackbox Processing
The following modules are automatically converted to blackboxes:

- addi, cmpi, subi: For DATA_TYPE > 4, removes .names lines (except ready/valid signals) in the BLIF.
- muli: Removes all .names and .latch lines for all DATA_TYPEs.
- divsi and divui: BLIF file is copied from the BLIFs generated fro muli.


## Extending the Script with New Hardware Modules
If a new hardware module is added to Dynamatic, for most cases, it is sufficient to simply add the module in the JSON configuration. Therefore no script modifications are required. However, if the module is not mapped to LUTs but mapped to carry-chains or DSPs (e.g., addi, muli units), an additional step is necessary. The module's name must be added to the BLACKBOX_COMPONENTS list. Once this is done, the script can be run as usual.

```
$ python3 tools/blif-generator.py {new_module}
```

### Yosys Commands
```
yosys -p
read_verilog -defer <verilog_files>
chparam -set <parameters> <module_name>
hierarchy -top <module_name>;
proc;
opt -nodffe -nosdff;
memory -nomap;
techmap;
flatten;
clean;
write_blif <dest_file>
```

### ABC Commands
```
abc -c "read_blif <source_file>;
strash;
rewrite;
b;
refactor;
b;
rewrite;
b;
refactor;
b;
rewrite;
b;
refactor;
b;
rewrite;
b;
refactor;
b;
rewrite;
b;
refactor;
b;
rewrite;
b;
refactor;
b;
write_blif <dest_file>"
```

