# BlifGenerator

MapBuf Buffer Placement Algorithm needs AIGs (AND-Invert Graphs) of all hardware modules. To automate the AIG generation, a script is provided to convert Verilog modules into BLIF (Berkeley Logic Interchange Format) files.

This document explains how to use and extend this script.

## Requirements
- Python **3.6** or later.
- YOSYS  **0.44** or later.
- ABC  **1.01** or later.

## Running the Script
The script accepts an optional argument specifying a hardware module name. If provided, only that module’s BLIF will be generated. Otherwise, BLIF files will be created for all supported modules.


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
$ python3 tools/blif-generator.py addi
```

## Directory Structure
Generated BLIF files are stored under:
```
/data/blif/<module_name>/<param1>/<param2>/.../<module_name>.blif
```
Parameter subdirectories are created based on the order of definition in the Verilog files.

**Example:**
For mux with SIZE=2, DATA_TYPE=5, SELECT_TYPE=1:
```
/data/blif/mux/2/5/1/mux.blif
```


## BLIF Generation Flow

1) The script loops over all modules one by one.

2) All parameter combinations of a module is iterated over, retrieved by get_range_for_param() function.

3) It creates the corresponding YOSYS script to synthesize the module.

4) An ABC script then generates the AIG of the module.

5) Both Yosys and ABC scripts as well as Yosys output is saved alongside the final BLIF file for debugging.

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

## Extending the Script with New Hardware Modules
If a new hardware module is added to Dynamatic, the script needs to be extended with this module. 

1) Add it to the appropriate dictionary:
  - ARITH_MODULES for arithmetic modules
  - HANDSHAKE_MODULES for handshake modules

Add the module to the corresponding dictionary, ARITH_MODULES for arith modules and HANDSHAKE_MODULES for handshake modules.

2) Use the module name as a key, and a list of parameter names (in Verilog declaration order) as the value.

3) No new additions are needed for the Verilog paths. The script automatically reads all Verilog files.

**Example:**
If a new handshake module with name TEST_MODULE and with parameters SIZE and DATA_TYPE is added to Dynamatic, HANDSHAKE_MODULES will be extended with:
```
HANDSHAKE_MODULES = {
  ...,
  'TEST_MODULE': ['SIZE', 'DATA_TYPE']
}
```

Then the script can be run with:
```
$ python3 tools/blif-generator.py TEST_MODULE
```

to generate the BLIF files.

