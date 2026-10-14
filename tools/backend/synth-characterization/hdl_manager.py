# This script is used to move the HDL files in the hdl_out_dir and generate the necessary HDL files for the unit.
import os
from utils import ip_library_sources, GENERATOR_EXTRA_SIGNALS

def copy_dependency_rtl_files(unit_name, dependency_dict, hdl_out_dir, dynamatic_dir):
    """
    Copy the RTL files of the dependencies of the given unit to the output directory.

    Args:
        unit_name (str): Name of the unit for which dependencies are to be copied.
        dependency_dict (dict): Dictionary containing the list of dependencies for all units.
        hdl_out_dir (str): Directory where HDL files should be stored.
        dynamatic_dir (str): Path to the Dynamatic directory.

    Returns:
        list: Xilinx IP libraries needed by the dependencies, as (library, source file relative to the Vivado installation).
    """
    ip_libraries = []
    # Add dependency RTL files to the output directory
    remaining_dependencies = dependency_dict[unit_name]["dependencies"].copy()
    while remaining_dependencies:
        dependency_unit = remaining_dependencies.pop(0)
        for ip_library in ip_library_sources.get(dependency_unit, []):
            if ip_library not in ip_libraries:
                ip_libraries.append(ip_library)
        # Find the dependecy unit location
        assert dependency_unit in dependency_dict, f"Dependency {dependency_unit} not found in dependency list."
        dependency_info = dependency_dict[dependency_unit]
        dependency_rtl = dependency_info["RTL"]
        dependency_rtl = dependency_rtl.replace("$DYNAMATIC", dynamatic_dir)
        if "_dataless" in dependency_unit:
            # If the dependency is a dataless unit, we copy it with a different name
            rtl_filename = dependency_rtl.split("/")[-1].replace(".vhd", "_dataless.vhd")
            os.system(f"cp {dependency_rtl} {hdl_out_dir}/{rtl_filename}")
        else:
            os.system(f"cp {dependency_rtl} {hdl_out_dir}")
        # Add the dependencies of this dependency to the remaining dependencies
        if "dependencies" in dependency_info:
            remaining_dependencies.extend(dependency_info["dependencies"])

    extra_dependencies = ["types", "logic", "oehb", "oehb_dataless", "br_dataless"]
    # Add extra dependencies that are not in the dependency list
    for extra_dependency in extra_dependencies:
        # Generated units include these in their own RTL
        if extra_dependency not in dependency_dict:
            continue
        extra_rtl = dependency_dict[extra_dependency]["RTL"]
        extra_rtl = extra_rtl.replace("$DYNAMATIC", dynamatic_dir)
        rtl_filename = extra_rtl.split("/")[-1]
        if "_dataless" in extra_dependency:
            os.system(f"cp {extra_rtl} {hdl_out_dir}/{rtl_filename.replace('.vhd', '_dataless.vhd')}")
        else:
            os.system(f"cp {extra_rtl} {hdl_out_dir}")

    return ip_libraries

def get_hdl_files(unit_name, generic, generator, hdl_out_dir, dynamatic_dir, dependency_dict, generator_params={}):
    """
    Generate or copy the HDL files for the given unit.
    
    Args:
        unit_name (str): Name of the unit.
        generic (str): Generic information for the unit.
        generator (str): Generator information for the unit.
        hdl_out_dir (str): Directory where HDL files should be stored.
        dynamatic_dir (str): Path to the Dynamatic directory.
        dependency_dict (dict): Dictionary containing the list of dependencies for all units.
        generator_params (dict): Values of the parameters of the generator command (e.g., {"BITWIDTH": 32}).

    Returns:
        str: Path to the generated or copied HDL file.
        list: Xilinx IP libraries needed by the unit, as (library, source file relative to the Vivado installation).
    """
    # Ensure the output directory exists
    if not os.path.exists(hdl_out_dir):
        os.makedirs(hdl_out_dir)
    # Copy the RTL files of the dependencies to the output directory
    ip_libraries = copy_dependency_rtl_files(unit_name, dependency_dict, hdl_out_dir, dynamatic_dir)
    # Check if the unit has RTL file
    if generic:
        # If generic is provided, copy the RTL file to the output directory
        rtl_file = generic
        rtl_file = rtl_file.replace("$DYNAMATIC", dynamatic_dir)
        os.system(f"cp {rtl_file} {hdl_out_dir}")
    else:
        assert generator, "Unit must have either a generic RTL file or a generator."
        simplified_unit_name = unit_name.split(".")[-1]  # Get the last part of the unit name
        cmd = generator.replace("$DYNAMATIC", dynamatic_dir).replace("$OUTPUT_DIR", hdl_out_dir).replace("$MODULE_NAME", simplified_unit_name).replace("$PREDICATE", "ne").replace("$EXTRA_SIGNALS", GENERATOR_EXTRA_SIGNALS)
        for param_name, param_value in generator_params.items():
            cmd = cmd.replace(f"${param_name}", str(param_value))
        # If a generator is provided, run the generator command
        print(f"Running generator command: {cmd}")
        os.system(cmd)        
        # Assert that the RTL file was generated
        rtl_file_vhdl = f"{hdl_out_dir}/{simplified_unit_name}.vhd"  # Assuming
        rtl_file_verilog = f"{hdl_out_dir}/{simplified_unit_name}.v"
        assert os.path.exists(rtl_file_vhdl) or os.path.exists(rtl_file_verilog), f"RTL file for unit {unit_name} was not generated or copied successfully."
        rtl_file = rtl_file_vhdl if os.path.exists(rtl_file_vhdl) else rtl_file_verilog
    
    return rtl_file, ip_libraries