# Script to run characterization of dataflow units
import argparse
import json
import os
from itertools import product
import re
from typing import List, Tuple

skipping_units = [
    "handshake.constant",
    "handshake.lsq",
    "handshake.sharing_wrapper",
    "handshake.ready_remover",
    "handshake.valid_merger",
    "handshake.extsi",
    "handshake.extui",
    "handshake.trunci",
    "handshake.truncf",
    "handshake.buffer",
    "handshake.ndwire",
    "handshake.fork",
    "handshake.lazy_fork",
    "handshake.sink",
    "handshake.mem_controller"]

parameters_ranges = { 
    "DATA_TYPE": [1, 2, 4, 8, 16, 32, 64],
    "SIZE": [2],
    "SELECT_TYPE": [2],
    "INDEX_TYPE": [2],
    "ADDR_TYPE": [64]
    }

def extract_generics_ports(vhdl_code):
    """
    Extract generics and ports from a VHDL entity block.
    Args:
        vhdl_code (str): VHDL code as a string.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of generics
            - List of ports
    """

    # Remove comments
    vhdl_code = re.sub(r'--.*', '', vhdl_code)

    # Match the entity block
    entity_match = re.search(r'entity\s+\w+\s+is(.*?)end\s+entity', vhdl_code, re.DOTALL | re.IGNORECASE)
    if not entity_match:
        raise ValueError("Could not find VHDL entity block.")

    match = re.search(r'\bentity\s+(\w+)\s+is', vhdl_code, re.IGNORECASE)
    entity_name = match.group(1) if match else "unknown_entity"
    entity_block = entity_match.group(1)

    # Extract generics and ports
    generics_match = re.search(r'generic\s*\((.*?)\)\s*;', entity_block, re.DOTALL | re.IGNORECASE)
    ports_match    = re.search(r'port\s*\(((?:[^()]*|\([^()]*\))*)\)\s*;', entity_block, re.DOTALL | re.IGNORECASE)

    generics_raw = generics_match.group(1).strip() if generics_match else ''
    ports_raw    = ports_match.group(1).strip() if ports_match else ''

    # Split on semicolon while keeping line breaks (in case of multiple declarations)
    def split_definitions(raw: str) -> List[str]:
        lines = re.split(r';\s*\n', raw)
        return [line.strip() for line in lines if line.strip()]

    generics = split_definitions(generics_raw)
    ports    = split_definitions(ports_raw)

    return entity_name, generics, ports


def extract_template_top(top_def_file, param_names):
    """
    Extract the template for the top file from the given top definition file.
    
    Args:
        top_def_file (str): Path to the top definition file.
        
    Returns:
        str: The template for the top file.
    """
    print(f"Extracting template top from {top_def_file}")
    with open(top_def_file, 'r') as f:
        vhdl_code = f.read()
    
    entity_name, generics, ports = extract_generics_ports(vhdl_code)
    for _generic in generics:
        generic = _generic.split(":")[0].strip()  # Get the generic name before the colon
        assert generic in param_names, f"Generic `{generic}` not found in parameter names."

    # Create constants for each generic
    constants = ""
    for param in param_names:
        constants += f"constant {param}_const : integer := {param}_const_value;\n"
    # Create wires to connect the ports
    wires = ""
    for port in ports:
        port_name = port.replace("in", "").replace("out", "").strip()
        port_name = f"signal {port_name}"
        for param in param_names:
            port_name = port_name.replace(f"{param}", f"{param}_const")
        wires += f"{port_name};\n"
    template_top = "library ieee;\n" \
                     "use ieee.std_logic_1164.all;\n" \
                        "use ieee.numeric_std.all;\n" \
                        "use ieee.math_real.all;\n" \
                        "use work.types.all;\n\n" \
                        f"{constants}" \
                        f"{wires}" \
                        f"dut: entity work.{entity_name}\n" \
                        "generic map (\n"
    for param in param_names:
        template_top += f"    {param} => {param}_const,\n"
    template_top = template_top.rstrip(",\n") + "\n" \
                     ")\n" \
                        "port map (\n"
    for port in ports:
        port_name = port.split(":")[0].strip()  # Get the port name before the colon
        template_top += f"    {port_name} => {port_name},\n"
    template_top = template_top.rstrip(",\n") + "\n" \
                     ");\n"
    return template_top



def run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool, top_def_file):
    """
    Run characterization for a single unit using the specified synthesis tool.
    
    Args:
        unit_name (str): Name of the unit to characterize.
        list_params (list): List of parameters for the unit.
        hdl_out_dir (str): Directory where HDL files are stored.
        synth_tool (str): Synthesis tool to use for characterization (e.g., 'vivado').
    """
    # Generate top files for all the combination of parameters
    params_charact = {}
    for param in list_params:
        param_name = param["name"]
        assert param_name in parameters_ranges, f"Parameter {param_name} not found in parameters_ranges."
        param_values = parameters_ranges[param_name]
        params_charact[param_name] = param_values
    # Compute all combinations of parameters
    param_combinations = list(product(*params_charact.values()))
    param_names = list(params_charact.keys())
    # Extract the template for the top file
    template_top = extract_template_top(top_def_file, param_names)
    # Create a top file for each combination of parameters
    id = 0
    for combination in param_combinations:
        top = f"{hdl_out_dir}/{unit_name}_top_{id}.vhd" 
        template_top_combined = template_top
        for param_name, param_value in zip(param_names, combination):
            # Replace the constant value in the template
            template_top_combined = template_top_combined.replace(f"{param_name}_const_value", str(param_value))
        with open(top, 'w') as f:
            f.write(template_top_combined)
        id += 1
        assert False, "Implementation not complete yet."

def get_hdl_files(unit_name, generic, generator, dependencies, hdl_out_dir, dynamatic_dir, dependency_list):
    """
    Generate or copy the HDL files for the given unit.
    
    Args:
        unit_name (str): Name of the unit.
        generic (str): Generic information for the unit.
        generator (str): Generator information for the unit.
        dependencies (list): List of dependencies for the unit.
        hdl_out_dir (str): Directory where HDL files should be stored.
        
    Returns:
        str: Path to the generated or copied HDL file.
    """
    # Add dependency RTL files to the output directory
    remaining_dependencies = dependencies.copy()
    while remaining_dependencies:
        dependency_unit = remaining_dependencies.pop(0)
        # Find the dependecy unit location
        assert dependency_unit in dependency_list, f"Dependency {dependency_unit} not found in dependency list."
        dependency_info = dependency_list[dependency_unit]
        dependency_rtl = dependency_info["RTL"]
        dependency_rtl = dependency_rtl.replace("$DYNAMATIC", dynamatic_dir)
        os.system(f"cp {dependency_rtl} {hdl_out_dir}")
        # Add the dependencies of this dependency to the remaining dependencies
        if "dependencies" in dependency_info:
            remaining_dependencies.extend(dependency_info["dependencies"])

    # Check if the unit has RTL file
    if generic:
        # If generic is provided, copy the RTL file to the output directory
        rtl_file = generic
        rtl_file = rtl_file.replace("$DYNAMATIC", dynamatic_dir)
        os.system(f"cp {rtl_file} {hdl_out_dir}")
    else:
        assert generator, "Unit must have either a generic RTL file or a generator."
        cmd = generator.replace("$DYNAMATIC", dynamatic_dir).replace("$OUTPUT_DIR", hdl_out_dir).replace("$MODULE_NAME", unit_name).replace("$PREDICATE", "ne")
        # If a generator is provided, run the generator command
        print(f"Running generator command: {cmd}")
        os.system(cmd)        
        # Assert that the RTL file was generated
        rtl_file_vhdl = f"{hdl_out_dir}/{unit_name}.vhd"  # Assuming
        rtl_file_verilog = f"{hdl_out_dir}/{unit_name}.v"
        assert os.path.exists(rtl_file_vhdl) or os.path.exists(rtl_file_verilog), f"RTL file for unit {unit_name} was not generated or copied successfully."
        rtl_file = rtl_file_vhdl if os.path.exists(rtl_file_vhdl) else rtl_file_verilog
    
    return rtl_file

def extract_rtl_info(unit_info):
    """
    Extract RTL information from the unit_info dictionary.
    
    Args:
        unit_info (dict): Dictionary containing unit information.
        
    Returns:
        tuple: A tuple containing the unit name, list of parameters, generic, generator, and dependencies.
    """
    unit_name = unit_info.get("name")
    list_params = unit_info["parameters"]
    generic = unit_info.get("generic", None)
    generator = unit_info.get("generator", None)
    dependencies = unit_info.get("dependencies")

    return unit_name, list_params, generic, generator, dependencies

def get_dependency_list(dataflow_units):
    """
    Extract a list RTLs of all possible dependencies from the dataflow units.
    Args:
        dataflow_units (list): List of dataflow unit dictionaries.
    Returns:
        list: A list of unique dependencies.
    """
    dependency_list = {}
    for unit_info in dataflow_units:
        if not "name" in unit_info:
            rtl_file = unit_info.get("generic")
            assert rtl_file, "Unit info must contain a 'generic' field for RTL file."
            unit_name = rtl_file.split("/")[-1].split(".")[0]
            if "dependencies" in unit_info:
                dependencies = unit_info["dependencies"]
            else:
                dependencies = []
            dependency_list[unit_name] = {"RTL": rtl_file, "dependencies": dependencies}
        elif "module-name" in unit_info:
            unit_name = unit_info["module-name"]
            rtl_file = unit_info["generic"]
            if "dependencies" in unit_info:
                dependencies = unit_info["dependencies"]
            else:
                dependencies = []
            dependency_list[unit_name] = {"RTL": rtl_file, "dependencies": dependencies}
        else:
            continue
    return dependency_list

def run_characterization(json_input, json_output, dynamatic_dir, synth_tool):
    """
    Run characterization of dataflow units based on the provided JSON input.
    
    Args:
        json_input (str): Path to the input JSON file containing dataflow unit RTL information.
        json_output (str): Path to the output JSON file where characterization results will be saved.
        dynamatic_dir (str): Path to the DYNAMATIC home directory.
        synth_tool (str): Synthesis tool to use for characterization (e.g., 'vivado').
    """
    # Load the input JSON file
    with open(json_input, 'r') as f:
        dataflow_units = json.load(f)

    tmp_dir = f"{dynamatic_dir}/tools/backend/synth-characterization/tmp"
    # Generate the temporary directory if it does not exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # Generate hdl directory
    hdl_dir = f"{tmp_dir}/hdl"
    if not os.path.exists(hdl_dir):
        os.makedirs(hdl_dir)

    dependency_list = get_dependency_list(dataflow_units)

    for unit_info in dataflow_units:
        # Extract the unit name and its RTL information
        unit_name, list_params, generic, generator, dependencies = extract_rtl_info(unit_info)
        if unit_name in skipping_units:
            print(f"Skipping unit {unit_name} as it is in the skipping list.")
            continue
        # We assume that units with no unit_name are just for dependencies
        if unit_name == None:
            continue
        # If the unit has a "DATA_TYPE" parameter with data = 0, skip since it's a dataless unit
        if len(list_params) > 0:
            skip_unit = False
            for param in list_params:
                if param["name"] == "DATA_TYPE" and "data-eq" in param and param["data-eq"] == 0:
                    skip_unit = True
                    break
            if skip_unit:
                continue
        # Clean previous RTL files
        os.system(f"rm -rf {hdl_dir}/*")
        # Copy the RTL files or generate them if necessary
        print(f"Processing unit: {unit_name}")
        top_def = get_hdl_files(unit_name, generic, generator, dependencies, hdl_dir, dynamatic_dir, dependency_list)
        # After generating the HDL files, we can proceed with characterization
        results = run_unit_characterization(unit_name, list_params, hdl_dir, synth_tool, top_def)

    # Save the results to the output JSON file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run characterization of dataflow units")
    parser.add_argument(
        "--json-input",
        type=str,
        default=None,
        help="Path to the JSON file containing the dataflow unit RTL information (if unspecified, $DYNAMATIC_DIR/data/rtl-config-vhdl-vivado.json)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        required=True,
        help="Path to the output JSON file where the characterization results will be saved",
    )
    parser.add_argument(
        "--dynamatic-dir",
        type=str,
        required=True,
        help="Path to the DYNAMATIC home directory",
    )
    parser.add_argument(
        "--synth-tool",
        type=str,
        default="vivado",
        help="Synthesis tool to use for characterization (default: vivado)",
    )
    args = parser.parse_args()
    json_input = args.json_input
    json_output = args.json_output
    dynamatic_dir = args.dynamatic_dir
    synth_tool = args.synth_tool
    if not json_input:
        json_input = f"{dynamatic_dir}/data/rtl-config-vhdl-vivado.json"
    run_characterization(json_input, json_output, dynamatic_dir, synth_tool)