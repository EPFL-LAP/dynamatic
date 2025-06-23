# Script to run characterization of dataflow units
import argparse
import json
import os

skipping_units = [
    "handshake.constant",
    "handshake.lsq",
    "handshake.sharing_wrapper",
    "handshake.ready_remover",
    "handshake.valid_merger"]

def run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool):
    """
    Run characterization for a single unit using the specified synthesis tool.
    
    Args:
        unit_name (str): Name of the unit to characterize.
        list_params (list): List of parameters for the unit.
        hdl_out_dir (str): Directory where HDL files are stored.
        synth_tool (str): Synthesis tool to use for characterization (e.g., 'vivado').
    """
    # Generate all the combination of parameters
    pass

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
        os.system(f"cp {dependency_rtl} {hdl_out_dir}")
        # Add the dependencies of this dependency to the remaining dependencies
        if "dependencies" in dependency_info:
            remaining_dependencies.extend(dependency_info["dependencies"])

    # Check if the unit has RTL file
    if generic:
        # If generic is provided, copy the RTL file to the output directory
        rtl_file = generic
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

def extract_rtl_info(unit_info):
    """
    Extract RTL information from the unit_info dictionary.
    
    Args:
        unit_info (dict): Dictionary containing unit information.
        
    Returns:
        tuple: A tuple containing the unit name, list of parameters, generic, generator, and dependencies.
    """
    unit_name = unit_info.get("name")
    list_params = unit_info.get("params", [])
    generic = unit_info.get("generic", None)
    generator = unit_info.get("generator", None)
    dependencies = unit_info.get("dependencies", [])
    
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
        # Clean previous RTL files
        os.system(f"rm -rf {hdl_dir}/*")
        # Copy the RTL files or generate them if necessary
        print(f"Processing unit: {unit_name}")
        get_hdl_files(unit_name, generic, generator, dependencies, hdl_dir, dynamatic_dir, dependency_list)
        # After generating the HDL files, we can proceed with characterization
        results = run_unit_characterization(unit_name, list_params, hdl_dir, synth_tool)

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