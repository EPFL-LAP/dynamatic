# Script to run characterization of dataflow units
import argparse
import json
import os
from report_parser import extract_rpt_data
from hdl_manager import get_hdl_files
from utils import VhdlInterfaceInfo, parameters_ranges, skipping_units
from unit_characterization import run_unit_characterization

def extract_rtl_info(unit_info):
    """
    Extract RTL information from the unit_info dictionary.
    
    Args:
        unit_info (dict): Dictionary containing unit information.
        
    Returns:
        tuple: A tuple containing the unit name, list of parameters, generic, generator, and dependencies.
    """
    unit_name = unit_info.get("name")
    list_params = unit_info.get("parameters", [])
    generic = unit_info.get("generic", None)
    generator = unit_info.get("generator", None)
    dependencies = unit_info.get("dependencies", [])

    return unit_name, list_params, generic, generator, dependencies

# Extract dependencies from the dataflow units
# This function is essential to move the right RTL files
# Example of a dependency dictionary:
# {
#     "unit_name": {"RTL": "path/to/rtl_file.vhd", "dependencies": ["dep1", "dep2"]},
#     "dep1": {"RTL": "path/to/dep1.vhd", "dependencies": []},
#     "dep2": {"RTL": "path/to/dep2.vhd", "dependencies": []},
#     ...
# }
# The dependencies are used to copy the necessary RTL files to the output directory
def get_dependency_dict(dataflow_units):
    """
    Extract the RTLs of all possible dependencies from the dataflow units.
    Args:
        dataflow_units (list): List of dataflow unit dictionaries.
    Returns:
        list: A list of unique dependencies.
    """
    dependency_dict = {}
    for unit_info in dataflow_units:
        if not "name" in unit_info:
            rtl_file = unit_info.get("generic")
            assert rtl_file, "Unit info must contain a 'generic' field for RTL file."
            unit_name = rtl_file.split("/")[-1].split(".")[0]
            if "dependencies" in unit_info:
                dependencies = unit_info["dependencies"]
            else:
                dependencies = []
            dependency_dict[unit_name] = {"RTL": rtl_file, "dependencies": dependencies}
        elif "module-name" in unit_info:
            unit_name = unit_info["module-name"]
            rtl_file = unit_info["generic"]
            if "dependencies" in unit_info:
                dependencies = unit_info["dependencies"]
            else:
                dependencies = []
            dependency_dict[unit_name] = {"RTL": rtl_file, "dependencies": dependencies}
        else:
            rtl_file = unit_info.get("generic")
            if not rtl_file:
                rtl_file = unit_info.get("generator")
            dependencies = unit_info.get("dependencies", [])
            unit_name = unit_info["name"]
            dependency_dict[unit_name] = {"RTL": rtl_file, "dependencies": dependencies}
    return dependency_dict


def run_characterization(json_input, json_output, dynamatic_dir, synth_tool, clock_period):
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
    # Generate tcl directory
    tcl_dir = f"{tmp_dir}/tcl"
    if not os.path.exists(tcl_dir):
        os.makedirs(tcl_dir)
    # Generate report directory
    rpt_dir = f"{tmp_dir}/reports"
    if not os.path.exists(rpt_dir):
        os.makedirs(rpt_dir)
    # Generate logs directory
    log_dir = f"{tmp_dir}/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    all_dependencies_dict = get_dependency_dict(dataflow_units)

    map_unit_to_list_unit_chars = {}

    for unit_info in dataflow_units:
        # Extract the unit name and its RTL information
        unit_name, list_params, generic, generator, dependencies = extract_rtl_info(unit_info)
        if unit_name == None:
            print("Skipping unit with no name.")
            continue
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
        # Clean previous RTL files and tcl files
        os.system(f"rm -rf {hdl_dir}/*")
        os.system(f"rm -rf {tcl_dir}/*")
        # Copy the RTL files or generate them if necessary
        print(f"Processing unit: {unit_name}")
        top_def_file = get_hdl_files(unit_name, generic, generator, hdl_dir, dynamatic_dir, all_dependencies_dict)
        # After generating the HDL files, we can proceed with characterization
        list_unit_chars = run_unit_characterization(unit_name, list_params, hdl_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir, clock_period)
        # Store the results in the map_unit_to_list_unit_chars dictionary
        map_unit_to_list_unit_chars[unit_name] = list_unit_chars
    
    # Save the results to the output JSON file
    extract_rpt_data(map_unit_to_list_unit_chars, json_output)

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
    parser.add_argument(
        "--clock-period",
        type=float,
        default=4.0,
        help="Clock period in nanoseconds to use for synthesis (default: 4.0 ns)",
    )
    args = parser.parse_args()
    json_input = args.json_input
    json_output = args.json_output
    dynamatic_dir = args.dynamatic_dir
    synth_tool = args.synth_tool
    clock_period = args.clock_period
    if not json_input:
        json_input = f"{dynamatic_dir}/data/rtl-config-vhdl-vivado.json"
    run_characterization(json_input, json_output, dynamatic_dir, synth_tool, clock_period)