# Script to run characterization of dataflow units
import argparse
import json
import os
from itertools import product
import re
from typing import List, Tuple
from multiprocessing import Pool

NUM_CORES = 10 # Number of cores to use for parallel synthesis (if applicable)

# List of units to skip during characterization 
# These units are either empty, unused, or characterized by other scripts
skipping_units = [
    # empty units
    "handshake.constant", 
    "handshake.br",
    "handshake.source",
    "handshake.extsi",
    "handshake.extui",
    "handshake.trunci",
    "handshake.truncf",
    "handshake.sink",
    "handshake.store",
    "handshake.maximumf",
    "handshake.minimumf",
    "handshake.extf",
    "handshake.divsi",
    "handshake.divui",
    # unused units
    "handshake.join",
    "handshake.sharing_wrapper",
    "handshake.ready_remover",
    "handshake.valid_merger",
    "handshake.ndwire",
    "handshake.mem_controller",
    "mem_to_bram",
    # units characterized by other scripts
    "handshake.fork",
    "handshake.lazy_fork",
    "handshake.lsq",
    "handshake.mulf",
    "handshake.negf",
    "handshake.buffer",
    "handshake.addf",
    "handshake.cmpf",
    "handshake.divf",
    "handshake.subf",
    "handshake.not"]

# List of parameters and their ranges for characterization
# This is used to generate the top files for characterization
parameters_ranges = { 
    "DATA_TYPE": [1, 2, 4, 8, 16, 32, 64],
    "SIZE": [2],
    "SELECT_TYPE": [2],
    "INDEX_TYPE": [2],
    "ADDR_TYPE": [64],
    "PREDICATE": ["ne"]
    }

class VhdlInterfaceInfo:
    """
    Class to hold VHDL interface information.
    This class is used to store the generics and ports of a VHDL entity.
    """
    def __init__(self, generics: List[str], ports: List[str]):
        self.generics = generics
        self.ports = ports
        self.ins, self.outs = self.extract_ins_outs()

    def __repr__(self):
        return f"VhdlInterfaceInfo(generics={self.generics}, ports={self.ports})"

    def extract_ins_outs(self) -> Tuple[List[str], List[str]]:
        """
        Extract input and output ports from the VHDL interface.
        
        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists:
                - List of input ports
                - List of output ports
        """
        ins = [port.split(":")[0].strip() for port in self.ports if "in" in port and not "data_array" in port]
        # Add 2d data_array ports to ins
        ins.extend(add_2d_ports(self.ports, "in"))
        outs = [port.split(":")[0].strip() for port in self.ports if "out" in port and not "data_array" in port]
        # Add 2d data_array ports to outs
        outs.extend(add_2d_ports(self.ports, "out"))
        return ins, outs
    
    def get_input_ports(self) -> List[str]:
        """
        Get the input ports of the VHDL interface.
        
        Returns:
            List[str]: List of input ports.
        """
        return self.ins
    
    def get_output_ports(self) -> List[str]:
        """
        Get the output ports of the VHDL interface.
        
        Returns:
            List[str]: List of output ports.
        """
        return self.outs
    
    def get_list_ports(self) -> List[str]:
        """
        Get the list of all ports (input and output) of the VHDL interface.
        
        Returns:
            List[str]: List of all ports.
        """
        return self.ports

    def get_list_generics(self) -> List[str]:
        """
        Get the list of generics of the VHDL interface.
        
        Returns:
            List[str]: List of generics.
        """
        return self.generics

def extract_generics_ports(vhdl_code, entity_name):
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
    entity_match = re.search(r'entity\s+(\w+)\s+is(.*?)end\s+entity', vhdl_code, re.DOTALL | re.IGNORECASE)
    if not entity_match:
        raise ValueError("Could not find VHDL entity block.")

    # If there are multiple entities, we remove the one that does not match the entity_name
    is_right_entity = entity_match.group(1) in entity_name
    while not is_right_entity and entity_match:
        entity_to_remove = entity_match.group(1)
        # Remove the first entity block that does not match the entity_name
        first_entity = re.search(fr'entity\s+{entity_to_remove}\s+is(.*?)end\s+architecture', vhdl_code, re.DOTALL | re.IGNORECASE)
        assert first_entity, f"Could not find VHDL entity block for {entity_to_remove} despite being in the code."
        vhdl_code = vhdl_code.replace(first_entity.group(0), "")
        entity_match = re.search(r'entity\s+(\w+)\s+is(.*?)end\s+entity', vhdl_code, re.DOTALL | re.IGNORECASE)
        entity_extracted = entity_match.group(1)
        # Check if the entity name matches the one we are looking for # Selector is a special case since its handshake name is handshake.select
        is_right_entity = entity_extracted in entity_name if entity_extracted != "selector" else True

    assert entity_match, f"Entity {entity_name} not found in the VHDL code."

    entity_name = entity_match.group(1) if entity_match else "unknown_entity"
    entity_block = entity_match.group(2)

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

    return entity_name, VhdlInterfaceInfo(generics, ports)


def extract_template_top(entity_name, vhdl_interface_info, param_names):
    """
    Extract the template for the top file from the given top definition file.
    
    Args:
        entity_name (str): Name of the top entity.
        vhdl_interface_info (VhdlInterfaceInfo): VHDL interface information containing generics and ports.
        param_names (List[str]): List of parameter names to be used in the template.
        
    Returns:
        str: The template for the top file.
    """
    generics = vhdl_interface_info.get_list_generics()
    ports = vhdl_interface_info.get_list_ports()
    for _generic in generics:
        generic = _generic.split(":")[0].strip()  # Get the generic name before the colon
        assert generic in param_names, f"Generic `{generic}` not found in parameter names."

    # Create ports for the top file
    tb_ports = []
    for _port in ports:
        port = _port
        for param in param_names:
            port = port.replace(f"{param}", f"{param}_const_value")
        tb_ports.append(port)
    # Create the template for the top file
    template_top = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;
entity tb is
port (
{';\n'.join(tb_ports)}
);
end entity;
architecture tb_arch of tb is
begin
dut: entity work.{entity_name}
generic map (
"""

    for param in param_names:
        if param != "PREDICATE":
            template_top += f"{param} => {param}_const_value,\n"
    template_top = template_top.rstrip(",\n") + "\n" \
                     ")\n" \
                        "port map (\n"
    for port in ports:
        port_name = port.split(":")[0].strip()  # Get the port name before the colon
        template_top += f"{port_name} => {port_name},\n"
    template_top = template_top.rstrip(",\n") + "\n" \
                     ");\n" \
                        "end architecture;\n"

    return template_top, entity_name

def _synth_worker(args):
    synth_tool, tcl_file, log_file = args
    os.system(f"{synth_tool} -mode batch -source {tcl_file} > {log_file}")

def run_synthesis(tcl_files, synth_tool, log_file):
    """
    Run synthesis for the given TCL files using the specified synthesis tool in parallel (if NUM_CORES > 1).
    Args:
        tcl_files (list): List of TCL files to run synthesis on.
        synth_tool (str): Synthesis tool to use (e.g., 'vivado').
    """
    # Run synthesis in parallel using Vivado
    args_list = [(synth_tool, tcl_file, f"{log_file}{i}") for i, tcl_file in enumerate(tcl_files)]
    with Pool(processes=NUM_CORES) as pool:
        pool.map(_synth_worker, args_list)

def add_2d_ports(ports, direction):
    """
    Add 2D data_array ports to the list of ports based on the direction.

    Args:
        ports (list): List of ports to process.
        direction (str): Direction of the ports ('in' or 'out').
    Returns:
        list: List of 2D data_array ports.
    """

    result = []
    for port in ports:
        if direction in port and "data_array" in port:
            match = re.search(r'data_array\((\w+)\s*-\s*1\s*downto\s*0\)', port)
            assert match, f"Could not find data_array port {port}."
            size_name = match.group(1)
            # Get corresponding size # Assuming only one value in parameters allowed
            size_value = parameters_ranges[size_name][0]
            for i in range(size_value):
                result.append(f"{port.split(':')[0].strip()}[{i}]")
    return result

def write_tcl(top_file, top_entity_name, hdl_files, tcl_file, sdc_file, rpt_timing, vhdl_interface_info):
    """
    Write the TCL file for synthesis based on the top file and HDL files.
    
    Args:
        top_file (str): Path to the top file.
        top_entity_name (str): Name of the top entity.
        hdl_files (list): List of HDL files needed for synthesis.
        tcl_file (str): Path to the output TCL file.
        sdc_file (str): Path to the SDC file for constraints.
        rpt_timing (str): Path to the output timing report file.
        vhdl_interface_info (VhdlInterfaceInfo): VHDL interface information containing generics and ports.
    """
    input_ports = vhdl_interface_info.get_input_ports()
    output_ports = vhdl_interface_info.get_output_ports()
    with open(tcl_file, 'w') as f:
        f.write(f"read_vhdl -vhdl2008 {top_file}\n")
        for hdl_file in hdl_files:
            f.write(f"read_vhdl -vhdl2008 {hdl_file}\n")
        f.write(f"read_xdc {sdc_file}\n")
        f.write("synth_design -top tb -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context\n")
        f.write("opt_design\n")
        f.write("place_design\n")
        f.write("phys_opt_design\n")
        f.write("route_design\n")
        f.write("phys_opt_design\n")
        for iport in input_ports:
            if "clk" in iport or "clock" in iport or "rst" in iport or "reset" in iport:
                continue  # Skip clock and reset ports
            for oport in output_ports:
                f.write(f"report_timing -from [get_ports {iport}] -to [get_ports {oport}] >> {rpt_timing}\n")

def write_sdc_constraints(sdc_file, period_ns):
    """
    Write the SDC constraints file with the specified period.
    
    Args:
        sdc_file (str): Path to the SDC file.
        period_ns (float): Period in nanoseconds.
    """
    with open(sdc_file, 'w') as f:
        f.write(f"create_clock -name clk -period {period_ns} -waveform {{0.000 {period_ns/2}}} [get_ports clk]\n")
        f.write("set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]\n")

def run_unit_characterization(unit_name, list_params, hdl_out_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir):
    """
    Run characterization for a single unit using the specified synthesis tool.
    
    Args:
        unit_name (str): Name of the unit to characterize.
        list_params (list): List of parameters for the unit.
        hdl_out_dir (str): Directory where HDL files are stored.
        synth_tool (str): Synthesis tool to use for characterization (e.g., 'vivado').
        top_def_file (str): Path to the top definition file.
        tcl_dir (str): Directory where TCL files will be stored.
        rpt_dir (str): Directory where reports will be stored.
        log_dir (str): Directory where logs will be stored.
    """
    # Get list of hdl files needed for the unit already present in the hdl_out_dir
    hdl_files = [f"{hdl_out_dir}/{file}" for file in os.listdir(hdl_out_dir) if os.path.isfile(os.path.join(hdl_out_dir, file))]
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
    # Extract generics and ports from the top definition file
    print(f"Extracting generics and ports from {top_def_file}")
    with open(top_def_file, 'r') as f:
        vhdl_code = f.read()
    top_entity_name, vhdl_interface_info = extract_generics_ports(vhdl_code, unit_name)
    # Extract the template for the top file
    template_top, top_entity_name = extract_template_top(top_entity_name, vhdl_interface_info, param_names)
    # Create sdc constraints file
    sdc_file = f"{tcl_dir}/period.sdc"
    write_sdc_constraints(sdc_file, 4.0)  # Set a default period of 4 ns
    # Create a top file for each combination of parameters and the corresponding tcl file
    list_tcls = []
    map_rpt2params = {}
    id = 0
    for combination in param_combinations:
        top_file = f"{hdl_out_dir}/{top_entity_name}_top_{id}.vhd" 
        template_top_combined = template_top
        for param_name, param_value in zip(param_names, combination):
            # Replace the constant value in the template
            template_top_combined = template_top_combined.replace(f"{param_name}_const_value", str(param_value))
        with open(top_file, 'w') as f:
            f.write(template_top_combined)
        # Write the tcl file for synthesis
        tcl_file = f"{tcl_dir}/synth_{top_entity_name}_top_{id}.tcl"
        list_tcls.append(tcl_file)
        rpt_timing = f"{rpt_dir}/rpt_timing_{top_entity_name}_top_{id}.txt"
        # Remove previous rpt_timing file if it exists
        if os.path.exists(rpt_timing):
            os.remove(rpt_timing)
        write_tcl(top_file, top_entity_name, hdl_files, tcl_file, sdc_file, rpt_timing, vhdl_interface_info)
        id += 1
        # Map the report file to the parameters used
        map_rpt2params[rpt_timing] = {}
        for param_name, param_value in zip(param_names, combination):
            map_rpt2params[rpt_timing][param_name] = param_value

    # Run the synthesis tool for each tcl file
    log_file = f"{log_dir}/synth_{unit_name}_log.txt"
    run_synthesis(list_tcls, synth_tool, log_file)    

    return map_rpt2params

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
        extra_rtl = dependency_list[extra_dependency]["RTL"]
        extra_rtl = extra_rtl.replace("$DYNAMATIC", dynamatic_dir)
        rtl_filename = extra_rtl.split("/")[-1]
        if "_dataless" in extra_dependency:
            os.system(f"cp {extra_rtl} {hdl_out_dir}/{rtl_filename.replace('.vhd', '_dataless.vhd')}")
        else:
            os.system(f"cp {extra_rtl} {hdl_out_dir}")    

    # Check if the unit has RTL file
    if generic:
        # If generic is provided, copy the RTL file to the output directory
        rtl_file = generic
        rtl_file = rtl_file.replace("$DYNAMATIC", dynamatic_dir)
        os.system(f"cp {rtl_file} {hdl_out_dir}")
    else:
        assert generator, "Unit must have either a generic RTL file or a generator."
        simplified_unit_name = unit_name.split(".")[-1]  # Get the last part of the unit name
        cmd = generator.replace("$DYNAMATIC", dynamatic_dir).replace("$OUTPUT_DIR", hdl_out_dir).replace("$MODULE_NAME", simplified_unit_name).replace("$PREDICATE", "ne")
        # If a generator is provided, run the generator command
        print(f"Running generator command: {cmd}")
        os.system(cmd)        
        # Assert that the RTL file was generated
        rtl_file_vhdl = f"{hdl_out_dir}/{simplified_unit_name}.vhd"  # Assuming
        rtl_file_verilog = f"{hdl_out_dir}/{simplified_unit_name}.v"
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
    list_params = unit_info.get("parameters", [])
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

def extract_data_from_report(rpt_file):
    """
    Extract data from the report file.
    
    Args:
        rpt_file (str): Path to the report file.
        
    Returns:
        tuple: A tuple containing dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay.
    """
    # Initialize variables
    dataDelay = -1
    validDelay = -1
    readyDelay = -1
    VRDelay = -1
    CVDelay = -1
    CRDelay = -1
    VCDelay = -1
    VDDelay = -1
    
    # Read the report file and extract the required data
    with open(rpt_file, 'r') as f:
        for line in f:
            # Extract connection type
            if "Command      :" in line:
                match = re.search(r'report_timing\s+-from\s+\[get_ports\s+{?([\w\[\]]+)}?\]\s+-to\s+\[get_ports\s+{?([\w\[\]]+)}?\]', line)
                assert match, f"Could not find connection type in line: {line}"
                from_port = match.group(1)
                to_port = match.group(2)
            # Extract delay of the data path
            if "Data Path Delay:" in line:
                match = re.search(r'Data Path Delay:\s+([\d.]+)ns', line)
                assert match, f"Could not find data path delay in line: {line}"
                delay = float(match.group(1))
                connection_found = False
                # Determine the connection type based on the ports names
                validSignalFrom = "_valid" in from_port
                validSignalTo = "_valid" in to_port
                readySignalFrom = "_ready" in from_port
                readySignalTo = "_ready" in to_port
                conditionFrom = ("condition" in from_port or "index" in from_port)and not validSignalFrom and not readySignalFrom
                conditionTo = ("condition" in to_port or "index" in to_port) and not validSignalTo and not readySignalTo
                dataFrom = False
                dataTo = False
                if not validSignalFrom and not readySignalFrom and not conditionFrom:
                    assert "lhs" in from_port or "rhs" in from_port or "trueValue" in from_port or "falseValue" in from_port or "ins" in from_port or "data" in from_port or "addrIn" in from_port, f"Unexpected port `{from_port}` without valid or ready signal."
                    dataFrom = True
                if not validSignalTo and not readySignalTo and not conditionTo:
                    assert "result" in to_port or "outs" in to_port or "trueOut" in to_port or "falseOut" in to_port or "addrOut" in to_port or "dataOut" in to_port, f"Unexpected port `{to_port}` without valid or ready signal."
                    dataTo = True

                if (dataFrom and dataTo) or (conditionFrom and dataTo):
                    # Data to data connection
                    connection_found = True
                    dataDelay = max(dataDelay, delay)
                elif validSignalFrom and validSignalTo:
                    # Valid to valid connection
                    connection_found = True
                    validDelay = max(validDelay, delay)
                elif validSignalFrom and dataTo:
                    # Valid to data connection (VD)
                    connection_found = True
                    VDDelay = max(VDDelay, delay)
                elif validSignalFrom and readySignalTo:
                    # Valid to ready connection (VR)
                    connection_found = True
                    VRDelay = max(VRDelay, delay)
                elif validSignalFrom and conditionTo:
                    # Valid to condition connection (CV)
                    connection_found = True
                    VCDelay = max(VCDelay, delay)
                elif readySignalFrom and readySignalTo:
                    # Ready to ready connection
                    connection_found = True
                    readyDelay = max(readyDelay, delay)
                elif conditionFrom and validSignalTo:
                    # Condition to valid connection (CV)
                    connection_found = True
                    CVDelay = max(CVDelay, delay)
                elif conditionFrom and readySignalTo:
                    # Condition to ready connection (CR)
                    connection_found = True
                    CRDelay = max(CRDelay, delay)

                assert connection_found, f"Could not determine connection type for ports `{from_port}` and `{to_port}`"

    return dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay
                
            

def extract_data(map_unit2rpts, json_output):
    """
    Extract the data from the map_unit2rpts dictionary and save it to a JSON file.
    IMPORTANT: For now we assume that only DATA_TYPE is the only parameter that can be used to characterize the unit.
    
    Args:
        map_unit2rpts (dict): Dictionary containing unit names as keys and their reports as values.
        json_output (str): Path to the output JSON file.
    """
    # Create the output data structure
    output_data = {}
    for unit_name, map_rpt2params in map_unit2rpts.items():
        dataDict = {}
        validDict = {"1": 0.0}
        readyDict = {"1": 0.0}
        VRDelayFinal = 0.0
        CVDelayFinal = 0.0
        CRDelayFinal = 0.0
        VCDelayFinal = 0.0
        VDDelayFinal = 0.0
        traversedOnce = False
        # Extract the data from the reports
        for rpt_file, params in map_rpt2params.items():
            # Check if the report file exists
            if not os.path.exists(rpt_file):
                print("\033[93m" + f"[WARNING] Report file for unit {unit_name} parameters {params} does not exist({rpt_file}). Skipping." + "\033[0m")
                continue
            traversedOnce = True
            # Extract data2data, valid2valid, ready2ready, VR, CV, CR, VC and VD
            dataDelay, validDelay, readyDelay, VRDelay, CVDelay, CRDelay, VCDelay, VDDelay = extract_data_from_report(rpt_file)
            dataDict[str(params["DATA_TYPE"])] = dataDelay
            validDict["1"] = max(validDict["1"], validDelay)
            readyDict["1"] = max(readyDict["1"], readyDelay)
            VRDelayFinal = max(VRDelayFinal, VRDelay)
            CVDelayFinal = max(CVDelayFinal, CVDelay)
            CRDelayFinal = max(CRDelayFinal, CRDelay)
            VCDelayFinal = max(VCDelayFinal, VCDelay)
            VDDelayFinal = max(VDDelayFinal, VDDelay)

        if traversedOnce == False:
            print("\033[91m" + f"[ERROR] No reports found for unit {unit_name}." + "\033[0m")
            continue

        output_data[unit_name] = {"delay":{"data": dataDict,
                                       "valid": validDict,
                                       "ready": readyDict,
                                       "VR": VRDelayFinal,
                                       "CV": CVDelayFinal,
                                       "CR": CRDelayFinal,
                                       "VC": VCDelayFinal,
                                       "VD": VDDelayFinal}}


    # Save the output data to the JSON file
    with open(json_output, 'w') as f:
        json.dump(output_data, f, indent=2)

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

    dependency_list = get_dependency_list(dataflow_units)

    map_unit2rpts = {}

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
        top_def_file = get_hdl_files(unit_name, generic, generator, dependencies, hdl_dir, dynamatic_dir, dependency_list)
        # After generating the HDL files, we can proceed with characterization
        map_rpt2params = run_unit_characterization(unit_name, list_params, hdl_dir, synth_tool, top_def_file, tcl_dir, rpt_dir, log_dir)
        # Store the results in the map_unit2rpts dictionary
        map_unit2rpts[unit_name] = map_rpt2params
    
    # Save the results to the output JSON file
    extract_data(map_unit2rpts, json_output)

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