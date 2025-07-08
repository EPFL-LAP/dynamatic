import os
import re
from typing import List, Tuple, Dict

# Constants for the characterization process

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

# Function to write the TCL file for synthesis
def write_tcl(top_entity_name, hdl_files, tcl_file, sdc_file, map_rpt_to_ports):
    """
    Write the TCL file for synthesis based on the top file and HDL files.
    
    Args:
        top_entity_name (str): Name of the top entity.
        hdl_files (list): List of HDL files needed for synthesis.
        tcl_file (str): Path to the output TCL file.
        sdc_file (str): Path to the SDC file for constraints.
        map_rpt_to_ports (dict): Dictionary mapping report files to input and output ports. Expected format: {rpt_filename1: {"input_ports": [...], "output_ports": [...]}, ...}.
    """
    with open(tcl_file, 'w') as f:
        for hdl_file in hdl_files:
            f.write(f"read_vhdl -vhdl2008 {hdl_file}\n")
        f.write(f"read_xdc {sdc_file}\n")
        f.write("synth_design -top tb -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context\n")
        f.write("opt_design\n")
        f.write("place_design\n")
        f.write("phys_opt_design\n")
        f.write("route_design\n")
        f.write("phys_opt_design\n")
        for rpt_timing, ports_info in map_rpt_to_ports.items():
            input_ports = ports_info["input_ports"]
            output_ports = ports_info["output_ports"]
            for iport in input_ports:
                if "clk" in iport or "clock" in iport or "rst" in iport or "reset" in iport:
                    continue  # Skip clock and reset ports
                for oport in output_ports:
                    f.write(f"report_timing -from [get_ports {iport}] -to [get_ports {oport}] >> {rpt_timing}\n")


# Class to hold VHDL interface information
# This class is used to store the generics and ports of a VHDL entity.
class VhdlInterfaceInfo:
    """
    Class to hold VHDL interface information.
    This class is used to store the generics and ports of a VHDL entity.
    """
    generics: List[str]  # List of generics in the VHDL entity
    ports: List[str]     # List of ports in the VHDL entity
    ins_per_type: Dict[str, str] # Dictionary of input ports with their types
    outs_per_type: Dict[str, str] # Dictionary of output ports with their types

    def __init__(self, generics: List[str], ports: List[str]):
        self.generics = generics
        self.ports = ports
        ins, outs = self.extract_ins_outs()
        self.ins_per_type = self.categorize_ports(ins)
        self.outs_per_type = self.categorize_ports(outs)

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
    
    def categorize_ports(self, ports: List[str]) -> Dict[str, str]:
        """
        Categorize ports based on their types.
        
        Args:
            ports (List[str]): List of ports to categorize.
        
        Returns:
            Dict[str, str]: Dictionary with port names as keys and their types as values.
        """
        categorized_ports = {}
        for port in ports:
            if "_valid" in port:
                categorized_ports[port] = "valid"
            elif "_ready" in port:
                categorized_ports[port] = "ready"
            elif "condition" in port or "index" in port:
                categorized_ports[port] = "condition"
            elif "lhs" in port or "rhs" in port or "trueValue" in port or "falseValue" in port or "ins" in port or "data" in port or "addrIn" in port:
                categorized_ports[port] = "data"
            elif "result" in port or "outs" in port or "trueOut" in port or "falseOut" in port or "addrOut" in port or "dataOut" in port:
                categorized_ports[port] = "data"
            elif "clk" in port or "clock" in port or "rst" in port or "reset" in port:
                categorized_ports[port] = "control_signal"
            else:
                assert False, f"Unexpected port type for port {port}. Please check the VHDL interface."
        return categorized_ports

    def get_input_ports(self) -> List[str]:
        """
        Get the input ports of the VHDL interface.
        
        Returns:
            List[str]: List of input ports.
        """
        return list(self.ins_per_type.keys())
    
    def get_input_ports_by_type(self, port_type: str) -> List[str]:
        """
        Get the input ports of a specific type from the VHDL interface.
        
        Args:
            port_type (str): Type of the input ports to retrieve (e.g., "valid", "ready", "data", "condition").
        
        Returns:
            List[str]: List of input ports of the specified type.
        """
        assert port_type in ["valid", "ready", "data", "condition"], f"Invalid port type: {port_type}. Valid types are 'valid', 'ready', 'data', 'condition'."
        # Return ports that match the specified type
        return [port for port, ptype in self.ins_per_type.items() if ptype == port_type]
    
    def get_output_ports_by_type(self, port_type: str) -> List[str]:
        """
        Get the output ports of a specific type from the VHDL interface.
        
        Args:
            port_type (str): Type of the output ports to retrieve (e.g., "valid", "ready", "data", "condition").
        
        Returns:
            List[str]: List of output ports of the specified type.
        """
        assert port_type in ["valid", "ready", "data", "condition"], f"Invalid port type: {port_type}. Valid types are 'valid', 'ready', 'data', 'condition'."
        # Return ports that match the specified type
        return [port for port, ptype in self.outs_per_type.items() if ptype == port_type]

    def get_output_ports(self) -> List[str]:
        """
        Get the output ports of the VHDL interface.
        
        Returns:
            List[str]: List of output ports.
        """
        return list(self.outs_per_type.keys())
    
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

# Class that contains all information for a single unit characterization
class UnitCharacterization:
    """
    Class to hold the characterization information for a single unit.
    This class is used to store the unit name, its VHDL interface information, and the parameters used for characterization.
    """
    unit_name: str # Name of the unit being characterized
    top_entity_name: str # Name of the top entity for this unit
    params: dict # Dictionary of parameters used for characterization
    hdl_files: List[str] # List to hold HDL files generated or copied for this unit
    map_signals_type_to_delay_rpt: dict # Dictionary to hold delay reports for each signal
    vhdl_interface_info: VhdlInterfaceInfo # VHDL interface information containing generics and ports
    unique_id: int # Unique identifier for the characterization instance
    tcl_file: str # Path to the last generated TCL file for synthesis
    
    # List of delay types to be characterized
    # Each tuple contains the input and output port types to be characterized in this exact order
    list_delay_types = [("data", "data"), ("valid", "valid"), ("ready", "ready"), 
                         ("valid", "ready"), ("condition", "valid"), ("condition", "ready"),
                         ("valid", "condition"), ("valid", "data")] 

    def __init__(self, unit_name: str, top_entity_name: str, params: dict, hdl_files: List[str], vhdl_interface_info: VhdlInterfaceInfo, unique_id: int):
        """
        Initialize the UnitCharacterization object.
        
        Args:
            unit_name (str): Name of the unit being characterized.
            top_entity_name (str): Name of the top entity for this unit.
            params (dict): Dictionary of parameters used for characterization.
        """
        self.unit_name = unit_name
        self.top_entity_name = top_entity_name
        self.params = params
        self.hdl_files = hdl_files
        self.unique_id = unique_id
        self.vhdl_interface_info = vhdl_interface_info
        self.map_signals_type_to_delay_rpt = {}
        self.tcl_file = None

    def generate_tcl(self, tcl_dir: str, rpt_dir: str, sdc_file: str) -> List[str]:
        """
        Generate a TCL files for synthesis.
        
        Args:
            tcl_dir (str): Directory where TCL files should be stored.
            rpt_dir (str): Directory where report files should be stored.
            sdc_file (str): Path to the SDC file for constraints.
        Returns:
            str: Path to the generated TCL file.
        """
        # Create a TCL file for each delay type
        tcl_file = f"{tcl_dir}/synth_{self.top_entity_name}_top_{self.unique_id}.tcl"
        # Create a map to hold input and output ports for each report file
        map_rpt_to_ports = {}
        for delay_type in self.list_delay_types:
            # Obtain name of the report file for this delay type
            delay_type_str = f"{delay_type[0]}_{delay_type[1]}"
            rpt_timing = f"{rpt_dir}/rpt_timing_{self.top_entity_name}_top_{self.unique_id}_{delay_type_str}.txt"
            self.map_signals_type_to_delay_rpt[delay_type] = rpt_timing
            # Find the input and output ports for this delay type
            input_ports = self.vhdl_interface_info.get_input_ports_by_type(delay_type[0])
            output_ports = self.vhdl_interface_info.get_output_ports_by_type(delay_type[1])
            map_rpt_to_ports[rpt_timing] = {
                "input_ports": input_ports,
                "output_ports": output_ports
            }
        # Write the TCL file
        write_tcl(self.top_entity_name, self.hdl_files, tcl_file, sdc_file, map_rpt_to_ports)
        self.tcl_file = tcl_file
        return tcl_file
    
    def get_signals_type_to_rpt(self) -> dict:
        """
        Get the dictionary mapping signal types to their delays.
        
        Returns:
            dict: Dictionary mapping signal types to their delays.
        """
        return self.map_signals_type_to_delay_rpt
    
    def get_parameter_value(self, param_name: str) -> str:
        """
        Get the value of a specific parameter.
        
        Args:
            param_name (str): Name of the parameter to retrieve.
        
        Returns:
            str: Value of the specified parameter.
        """
        assert param_name in self.params, f"Parameter {param_name} not found in parameters."
        return self.params[param_name]