import os
import re
from typing import List, Tuple

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

# Class to hold VHDL interface information
# This class is used to store the generics and ports of a VHDL entity.
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