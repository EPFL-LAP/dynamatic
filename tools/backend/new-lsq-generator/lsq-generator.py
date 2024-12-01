# This file generate the entire design of the new lsq, including:
#   - Core LSQ design
#   - Wrapper with extra peripheral logic for connecting the lsq
import math
import argparse
import os
import sys

from configs import *
from lsq_core import *

#===----------------------------------------------------------------------===#
# Parser Definition
#===----------------------------------------------------------------------===#
parser = argparse.ArgumentParser(description='Please specify the output path and lsq config file')
parser.add_argument('--output-dir', '-o', dest='output_path', default='.', type=str)
parser.add_argument('--config-file', '-c', required=True, dest='config_files', default='', type=str)

# Build the target
args = parser.parse_args()

#===----------------------------------------------------------------------===#
# Wrapper Generation
#===----------------------------------------------------------------------===#

class LSQWrapper:
    """Class used to generate the top level wrapper for the LSQ
    """
    
    def __init__(self, path_rtl: str, suffix: str, configs: Configs):
        # Store the global information
        self.output_folder = path_rtl
        self.lsq_name = configs.name
        self.module_suffix = suffix
        self.lsq_config = configs
        
        # Define information needed for VHDL file generation
        # This part is inherited from the design of the original lsq_generator
        self.library_header = 'library IEEE;\nuse IEEE.std_logic_1164.all;\nuse IEEE.numeric_std.all;\n\n'
        self.tab_level = 1
        self.temp_count = 0
        self.signal_init_str = ''
        self.port_init_str = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        self.reg_init_str = '\tprocess (clk, rst) is\n' + '\tbegin\n'
        self.arch = ''
        
        # Define the final output string
        self.lsq_wrapper_str = "\n\n"
        
    def genWrapper(self):
        """This function generates the desired wrapper for the LSQ
        """
        
        # PART 1: Add library information to the VHDL module
        self.lsq_wrapper_str += self.library_header
        
        # PART 2: Define the entity
        self.lsq_wrapper_str += f'entity {self.lsq_name} is\n'
        
        # PART 3: Add the module port definition
        self.lsq_wrapper_str += self.port_init_str
        
        ##
        ## Define all the IOs
        ##! Now for storeData and loadData related IO, we assume there's only one channel, thus we don't use the *Array class
        ### io_storeData: output
        io_storeData = VHDLLogicVecType('io_storeData', 'o', self.lsq_config.dataW)
        # io_storeData = VHDLLogicVecTypeArray('io_storeData', 'o', self.lsq_config.numStMem, self.lsq_config.dataW)
        self.lsq_wrapper_str += io_storeData.signalInit()
        
        ### io_storeAddr: output
        io_storeAddr = VHDLLogicVecType('io_storeAddr', 'o', self.lsq_config.addrW)
        # io_storeAddr = VHDLLogicVecTypeArray('io_storeAddr', 'o', self.lsq_config.numStMem, self.lsq_config.addrW)
        self.lsq_wrapper_str += io_storeAddr.signalInit()
        
        ### io_storeEn: output
        io_storeEn = VHDLLogicType('io_storeEn', 'o')
        # io_storeEn = VHDLLogicTypeArray('io_storeEn', 'o', self.lsq_config.numStMem)
        self.lsq_wrapper_str += io_storeEn.signalInit()
        
        ### io_loadData: input
        io_loadData = VHDLLogicVecType('io_loadData', 'i', self.lsq_config.dataW)
        # io_loadData = VHDLLogicVecTypeArray('io_loadData', 'i', self.lsq_config.numLdMem, self.lsq_config.dataW)
        self.lsq_wrapper_str += io_loadData.signalInit()
        
        ### io_loadAddr: output
        io_loadAddr = VHDLLogicVecType('io_loadAddr', 'o', self.lsq_config.addrW)
        # io_loadAddr = VHDLLogicVecTypeArray('io_loadAddr', 'o', self.lsq_config.numLdMem, self.lsq_config.addrW)
        self.lsq_wrapper_str += io_loadAddr.signalInit()
        
        ### io_loadEn: output
        io_loadEn = VHDLLogicType('io_loadEn', 'o')
        # io_loadEn = VHDLLogicTypeArray('io_loadEn', 'o', self.lsq_config.numLdMem)
        self.lsq_wrapper_str += io_loadEn.signalInit()
        
        ### io_ctrl_*_ready: output
        io_ctrl_ready = VHDLLogicTypeArray('io_ctrl_ready', 'o', self.lsq_config.numGroups)
        self.lsq_wrapper_str += io_ctrl_ready.signalInit()
        
        ### io_ctrl_*_valid: input
        io_ctrl_valid = VHDLLogicTypeArray('io_ctrl_valid', 'i', self.lsq_config.numGroups)
        self.lsq_wrapper_str += io_ctrl_valid.signalInit()
        
        ### io_ldAddr_*_ready: output
        io_ldAddr_ready  = VHDLLogicTypeArray('io_ldAddr_ready', 'o', self.lsq_config.numLdPorts)
        self.lsq_wrapper_str += io_ldAddr_ready.signalInit()
        
        ### io_ldAddr_*_valid: input
        io_ldAddr_valid  = VHDLLogicTypeArray('io_ldAddr_valid', 'i', self.lsq_config.numLdPorts)
        self.lsq_wrapper_str += io_ldAddr_valid.signalInit()
        
        ### io_ldAddr_*_bits: input
        io_ldAddr_bits = VHDLLogicVecTypeArray('io_ldAddr_bits', 'i', self.lsq_config.numLdPorts, self.lsq_config.addrW)
        self.lsq_wrapper_str += io_ldAddr_bits.signalInit()
        
        ### io_ldData_*_ready: input
        io_ldData_ready = VHDLLogicTypeArray('io_ldData_ready', 'i', self.lsq_config.numLdPorts)
        self.lsq_wrapper_str += io_ldData_ready.signalInit()
        
        ### io_ldData_*_valid: output
        io_ldData_valid = VHDLLogicTypeArray('io_ldData_valid', 'o', self.lsq_config.numLdPorts)
        self.lsq_wrapper_str += io_ldData_valid.signalInit()
        
        ### io_ldData_*_bits: output
        io_ldData_bits = VHDLLogicVecTypeArray('io_ldData_bits', 'o', self.lsq_config.numLdPorts, self.lsq_config.dataW)
        self.lsq_wrapper_str += io_ldData_bits.signalInit()
        
        ## IO Definition finished
        self.lsq_wrapper_str += '\nend entity;\n\n'
        
        return self.lsq_wrapper_str
        
        
        
        
        
    
    

#===----------------------------------------------------------------------===#
# Main Function
#===----------------------------------------------------------------------===#

def main():
    """ Main function for lsq generation, expecting two arguments from the CLI
        - Output folder path
        - Config file(s) path
    """
    # Check the existence of the output folder
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # Parse the config file
    lsqConfigsList = GetConfigs(args.config_files)
    
    # STEP 1: Generate the desired core lsq logic
    for lsqConfigs in lsqConfigsList:
        codeGen(args.output_path, lsqConfigs)
        
    # STEP 2: Generate the wrapper to be connected with circuits generated by Dynamatic
    lsq_wrapper_module = LSQWrapper(args.output_path, '_wrapper', lsqConfigsList[0])
    
    print(lsq_wrapper_module.genWrapper())
    
if __name__ == '__main__':
    main()

