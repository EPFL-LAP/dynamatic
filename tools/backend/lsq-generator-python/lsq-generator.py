#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# This file generate the entire design of the new lsq, including:
#   - Core LSQ design
#   - Wrapper with extra peripheral logic for connecting the lsq
import math
import argparse
import os
import sys

from verilog_gen.signals import Logic, LogicVec, LogicArray, LogicVecArray
from verilog_gen.configs import Configs, GetConfigs
from verilog_gen.codegen import codeGen
from verilog_gen.emitters import Emitter, VHDLEmitter, VerilogEmitter
from verilog_gen.ir import Val, CustomStatement, Bit

# ===----------------------------------------------------------------------===#
# Parser Definition
# ===----------------------------------------------------------------------===#
parser = argparse.ArgumentParser(
    description="Please specify the output path and lsq config file"
)
parser.add_argument("--output-dir", "-o",
                    dest="output_path", default=".", type=str)
parser.add_argument(
    "--config-file", "-c", required=True, dest="config_files", default="", type=str
)
parser.add_argument("--language", "-l", dest="language", default="vhdl", type=str)

# Build the target
args = parser.parse_args()

# ===----------------------------------------------------------------------===#
# Wrapper Generation
# ===----------------------------------------------------------------------===#


class LSQWrapper:
    """This class adapts the LSQ core module to handle the following:
      - The naming difference between the IOs of the core module and in dataflow circuits.
      - All needed logic to handle the AXI memory interfaces assumed by the core lsq logic

    Terminologies:
    We call the LSQ module a "master module" for the following case:
      kernel -> LSQ -> AXI (now assuming it always connects to a BRAM),
    The wrapper for this case is generated using `genWrapperMaster` method.

    We call the LSQ module a "slave module" for the following case:
      kernel -> LSQ -> MC -> AXI,
    The wrapper for this case is generated in `genWrapperSlave` method.

    Signal name mappings documentation:
    1. Mapping of the master module:
      | "names in the wrapper"             | "names in LSQ core"                     |
      | ---------------------------------- | --------------------------------------- |
      | io_ldAddr_<id>_(bits|valid|ready)  | ldp_addr_(|valid|ready)_<id>_(i|i|o)    |
      | io_ldData_<id>_(bits|valid|ready)  | ldp_data_(|valid|ready)_<id>_(o|o|i)    |
      | io_stAddr_<id>_(bits|valid|ready)  | stp_addr_(|valid|ready)_<id>_(i|i|o)    |
      | io_stData_<id>_(bits|valid|ready)  | stp_data_(|valid|ready)_<id>_(i|i|o)    |
      | io_storeData                       | wreq_data_0_o                           |
      | io_storeAddr                       | wreq_addr_0_o                           |
      | N/A                                | wreq_id_0_o                             |
      | io_storeEn                         | wreq_valid_0_o                          |
      | N/A                                | wreq_ready_0_i                          |
      | io_loadData                        | rresp_data_0_i                          |
      | N/A                                | rresp_id_0_i                            |
      | N/A                                | rresp_valid_0_i                         |
      | N/A                                | rresp_ready_0_o                         |
      | io_loadAddr                        | rreq_addr_0_o                           |
      | N/A                                | rreq_id_0_o                             |
      | io_loadEn                          | rreq_valid_0_o                          |
      | N/A                                | rreq_ready_0_i                          |
      | N/A                                | wresp_id_0_i                            |
      | N/A                                | wresp_valid_0_i                         |
      | N/A                                | wresp_ready_0_o                         |
      | io_ctrl_<id>_ready                 | group_init_ready_<id>_o                 |
      | io_ctrl_<id>_valid                 | group_init_valid_<id>_i                 |
      | io_memStart_valid                  | memStart_valid_i                        |
      | io_memStart_ready                  | memStart_ready_o                        |
      | io_ctrlEnd_valid                   | ctrlEnd_valid_i                         |
      | io_ctrlEnd_ready                   | ctrlEnd_ready_o                         |
      | io_memEnd_valid                    | memEnd_valid_o                          |
      | io_memEnd_ready                    | memEnd_ready_i                          |
      | ---------------------------------- | --------------------------------------- |

      (*Most of the N/A ports are handled by separate signals defined in the wrapper)

    2. Mapping of the slave module:
      | "names in the wrapper"             | "names in LSQ core"                     |
      | ---------------------------------- | --------------------------------------- |
      | io_ldAddr_<id>_(bits|valid|ready)  | ldp_addr_(|valid|ready)_<id>_(i|i|o)    |
      | io_ldData_<id>_(bits|valid|ready)  | ldp_data_(|valid|ready)_<id>_(o|o|i)    |
      | io_stAddr_<id>_(bits|valid|ready)  | stp_addr_(|valid|ready)_<id>_(i|i|o)    |
      | io_stData_<id>_(bits|valid|ready)  | stp_data_(|valid|ready)_<id>_(i|i|o)    |
      | io_stDataToMC_bits                 | wreq_data_0_o                           |
      | io_stDataToMC_valid                | N/A                                     |
      | io_stDataToMC_ready                | N/A                                     |
      | io_stAddrToMC_bits                 | wreq_addr_0_o                           |
      | io_stAddrToMC_valid                | N/A                                     |
      | io_stAddrToMC_ready                | N/A                                     |
      | io_ldDataFromMC_bits               | rresp_data_0_i                          |
      | io_ldDataFromMC_valid              | rresp_valid_0_i                         |
      | io_ldDataFromMC_ready              | rresp_ready_0_o                         |
      | io_ldAddrToMC_bits                 | rreq_addr_0_o                           |
      | io_ldAddrToMC_valid                | N/A                                     |
      | io_ldAddrToMC_ready                | rreq_ready_0_i                          |
      | io_ctrl_<id>_ready                 | group_init_ready_<id>_o                 |
      | io_ctrl_<id>_valid                 | group_init_valid_<id>_i                 |
      | N/A                                | rreq_valid_0_o                          |
      | N/A                                | wreq_id_0_o                             |
      | N/A                                | wreq_valid_0_o                          |
      | N/A                                | wreq_ready_0_i                          |
      | N/A                                | rresp_id_0_i                            |
      | N/A                                | rreq_id_0_o                             |
      | N/A                                | wresp_id_0_i                            |
      | N/A                                | wresp_valid_0_i                         |
      | N/A                                | wresp_ready_0_o                         |
      | ---------------------------------- | --------------------------------------- |

      (*Most of the N/A ports are handled by separate signals defined in the wrapper)

    """

    def __init__(self, path_rtl: str, suffix: str, configs: Configs):
        # Store the global information
        self.output_folder = path_rtl
        self.lsq_name = configs.name
        self.module_suffix = suffix
        self.lsq_config = configs

        # Define the final output string
        self.lsq_wrapper_str = "\n\n"

    def genWrapper(self, em: Emitter):
        """This function generates the desired wrapper for the LSQ"""
        
        em.clock_name = "clock"
        em.reset_name = "reset"

        ##
        # Define all the IOs, details can be found in the table above
        # ! Now for storeData and loadData related IO, we assume there's only one channel, thus we don't use the *Array class
        io_storeData = LogicVec(em, "io_storeData", 'o', self.lsq_config.dataW, dyn_comp=True)
        io_storeAddr = LogicVec(em, "io_storeAddr", 'o', self.lsq_config.addrW, dyn_comp=True)
        io_storeEn = Logic(em, "io_storeEn", 'o', dyn_comp=True)
        io_loadData = LogicVec(em, "io_loadData", 'i', self.lsq_config.dataW, dyn_comp=True)
        io_loadAddr = LogicVec(em, "io_loadAddr", 'o', self.lsq_config.addrW, dyn_comp=True)
        io_loadEn = Logic(em, "io_loadEn", 'o', dyn_comp=True)
        io_ctrl_ready = LogicArray(em, "io_ctrl_ready", 'o', self.lsq_config.numGroups, dyn_comp=True)
        io_ctrl_valid = LogicArray(em, "io_ctrl_valid", 'i', self.lsq_config.numGroups, dyn_comp=True)
        io_ldAddr_ready = LogicArray(em, "io_ldAddr_ready", 'o', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldAddr_valid = LogicArray(em, "io_ldAddr_valid", 'i', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldAddr_bits = LogicVecArray(em, "io_ldAddr_bits", 'i', self.lsq_config.numLdPorts, self.lsq_config.addrW, dyn_comp=True)
        io_ldData_ready = LogicArray(em, "io_ldData_ready", 'i', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldData_valid = LogicArray(em, "io_ldData_valid", 'o', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldData_bits = LogicVecArray(em, "io_ldData_bits", 'o', self.lsq_config.numLdPorts, self.lsq_config.dataW, dyn_comp=True)
        io_stAddr_ready = LogicArray(em, "io_stAddr_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True)
        io_stAddr_valid = LogicArray(em, "io_stAddr_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True)
        io_stAddr_bits = LogicVecArray(em, "io_stAddr_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.addrW, dyn_comp=True)
        io_stData_ready = LogicArray(em, "io_stData_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True)
        io_stData_valid = LogicArray(em, "io_stData_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True)
        io_stData_bits = LogicVecArray(em, "io_stData_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.dataW, dyn_comp=True)
        io_memStart_ready = Logic(em, "io_memStart_ready", 'o', dyn_comp=True)
        io_memStart_valid = Logic(em, "io_memStart_valid", 'i', dyn_comp=True)
        io_ctrlEnd_ready = Logic(em, "io_ctrlEnd_ready", 'o', dyn_comp=True)
        io_ctrlEnd_valid = Logic(em, "io_ctrlEnd_valid", 'i', dyn_comp=True)
        io_memEnd_ready = Logic(em, "io_memEnd_ready", 'i', dyn_comp=True)
        io_memEnd_valid = Logic(em, "io_memEnd_valid", 'o', dyn_comp=True)

        ##
        # Architecture definition start
        ##

        # Define internal signals
        rreq_ready = LogicArray(em, "rreq_ready", "w", self.lsq_config.numLdMem, dyn_comp=True)
        rresp_valid = LogicArray(em, "rresp_valid", 'w', self.lsq_config.numLdMem, dyn_comp=True)
        rresp_id = LogicVecArray(em, "rresp_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True)
        wreq_ready = LogicArray(em, "wreq_ready", 'w', self.lsq_config.numStMem, dyn_comp=True)
        wresp_valid = LogicArray(em, "wresp_valid", 'w', self.lsq_config.numStMem, dyn_comp=True)
        wresp_id = LogicVecArray(em, "wresp_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True)
        rreq_id = LogicVecArray(em, "rreq_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True)
        wreq_id = LogicVecArray(em, "wreq_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True)
        

        # Define the process to update
        # rreq_ready, rresp_valid
        em.add_comment('--------------------------------------------------------------------------')
        em.add_comment('Process for rreq_ready, rresp_valid and rresp_id')
        
        em.add_statement(em.get_reg_init_str())
        em.increase_indent()
        em.add_custom_statement(CustomStatement("if reset = '1' then", "if (reset) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rreq_ready[i], Bit(0), in_process=True)
            em.add_assignment(rresp_valid[i], Bit(0), in_process=True)
            em.add_assignment(rresp_id[i], Val(0), in_process=True)

    
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("elsif rising_edge(clock) then", "end\nelse begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rreq_ready[i], Bit(1), in_process=True)

        em.add_custom_statement(CustomStatement(f"if {io_loadEn.getNameWrite()} = '1' then", f"if ({io_loadEn.getNameWrite()}) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rresp_valid[i], Bit(1), in_process=True)
            em.add_assignment(rresp_id[i], rreq_id[i], in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("else", "end\nelse begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rresp_valid[i], Bit(0), in_process=True)
        
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end process;", "end"))

        em.add_comment('--------------------------------------------------------------------------')

        # Define the process to update
        # wreq_ready, wresp_valid, wresp_id
        em.add_comment('--------------------------------------------------------------------------')
        em.add_comment('Process for wreq_ready, wresp_valid and wresp_id')

        em.add_statement(em.get_reg_init_str())
        em.increase_indent()
        em.add_custom_statement(CustomStatement("if reset = '1' then", "if (reset) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wreq_ready[i], Bit(0), in_process=True)
            em.add_assignment(wresp_valid[i], Bit(0), in_process=True)
            em.add_assignment(wresp_id[i], Val(0), in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("elsif rising_edge(clock) then", "end\nelse begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wreq_ready[i], Bit(1), in_process=True)

        em.add_custom_statement(CustomStatement(f"if {io_storeEn.getNameWrite()} = '1' then", f"if ({io_storeEn.getNameWrite()}) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wresp_valid[i], Bit(1), in_process=True)
            em.add_assignment(wresp_id[i], rreq_id[i], in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("else", "end\nelse begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wresp_valid[i], Bit(0), in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end process;", "end"))

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        ###
        # Instantiate the LSQ_core module
        ###
        em.add_comment('Instantiate the core LSQ logic')
        em.start_instantiation(self.lsq_name + "_core")

        em.add_map("rst", "reset")
        em.add_map("clk", "clock")

        em.add_map("wreq_data_0_o", io_storeData.getNameWrite())
        em.add_map("wreq_addr_0_o", io_storeAddr.getNameWrite())
        em.add_map("wreq_valid_0_o", io_storeEn.getNameWrite())

        em.add_map("rresp_data_0_i", io_loadData.getNameRead())
        em.add_map("rreq_addr_0_o", io_loadAddr.getNameWrite())
        em.add_map("rreq_valid_0_o", io_loadEn.getNameWrite())

        em.add_map("memStart_ready_o", io_memStart_ready.getNameWrite())
        em.add_map("memStart_valid_i", io_memStart_valid.getNameRead())

        em.add_map("ctrlEnd_ready_o", io_ctrlEnd_ready.getNameWrite())
        em.add_map("ctrlEnd_valid_i", io_ctrlEnd_valid.getNameRead())
        em.add_map("memEnd_ready_i", io_memEnd_ready.getNameRead())
        em.add_map("memEnd_valid_o", io_memEnd_valid.getNameWrite())

        for i in range(self.lsq_config.numGroups):
            em.add_map(f"group_init_ready_{i}_o", io_ctrl_ready[i].getNameWrite())
            em.add_map(f"group_init_valid_{i}_i", io_ctrl_valid[i].getNameRead())
        for i in range(self.lsq_config.numLdPorts):
            em.add_map(f"ldp_addr_ready_{i}_o", io_ldAddr_ready[i].getNameWrite())
            em.add_map(f"ldp_addr_valid_{i}_i", io_ldAddr_valid[i].getNameRead())
            em.add_map(f"ldp_addr_{i}_i", io_ldAddr_bits[i].getNameRead())
            em.add_map(f"ldp_data_ready_{i}_i", io_ldData_ready[i].getNameRead())
            em.add_map(f"ldp_data_valid_{i}_o", io_ldData_valid[i].getNameWrite())
            em.add_map(f"ldp_data_{i}_o", io_ldData_bits[i].getNameWrite())

        for i in range(self.lsq_config.numStPorts):
            em.add_map(f"stp_addr_ready_{i}_o", io_stAddr_ready[i].getNameWrite())
            em.add_map(f"stp_addr_valid_{i}_i", io_stAddr_valid[i].getNameRead())
            em.add_map(f"stp_addr_{i}_i", io_stAddr_bits[i].getNameRead())
            em.add_map(f"stp_data_ready_{i}_o", io_stData_ready[i].getNameWrite())
            em.add_map(f"stp_data_valid_{i}_i", io_stData_valid[i].getNameRead())
            em.add_map(f"stp_data_{i}_i", io_stData_bits[i].getNameRead())

        # Define all AXI ports, we assume there is only 1 channel
        for i in range(self.lsq_config.numLdMem):
            em.add_map(f"rreq_ready_{i}_i", rreq_ready[i].getNameRead())
            em.add_map(f"rresp_valid_{i}_i", rresp_valid[i].getNameRead())
            em.add_map(f"rresp_id_{i}_i", rresp_id[i].getNameRead())
            em.add_map(f"rreq_id_0_o", rreq_id[i].getNameWrite())

        for i in range(self.lsq_config.numStMem):
            em.add_map(f"wreq_ready_{i}_i", wreq_ready[i].getNameRead())
            em.add_map(f"wresp_valid_{i}_i", wresp_valid[i].getNameRead())
            em.add_map(f"wresp_id_{i}_i", wresp_id[i].getNameRead())
            em.add_map(f"wreq_id_{i}_o", wreq_id[i].getNameWrite())

        em.complete_instantiation()

        # Write to the file
        output_str = em.get_definition_str(self.lsq_name)
        with open(f"{self.output_folder}/{self.lsq_name}.vhd", 'w') as file:
            file.write(output_str)

        return self.lsq_wrapper_str

    def genWrapperSlave(self, em: Emitter):
        """This function generates the desired wrapper for the LSQ"""

        ##
        # Define all the IOs
        em.clock_name = "clock"
        em.reset_name = "reset"

        io_storeData = LogicVec(em, "io_stDataToMC_bits", 'o', self.lsq_config.dataW, dyn_comp=True)
        io_storeAddr = LogicVec(em, "io_stAddrToMC_bits", 'o', self.lsq_config.addrW, dyn_comp=True)
        io_loadData = LogicVec(em, "io_ldDataFromMC_bits", 'i', self.lsq_config.dataW, dyn_comp=True)
        io_loadAddr = LogicVec(em, "io_ldAddrToMC_bits", 'o', self.lsq_config.addrW, dyn_comp=True)
        io_ctrl_ready = LogicArray(em, "io_ctrl_ready", 'o', self.lsq_config.numGroups, dyn_comp=True)
        io_ctrl_valid = LogicArray(em, "io_ctrl_valid", 'i', self.lsq_config.numGroups, dyn_comp=True)
        io_ldAddr_ready = LogicArray(em, "io_ldAddr_ready", 'o', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldAddr_valid = LogicArray(em, "io_ldAddr_valid", 'i', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldAddr_bits = LogicVecArray(em, "io_ldAddr_bits", 'i', self.lsq_config.numLdPorts, self.lsq_config.addrW, dyn_comp=True)
        io_ldData_ready = LogicArray(em, "io_ldData_ready", 'i', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldData_valid = LogicArray(em, "io_ldData_valid", 'o', self.lsq_config.numLdPorts, dyn_comp=True)
        io_ldData_bits = LogicVecArray(em, "io_ldData_bits", 'o', self.lsq_config.numLdPorts, self.lsq_config.dataW, dyn_comp=True)
        io_stAddr_ready = LogicArray(em, "io_stAddr_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True)
        io_stAddr_valid = LogicArray(em, "io_stAddr_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True)
        io_stAddr_bits = LogicVecArray(em, "io_stAddr_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.addrW, dyn_comp=True)
        io_stData_ready = LogicArray(em, "io_stData_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True)
        io_stData_valid = LogicArray(em, "io_stData_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True)
        io_stData_bits = LogicVecArray(em, "io_stData_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.dataW, dyn_comp=True)
        io_ldAddrToMC_ready = Logic(em, "io_ldAddrToMC_ready", 'i', dyn_comp=True)
        io_ldAddrToMC_valid = Logic(em, "io_ldAddrToMC_valid", 'o', dyn_comp=True)
        io_ldDataFromMC_ready = Logic(em, "io_ldDataFromMC_ready", 'o', dyn_comp=True)
        io_ldDataFromMC_valid = Logic(em, "io_ldDataFromMC_valid", 'i', dyn_comp=True)
        io_stAddrToMC_ready = Logic(em, "io_stAddrToMC_ready", 'i', dyn_comp=True)
        io_stAddrToMC_valid = Logic(em, "io_stAddrToMC_valid", 'o', dyn_comp=True)
        io_stDataToMC_ready = Logic(em, "io_stDataToMC_ready", 'i', dyn_comp=True)
        io_stDataToMC_valid = Logic(em, "io_stDataToMC_valid", 'o', dyn_comp=True)

        ##
        # IO Definition finished
        ##

        ##
        # Architecture definition start
        ##

        # Define internal signals
        io_loadEn = Logic(em, "io_loadEn", 'w', dyn_comp=True)
        io_storeEn = Logic(em, "io_storeEn", 'w', dyn_comp=True)

        rresp_id = LogicVecArray(em, "rresp_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True)

        wreq_ready = LogicArray(em, "wreq_ready", 'w', self.lsq_config.numStMem, dyn_comp=True)
        wresp_valid = LogicArray(em, "wresp_valid", 'w', self.lsq_config.numStMem, dyn_comp=True)
        wresp_id = LogicVecArray(em, "wresp_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True)

        rreq_id = LogicVecArray(em, "rreq_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True)
        wreq_id = LogicVecArray(em, "wreq_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True)

        # Define the process to update
        # rresp_id
        em.add_comment("----------------------------------------------------------------------------\n")
        em.add_comment("Process for rresp_id\n")
        em.add_statement(em.get_reg_init_str())
        em.add_custom_statement(CustomStatement("if reset = '1' then", "if (reset) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rresp_id[i], Bit(0), in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("elsif rising_edge(clock) then", "end\nelse begin"))
        em.increase_indent()

        em.add_custom_statement(CustomStatement(f"if {io_loadEn.getNameWrite()} = '1' then", f"if ({io_loadEn.getNameWrite()}) begin"))

        for i in range(self.lsq_config.numLdMem):
            em.add_assignment(rresp_id[i], rreq_id[i], in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end process;", "end"))

        em.add_comment("--------------------------------------------------------------------------\n")

        # Define the process to update
        # wresp_valid, wresp_id
        em.add_comment("----------------------------------------------------------------------------\n")
        em.add_comment("Process for wreq_ready, wresp_valid and wresp_id\n")
        em.add_statement(em.get_reg_init_str())
        em.add_custom_statement(CustomStatement("if reset = '1' then", "if (reset) begin"))
        em.increase_indent()

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wresp_valid[i], Bit(0), in_process=True)
            em.add_assignment(wresp_id[i], Bit(0), in_process=True)

        em.add_custom_statement(CustomStatement("elsif rising_edge(clock) then", "end\nelse begin"))

        em.add_custom_statement(CustomStatement(f"if {io_storeEn.getNameWrite()} = '1' then", f"if ({io_storeEn.getNameWrite()}) begin"))

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wresp_valid[i], Bit(1), in_process=True)
            em.add_assignment(wresp_id[i], rreq_id[i], in_process=True)

        em.add_custom_statement(CustomStatement("else", "end\nelse begin"))

        for i in range(self.lsq_config.numStMem):
            em.add_assignment(wresp_valid[i], Bit(0), in_process=True)

        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end if;", "end"))
        em.decrease_indent()
        em.add_custom_statement(CustomStatement("end process;", "end"))

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        ###
        # Signal Assignment
        ###
        em.add_comment("Signal Assignment")
        em.add_assignment(io_ldAddrToMC_valid, io_loadEn)
        em.add_assignment(io_stAddrToMC_valid, io_storeEn)
        em.add_assignment(io_stDataToMC_valid, io_storeEn)
        em.add_assignment(wreq_ready[0], io_stAddrToMC_ready & io_stDataToMC_ready)

        ###
        # Instantiate the LSQ_core module
        ###
        em.add_comment("Instantiate the core LSQ logic\n")
        em.start_instantiation(self.lsq_name + "_core")
        
        em.add_map("rst", "reset")
        em.add_map("clk", "clock")

        em.add_map("wreq_data_0_o", io_storeData.getNameWrite())
        em.add_map("wreq_addr_0_o", io_storeAddr.getNameWrite())
        em.add_map("wreq_valid_0_o", io_storeEn.getNameWrite())

        em.add_map("rresp_data_0_i", io_loadData.getNameRead())
        em.add_map("rreq_addr_0_o", io_loadAddr.getNameWrite())
        em.add_map("rreq_valid_0_o", io_loadEn.getNameWrite())

        for i in range(self.lsq_config.numGroups):
            em.add_map(f"group_init_ready_{i}_o", io_ctrl_ready[i].getNameWrite())
            em.add_map(f"group_init_valid_{i}_i", io_ctrl_valid[i].getNameRead())

        for i in range(self.lsq_config.numLdPorts):
            em.add_map(f"ldp_addr_ready_{i}_o", io_ldAddr_ready[i].getNameWrite())
            em.add_map(f"ldp_addr_valid_{i}_i", io_ldAddr_valid[i].getNameRead())
            em.add_map(f"ldp_addr_{i}_i", io_ldAddr_bits[i].getNameRead())
            em.add_map(f"ldp_data_ready_{i}_i", io_ldData_ready[i].getNameRead())
            em.add_map(f"ldp_data_valid_{i}_o", io_ldData_valid[i].getNameWrite())
            em.add_map(f"ldp_data_{i}_o", io_ldData_bits[i].getNameWrite())

        for i in range(self.lsq_config.numStPorts):
            em.add_map(f"stp_addr_ready_{i}_o", io_stAddr_ready[i].getNameWrite())
            em.add_map(f"stp_addr_valid_{i}_i", io_stAddr_valid[i].getNameRead())
            em.add_map(f"stp_addr_{i}_i", io_stAddr_bits[i].getNameRead())
            em.add_map(f"stp_data_ready_{i}_o", io_stData_ready[i].getNameWrite())
            em.add_map(f"stp_data_valid_{i}_i", io_stData_valid[i].getNameRead())
            em.add_map(f"stp_data_{i}_i", io_stData_bits[i].getNameRead())

        # Define all AXI ports, we assume there is only 1 channel
        for i in range(self.lsq_config.numLdMem):
            em.add_map(f"rreq_ready_{i}_i", io_ldAddrToMC_ready.getNameRead())
            em.add_map(f"rresp_valid_{i}_i", io_ldDataFromMC_valid.getNameRead())
            em.add_map(f"rresp_ready_{i}_o", io_ldDataFromMC_ready.getNameWrite())
            em.add_map(f"rresp_id_{i}_i", rresp_id[i].getNameRead())
            em.add_map(f"rreq_id_0_o", rreq_id[i].getNameWrite())

        for i in range(self.lsq_config.numStMem):
            em.add_map(f"wreq_ready_{i}_i", wreq_ready[i].getNameRead())
            em.add_map(f"wresp_valid_{i}_i", wresp_valid[i].getNameRead())
            em.add_map(f"wresp_id_{i}_i", wresp_id[i].getNameRead())
            em.add_map(f"wreq_id_{i}_o", wreq_id[i].getNameWrite())


        em.complete_instantiation()

        # Write to the file
        output_str = em.get_definition_str(self.lsq_name)
        with open(f"{self.output_folder}/{self.lsq_name}.vhd", 'w') as file:
            file.write(output_str)

        return output_str

# ===----------------------------------------------------------------------===#
# Main Function
# ===----------------------------------------------------------------------===#


def main():
    """Main function for lsq generation, expecting two arguments from the CLI
    - Output folder path
    - Config file(s) path
    """
    # Check the existence of the output folder
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.language == "vhdl":
        emitter = VHDLEmitter()
    elif args.language == "verilog":
        emitter = VerilogEmitter()
    else:
        raise ValueError("Unsupported language specified. Use 'vhdl' or 'verilog'.")

    # Parse the config file
    lsqConfig = GetConfigs(args.config_files)

    # STEP 1: Generate the desired core lsq logic
    codeGen(emitter.new(), args.output_path, lsqConfig)

    # STEP 2: Generate the wrapper to be connected with circuits generated by Dynamatic
    lsq_wrapper_module = LSQWrapper(args.output_path, "_wrapper", lsqConfig)

    # Step 3: Generate the corresponding wrapper based on the config.master
    if lsqConfig.master:
        lsq_wrapper_module.genWrapper(emitter)
    else:
        lsq_wrapper_module.genWrapperSlave(emitter)


if __name__ == "__main__":
    main()
