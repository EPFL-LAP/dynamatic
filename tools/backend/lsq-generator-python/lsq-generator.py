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

from vhdl_gen.signals import Logic, LogicVec, LogicArray, LogicVecArray
from vhdl_gen.configs import Configs, GetConfigs
from vhdl_gen.codegen import codeGen
from vhdl_gen.operators import Op
from vhdl_gen.context import VHDLContext

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

    def genWrapper(self):
        """This function generates the desired wrapper for the LSQ"""
        
        ctx = VHDLContext()
        ctx.tabLevel = 1
        ctx.tempCount = 0
        ctx.signalInitString = ''
        ctx.portInitString = '\tport(\n\t\treset : in std_logic;\n\t\tclock : in std_logic'
        ctx.regInitString = '\tprocess (clock, reset) is\n' + '\tbegin\n'
        arch = ''


        # PART 1: Add library information to the VHDL module
        self.lsq_wrapper_str += ctx.library

        # PART 2: Define the entity
        self.lsq_wrapper_str += f"entity {self.lsq_name} is\n"


        ##
        # Define all the IOs, details can be found in the table above
        # ! Now for storeData and loadData related IO, we assume there's only one channel, thus we don't use the *Array class
        # io_storeData: output
        io_storeData = LogicVec(ctx, "io_storeData", 'o', self.lsq_config.dataW, dyn_comp=True)


        # io_storeAddr: output
        io_storeAddr = LogicVec(ctx, "io_storeAddr", 'o', self.lsq_config.addrW, dyn_comp=True)


        # io_storeEn: output
        io_storeEn = Logic(ctx, "io_storeEn", 'o', dyn_comp=True)


        # io_loadData: input
        io_loadData = LogicVec(ctx, "io_loadData", 'i', self.lsq_config.dataW, dyn_comp=True)

        # io_loadAddr: output
        io_loadAddr = LogicVec(ctx, "io_loadAddr", 'o', self.lsq_config.addrW, dyn_comp=True)


        # io_loadEn: output
        io_loadEn = Logic(ctx, "io_loadEn", 'o', dyn_comp=True)


        # io_ctrl_*_ready: output
        io_ctrl_ready = LogicArray(ctx, 
            "io_ctrl_ready", 'o', self.lsq_config.numGroups, dyn_comp=True
        )

        # io_ctrl_*_valid: input
        io_ctrl_valid = LogicArray(ctx, 
            "io_ctrl_valid", 'i', self.lsq_config.numGroups, dyn_comp=True
        )

        # io_ldAddr_*_ready: output
        io_ldAddr_ready = LogicArray(ctx, 
            "io_ldAddr_ready", 'o', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldAddr_*_valid: input
        io_ldAddr_valid = LogicArray(ctx, 
            "io_ldAddr_valid", 'i', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldAddr_*_bits: input
        io_ldAddr_bits = LogicVecArray(ctx, 
            "io_ldAddr_bits", 'i', self.lsq_config.numLdPorts, self.lsq_config.addrW, dyn_comp=True
        )

        # io_ldData_*_ready: input
        io_ldData_ready = LogicArray(ctx, 
            "io_ldData_ready", 'i', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldData_*_valid: output
        io_ldData_valid = LogicArray(ctx, 
            "io_ldData_valid", 'o', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldData_*_bits: output
        io_ldData_bits = LogicVecArray(ctx, 
            "io_ldData_bits", 'o', self.lsq_config.numLdPorts, self.lsq_config.dataW, dyn_comp=True
        )

        # io_stAddr_ready: output
        io_stAddr_ready = LogicArray(ctx, 
            "io_stAddr_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stAddr_valid: input
        io_stAddr_valid = LogicArray(ctx, 
            "io_stAddr_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stAddr_bits: input
        io_stAddr_bits = LogicVecArray(ctx, 
            "io_stAddr_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.addrW, dyn_comp=True
        )

        # io_stData_ready: output
        io_stData_ready = LogicArray(ctx, 
            "io_stData_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stData_valid: input
        io_stData_valid = LogicArray(ctx, 
            "io_stData_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stData_bits: input
        io_stData_bits = LogicVecArray(ctx, 
            "io_stData_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.dataW, dyn_comp=True
        )

        # io_memStart_ready: output
        io_memStart_ready = Logic(ctx, "io_memStart_ready", 'o', dyn_comp=True)

        # io_memStart_valid: input
        io_memStart_valid = Logic(ctx, "io_memStart_valid", 'i', dyn_comp=True)

        # io_ctrlEnd_ready: output
        io_ctrlEnd_ready = Logic(ctx, "io_ctrlEnd_ready", 'o', dyn_comp=True)

        # io_ctrlEnd_valid: input
        io_ctrlEnd_valid = Logic(ctx, "io_ctrlEnd_valid", 'i', dyn_comp=True)

        # io_memEnd_ready: input
        io_memEnd_ready = Logic(ctx, "io_memEnd_ready", 'i', dyn_comp=True)

        # io_memEnd_valid: output
        io_memEnd_valid = Logic(ctx, "io_memEnd_valid", 'o', dyn_comp=True)

        ##
        # IO Definition finished
        ##
        self.lsq_wrapper_str += ctx.portInitString
        self.lsq_wrapper_str += "\n\t);"
        self.lsq_wrapper_str += "\nend entity;\n\n"

        ##
        # Architecture definition start
        ##
        self.lsq_wrapper_str += f"architecture arch of {self.lsq_name} is\n"

        # Define internal signals
        rreq_ready = LogicArray(ctx,
            "rreq_ready", "w", self.lsq_config.numLdMem, dyn_comp=True
        )

        rresp_valid = LogicArray(ctx, 
            "rresp_valid", 'w', self.lsq_config.numLdMem, dyn_comp=True
        )

        rresp_id = LogicVecArray(ctx, 
            "rresp_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True
        )

        wreq_ready = LogicArray(ctx, 
            "wreq_ready", 'w', self.lsq_config.numStMem, dyn_comp=True
        )

        wresp_valid = LogicArray(ctx, 
            "wresp_valid", 'w', self.lsq_config.numStMem, dyn_comp=True
        )

        wresp_id = LogicVecArray(ctx, 
            "wresp_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True
        )

        rreq_id = LogicVecArray(ctx, 
            "rreq_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True
        )

        wreq_id = LogicVecArray(ctx, 
            "wreq_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True
        )
        
        self.lsq_wrapper_str += ctx.signalInitString

        # Begin actual arch logic definition
        self.lsq_wrapper_str += "begin\n"

        # Define the process to update
        # rreq_ready, rresp_valid
        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"
        self.lsq_wrapper_str += (
            "\t-- Process for rreq_ready, rresp_valid and rresp_id\n"
        )
        self.lsq_wrapper_str += ctx.regInitString
        self.lsq_wrapper_str += "\t" * \
            (ctx.tabLevel + 1) + "if reset = '1' then\n"

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rreq_ready[i], "'0'")
            self.lsq_wrapper_str += Op(ctx, rresp_valid[i], "'0'")
            self.lsq_wrapper_str += Op(ctx, rresp_id[i], "(", "others", "=>", "'0'", ")")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 1) + "elsif rising_edge(clock) then\n"
        )

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rreq_ready[i], "'1'")

        self.lsq_wrapper_str += (
            "\n"
            + "\t" * (ctx.tabLevel + 2)
            + "if "
            + io_loadEn.getNameWrite()
            + " = '1' then\n"
        )

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rresp_valid[i], "'1'")
            self.lsq_wrapper_str += Op(ctx, rresp_id[i], rreq_id[i])

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + "else\n"

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rresp_valid[i], "'0'")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + "end if;\n"
            + "\t" * 2
            + "end if;\n"
            + "\tend process;\n"
        )

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        # Define the process to update
        # wreq_ready, wresp_valid, wresp_id
        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"
        self.lsq_wrapper_str += (
            "\t-- Process for wreq_ready, wresp_valid and wresp_id\n"
        )
        self.lsq_wrapper_str += ctx.regInitString
        self.lsq_wrapper_str += "\t" * \
            (ctx.tabLevel + 1) + "if reset = '1' then\n"

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wreq_ready[i], "'0'")
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'0'")
            self.lsq_wrapper_str += Op(ctx, wresp_id[i], "(", "others", "=>", "'0'", ")")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 1) + "elsif rising_edge(clock) then\n"
        )

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wreq_ready[i], "'1'")

        self.lsq_wrapper_str += (
            "\n"
            + "\t" * (ctx.tabLevel + 2)
            + "if "
            + io_storeEn.getNameWrite()
            + " = '1' then\n"
        )

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'1'")
            self.lsq_wrapper_str += Op(ctx, wresp_id[i], rreq_id[i])

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + "else\n"

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'0'")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + "end if;\n"
            + "\t" * 2
            + "end if;\n"
            + "\tend process;\n"
        )

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        ###
        # Instantiate the LSQ_core module
        ###
        self.lsq_wrapper_str += "\t-- Instantiate the core LSQ logic\n"
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel)
            + f"{self.lsq_name}_core : entity work.{self.lsq_name}_core\n"
        )
        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 1) + f"port map(\n"

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + f"rst => reset,\n"
        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + f"clk => clock,\n"

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_data_0_o => {io_storeData.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_addr_0_o => {io_storeAddr.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_valid_0_o => {io_storeEn.getNameWrite()},\n"
        )

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rresp_data_0_i => {io_loadData.getNameRead()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rreq_addr_0_o => {io_loadAddr.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rreq_valid_0_o => {io_loadEn.getNameWrite()},\n"
        )

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"memStart_ready_o => {io_memStart_ready.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"memStart_valid_i => {io_memStart_valid.getNameRead()},\n"
        )

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"ctrlEnd_ready_o => {io_ctrlEnd_ready.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"ctrlEnd_valid_i => {io_ctrlEnd_valid.getNameRead()},\n"
        )

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"memEnd_ready_i => {io_memEnd_ready.getNameRead()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"memEnd_valid_o => {io_memEnd_valid.getNameWrite()},\n"
        )

        for i in range(self.lsq_config.numGroups):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"group_init_ready_{i}_o => {io_ctrl_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"group_init_valid_{i}_i => {io_ctrl_valid[i].getNameRead()},\n"
            )

        for i in range(self.lsq_config.numLdPorts):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_ready_{i}_o => {io_ldAddr_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_valid_{i}_i => {io_ldAddr_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_{i}_i => {io_ldAddr_bits[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_ready_{i}_i => {io_ldData_ready[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_valid_{i}_o => {io_ldData_valid[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_{i}_o => {io_ldData_bits[i].getNameWrite()},\n"
            )

        for i in range(self.lsq_config.numStPorts):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_ready_{i}_o => {io_stAddr_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_valid_{i}_i => {io_stAddr_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_{i}_i => {io_stAddr_bits[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_ready_{i}_o => {io_stData_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_valid_{i}_i => {io_stData_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_{i}_i => {io_stData_bits[i].getNameRead()},\n"
            )

        # Define all AXI ports, we assume there is only 1 channel
        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rreq_ready_{i}_i => {rreq_ready[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rresp_valid_{i}_i => {rresp_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rresp_id_{i}_i => {rresp_id[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rreq_id_0_o => {rreq_id[i].getNameWrite()},\n"
            )

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wreq_ready_{i}_i => {wreq_ready[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wresp_valid_{i}_i => {wresp_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wresp_id_{i}_i => {wresp_id[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wreq_id_{i}_o => {wreq_id[i].getNameWrite()}\n"
            )

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 1) + ");\n"

        # End module definition
        self.lsq_wrapper_str += "end architecture;\n"

        # Write to the file
        with open(f"{self.output_folder}/{self.lsq_name}.vhd", 'w') as file:
            file.write(self.lsq_wrapper_str)

        return self.lsq_wrapper_str

    def genWrapperSlave(self):
        """This function generates the desired wrapper for the LSQ"""

        # PART 1: Add library information to the VHDL module
        self.lsq_wrapper_str += ctx.library

        # PART 2: Define the entity
        self.lsq_wrapper_str += f"entity {self.lsq_name} is\n"

        # PART 3: Add the module port definition
        self.lsq_wrapper_str += ctx.portInitString

        ##
        # Define all the IOs

        # io_stDataToMC_bits: output
        io_storeData = LogicVec(ctx, 
            "io_stDataToMC_bits", 'o', self.lsq_config.dataW, dyn_comp=True
        )

        # io_stAddrToMC_bits: output
        io_storeAddr = LogicVec(ctx, 
            "io_stAddrToMC_bits", 'o', self.lsq_config.addrW, dyn_comp=True
        )

        # io_ldDataFromMC_bits: input
        io_loadData = LogicVec(ctx, 
            "io_ldDataFromMC_bits", 'i', self.lsq_config.dataW, dyn_comp=True
        )

        # io_ldAddrToMC_bits: output
        io_loadAddr = LogicVec(ctx, 
            "io_ldAddrToMC_bits", 'o', self.lsq_config.addrW, dyn_comp=True
        )

        # io_ctrl_*_ready: output
        io_ctrl_ready = LogicArray(ctx, 
            "io_ctrl_ready", 'o', self.lsq_config.numGroups, dyn_comp=True
        )

        # io_ctrl_*_valid: input
        io_ctrl_valid = LogicArray(ctx, 
            "io_ctrl_valid", 'i', self.lsq_config.numGroups, dyn_comp=True
        )

        # io_ldAddr_*_ready: output
        io_ldAddr_ready = LogicArray(ctx, 
            "io_ldAddr_ready", 'o', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldAddr_*_valid: input
        io_ldAddr_valid = LogicArray(ctx, 
            "io_ldAddr_valid", 'i', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldAddr_*_bits: input
        io_ldAddr_bits = LogicVecArray(ctx, 
            "io_ldAddr_bits", 'i', self.lsq_config.numLdPorts, self.lsq_config.addrW, dyn_comp=True
        )

        # io_ldData_*_ready: input
        io_ldData_ready = LogicArray(ctx, 
            "io_ldData_ready", 'i', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldData_*_valid: output
        io_ldData_valid = LogicArray(ctx, 
            "io_ldData_valid", 'o', self.lsq_config.numLdPorts, dyn_comp=True
        )

        # io_ldData_*_bits: output
        io_ldData_bits = LogicVecArray(ctx, 
            "io_ldData_bits", 'o', self.lsq_config.numLdPorts, self.lsq_config.dataW, dyn_comp=True
        )

        # io_stAddr_ready: output
        io_stAddr_ready = LogicArray(ctx, 
            "io_stAddr_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stAddr_valid: input
        io_stAddr_valid = LogicArray(ctx, 
            "io_stAddr_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stAddr_bits: input
        io_stAddr_bits = LogicVecArray(ctx, 
            "io_stAddr_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.addrW, dyn_comp=True
        )

        # io_stData_ready: output
        io_stData_ready = LogicArray(ctx, 
            "io_stData_ready", 'o', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stData_valid: input
        io_stData_valid = LogicArray(ctx, 
            "io_stData_valid", 'i', self.lsq_config.numStPorts, dyn_comp=True
        )

        # io_stData_bits: input
        io_stData_bits = LogicVecArray(ctx, 
            "io_stData_bits", 'i', self.lsq_config.numStPorts, self.lsq_config.dataW, dyn_comp=True
        )

        # io_ldAddrToMC_ready: input
        io_ldAddrToMC_ready = Logic(ctx, "io_ldAddrToMC_ready", 'i', dyn_comp=True)

        # io_ldAddrToMC_valid
        io_ldAddrToMC_valid = Logic(ctx, "io_ldAddrToMC_valid", 'o', dyn_comp=True)

        # io_ldDataFromMC_ready
        io_ldDataFromMC_ready = Logic(ctx, "io_ldDataFromMC_ready", 'o', dyn_comp=True)

        # io_ldDataFromMC_valid
        io_ldDataFromMC_valid = Logic(ctx, "io_ldDataFromMC_valid", 'i', dyn_comp=True)

        # io_stAddrToMC_ready
        io_stAddrToMC_ready = Logic(ctx, "io_stAddrToMC_ready", 'i', dyn_comp=True)

        # io_stAddrToMC_valid
        io_stAddrToMC_valid = Logic(ctx, "io_stAddrToMC_valid", 'o', dyn_comp=True)

        # io_stDataToMC_ready
        io_stDataToMC_ready = Logic(ctx, "io_stDataToMC_ready", 'i', dyn_comp=True)

        # io_stDataToMC_valid
        io_stDataToMC_valid = Logic(ctx, "io_stDataToMC_valid", 'o', dyn_comp=True)

        ##
        # IO Definition finished
        ##
        self.lsq_wrapper_str += "\n\t);"
        self.lsq_wrapper_str += "\nend entity;\n\n"

        ##
        # Architecture definition start
        ##
        self.lsq_wrapper_str += f"architecture arch of {self.lsq_name} is\n"

        # Define internal signals
        io_loadEn = Logic(ctx, "io_loadEn", 'w', dyn_comp=True)

        io_storeEn = Logic(ctx, "io_storeEn", 'w', dyn_comp=True)

        rresp_id = LogicVecArray(ctx, 
            "rresp_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True
        )

        wreq_ready = LogicArray(ctx, 
            "wreq_ready", 'w', self.lsq_config.numStMem, dyn_comp=True
        )

        wresp_valid = LogicArray(ctx, 
            "wresp_valid", 'w', self.lsq_config.numStMem, dyn_comp=True
        )

        wresp_id = LogicVecArray(ctx, 
            "wresp_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True
        )

        rreq_id = LogicVecArray(ctx, 
            "rreq_id", 'w', self.lsq_config.numLdMem, self.lsq_config.idW, dyn_comp=True
        )

        wreq_id = LogicVecArray(ctx, 
            "wreq_id", 'w', self.lsq_config.numStMem, self.lsq_config.idW, dyn_comp=True
        )

        # Begin actual arch logic definition
        self.lsq_wrapper_str += "begin\n"

        # Define the process to update
        # rresp_id
        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"
        self.lsq_wrapper_str += "\t-- Process for rresp_id\n"
        self.lsq_wrapper_str += ctx.regInitString
        self.lsq_wrapper_str += "\t" * \
            (ctx.tabLevel + 1) + "if reset = '1' then\n"

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rresp_id[i], "(", "others", "=>", "'0'", ")")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 1) + "elsif rising_edge(clock) then\n"
        )
        self.lsq_wrapper_str += (
            "\n"
            + "\t" * (ctx.tabLevel + 2)
            + "if "
            + io_loadEn.getNameWrite()
            + " = '1' then\n"
        )

        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += Op(ctx, rresp_id[i], rreq_id[i])

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + "end if;\n"
            + "\t" * 2
            + "end if;\n"
            + "\tend process;\n"
        )

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        # Define the process to update
        # wresp_valid, wresp_id
        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"
        self.lsq_wrapper_str += (
            "\t-- Process for wreq_ready, wresp_valid and wresp_id\n"
        )
        self.lsq_wrapper_str += ctx.regInitString
        self.lsq_wrapper_str += "\t" * \
            (ctx.tabLevel + 1) + "if reset = '1' then\n"

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'0'")
            self.lsq_wrapper_str += Op(ctx, wresp_id[i], "(", "others", "=>", "'0'", ")")
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 1) + "elsif rising_edge(clock) then\n"
        )

        self.lsq_wrapper_str += (
            "\n"
            + "\t" * (ctx.tabLevel + 2)
            + "if "
            + io_storeEn.getNameWrite()
            + " = '1' then\n"
        )

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'1'")
            self.lsq_wrapper_str += Op(ctx, wresp_id[i], rreq_id[i])

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + "else\n"

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += Op(ctx, wresp_valid[i], "'0'")

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + "end if;\n"
            + "\t" * 2
            + "end if;\n"
            + "\tend process;\n"
        )

        self.lsq_wrapper_str += "\t----------------------------------------------------------------------------\n"

        ###
        # Signal Assignment
        ###
        self.lsq_wrapper_str += "\t-- Signal Assignment\n"
        self.lsq_wrapper_str += Op(ctx, io_ldAddrToMC_valid, io_loadEn)
        self.lsq_wrapper_str += Op(ctx, io_stAddrToMC_valid, io_storeEn)
        self.lsq_wrapper_str += Op(ctx, io_stDataToMC_valid, io_storeEn)
        self.lsq_wrapper_str += Op(
            ctx,
            wreq_ready[0],
            io_stAddrToMC_ready,
            "and",
            io_stDataToMC_ready,
        )

        ###
        # Instantiate the LSQ_core module
        ###
        self.lsq_wrapper_str += "\t-- Instantiate the core LSQ logic\n"
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel)
            + f"{self.lsq_name}_core : entity work.{self.lsq_name}_core\n"
        )
        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 1) + f"port map(\n"

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + f"rst => reset,\n"
        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 2) + f"clk => clock,\n"

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_data_0_o => {io_storeData.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_addr_0_o => {io_storeAddr.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"wreq_valid_0_o => {io_storeEn.getNameWrite()},\n"
        )

        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rresp_data_0_i => {io_loadData.getNameRead()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rreq_addr_0_o => {io_loadAddr.getNameWrite()},\n"
        )
        self.lsq_wrapper_str += (
            "\t" * (ctx.tabLevel + 2)
            + f"rreq_valid_0_o => {io_loadEn.getNameWrite()},\n"
        )

        for i in range(self.lsq_config.numGroups):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"group_init_ready_{i}_o => {io_ctrl_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"group_init_valid_{i}_i => {io_ctrl_valid[i].getNameRead()},\n"
            )

        for i in range(self.lsq_config.numLdPorts):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_ready_{i}_o => {io_ldAddr_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_valid_{i}_i => {io_ldAddr_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_addr_{i}_i => {io_ldAddr_bits[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_ready_{i}_i => {io_ldData_ready[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_valid_{i}_o => {io_ldData_valid[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"ldp_data_{i}_o => {io_ldData_bits[i].getNameWrite()},\n"
            )

        for i in range(self.lsq_config.numStPorts):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_ready_{i}_o => {io_stAddr_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_valid_{i}_i => {io_stAddr_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_addr_{i}_i => {io_stAddr_bits[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_ready_{i}_o => {io_stData_ready[i].getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_valid_{i}_i => {io_stData_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"stp_data_{i}_i => {io_stData_bits[i].getNameRead()},\n"
            )

        # Define all AXI ports, we assume there is only 1 channel
        for i in range(self.lsq_config.numLdMem):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rreq_ready_{i}_i => {io_ldAddrToMC_ready.getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rresp_valid_{i}_i => {io_ldDataFromMC_valid.getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rresp_ready_{i}_o => {io_ldDataFromMC_ready.getNameWrite()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rresp_id_{i}_i => {rresp_id[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"rreq_id_0_o => {rreq_id[i].getNameWrite()},\n"
            )

        for i in range(self.lsq_config.numStMem):
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wreq_ready_{i}_i => {wreq_ready[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wresp_valid_{i}_i => {wresp_valid[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wresp_id_{i}_i => {wresp_id[i].getNameRead()},\n"
            )
            self.lsq_wrapper_str += (
                "\t" * (ctx.tabLevel + 2)
                + f"wreq_id_{i}_o => {wreq_id[i].getNameWrite()}\n"
            )

        self.lsq_wrapper_str += "\t" * (ctx.tabLevel + 1) + ");\n"

        # End module definition
        self.lsq_wrapper_str += "end architecture;\n"

        # Write to the file
        with open(f"{self.output_folder}/{self.lsq_name}.vhd", 'w') as file:
            file.write(self.lsq_wrapper_str)

        return self.lsq_wrapper_str


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

    # Parse the config file
    lsqConfig = GetConfigs(args.config_files)

    # STEP 1: Generate the desired core lsq logic
    codeGen(args.output_path, lsqConfig)

    # STEP 2: Generate the wrapper to be connected with circuits generated by Dynamatic
    lsq_wrapper_module = LSQWrapper(args.output_path, "_wrapper", lsqConfig)

    # Step 3: Generate the corresponding wrapper based on the config.master
    if lsqConfig.master:
        lsq_wrapper_module.genWrapper()
    else:
        lsq_wrapper_module.genWrapperSlave()


if __name__ == "__main__":
    main()
