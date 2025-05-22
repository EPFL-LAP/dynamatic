#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
import json
import sys

# read and parse the json file
# example json format in README


def GetConfigs(config_json_path: str):
    with open(config_json_path, 'r') as file:
        configString = file.read()
        configs = json.loads(configString)
        return Configs(configs)


class Configs:
    """
    Configuration object for LSQ code generation.

    This class is instantiated using 'GetConfigs(path_to_json_file)', which loads a JSON file
    and overwrites all the values below using user-defined parameters.

    The values shown below are NOT the actual values used during generation;
    they are one of possible default configurations.
    """

    name:          str = 'test'     # Name prefix used for generated VHDL files
    dataW:         int = 16         # Data width        (Number of bits for load/store data)
    addrW:         int = 13         # Address width     (Number of bits for memory address)
    idW:           int = 2          # ID width          (Number of bits for ID in the memory interface)
    numLdqEntries: int = 3          # Load queue size   (Number of entries in the load queue)
    numStqEntries: int = 10         # Store queue size  (Number of entries in the store queue)
    numLdPorts:    int = 3          # Number of load access ports
    numStPorts:    int = 3          # Number of store access ports
    numGroups:     int = 2          # Number of total Basic Blocks (BBs)
    numLdMem:      int = 1          # Number of load channels at memory interface (Fixed to 1)
    numStMem:      int = 1          # Number of store channels at memory interface (Fixed to 1)

    gaNumLoads:    list = [2, 1]    # Number of loads in each BB
    gaNumStores:   list = [2, 1]    # Number of stores in each BB
    gaLdOrder:     list = [[2, 2],  # The order matrix for each group
                           [0]]     # Outer list (Row): Index for each BB
    # Inner list (Column): List of store counts ahead of each load
    # In this example -> BB0=[st0,st1,ld0,ld1], BB1=[ld2,st2]
    gaLdPortIdx:   list = [[0, 1],   # The related access port index for each load in BB
                           [2]]
    gaStPortIdx:   list = [[0, 1],   # The related access port index for each store in BB
                           [2]]
    ldqAddrW:      int = 2          # Load queue address width
    stqAddrW:      int = 4          # Store queue address width
    ldpAddrW:      int = 2          # Load port address width
    stpAddrW:      int = 2          # Store port address width

    pipe0:        bool = False      # Enable pipeline register 0
    pipe1:        bool = False      # Enable pipeline register 1
    pipeComp:     bool = False      # Enable pipeline register pipeComp
    headLag:      bool = False      # Whether the head pointer of the load queue is updated
    # one cycle later than the valid bits of entries
    stResp:        bool = False     # Whether store response channel in store access port is enabled
    gaMulti:       bool = False     # Whether multiple groups are allowed to request an allocation at the same cycle

    def __init__(self, config: dict) -> None:
        self.name = config["name"]
        self.dataW = config["dataWidth"]
        self.addrW = config["addrWidth"]
        self.idW = config["indexWidth"]
        self.numLdqEntries = config["fifoDepth_L"]
        self.numStqEntries = config["fifoDepth_S"]
        self.numLdPorts = config["numLoadPorts"]
        self.numStPorts = config["numStorePorts"]
        self.numGroups = config["numBBs"]
        self.numLdMem = config["numLdChannels"]
        self.numStMem = config["numStChannels"]
        self.master = bool(config["master"])

        self.stResp = bool(config["stResp"])
        self.gaMulti = bool(config["groupMulti"])

        self.gaNumLoads = config["numLoads"]
        self.gaNumStores = config["numStores"]
        self.gaLdOrder = config["ldOrder"]
        self.gaLdPortIdx = config["ldPortIdx"]
        self.gaStPortIdx = config["stPortIdx"]

        self.ldqAddrW = math.ceil(math.log2(self.numLdqEntries))
        self.stqAddrW = math.ceil(math.log2(self.numStqEntries))
        self.emptyLdAddrW = math.ceil(math.log2(self.numLdqEntries+1))
        self.emptyStAddrW = math.ceil(math.log2(self.numStqEntries+1))
        # Check the number of ports, if num*Ports == 0, set it to 1
        self.ldpAddrW = math.ceil(math.log2(self.numLdPorts if self.numLdPorts > 0 else 1))
        self.stpAddrW = math.ceil(math.log2(self.numStPorts if self.numStPorts > 0 else 1))

        self.pipe0 = bool(config["pipe0En"])
        self.pipe1 = bool(config["pipe1En"])
        self.pipeComp = bool(config["pipeCompEn"])
        self.headLag = bool(config["headLagEn"])

        assert (self.idW >= self.ldqAddrW)

        # list size checking
        assert (len(self.gaNumLoads) == self.numGroups)
        assert (len(self.gaNumStores) == self.numGroups)
        assert (len(self.gaLdOrder) == self.numGroups)
        assert (len(self.gaLdPortIdx) == self.numGroups)
        assert (len(self.gaStPortIdx) == self.numGroups)
