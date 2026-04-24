#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
import json
import os
import sys
from pprint import pprint

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
    bypass:        bool = True      # Whether bypassing (store-to-load forwarding) is enabled

    # guarantees execution of the oldest pending memory operation (load or store) in the presence of false conflicts
    # (which can happen with approximate address comparison)
    fallbackIssueLoad: bool = False
    fallbackIssueStore: bool = False

    # fully in-order issue: operations are issued in program order (oldest first, globally across loads and stores;
    # implies fallbackIssueLoad=True, fallbackIssueStore=True, bypass=False)
    inOrder: bool = False

    # synthetic issue restrictions: restricts which loads can be issued
    # If not None, only allow the N oldest pending (allocated but not issued) loads to be issued
    issueOldestLoads: int | None = None

    def __init__(self, config: dict) -> None:
        self.name = config["name"]
        self.dataW = config["dataWidth"]
        self.addrW = config["addrWidth"]
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

        # TODO: set based on requested LSQ model
        self.bypass = True
        self.fallbackIssueLoad = False
        self.fallbackIssueStore = False
        self.inOrder = False

        self.gaNumLoads = config["numLoads"]
        self.gaNumStores = config["numStores"]
        self.gaLdOrder = config["ldOrder"]
        self.gaLdPortIdx = config["ldPortIdx"]
        self.gaStPortIdx = config["stPortIdx"]

        self.pipe0 = bool(config["pipe0En"])
        self.pipe1 = bool(config["pipe1En"])
        self.pipeComp = bool(config["pipeCompEn"])
        self.headLag = bool(config["headLagEn"])

        def get_env(name: str) -> int | None:
            value = os.environ.get(name, "")
            if value == "":
                return None
            else:
                return int(value)

        numLdqEntries = get_env("LSQ_NUM_LDQ_ENTRIES")
        if numLdqEntries is not None:
            self.numLdqEntries = numLdqEntries
        numStqEntries = get_env("LSQ_NUM_STQ_ENTRIES")
        if numStqEntries is not None:
            self.numStqEntries = numStqEntries
        noBypass = get_env("LSQ_NO_BYPASS")
        if noBypass is not None:
            self.bypass = not bool(noBypass)
        inOrder = get_env("LSQ_IN_ORDER")
        if inOrder is not None:
            self.inOrder = bool(inOrder)

        pipeComp = get_env("LSQ_PIPE_COMP_EN")
        if pipeComp is not None:
            self.pipeComp = bool(pipeComp)
        pipe0 = get_env("LSQ_PIPE0_EN")
        if pipe0 is not None:
            self.pipe0 = bool(pipe0)
        pipe1 = get_env("LSQ_PIPE1_EN")
        if pipe1 is not None:
            self.pipe1 = bool(pipe1)
        headLag = get_env("LSQ_HEAD_LAG_EN")
        if headLag is not None:
            self.headLag = bool(headLag)

        ### ISSUE RESTRICTION ###
        self.issueOldestLoads = None
        issueOldestLoads = get_env("LSQ_ISSUE_OLDEST_LOADS")
        if issueOldestLoads is not None:
            self.issueOldestLoads = int(issueOldestLoads)
            if self.issueOldestLoads >= self.numLdqEntries:
                self.issueOldestLoads = None  # not needed

        ### COMPUTED VALUES ###

        if self.inOrder:
            # in-order requires bypass to be disabled and fallback issue to be enabled for both loads and stores
            self.bypass = False
            self.fallbackIssueLoad = True
            self.fallbackIssueStore = True

        self.ldqAddrW = math.ceil(math.log2(self.numLdqEntries))
        self.stqAddrW = math.ceil(math.log2(self.numStqEntries))
        self.idW = max(self.ldqAddrW, self.stqAddrW)

        # Use one more bit to be able to represent the empty state of the queue when the number of entries is a power of 2
        self.emptyLdAddrW = self.ldqAddrW + 1
        self.emptyStAddrW = self.stqAddrW + 1

        # Check the number of ports, if num*Ports == 0, set it to 1
        self.ldpAddrW = math.ceil(math.log2(self.numLdPorts if self.numLdPorts > 0 else 1))
        self.stpAddrW = math.ceil(math.log2(self.numStPorts if self.numStPorts > 0 else 1))

        pprint(self.__dict__)

        ### CHECKS ###

        assert (self.idW >= self.ldqAddrW)

        # list size checking
        assert (len(self.gaNumLoads) == self.numGroups)
        assert (len(self.gaNumStores) == self.numGroups)
        assert (len(self.gaLdOrder) == self.numGroups)
        assert (len(self.gaLdPortIdx) == self.numGroups)
        assert (len(self.gaStPortIdx) == self.numGroups)

        # An LSQ with N load/store entries can only support up to N-1 loads/stores per group.
        for i in range(self.numGroups):
            assert self.gaNumLoads[i] < self.numLdqEntries, f"group {i}: too many loads ({self.gaNumLoads[i]}) for load queue with {self.numLdqEntries} entries!"
            assert self.gaNumStores[i] < self.numStqEntries, f"group {i}: too many stores ({self.gaNumStores[i]}) for store queue with {self.numStqEntries} entries!"

        if self.fallbackIssueLoad or self.fallbackIssueStore:
            assert not self.bypass, "Fallback issue is not compatible with bypassing."
        if self.fallbackIssueLoad:
            # TODO: To properly support multiple load channels, we need to ensure that the fallback load is not
            # duplicated a load # issued by another load channel in the same cycle. Multiple load channels are not
            # currently used by Dynamatic, so this is left as future work.
            assert self.numLdMem == 1, "Fallback issue is only supported for single load port configuration."

        # synthetic issue restrictions
        if self.issueOldestLoads is not None:
            assert self.issueOldestLoads > 0, "issueOldestLoads must be positive."
            assert self.issueOldestLoads <= self.numLdqEntries, "issueOldestLoads cannot be greater than the number of load queue entries."
            assert not self.fallbackIssueLoad, "issueOldestLoads is not compatible with fallback issue for loads."
