#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
import json

class Config:
    """
    Configuration object for LSQ code generation.

    This class is instantiated using 'GetConfigs(path_to_json_file)', which loads a JSON file
    and overwrites all the values below using user-defined parameters.

    The values shown below are NOT the actual values used during generation;
    they are one of possible default configurations.
    """

    name:          str = 'test'     # Name prefix used for generated VHDL files
    payload_bitwidth:         int = 16         # Data width        (Number of bits for load/store data)
    addrW:         int = 13         # Address width     (Number of bits for memory address)
    idW:           int = 2          # ID width          (Number of bits for ID in the memory interface)
    numLdqEntries: int = 3          # Load queue size   (Number of entries in the load queue)
    numStqEntries: int = 10         # Store queue size  (Number of entries in the store queue)
    numLdPorts:    int = 3          # Number of load access ports
    numStPorts:    int = 3          # Number of store access ports
    num_groups:     int = 2          # Number of total Basic Blocks (BBs)
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

    def __init__(self, json_file) -> None:
        with open(json_file, "r") as f:
            obj = json.load(f)
            
            self.name = obj["name"]
            # self.dataW
            self.payload_bitwidth = obj["dataWidth"]
            self.addrW = obj["addrWidth"]
            self.idW = obj["indexWidth"]

            #self.numLdqEntries
            self._ldq_num_entries = obj["fifoDepth_L"]

            # numStqEntries
            self._stq_num_entries = obj["fifoDepth_S"]

            # numLdPorts
            self._num_ld_ports = obj["numLoadPorts"]
            
            # numStPorts
            self._num_st_ports = obj["numStorePorts"]

            self._num_groups = obj["numBBs"]

            self.numLdMem = obj["numLdChannels"]
            self.numStMem = obj["numStChannels"]
            self.master = bool(obj["master"])

            self.stResp = bool(obj["stResp"])
            self.gaMulti = bool(obj["groupMulti"])

            self.gaNumLoads = obj["numLoads"]
            self.gaNumStores = obj["numStores"]
            self.gaLdOrder = obj["ldOrder"]
            self.gaLdPortIdx = obj["ldPortIdx"]
            self.gaStPortIdx = obj["stPortIdx"]

            #self.ldqAddrW
            self._ldq_idx_w = math.ceil(math.log2(self.numLdqEntries))
            
            #self.sdqAddrW
            self._stq_idx_w = math.ceil(math.log2(self.numStqEntries))
            
            # emptyLdAddrW
            self._empty_ldw_idx_w = math.ceil(math.log2(self.numLdqEntries+1))

            # emptyStAddrW
            self._empty_stq_idx_w = math.ceil(math.log2(self.numStqEntries+1))
            
            # Check the number of ports, if num*Ports == 0, set it to 1

            # self.ldpAddrW
            self._ldp_idx_w = math.ceil(math.log2(self.numLdPorts if self.numLdPorts > 0 else 1))
            
            # self.stpAddrW
            self._stp_idx_w = math.ceil(math.log2(self.numStPorts if self.numStPorts > 0 else 1))

            self.pipe0 = bool(obj["pipe0En"])
            self.pipe1 = bool(obj["pipe1En"])
            self.pipeComp = bool(obj["pipeCompEn"])
            self.headLag = bool(obj["headLagEn"])

            assert (self.idW >= self.ldqAddrW)

            # list size checking
            assert (len(self.gaNumLoads) == self.num_groups())
            assert (len(self.gaNumStores) == self.num_groups())
            assert (len(self.gaLdOrder) == self.num_groups())
            assert (len(self.gaLdPortIdx) == self.num_groups())
            assert (len(self.gaStPortIdx) == self.num_groups())

    def num_groups(self) -> int:
        """
        Number of individual groups in the group allocator.
        By definition is equal to the number of basic blocks 
        which have loads or stores to this LSQ.
        """
        return self._num_groups
    
    def load_queue_idx_bitwidth(self) -> int:
        """
        Bitwidth for a pointer into the load queue.
        Calculated by ceil(log2(num_entries))
        """
        return self._ldq_idx_w
    
    def store_queue_idx_bitwidth(self) -> int:
        """
        Bitwidth for a pointer into the store queue.
        Calculated by ceil(log2(num_entries))
        """
        return self._stq_idx_w
    
    def load_queue_num_entries(self) -> int:
        """
        Number of queue entries in the load queue.
        """
        return self._ldq_num_entries
    
    def load_ports_idx_bitwidth(self) -> int:
        """
        Bitwidth required to identify a load port.
        Calculated by ceil(log2(num_load_ports))
        Inconsistant code for this: 
        If there are no load ports, the bitwidth is overriden to 1,
        but there are checks for if it is equal to 0 across the code
        """
        return self._ldp_idx_w
    
    def store_queue_num_entries(self) -> int:
        """
        Number of queue entries in the store queue.
        """
        return self._stq_num_entries
    
    def store_ports_idx_bitwidth(self) -> int:
        """
        Bitwidth required to identify a store port.
        Calculated by ceil(log2(num_store_ports))
        Inconsistant code for this: 
        If there are no load ports, the bitwidth is overriden to 1,
        but there are checks for if it is equal to 0 across the code
        """
        return self._stp_idx_w
    
    def store_ports_num(self) -> int:
        """
        Number of store ports.
        """
        return self._num_st_ports
    
    def load_ports_num(self) -> int:
        """
        Number of load ports.
        """
        return self._num_ld_ports