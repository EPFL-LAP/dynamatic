#
# Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import math
import json

from typing import List

class Config:
    """
    Configuration object for LSQ code generation.

    This class is instantiated using 'GetConfigs(path_to_json_file)', which loads a JSON file
    and overwrites all the values below using user-defined parameters.

    The values shown below are NOT the actual values used during generation;
    they are one of possible default configurations.
    """

    num_groups: int
    """
    Number of individual groups in the group allocator.
    By definition is equal to the number of basic blocks 
    which have loads or stores to this LSQ.
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

    _group_num_loads:    list = [2, 1]    # Number of loads in each BB
    _group_num_stores:   list = [2, 1]    # Number of stores in each BB
    _group_store_order:     list = [[2, 2],  # The order matrix for each group
                           [0]]     # Outer list (Row): Index for each BB
    # Inner list (Column): List of store counts ahead of each load
    # In this example -> BB0=[st0,st1,ld0,ld1], BB1=[ld2,st2]
    _group_load_port_idxs:   list = [[0, 1],   # The related access port index for each load in BB
                           [2]]
    _group_store_port_idxs:   list = [[0, 1],   # The related access port index for each store in BB
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
            self.dataW = self.payload_bitwidth

            self.addrW = obj["addrWidth"]
            self.idW = obj["indexWidth"]

            #self.numLdqEntries
            self._ldq_num_entries = obj["fifoDepth_L"]
            self.numLdqEntries = self._ldq_num_entries

            # numStqEntries
            self._stq_num_entries = obj["fifoDepth_S"]
            self.numStqEntries = self._stq_num_entries

            # numLdPorts
            self._num_ld_ports = obj["numLoadPorts"]
            self.numLdPorts = self._num_ld_ports
            
            # numStPorts
            self._num_st_ports = obj["numStorePorts"]
            self.numStPorts = self._num_st_ports

            self._num_groups = obj["numBBs"]

            self.numLdMem = obj["numLdChannels"]
            self.numStMem = obj["numStChannels"]
            self.master = bool(obj["master"])

            self.stResp = bool(obj["stResp"])
            self.gaMulti = bool(obj["groupMulti"])

            #gaNumLoads
            self._group_num_loads = obj["numLoads"]
            self.gaNumLoads = self._group_num_loads

            #gaNumStores
            self._group_num_stores = obj["numStores"]
            self.gaNumStores = self._group_num_stores

            self._group_store_order = obj["ldOrder"]
            self._group_load_port_idxs = obj["ldPortIdx"]
            self._group_store_port_idxs = obj["stPortIdx"]

            #self.ldqAddrW
            self._ldq_idx_w = math.ceil(math.log2(self._ldq_num_entries))
            self.ldqAddrW = self._ldq_idx_w
            
            #self.stqAddrW
            self._stq_idx_w = math.ceil(math.log2(self._stq_num_entries))
            self.stqAddrW = self._stq_idx_w
            
            # emptyLdAddrW
            self._ldq_size_w = math.ceil(math.log2(self._ldq_num_entries+1))
            self.emptyLdAddrW = self._ldq_size_w

            # emptyStAddrW
            self._stq_size_w = math.ceil(math.log2(self._stq_num_entries+1))
            self.emptyStAddrW = self._stq_size_w

            # Check the number of ports, if num*Ports == 0, set it to 1

            # self.ldpAddrW
            self._ldp_idx_w = math.ceil(math.log2(self._num_ld_ports if self._num_ld_ports > 0 else 1))
            self.ldpAddrW = self._ldp_idx_w
            
            # self.stpAddrW
            self._stp_idx_w = math.ceil(math.log2(self._num_st_ports if self._num_st_ports > 0 else 1))
            print("Store port index:", self._stp_idx_w, self._num_st_ports, math.log2(self._num_st_ports))
            self.stpAddrW = self._stp_idx_w

            self.pipe0 = bool(obj["pipe0En"])
            self.pipe1 = bool(obj["pipe1En"])
            self.pipeComp = bool(obj["pipeCompEn"])
            self.headLag = bool(obj["headLagEn"])

            assert (self.idW >= self.ldqAddrW)

            # list size checking
            assert (len(self._group_num_loads) == self.num_groups())
            assert (len(self._group_num_stores) == self.num_groups())
            assert (len(self._group_store_order) == self.num_groups())
            assert (len(self._group_load_port_idxs) == self.num_groups())
            assert (len(self._group_store_port_idxs) == self.num_groups())

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
    
    def load_queue_size_w(self) -> int:
        """
        Bitwidth required to represent 
        the total number of queue entries.

        Calculated by ceil(log2(num_entries + 1))
        """
        return self._ldq_size_w

    def store_queue_size_w(self) -> int:
        """
        Bitwidth required to represent 
        the total number of queue entries.

        Calculated by Calculated by ceil(log2(num_entries + 1))
        """
        return self._stq_size_w

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
    
    def group_num_loads(self, group_idx) -> int:
        """
        Number of loads in a group
        """
        return self._group_num_loads[group_idx]
    
    def group_num_stores(self, group_idx) -> int:
        """
        Number of stores in a group
        """
        return self._group_num_stores[group_idx]
    
    def group_load_ports(self, group_idx) -> List[int]:
        """
        List of load port indices (1 per load) in a group
        """
        return self._group_load_port_idxs[group_idx]
    
    def group_store_ports(self, group_idx) -> List[int]:
        """
        List of store port indices (1 per store) in a group
        """
        return self._group_store_port_idxs[group_idx]
    
    def group_store_order(self, group_idx) -> List[int]:
        """
        List of store orders (1 per load) in a group
        """
        return self._group_store_order[group_idx]
    
    def group_num_loads(self, group_idx) -> int:
        """
        Number of loads in a group
        """
        return self._group_num_loads[group_idx]
    
    def group_num_stores(self, group_idx) -> int:
        """
        Number of stores in a group
        """
        return self._group_num_stores[group_idx]