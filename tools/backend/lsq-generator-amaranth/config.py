from dataclasses import dataclass

@dataclass
class LsqConfig:
    """Configuration object for LSQ code generation."""

    name:          str # Name prefix used for generated files
    dataW:         int # Data width        (Number of bits for load/store data)
    addrW:         int # Address width     (Number of bits for memory address)
    idW:           int # ID width          (Number of bits for ID in the memory interface)
    numLdqEntries: int # Load queue size   (Number of entries in the load queue)
    numStqEntries: int # Store queue size  (Number of entries in the store queue)
    numLdPorts:    int # Number of load access ports
    numStPorts:    int # Number of store access ports
    numGroups:     int # Number of total Basic Blocks (BBs)
    numLdMem:      int # Number of load channels at memory interface (Fixed to 1)
    numStMem:      int # Number of store channels at memory interface (Fixed to 1)

    gaNumLoads:    list[int] # Number of loads in each BB
    gaNumStores:   list[int] # Number of stores in each BB
    gaLdOrder:     list[list[int]] # The order matrix for each group
    # Outer list (Row): Index for each BB
    # Inner list (Column): List of store counts ahead of each load
    # In this example -> BB0=[st0,st1,ld0,ld1], BB1=[ld2,st2]
    gaLdPortIdx:   list[list[int]] # The related access port index for each load in BB
    gaStPortIdx:   list[list[int]] # The related access port index for each store in BB
    ldqAddrW:      int # Load queue address width
    stqAddrW:      int # Store queue address width
    ldpAddrW:      int # Load port address width
    stpAddrW:      int # Store port address width

    pipe0:        bool # Enable pipeline register 0
    pipe1:        bool # Enable pipeline register 1
    pipeComp:     bool # Enable pipeline register pipeComp
    headLag:      bool # Whether the head pointer of the load queue is updated
    # one cycle later than the valid bits of entries
    stResp:        bool # Whether store response channel in store access port is enabled
    gaMulti:       bool # Whether multiple groups are allowed to request an allocation at the same cycle
    bypass:        bool # Whether bypassing (store-to-load forwarding) is enabled
