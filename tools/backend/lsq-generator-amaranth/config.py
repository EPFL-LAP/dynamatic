import math
from dataclasses import dataclass


@dataclass
class LsqConfig:
    """Configuration object for LSQ code generation."""

    name:          str  # Name prefix used for generated files
    dataW:         int  # Data width        (Number of bits for load/store data)
    addrW:         int  # Address width     (Number of bits for memory address)
    idW:           int  # ID width          (Number of bits for ID in the memory interface)
    numLdqEntries: int  # Load queue size   (Number of entries in the load queue)
    numStqEntries: int  # Store queue size  (Number of entries in the store queue)
    numLdPorts:    int  # Number of load access ports
    numStPorts:    int  # Number of store access ports
    numGroups:     int  # Number of total Basic Blocks (BBs)
    numLdMem:      int  # Number of load channels at memory interface (Fixed to 1)
    numStMem:      int  # Number of store channels at memory interface (Fixed to 1)

    master: bool  # Whether this LSQ drives the memory directly or is a slave to a regular memory controller

    gaNumLoads:    list[int]  # Number of loads in each BB
    gaNumStores:   list[int]  # Number of stores in each BB
    gaLdOrder:     list[list[int]]  # The order matrix for each group
    # Outer list (Row): Index for each BB
    # Inner list (Column): List of store counts ahead of each load
    # In this example -> BB0=[st0,st1,ld0,ld1], BB1=[ld2,st2]
    gaLdPortIdx:   list[list[int]]  # The related access port index for each load in BB
    gaStPortIdx:   list[list[int]]  # The related access port index for each store in BB

    pipe0:        bool  # Enable pipeline register 0
    pipe1:        bool  # Enable pipeline register 1
    pipeComp:     bool  # Enable pipeline register pipeComp
    headLag:      bool  # Whether the head pointer of the load queue is updated
    # one cycle later than the valid bits of entries
    stResp:        bool  # Whether store response channel in store access port is enabled
    gaMulti:       bool  # Whether multiple groups are allowed to request an allocation at the same cycle
    bypass:        bool  # Whether bypassing (store-to-load forwarding) is enabled

    @classmethod
    def from_json(cls, data: dict):
        """Create an instance of LsqConfig from a JSON dictionary."""
        config = cls(
            name=data["name"],
            dataW=data["dataWidth"],
            addrW=data["addrWidth"],
            idW=data["indexWidth"],
            numLdqEntries=data["fifoDepth_L"],
            numStqEntries=data["fifoDepth_S"],
            numLdPorts=data["numLoadPorts"],
            numStPorts=data["numStorePorts"],
            numGroups=data["numBBs"],
            numLdMem=data["numLdChannels"],
            numStMem=data["numStChannels"],
            master=bool(data["master"]),
            stResp=bool(data["stResp"]),
            gaMulti=bool(data["groupMulti"]),
            bypass=True,
            gaNumLoads=data["numLoads"],
            gaNumStores=data["numStores"],
            gaLdOrder=data["ldOrder"],
            gaLdPortIdx=data["ldPortIdx"],
            gaStPortIdx=data["stPortIdx"],
            pipe0=bool(data.get("pipe0", False)),
            pipe1=bool(data.get("pipe1", False)),
            pipeComp=bool(data.get("pipeComp", False)),
            headLag=bool(data.get("headLag", False))
        )
        config.validate()
        return config

    @property
    def ldqAddrW(self) -> int:
        return math.ceil(math.log2(self.numLdqEntries))

    @property
    def stqAddrW(self) -> int:
        return math.ceil(math.log2(self.numStqEntries))

    @property
    def ldpAddrW(self) -> int:
        return math.ceil(math.log2(self.numLdPorts)) if self.numLdPorts > 0 else 0

    @property
    def stpAddrW(self) -> int:
        return math.ceil(math.log2(self.numStPorts)) if self.numStPorts > 0 else 0

    def validate(self):
        """Validate the configuration parameters."""
        assert self.idW >= self.ldqAddrW, "ID width must be greater than or equal to load queue address width."
        assert len(self.gaNumLoads) == self.numGroups, "Length of gaNumLoads must match numGroups."
        assert len(self.gaNumStores) == self.numGroups, "Length of gaNumStores must match numGroups."
        assert len(self.gaLdOrder) == self.numGroups, "Length of gaLdOrder must match numGroups."
        assert len(self.gaLdPortIdx) == self.numGroups, "Length of gaLdPortIdx must match numGroups."
        assert len(self.gaStPortIdx) == self.numGroups, "Length of gaStPortIdx must match numGroups."

        # An LSQ with N load/store entries can only support up to N-1 loads/stores per group.
        for i in range(self.numGroups):
            assert self.gaNumLoads[i] < self.numLdqEntries, f"group {i}: too many loads ({self.gaNumLoads[i]}) for load queue ({self.numLdqEntries} entries)."
            assert self.gaNumStores[i] < self.numStqEntries, f"group {i}: too many stores ({self.gaNumStores[i]}) for store queue ({self.numStqEntries} entries)."
