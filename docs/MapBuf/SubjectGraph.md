# Subject Graphs
Subject graphs are directed acyclic graphs composed of abstract logic operations (not actual gates). They serve as technology-independent representations of circuit logic, with common types including AND-Inverter Graphs (AIGs) and XOR-AND-Inverter Graphs (XAGs). In the implementation of MapBuf, we use AIGs for subject graphs.

While the Handshake dialect in Dynamatic is used to model the Dataflow circuits with Operations corresponding to Dataflow units, it falls short of providing the AIG structure required by MapBuf. Existing buffer placement algorithms (FPGA20, FPL22) use Dataflow graph channels (represented as Values in MLIR) as timing variables in the MILP formulation. However, the representation provided by the Handshake dialect is insufficient for MapBuf's MILP formulation, which requires AND-Inverter Graph (AIG) edges as timing variables to accurately model LUT-level timing constraints.

This creates a gap within Dynamatic, the high-level Handshake dialect cannot provide the low-level AIG representation needed for MapBuf. The Subject Graph class implementation fills this gap. While it is not a formal MLIR dialect, it functions conceptually as an AIG Dialect within Dynamatic. The Subject Graph implementation:

1) Creates AIG representations of Dataflow units by integrating BLIF files of the corresponding hardware units 
2) Constructs the complete Subject Graph of the entire circuit by connecting these individual unit representations
3) Provides bidirectional mapping between MLIR Operations and Subject Graphs through a static moduleMap, enabling efficient lookups in both directions
4) Enables buffer insertion at specific points in the circuit

# BaseSubjectGraph Class
BaseSubjectGraph class is an abstract class that provides base functionality for hardware module specific Subject Graph classes. 
## Key Variables
1) Operation *op: The MLIR Operation of the Dataflow unit that the Subject Graph represents
2) std::string uniqueName: Unique identifier used for node naming in the BLIF file
3) bool isBlackbox: Flag indicating if the module is not mapped to LUTs but DSPs or carry chains on the FPGA. No AIG is created for the logic part of these modules, but only channel signals are created.
4) std::vector<BaseSubjectGraph *> inputSubjectGraphs/outputSubjectGraphs: SubjectGraphs connected as inputs/outputs
5) DenseMap<BaseSubjectGraph *, unsigned int> inputSubjectGraphToResultNumber: Maps SubjectGraphs to their MLIR result numbers
6) static DenseMap<Operation *, BaseSubjectGraph *> moduleMap: A static variable that maps Operations to their SubjectGraphs
7) LogicNetwork *blifData: Pointer to the parsed BLIF file data, the AIG file is saved here. 

## Key Functions
1) void buildSubjectGraphConnections(): Populates input/output SubjectGraph vectors and maps of a SubjectGraph object
2) void connectInputNodesHelper(): Helper for connecting input nodes to outputs of preceding module. Used to connect AIGs of different units, so that we can have the AIG of the whole circuit.

### Virtual Functions
1) virtual void connectInputNodes() = 0: Connects the input nodes of the this SubjectGraph with another SubjectGraph
2) virtual ChannelSignals &returnOutputNodes(unsigned int resultNumber) = 0: Returns output nodes for a specific channel 

# Channel Signals
A struct that holds the different types of signals that a channel can have. It consists of a vector of Nodes for Data signals, and single Nodes for Valid and Ready signals. The input/output variables of the SubjectGraph classes consist of this struct.

# Derived SubjectGraph Classes
Each different dataflow unit has its own derived SubjectGraph class, extending BaseSubjectGraph to suit different number of I/O channels and naming conventions. These classes have different variables representing their I/O channels, and also specialized constructor to correctly parse the BLIF file generated from their hardware descriptions, in accordance with the naming convention used for these modules in their hardware descriptions.

## ArithSubjectGraph
Represents arithmetic operations in the Handshake dialect, which consists of AddIOp, AndIOp, CmpIOp, OrIOp, ShLIOp, ShRSIOp, ShRUIOp, SubIOp, XOrIOp, MulIOp, DivSIOp, DivUIOp. 

### Variables
1) unsigned int dataWidth: Bit width of the data signals (DATA_TYPE parameter in the HDL)
Corresponds to the DATA_TYPE parameter in the HDL implementation. 
2) std::unordered_map<unsigned int, ChannelSignals> inputNodes: Maps lhs and rhs inputs to their corresponding Channel Signals. lhs goes to inputNodes[0] and rhs goes to inputNodes[1].
3) ChannelSignals outputNodes: Output Channel Signals of the module.

### Functions
1) ArithSubjectGraph(Operation *op): 
    1) Retrieves the dataWidth of the module.
    2) Checks if dataWidth is greater than 4, if so, the module is a blackbox. 
    3) AIG is read into blifData variable.
    4) Loop over all of the nodes of AIG. Based on the names, populate the ChannelSignals structs of inputs and outputs. For example, if a node in the AIG file has the string "lhs" in it, it means that that node is an input node of the lhs. Then, assignSignals function is called on that node. If the Node has the strings "valid" or "ready, the corresponding Channel Signal is assigned to this Node. Else, it means the Node is a Data Signal. The naming convention in the generated BLIF files need to be read in order to determine how to parse the Nodes correctly.

2) void connectInputNodes(): Connects the input Nodes of this Subject Graph with the output Nodes of its predecessing Subject Graph

3) ChannelSignals & returnOutputNodes(): Returns the outputNodes of this module.

## ForkSubjectGraph
Represents fork_dataless and fork modules. 

### Variables
1) unsigned int size: Number of inputs of the Fork module (SIZE parameter in HDL)
2) unsigned int dataWidth: Bit width of the data signals (DATA_TYPE parameter in HDL)
3) std::vector<ChannelSignals> outputNodes: Vector of the outputs of fork.
4) ChannelSignals inputNodes: Input Nodes of the module.

### Functions:
1) ForkSubjectGraph(Operation *op): 
    1) Determines if the fork is dataless.
    2) Output Nodes have "outs" and Input Nodes have "ins" strings in them. 
    3) The generateNewName functions are used to differentiate different output channels from each other. In the hardware description, the output data bits are layed out. For example, for dataWidth = 16 and size = 3, the outs signals will be from outs_0 to outs_47. generateNewName functions transforms the names into more differentiable format, so the names are like outs_0_0 to outs_0_15, outs_1_0 to outs_1_15, and outs_2_0 to outs_2_15. With this the output nodes are easily assigned to their corresponding channels.

2) ChannelSignals & returnOutputNodes(unsigned int channelIndex): Returns the output nodes associated with the channelIndex. 

## MuxSubjectGraph
### Variables
1)  unsigned int size: Number of inputs.
2)  unsigned int dataWidth: Bit width of data signals.
3)  unsigned int selectType: Number of index inputs.

### Functions
1) MuxSubjectGraph(Operation *op): Similar to generateNewName functions in the ForkSubjectGraph, the input names are transformed into forms that allows them to be differentiated easier.


