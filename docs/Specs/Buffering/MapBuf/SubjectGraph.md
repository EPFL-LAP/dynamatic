# Subject Graphs
Subject graphs are directed acyclic graphs composed of abstract logic operations (not actual gates). They serve as technology-independent representations of circuit logic, with common types including AND-Inverter Graphs (AIGs) and XOR-AND-Inverter Graphs (XAGs). In the implementation of MapBuf, we use AIGs for subject graphs.

While the Handshake dialect in Dynamatic is used to model the Dataflow circuits with Operations corresponding to Dataflow units, it falls short of providing the AIG structure required by MapBuf. Existing buffer placement algorithms (FPGA20, FPL22) use Dataflow graph channels (represented as Values in MLIR) as timing variables in the MILP formulation. However, the representation provided by the Handshake dialect is insufficient for MapBuf's MILP formulation, which requires AND-Inverter Graph (AIG) edges as timing variables to accurately model LUT-level timing constraints.

This creates a gap within Dynamatic, the high-level Handshake dialect cannot provide the low-level AIG representation needed for MapBuf. The Subject Graph class implementation fills this gap. While it is not a formal MLIR dialect, it functions conceptually as an AIG Dialect within Dynamatic. The Subject Graph implementation:

1) It parses the AIG implementation of each dataflow unit in the dataflow circuit. 
2) Constructs the complete AIG of the entire dataflow circuit by connecting the AIG of each unit.
3) Provides bidirectional mapping between dataflow units and the nodes in the AIG through a static moduleMap, enabling efficient lookups in both directions.
4) Enables buffer insertion at specific points in the dataflow circuit.


# Implementation Overview

The code base data structure is **BaseSubjectGraph** which contains the AIG of each dataflow unit separately.

The core data structure that contains the list of the subject graph of all dataflow units is `subjectGraphVector` which is filled in the `BaseSubjectGraph` object generator.

The function that generates the Subject Graphs of dataflow units is `SubjectGraphGenerator`. The following is its pseudo-code:

```
DataflowCircuit DC;
std::vector<BaseSubjectGraph *> subjectGraphs;
for ( DataFlow unit: DC.get_dataflow_units() ){

  BaseSubjectGraph * unit_sg = BaseSubjectGraph(unit);
  subjectGraphs.append( unit_sg );

}

for ( BaseSubjectGraph * module: subjectGraphs){
  module->buildSubjectGraphConnections();
}

```

For each dataflow unit in the dataflow circuit, the SubjectGraphGenerator creates the corresponding derived BaseSubjectGraph object. Then, for each one of these, it calls the corresponding buildSubjectGraphConnections function, which establishes the input/output relations between Subject Graphs.

At this stage, Nodes of the neighbouring Subject Graphs are not connected. The connection is built by the function connectSubjectGraphs(). The following is its pseudo-code:

```
for ( BaseSubjectGraph * module: subjectGraphs){
  module->connectInputNodes();
}

LogicNetwork* mergedBlif = new LogicNetwork();

for ( BaseSubjectGraph * module: subjectGraphs){
  mergedBlif->addNodes(module->getNodes());
}

return mergedBlif;

```

The process of constructing a unified circuit graph begins with invoking the connectInputNodes() function for each SubjectGraph. This function establishes connections between adjacent graphs by merging their input and output nodes.

Next, a new LogicNetwork object—referred to as mergedBlif—is instantiated to serve as the container for the complete circuit. All nodes from the individual SubjectGraphs are then added to this new LogicNetwork. Because each node already encapsulates its connection information, simply aggregating them into a single network is sufficient to produce a fully connected representation of the circuit.

Separating the connection logic from the creation of the individual SubjectGraphs offers greater modularity and flexibility. This design makes it easy to insert or remove SubjectGraphs before finalizing the overall network, enabling more dynamic and maintainable circuit assembly.

# BaseSubjectGraph Class
The BaseSubjectGraph class is an abstract base class that provides shared functionality for generating the subject graph of a dataflow unit. Each major type of dataflow unit has its own subclass that extends BaseSubjectGraph. These subclasses implement their own constructors and are responsible for parsing the corresponding BLIF (Berkeley Logic Interchange Format) file to construct the unit's subject graph.

The following pseudocode illustrates the subject graph generation process within the dataflow unit class generator:

```
dataBitwidth = unit->getDataBitwidth();
loadBlifFile(dataBitwidth);

processOutOfRuleNodes();
NodeProcessingRule rules = ... // generated seprately for each dataflow unit type
processNodesWithRules(rules);
```

The process begins by retrieving the data bitwidth of the unit, which is used to select and load the appropriate BLIF file via the `loadBlifFile` functionThis file provides the AIG   representation for the specific unit at that bitwidth.

After parsing the BLIF, two functions are used to interpret and process the AIG nodes:
- `processOutOfRuleNodes`: A subclass-specific function that performs custom processing of AIG nodes, typically identifying matches between primary inputs (PIs) and primary outputs (POs) and the corresponding ports of the dataflow unit.
- `processNodesWithRules`: A generic function shared across all subclasses, which matches the PIs and POs of the AIG with the corresponding ports of the dataflow units applying the rules describes by `NodeProcessingRule` structure. 

An example of a NodeProcessingRule is `{"lhs", lhsNodes, false, nullptr}`. This rule instructs the system to collect AIG PIs or POs whose names contain the substring `"lhs"` into the set `lhsNodes`, without renaming them (`false` flag) and without applying additional processing (`nullptr` argument).


Another key step is handled by the `buildSubjectGraphConnections` function. It iterates over the dataflow unit's input and output ports and stores their corresponding subject graphs in two vectors—one for inputs and one for outputs.


Finally, the `connectInputNodes` function connects the different subject graphs together using the previously collected node information and the input/output subject graph vectors. This step completes the construction of the full subject graph.


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

# Derived BaseSubjectGraph Classes
As mentioned in the BaseSubjectGraph Class section, each different dataflow unit has its own derived SubjectGraph class. In this section, we mention in detail some of them.

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


