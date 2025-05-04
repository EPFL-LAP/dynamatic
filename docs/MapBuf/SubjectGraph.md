# Subject Graphs
Subject graphs are directed acyclic graphs composed of abstract logic operations (not actual gates). They serve as technology-independent representations of circuit logic, with common types including AND-Inverter Graphs (AIGs) and XOR-AND-Inverter Graphs (XAGs). In the implementation of MapBuf, we use AIGs for subject graphs.

While the Handshake dialect in Dynamatic is used to model the Dataflow circuits with Operations corresponding to Dataflow units, it falls short of providing the AIG structure required by MapBuf. Existing buffer placement algorithms (FPGA20, FPL22) use Dataflow graph channels (represented as Values in MLIR) as timing variables in the MILP formulation. However, the representation provided by the Handshake dialect is insufficient for MapBuf's MILP formulation, which requires AND-Inverter Graph (AIG) edges as timing variables to accurately model LUT-level timing constraints.

This creates a gap within Dynamatic, the high-level Handshake dialect cannot provide the low-level AIG representation needed for MapBuf. The Subject Graph class implementation fills this gap. While it is not a formal MLIR dialect, it functions conceptually as an AIG Dialect within Dynamatic. The Subject Graph implementation:

1) Creates AIG representations of Dataflow units by integrating BLIF files of the corresponding hardware units 
2) Constructs the complete Subject Graph of the entire circuit by connecting these individual unit representations
3) Provides bidirectional mapping between MLIR Operations and Subject Graphs through a static moduleMap, enabling efficient lookups in both directions
4) Enables buffer insertion at specific points in the circuit

