# Technology Mapping

This file provides support for technology mapping algorithm used in MapBuf that generates K-feasible cuts to map Subject Graph nodes to K-input LUTs.

## Implementation Overview
The core data structure of this code is [`Cut`](https://github.com/EPFL-LAP/dynamatic/blob/aafb8cab5705f0cd7f6e3b0660b4deeca776153c/experimental/include/experimental/Support/CutlessMapping.h#L35). This class represents a single cut of a node, containing the root node, leaf nodes, depth of the cut, and a Cut Selection Variable used in MILP formulation.

## Cutless FPGA Mapping
The technology mapping algorithm is implemented in the function [cutAlgorithm()](https://github.com/EPFL-LAP/dynamatic/blob/aafb8cab5705f0cd7f6e3b0660b4deeca776153c/experimental/lib/Support/CutlessMapping.cpp#L122). This algorithm is based on the paper [Cutless FPGA Mapping](https://people.eecs.berkeley.edu/~alanmi/publications/2007/tech07_fast.pdf). 

The algorithm uses a depth-oriented mapping strategy where nodes are grouped into "wavy lines" by depth. By definition, nodes in the n-th wavy line can be implemented using K or fewer nodes from any previous wavy line (0 to n-1). The 0th wavy line consists of Primary Inputs of the Subject Graph. The algorithm iterates over all AIG nodes continuously, until all nodes are mapped to a wavy line. 

For 6-input LUTs, exhaustive cut enumeration produces hundreds of cuts per node, which prevents MILP solver from finding a solution within a reasonable time. Therefore, we limit the enumeration to 3 cuts per node, which satisfies the requirements of our buffer placement algorithm:

1) Trivial cut: The cut that consists only of fanins of the node. 
2) Deepest cut: The cut that minimizes the number of logic levels.
3) Channel aware cut: Explained in next section

### Channel Aware Cut Generation
The Cut Selection Conflict Constraint in MapBuf enforces that a cut cannot be selected if it covers a channel edge where a buffer has been placed. If MapBuf only finds deepest cuts of the nodes, that would mean all channels are covered by cuts, preventing the MILP from placing buffers on the channels. This inability to place buffers would violate timing constraints, resulting in an infeasible problem. Therefore, for each node, MapBuf must find at least one cut that does not cover any channel. These cuts are not the deepest possible, but they enable MapBuf to place buffers on channels to satisfy timing constraints.

To generate these channel-aware cuts, we run the cut generation algorithm a second time with a key modification: Channel nodes are included as Primary Inputs of the Subject Graph. This way, Channel nodes are added to the 0th wavy line, enabling the production of cuts that terminate at channel boundaries rather than crossing them.
