# BlifReader

This file provides support for parsing and emitting BLIF (Berkeley Logic Interchange Format) files, enabling their conversion to and from a `LogicNetwork` data structure. It allows importing circuits into the Dynamatic framework, analyzing them, and exporting them back.

The core responsibilities include:

- Parsing .BLIF files into a LogicNetwork

- Computing and obtaining the topological order 

- Writing a LogicNetwork back to .BLIF format

## Implementation Overview

The core data structure of this code is [`LogicNetwork`](https://github.com/EPFL-LAP/dynamatic/blob/main/experimental/include/experimental/Support/BlifReader.h#L175). This class contains the logic network represented inside a BLIF file.

The pseudo-function for parsing a BLIF file ([`parseBlifFile`](https://github.com/EPFL-LAP/dynamatic/blob/main/experimental/lib/Support/BlifReader.cpp#L228)) is the following:

```
LogicNetwork *BlifParser::parseBlifFile(filename) {

  LogicNetwork data;
  string line;

  while(open(filename, line)){
    str type = line.split(0);
    switch(type){
      ".input" or ".output":
        data->addIONode(type, line);

      ".latch":
        data->addLatch(type, line);

      ".names":
        data->addLogicGate(type, line);

      ".end":
        break;
    } 
  }

  data->generateTopologicalOrder();
  return data;
}
```

This function iterates over the lines of the BLIF file and it adds to the logic network the different nodes. The node type added depends on the `type` variable. This variable exclusively depends on the first word of the `line` variable. This follows the expected structure of BLIF files.

After filling in the logic network, the function `generateTopologicalOrder` saves the topological order of the network in the vector [`nodesTopologicalOrder`](https://github.com/EPFL-LAP/dynamatic/blob/main/experimental/include/experimental/Support/BlifReader.h#L179).

The pseudo-function for exporting a logic network in a BLIF file ([`writeToFile`](https://github.com/EPFL-LAP/dynamatic/blob/main/experimental/lib/Support/BlifReader.cpp#L357)) is the following one:


```
void BlifWriter::writeToFile(LogicNetwork network, string filename) {

  FILE file = open(filename);

  file.write(".inputs");
  for(i : network.getInputs()){
    file.write(i);
  }

  file.write(".outputs");
  for(i : network.getOutputs()){
    file.write(i);
  }
  
  file.write(".latch");
  for(i : network.getLatches()){
    file.write(i);
  }

  for(node : network.getNodesInTopologicalOrder()){
    file.write(node);
  }

  file.close();
}

```

This function iterates over the different parts of a network and it writes them in the output file. 

## Key Classes

There are two main classes:

- `LogicNetwork`: it represents the logic network expressed in a BLIF file
- `Node`: it represents a node in the logic network

## Key Variables

### LogicNetwork

- `std::vector<std::pair<Node *, Node *>> latches` is a vector containing pairs of the input and output nodes of a latch (register).
- `std::unordered_map<std::string, Node *> nodes` is a map where the keys are the names of the nodes and the values are objects of the `Node` class. This map contains all the nodes in the logic network.
- `std::vector<Node *> nodesTopologicalOrder` is a vector of objects of the `Node` class placed in topological order.

### Node

- `MILPVarsSubjectGraph *gurobiVars` is a struct containing the Gurobi variables that will be used in the Buffer Placement pass.
- `std::set<Node *> fanins` is a set containing objects of the `Node` class representing the fanins of the node.
- `std::set<Node *> fanouts`:is a set containing objects of the `Node` class representing the fanouts of the node.  
- `std::string function` is a string containing the function of the node.
- `std::string name` is a string representing the name of the node.


## Key Functions

### LogicNetwork Class

#### Node Creation and Addition

- `void addIONode(const std::string &name, const std::string &type)`:
adds input/output nodes to the circuit where type specifies input or output.

- `void addLatch(const std::string &inputName, const std::string &outputName)`
adds latch nodes to the circuit by specifying input and output node.

- `void addConstantNode(const std::vector<std::string> &nodes, const std::string &function)`
adds constant nodes to the circuit with function specified in the string.

- `void addLogicGate(const std::vector<std::string> &nodes, const std::string &function)`
adds a logic gate to the circuit with function specified in the string.

- `Node *addNode(Node *node)`
adds a node to the circuit with conflict resolution (renaming if needed).

- `Node *createNode(const std::string &name)`
creates a node by name.

#### Querying the Circuit

- `std::set<Node *> getAllNodes()`
returns all nodes in the circuit.

- `std::set<Node *> getChannels()`
returns nodes corresponding to dataflow graph channel edges.

- `std::vector<std::pair<Node *, Node *>> getLatches() const`
returns the list of latches.

- `std::set<Node *> getPrimaryInputs()`
returns all primary input nodes.

- `std::set<Node *> getPrimaryOutputs()`
returns all primary output nodes.

- `std::vector<Node *> getNodesInTopologicalOrder()`
returns nodes in topological order (precomputed).

- `std::set<Node *> getInputs()`
returns declared inputs of the BLIF file.

- `std::set<Node *> getOutputs()`
returns declared outputs of the BLIF file.

#### Graph Analysis

- `std::vector<Node *> findPath(Node *start, Node *end)`
    finds a path from start to end using BFS.

### Node Class

- `void addFanin(Node *node)` adds a new fanin.
- `void addFanout(Node *node)` adds a new fanout.
- `static void addEdge(Node *fanin, Node *fanout)` adds an edge between fanin and fanout.
- `static void configureLatch(Node *regInputNode, Node *regOutputNode)` configures the node as a latch based on the input and output nodes.
- `void replaceFanin(Node *oldFanin, Node *newFanin)` replaces an existing fanin with a new one.
- `static void connectNodes(Node *currentNode, Node *previousNode)` connects two nodes by setting the pointer of current node to the previous node. 
- `void configureIONode(const std::string &type)` configures the node based on the type of I/O node.
- `void configureConstantNode()` configures the node as a constant node based on the function
- `bool isPrimaryInput()` returns if the node is a primary input
- `bool isPrimaryOutput()` returns if the node is a primary output
- `void convertIOToChannel()` used to merge I/O nodes. I/O is set false and isChannelEdge is set to true so that the node can be considered as a dataflow graph edge.
