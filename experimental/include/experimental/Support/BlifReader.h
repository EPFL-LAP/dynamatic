//===- BlifReader.h - Exp. support for MAPBUF -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the functionality for reading a BLIF file into data
// structures.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BLIF_READER_H
#define EXPERIMENTAL_SUPPORT_BLIF_READER_H

#include "dynamatic/Support/LLVM.h"
#include "gurobi_c++.h"
#include <boost/functional/hash/extensions.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class LogicNetwork;

struct MILPVarsSubjectGraph {
  GRBVar tIn;
  GRBVar tOut;
  std::optional<GRBVar> bufferVar;
};

/// Represents a node in an And-Inverter Graph (AIG) circuit representation.
///
/// Each node can represent different circuit elements including:
/// - Primary inputs and outputs
/// - Latch (register) inputs and outputs
/// - Constant values (0 or 1)
/// - Blackbox output nodes
/// - Channel edge nodes
///
/// The node maintains its connectivity through fanin and fanout relationships
/// with other nodes, stored as pointer sets.
/// Node identity and ordering is determined by the unique name attribute.
///
/// The class integrates with MILP through the gurobiVars member for buffer
/// placement.
class Node {
private:
  std::string name;
  bool isChannelEdge = false;
  bool isBlackboxOutput = false;
  bool isInputBool = false;
  bool isOutputBool = false;
  bool isLatchInputBool = false;
  bool isLatchOutputBool = false;
  bool isConstZeroBool = false;
  bool isConstOneBool = false;
  std::string function;
  std::set<Node *> fanins = {};
  std::set<Node *> fanouts = {};

public:
  MILPVarsSubjectGraph *gurobiVars;

  Node() = default;
  Node(const std::string &name, LogicNetwork *parent)
      : name(name), gurobiVars(new MILPVarsSubjectGraph()) {}

  Node &operator=(const Node &other) {
    if (this == &other) {
      // Handle self-assignment
      return *this;
    }

    // Copy the primitive types and strings
    name = other.name;
    isChannelEdge = other.isChannelEdge;
    isBlackboxOutput = other.isBlackboxOutput;
    isInputBool = other.isInputBool;
    isOutputBool = other.isOutputBool;
    isLatchInputBool = other.isLatchInputBool;
    isLatchOutputBool = other.isLatchOutputBool;
    isConstZeroBool = other.isConstZeroBool;
    isConstOneBool = other.isConstOneBool;
    function = other.function;
    gurobiVars = other.gurobiVars;

    // Clear and copy the fanins and fanouts sets (deep copy of pointers)
    fanins.clear();
    for (auto *nodePtr : other.fanins) {
      fanins.insert(nodePtr); // Copy pointers, not the objects themselves
    }

    fanouts.clear();
    for (auto *nodePtr : other.fanouts) {
      fanouts.insert(nodePtr); // Copy pointers, not the objects themselves
    }

    // Return *this to allow chain assignment
    return *this;
  }

  ~Node() {
    delete gurobiVars;
    fanins.clear();
    fanouts.clear();
  }

  const std::string &getName() const { return name; }
  void setName(const std::string &newName);
  void setChannelEdge(bool value) { isChannelEdge = value; }
  void setBlackboxOutput(bool value) { isBlackboxOutput = value; }
  void setInput(bool value) { isInputBool = value; }
  void setOutput(bool value) { isOutputBool = value; }
  void setLatchInput(bool value) { isLatchInputBool = value; }
  void setLatchOutput(bool value) { isLatchOutputBool = value; }
  void setConstZero(bool value) { isConstZeroBool = value; }
  void setConstOne(bool value) { isConstOneBool = value; }
  void setFunction(const std::string &func) { function = func; }

  bool isInput() const { return isInputBool; }
  bool isOutput() const { return isOutputBool; }
  bool isChannelEdgeNode() const { return isChannelEdge; }
  bool isBlackboxOutputNode() const { return isBlackboxOutput; }
  bool isPrimaryInput() const {
    return (isConstOneBool || isConstZeroBool || isInputBool ||
            isLatchOutputBool || isBlackboxOutput);
  }
  bool isPrimaryOutput() const { return (isOutputBool || isLatchInputBool); }
  bool isLatchInput() const { return isLatchInputBool; }
  bool isLatchOutput() const { return isLatchOutputBool; }
  bool isConstZero() const { return isConstZeroBool; }
  bool isConstOne() const { return isConstOneBool; }
  const std::string &getFunction() const { return function; }

  std::set<Node *> &getFanins() { return fanins; }
  std::set<Node *> &getFanouts() { return fanouts; }

  void addFanin(Node *node) { fanins.insert(node); }
  void addFanout(Node *node) { fanouts.insert(node); }
  void addFanin(std::set<Node *> &nodes) {
    fanins.insert(nodes.begin(), nodes.end());
  }
  void addFanout(std::set<Node *> &nodes) {
    fanouts.insert(nodes.begin(), nodes.end());
  }

  std::string str() const { return name; }

  friend class LogicNetwork;
};

/// Represents BLIF (Berkeley Logic Interchange Format) circuit data structure.
///
/// Manages a collection of interconnected nodes representing a digital
/// circuit, maintaining relationships between nodes including latches,
/// submodules, and topological ordering. The class provides functionality for:
/// - Node management (creation, modification, lookup)
/// - Circuit topology analysis (path finding, traversal)
/// - Circuit element categorization (inputs, outputs, channels)
/// - BLIF file generation
///
/// Node uniqueness is enforced through automatic name conflict resolution.
/// Memory ownership of nodes is maintained by this class through the nodes map.
class LogicNetwork {
private:
  std::vector<std::pair<Node *, Node *>> latches;
  std::string moduleName;
  std::unordered_map<std::string, Node *> nodes;
  std::vector<Node *> nodesTopologicalOrder;
  std::unordered_map<std::string, std::set<std::string>> submodules;

public:
  LogicNetwork() = default;

  /// Adds a latch to the circuit data structure.
  void addLatch(Node *input, Node *output) {
    latches.emplace_back(input, output);
  }

  /// Adds a node to the circuit data structure, resolving name conflicts. If a
  /// node with the same name already exists, counter value is appended to the
  /// name.
  Node *addNode(Node *node) {
    static unsigned int counter = 0;
    if (nodes.find(node->getName()) != nodes.end()) {
      if (node->isChannelEdgeNode()) {
        return nodes[node->getName()];
      }
      node->setName(node->getName() + "_" + std::to_string(counter++));
    }
    nodes[node->getName()] = node;
    return node;
  }

  /// Creates a new Node and returns it. If a Node with the same name already
  /// exists, the existing Node is returned.
  Node *createNode(const std::string &name) {
    if (nodes.find(name) != nodes.end()) {
      return nodes[name]; // Return existing node if name is already used
    }
    nodes[name] = new Node(name, this);
    return nodes[name];
  }

  // Finds the path from "start" to "end" using bfs.
  std::vector<Node *> findPath(Node *start, Node *end);

  // Implements the "Cutless FPGA Mapping" algorithm.Returns the nodes in the
  // circuit that can be implemented with "limit" number of nodes from the set
  // "wavyLine". For example, if the limit is 6 (6-input LUT), returns all the
  // nodes that can be implemented with 6 Nodes from wavyLine set.
  std::set<Node *> findNodesWithLimitedWavyInputs(size_t limit,
                                                  std::set<Node *> &wavyLine);

  // Helper function for findNodesWithLimitedWavyInputs. Finds the wavy inputs
  // using dfs.
  std::set<Node *> findWavyInputsOfNode(Node *node, std::set<Node *> &wavyLine);

  // Returns all of the Aig Nodes.
  std::set<Node *> getAllNodes() {
    std::set<Node *> result;
    for (auto &pair : nodes) {
      result.insert(pair.second);
    }
    return result;
  }

  // Returns latch map.
  std::vector<std::pair<Node *, Node *>> getLatches() const { return latches; }

  // Returns the Primary Inputs by looping over all the AigNodes and checking if
  // they are Primary Inputs by calling isPrimaryInput member function from
  // Node class.
  std::set<Node *> getPrimaryInputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isPrimaryInput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  // Returns the Primary Outputs by looping over all the AigNodes and checking
  // if they are Primary Outputs by calling isPrimaryOutput member function from
  // Node class.
  std::set<Node *> getPrimaryOutputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isPrimaryOutput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  // Returns the Nodes that correspond to Dataflow Graph Channels Edges.
  std::set<Node *> getChannels() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isChannelEdgeNode()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  // Returns the Module Name
  std::string getModuleName() { return moduleName; };

  // Returns the AigNodes in topological order. Nodes were sorted in topological
  // order when LogicNetwork class is instantiated.
  std::vector<Node *> getNodesInOrder() { return nodesTopologicalOrder; }

  // Returns Inputs of the Blif file.
  std::set<Node *> getInputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isInput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  // Returns Outputs of the Blif file.
  std::set<Node *> getOutputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isOutput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  // Sets the Module Name of the Blif file.
  void setModuleName(const std::string &moduleName) {
    this->moduleName = moduleName;
  }

  // Helper function for traverseNodes. Sorts the Nodes using dfs.
  void traverseUtil(Node *node, std::set<Node *> &visitedNodes);

  // Traverses the nodes, sorting them into topological order from
  // Primary Inputs to Primary Outputs.
  void traverseNodes();
};

// Generates a file in .blif format.
class BlifWriter {
public:
  BlifWriter() = default;
  void writeToFile(LogicNetwork &network, const std::string &filename);
};

/// Parses Berkeley Logic Interchange Format (BLIF) files into LogicNetwork
/// class.
class BlifParser {
public:
  BlifParser() = default;
  experimental::LogicNetwork *parseBlifFile(const std::string &filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BLIF_READER_H