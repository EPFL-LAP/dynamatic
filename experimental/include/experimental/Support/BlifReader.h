//===- BlifReader.h - Exp. support for MAPBUF buffer placement -------*- C++
//-*-===//
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

class BlifData;
class Node;

struct MILPVarsSubjectGraph {
  GRBVar tIn;
  GRBVar tOut;
  std::optional<GRBVar> bufferVar;
};

// Node class to encapsulate node properties
class Node {
private:
  std::string name;
  // BlifData *parent;
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
  Node(const std::string &name)
      : name(name), gurobiVars(new MILPVarsSubjectGraph()) {}
  Node(const std::string &name, BlifData *parent)
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

  bool operator==(const Node &other) const { return name == other.name; }
  bool operator==(const Node *other) const { return name == other->name; }
  bool operator!=(const Node &other) const { return name != other.name; }
  bool operator<(const Node &other) const { return name < other.name; }
  bool operator<(const Node *other) const { return name < other->name; }
  bool operator>(const Node &other) const { return name > other.name; }
  std::string str() const { return name; }

  friend class BlifData;
};

struct PointerCompare {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->getName() < rhs->getName();
  }
};

class BlifData {
private:
  std::string moduleName;
  std::unordered_map<std::string, Node *> nodes;
  std::vector<Node *> nodesTopologicalOrder;
  std::unordered_map<Node *, Node *, boost::hash<Node *>> latches;
  std::unordered_map<std::string, std::set<std::string>> submodules;

public:
  BlifData() = default;

  void addLatch(Node *input, Node *output) { latches[input] = output; }

  Node *getNodeByName(const std::string &name) {
    auto it = nodes.find(name);
    if (it != nodes.end()) {
      return it->second;
    }
    return nullptr;
  }

  Node *createNode(const std::string &name) {
    if (nodes.find(name) != nodes.end()) {
      return nodes[name]; // Return existing node if name is already used
    }
    nodes[name] = new Node(name, this);
    return nodes[name];
  }

  void renameNode(const std::string &oldName, const std::string &newName) {
    auto it = nodes.find(oldName);
    if (it != nodes.end()) {
      auto *node = it->second;
      nodes.erase(it);
      node->name = newName;
      nodes[newName] = node;
    }
  }

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

  std::set<Node *> getPrimaryInputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isPrimaryInput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  std::set<Node *> getPrimaryOutputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isPrimaryOutput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  std::set<Node *> getChannels() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isChannelEdgeNode()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  std::vector<Node *> getNodesInOrder() { return nodesTopologicalOrder; }

  std::set<Node *> getInputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isInput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  std::set<Node *> getOutputs() {
    std::set<Node *> result;
    for (const auto &pair : nodes) {
      if (pair.second->isOutput()) {
        result.insert(pair.second);
      }
    }
    return result;
  }

  void setModuleName(const std::string &moduleName) {
    this->moduleName = moduleName;
  }

  void traverseUtil(Node *node, std::set<Node *> &visitedNodes);
  void traverseNodes();
  void printModuleInfo();
  void generateBlifFile(const std::string &filename);

  std::set<Node *> getAllNodes() {
    std::set<Node *> result;
    for (auto &pair : nodes) {
      result.insert(pair.second);
    }
    return result;
  }

  std::unordered_map<Node *, Node *, boost::hash<Node *>> getLatches() const {
    return latches;
  }

  std::vector<Node *> findPath(Node *start, Node *end);

  std::set<Node *> findNodesWithLimitedWavyInputs(size_t limit,
                                                  std::set<Node *> &wavyLine);

  std::set<Node *> findWavyInputsOfNode(Node *node, std::set<Node *> &wavyLine);
};

class BlifParser {
public:
  BlifParser() = default;
  experimental::BlifData *parseBlifFile(const std::string &filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_BLIF_READER_H