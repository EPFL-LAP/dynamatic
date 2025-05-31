//===- BlifReader.cpp - Exp. support for MAPBUF -----------------*- C++ -*-===//
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
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "experimental/Support/BlifReader.h"
#include "gurobi_c++.h"
#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#include "experimental/Support/BlifReader.h"

using namespace dynamatic::experimental;

void Node::configureIONode(const std::string &type) {
  isChannelEdge = true;
  if (type == ".inputs")
    isInput = true;
  else if (type == ".outputs")
    isOutput = true;
}

void Node::configureConstantNode() {
  if (function == "0") {
    isConstZero = true;
  } else if (function == "1") {
    isConstOne = true;
  } else {
    llvm::errs() << "Unknown constant value: " << function << "\n";
  }
}

void Node::convertIOToChannel() {
  assert((isInput || isOutput) &&
         "The node should be an IO to convert it to a channel");
  isInput = false;
  isOutput = false;
  isChannelEdge = true;
};

void LogicNetwork::addConstantNode(const std::vector<std::string> &nodes,
                                   const std::string &function) {
  // Create the fanout node, which is the last element in the nodes vector.
  Node *fanOut = createNode(nodes.back());
  fanOut->function = function;
  fanOut->configureConstantNode();
}

void LogicNetwork::addLatch(const std::string &inputName,
                            const std::string &outputName) {
  // Create the input and output nodes for the latch. Nodes are created in
  // LogicNetwork to ensure uniqueness, and avoids circular dependencies.
  Node *regInputNode = createNode(inputName);
  Node *regOutputNode = createNode(outputName);

  // Configure the latch.
  Node::configureLatch(regInputNode, regOutputNode);

  // Add the latch to the latches vector.
  latches.emplace_back(regInputNode, regOutputNode);
}

void LogicNetwork::addIONode(const std::string &name, const std::string &type) {
  Node *node = createNode(name);
  node->configureIONode(type);
}

void LogicNetwork::addLogicGate(const std::vector<std::string> &nodes,
                                const std::string &function) {

  // Create the fanout node, which is the last element in the nodes vector.
  Node *fanOut = createNode(nodes.back());
  fanOut->function = function;

  // Create the fanin nodes. Set fanins and fanouts for each node.
  for (size_t i = 0; i < nodes.size() - 1; ++i) {
    Node *fanIn = createNode(nodes[i]);
    Node::addEdge(fanIn, fanOut);
  }
}

Node *LogicNetwork::addNode(Node *node) {
  static unsigned int counter = 0;
  if (nodes.find(node->name) !=
      nodes.end()) {           // Name conflict with another Node
    if (node->isChannelEdge) { // Channel nodes are unique, so return the node
      return nodes[node->name];
    }
    node->name = (node->name + "_" +
                  std::to_string(counter++)); // Otherwise, resolve conflict
  }
  nodes[node->name] = node; // Add the node to the map
  return node;
}

Node *LogicNetwork::createNode(const std::string &name) {
  if (nodes.find(name) != nodes.end()) {
    return nodes[name]; // Return existing node if name is already used
  }
  nodes[name] = new Node(name, this);
  return nodes[name];
}

std::set<Node *> LogicNetwork::getAllNodes() {
  std::set<Node *> allNodes;
  for (auto &pair : nodes) {
    allNodes.insert(pair.second);
  }
  return allNodes;
}

std::set<Node *> LogicNetwork::getChannels() {
  std::set<Node *> channels;
  for (const auto &pair : nodes) {
    if (pair.second->isChannelEdge) {
      channels.insert(pair.second);
    }
  }
  return channels;
}

// Returns Inputs of the Blif file.
std::set<Node *> LogicNetwork::getInputs() {
  std::set<Node *> inputs;
  for (const auto &pair : nodes) {
    if (pair.second->isInput) {
      inputs.insert(pair.second);
    }
  }
  return inputs;
}

// Returns Outputs of the Blif file.
std::set<Node *> LogicNetwork::getOutputs() {
  std::set<Node *> outputs;
  for (const auto &pair : nodes) {
    if (pair.second->isOutput) {
      outputs.insert(pair.second);
    }
  }
  return outputs;
}

std::set<Node *> LogicNetwork::getPrimaryInputs() {
  std::set<Node *> primaryInputs;
  for (const auto &pair : nodes) {
    if (pair.second->isPrimaryInput()) {
      primaryInputs.insert(pair.second);
    }
  }
  return primaryInputs;
}

std::set<Node *> LogicNetwork::getPrimaryOutputs() {
  std::set<Node *> primaryOutputs;
  for (const auto &pair : nodes) {
    if (pair.second->isPrimaryOutput()) {
      primaryOutputs.insert(pair.second);
    }
  }
  return primaryOutputs;
}

void LogicNetwork::topologicalOrderUtil(Node *node,
                                        std::set<Node *> &visitedNodes,
                                        std::vector<Node *> &order) {
  // if the node is already added to topological order, return.
  if (std::find(order.begin(), order.end(), node) != order.end()) {
    return;
  }

  // Search using DFS.
  visitedNodes.insert(node);
  for (auto *const fanin : node->fanins) {
    if (std::find(order.begin(), order.end(), fanin) == order.end() &&
        visitedNodes.count(fanin) > 0) {
      llvm::errs() << "Cyclic dependency detected!\n";
    } else if (std::find(order.begin(), order.end(), fanin) == order.end()) {
      topologicalOrderUtil(fanin, visitedNodes, order);
    }
  }

  visitedNodes.erase(node);
  order.push_back(node);
}

void LogicNetwork::generateTopologicalOrder() {
  // Create a new vector to store the topological order to ensure safety.
  std::vector<Node *> newTopologicalOrder;
  std::set<Node *> primaryInputs;
  std::set<Node *> primaryOutputs;

  for (auto &node : nodes) {
    if (node.second->isPrimaryInput()) {
      primaryInputs.insert(node.second);
    }
    if (node.second->isPrimaryOutput()) {
      primaryOutputs.insert(node.second);
    }
  }

  // Add primary inputs to the topological order as first elements.
  for (auto *input : primaryInputs) {
    newTopologicalOrder.push_back(input);
  }

  // Perform dfs on primary outputs to traverse the nodes in topological order.
  std::set<Node *> visitedNodes;
  for (const auto &node : primaryOutputs) {
    topologicalOrderUtil(node, visitedNodes, newTopologicalOrder);
  }

  nodesTopologicalOrder = std::move(newTopologicalOrder);
}

LogicNetwork *BlifParser::parseBlifFile(const std::string &filename) {
  LogicNetwork *data = new LogicNetwork();
  std::ifstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " << filename << "\n";
  }

  std::string line;
  // Loop over all the lines in .blif file.
  while (std::getline(file, line)) {
    // If comment or empty line, skip.
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    // If line ends with '\\', read the next line and append to the current
    // line.
    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
    }

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    // Model name
    if (type == ".model") {
      std::string moduleName;
      iss >> moduleName;
      data->moduleName = moduleName;
    }

    // Input/Output nodes. These are also Dataflow graph channels.
    else if ((type == ".inputs") || (type == ".outputs")) {
      std::string nodeName;
      while (iss >> nodeName) {
        data->addIONode(nodeName, type);
      }
    }

    // Latches.
    else if (type == ".latch") {
      std::string regInput, regOutput;
      iss >> regInput >> regOutput;
      data->addLatch(regInput, regOutput);
    }

    // .names stand for logic gates.
    else if (type == ".names") {
      std::vector<std::string> nodeNames;
      std::string currentNode;

      // Read node names from current line (e.g., "a b c")
      while (iss >> currentNode) {
        nodeNames.push_back(currentNode);
      }

      // Read logic function from next line (e.g., "11 1")
      std::string function;
      std::getline(file, line);
      std::istringstream functionStream(line);
      functionStream >> function;

      if (nodeNames.size() == 1) {
        data->addConstantNode(nodeNames, function);
      } else {
        data->addLogicGate(nodeNames, function);
      }
    }

    // Subcircuits. not used for now.
    else if (line.find(".subckt") == 0) {
      llvm::errs() << "Subcircuits not supported "
                   << "\n";
      continue;
    }

    // Ends the file.
    else if (line.find(".end") == 0) {
      break;
    }
  }

  // Builds topological order data structure
  data->generateTopologicalOrder();
  return data;
}

std::vector<Node *> LogicNetwork::findPath(Node *start, Node *end) {
  // BFS search to find the shortest path from start to end.
  std::queue<Node *> queue;
  std::unordered_map<Node *, Node *> parent;
  std::set<Node *> visited;

  queue.push(start);
  visited.insert(start);

  while (!queue.empty()) {
    Node *current = queue.front();
    queue.pop();

    if (current == end) {
      // Reconstruct the path
      std::vector<Node *> path;
      while (current != start) {
        path.push_back(current);
        current = parent[current];
      }
      path.push_back(start);
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (const auto &fanout : current->fanouts) {
      if (visited.count(fanout) == 0) {
        queue.push(fanout);
        visited.insert(fanout);
        parent[fanout] = current;
      }
    }
  }

  // If no path is found, return an empty vector
  return {};
}

void BlifWriter::writeToFile(LogicNetwork &network,
                             const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " << filename << "\n";
    return;
  }

  file << ".model " << network.moduleName << "\n";

  file << ".inputs";
  for (const auto &input : network.getInputs()) {
    file << " " << input->name;
  }
  file << "\n";

  file << ".outputs";
  for (const auto &output : network.getOutputs()) {
    file << " " << output->name;
  }
  file << "\n";

  for (const auto &latch : network.getLatches()) {
    file << ".latch " << latch.first->name << " " << latch.second->name << "\n";
  }

  for (const auto &node : network.getNodesInTopologicalOrder()) {
    if (node->isConstZero || node->isConstOne) {
      file << ".names " << node->name << "\n";
      file << (node->isConstZero ? "0" : "1") << "\n";
    } else if (node->fanins.size() == 1) {
      file << ".names " << (*node->fanins.begin())->name << " " << node->name
           << "\n";
      file << node->function << "\n";
    } else if (node->fanins.size() == 2) {
      auto fanins = node->fanins;
      auto it = fanins.begin();
      auto name1 = (*it)->name;
      ++it;
      auto name2 = (*it)->name;
      file << ".names " << name1 << " " << name2 << " " << node->name << "\n";
      file << node->function << "\n";
    }
  }

  file << ".end\n";
  file.close();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
