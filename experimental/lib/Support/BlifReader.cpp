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

#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#include "experimental/Support/BlifReader.h"

using namespace dynamatic::experimental;

void Node::setName(const std::string &newName) { this->name = newName; }

void LogicNetwork::topologicalOrderUtil(Node *node,
                                        std::set<Node *> &visitedNodes,
                                        std::vector<Node *> &order) {
  // if the node is already added to topological order, return.
  if (std::find(order.begin(), order.end(), node) != order.end()) {
    return;
  }

  // Search using DFS.
  visitedNodes.insert(node);
  for (auto *const fanin : node->getFanins()) {
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
      data->setModuleName(moduleName);
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

      data->addLogicGate(nodeNames, function);
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
  std::unordered_map<Node *, Node *, boost::hash<Node *>> parent;
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

    for (const auto &fanout : current->getFanouts()) {
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

std::set<Node *>
LogicNetwork::findNodesWithLimitedWavyInputs(size_t limit,
                                             std::set<Node *> &wavyLine) {
  std::set<Node *> nodesWithLimitedWavyInputs;

  for (auto &node : nodesTopologicalOrder) {
    bool erased = false;
    // Erase a channel node from the wavyLine temporarily, so the search does
    // not end prematurely.
    if (node->isChannelEdge) {
      if (wavyLine.count(node) > 0) {
        wavyLine.erase(node);
        erased = true;
      }
    }
    std::set<Node *> wavyInputs = findWavyInputsOfNode(node, wavyLine);
    // if the number of wavy inputs is less than or equal to the limit (less
    // than the LUT size), add to the set
    if (wavyInputs.size() <= limit) {
      nodesWithLimitedWavyInputs.insert(node);
    }

    if (erased) {
      wavyLine.insert(node);
    }
  }
  return nodesWithLimitedWavyInputs;
}

std::set<Node *>
LogicNetwork::findWavyInputsOfNode(Node *node, std::set<Node *> &wavyLine) {
  std::set<Node *> wavyInputs;
  std::set<Node *> visited;

  // DFS to find the wavy inputs of the node.
  std::function<void(Node *)> dfs = [&](Node *currentNode) {
    if (visited.count(currentNode) > 0) {
      return;
    }
    visited.insert(currentNode);

    if (wavyLine.count(currentNode) > 0) {
      wavyInputs.insert(currentNode);
      return;
    }

    for (const auto &fanin : currentNode->getFanins()) {
      dfs(fanin);
    }
  };

  dfs(node);
  return wavyInputs;
}

void BlifWriter::writeToFile(LogicNetwork &network,
                             const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " << filename << "\n";
    return;
  }

  file << ".model " << network.getModuleName() << "\n";

  file << ".inputs";
  for (const auto &input : network.getInputs()) {
    file << " " << input->getName();
  }
  file << "\n";

  file << ".outputs";
  for (const auto &output : network.getOutputs()) {
    file << " " << output->getName();
  }
  file << "\n";

  for (const auto &latch : network.getLatches()) {
    file << ".latch " << latch.first->getName() << " "
         << latch.second->getName() << "\n";
  }

  for (const auto &node : network.getNodesInOrder()) {
    if (node->isConstZero() || node->isConstOne()) {
      file << ".names " << node->getName() << "\n";
      file << (node->isConstZero() ? "0" : "1") << "\n";
    } else if (node->getFanins().size() == 1) {
      file << ".names " << (*node->getFanins().begin())->getName() << " "
           << node->getName() << "\n";
      file << node->getFunction() << "\n";
    } else if (node->getFanins().size() == 2) {
      auto fanins = node->getFanins();
      auto it = fanins.begin();
      auto name1 = (*it)->getName();
      ++it;
      auto name2 = (*it)->getName();
      file << ".names " << name1 << " " << name2 << " " << node->getName()
           << "\n";
      file << node->getFunction() << "\n";
    }
  }

  file << ".end\n";
  file.close();
}
