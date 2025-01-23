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
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

void Node::setName(const std::string &newName) { this->name = newName; }

void LogicNetwork::traverseUtil(Node *node, std::set<Node *> &visitedNodes) {
  // if the node is already added to topological order, return.
  if (std::find(nodesTopologicalOrder.begin(), nodesTopologicalOrder.end(),
                node) != nodesTopologicalOrder.end()) {
    return;
  }

  // Search using DFS.
  visitedNodes.insert(node);
  for (auto *const fanin : node->getFanins()) {
    if (std::find(nodesTopologicalOrder.begin(), nodesTopologicalOrder.end(),
                  fanin) == nodesTopologicalOrder.end() &&
        visitedNodes.count(fanin) > 0) {
      llvm::errs() << "Cyclic dependency detected!\n";
    } else if (std::find(nodesTopologicalOrder.begin(),
                         nodesTopologicalOrder.end(),
                         fanin) == nodesTopologicalOrder.end()) {
      traverseUtil(fanin, visitedNodes);
    }
  }

  visitedNodes.erase(node);
  nodesTopologicalOrder.push_back(node);
}

void LogicNetwork::traverseNodes() {
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
    nodesTopologicalOrder.push_back(input);
  }

  // Perform dfs on primary outputs to traverse the nodes in topological order.
  std::set<Node *> visitedNodes;
  for (const auto &node : primaryOutputs) {
    traverseUtil(node, visitedNodes);
  }
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
        Node *node = data->createNode(nodeName);
        node->setChannelEdge(true);

        if (type == ".inputs")
          node->setInput(true);
        else if (type == ".outputs")
          node->setOutput(true);
      }
    }

    // Latches.
    else if (type == ".latch") {
      std::string regInput;
      std::string regOutput;
      iss >> regInput;
      iss >> regOutput;

      Node *regInputNode = data->createNode(regInput);
      regInputNode->setLatchInput(true);

      Node *regOutputNode = data->createNode(regOutput);
      regOutputNode->setLatchOutput(true);

      data->addLatch(regInputNode, regOutputNode);
    }

    // .names stand for logic gates.
    else if (type == ".names") {
      std::string function;
      std::vector<std::string> currNodes;
      std::string node;

      while (iss >> node) { // Read the nodes. Example: a b c
        currNodes.push_back(node);
      }

      std::getline(file, line); // Read the function. Function is in the next
                                // line. Example: 11 1
      std::istringstream iss2(line);
      iss2 >> function;

      if (currNodes.empty()) {
        llvm::errs() << "No nodes found in .names\n";
      }

      Node *fanOut = nullptr;
      // If there is only one node, it is a constant node.
      if (currNodes.size() == 1) {
        fanOut = data->createNode(currNodes[0]);
        if (function == "0") {
          fanOut->setConstZero(true);
        } else if (function == "1") {
          fanOut->setConstOne(true);
        } else {
          llvm::errs() << "Unknown constant value: " << function << "\n";
        }
        // If there are two nodes, it is a wire.
      } else if (currNodes.size() == 2) {
        Node *fanin = data->createNode(currNodes.front());
        fanOut = data->createNode(currNodes.back());
        fanOut->addFanin(fanin);
        fanin->addFanout(fanOut);
        // If there are three nodes, it is a logic gate. First the nodes are
        // fanins, and the last node is fanout.
      } else if (currNodes.size() == 3) {
        Node *fanin1 = data->createNode(currNodes[0]);
        Node *fanin2 = data->createNode(currNodes[1]);
        fanOut = data->createNode(currNodes.back());
        fanOut->addFanin(fanin1);
        fanOut->addFanin(fanin2);
        fanin1->addFanout(fanOut);
        fanin2->addFanout(fanOut);

      } else {
        llvm::errs() << "Unknown number of nodes in .names: "
                     << currNodes.size() << "\n";
      }

      fanOut->setFunction(line);
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

  // Sorts the nodes in topological order.
  data->traverseNodes();
  return data;
}

void LogicNetwork::generateBlifFile(const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " << filename << "\n";
    return;
  }

  file << ".model " << moduleName << "\n";

  file << ".inputs";
  for (const auto &input : getInputs()) {
    file << " " << input->getName();
  }
  file << "\n";

  file << ".outputs";
  for (const auto &output : getOutputs()) {
    file << " " << output->getName();
  }
  file << "\n";

  for (const auto &latch : latches) {
    file << ".latch " << latch.first->getName() << " "
         << latch.second->getName() << "\n";
  }

  for (const auto &node : nodesTopologicalOrder) {
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
  std::set<Node *> nodesWithLimitedPrimaryInputs;

  for (auto &node : nodesTopologicalOrder) {
    bool erased = false;
    if (node->isChannelEdge) {
      if (wavyLine.count(node) > 0) {
        wavyLine.erase(node);
        erased = true;
      }
    }
    std::set<Node *> wavyInputs = findWavyInputsOfNode(node, wavyLine);
    if (wavyInputs.size() <= limit) {
      nodesWithLimitedPrimaryInputs.insert(node);
    }

    if (erased) {
      wavyLine.insert(node);
    }
  }
  return nodesWithLimitedPrimaryInputs;
}

std::set<Node *>
LogicNetwork::findWavyInputsOfNode(Node *node, std::set<Node *> &wavyLine) {
  std::set<Node *> primaryInputs;
  std::set<Node *> visited;
  std::function<void(Node *)> dfs = [&](Node *currentNode) {
    if (visited.count(currentNode) > 0) {
      return;
    }
    visited.insert(currentNode);

    if (wavyLine.count(currentNode) > 0) {
      primaryInputs.insert(currentNode);
      return;
    }

    for (const auto &fanin : currentNode->getFanins()) {
      dfs(fanin);
    }
  };

  dfs(node);
  return primaryInputs;
}
