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

void AigNode::setName(const std::string &newName) { this->name = newName; }

void BlifData::traverseUtil(AigNode *node, std::set<AigNode *> &visitedNodes) {
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

void BlifData::traverseNodes() {
  std::set<AigNode *> primaryInputs;
  std::set<AigNode *> primaryOutputs;

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
  std::set<AigNode *> visitedNodes;
  for (const auto &node : primaryOutputs) {
    traverseUtil(node, visitedNodes);
  }
}

BlifData *BlifParser::parseBlifFile(const std::string &filename) {
  BlifData *data = new BlifData();
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

    // Model name
    if (line.find(".model") == 0) {
      data->setModuleName(line.substr(7));
    }

    // Input nodes. These are also Dataflow graph channels.
    else if (line.find(".inputs") == 0) {
      std::string inputs = line.substr(8);
      std::istringstream iss(inputs);
      std::string input;
      while (iss >> input) {
        AigNode *inputNode = data->createNode(input);
        inputNode->setInput(true);
        inputNode->setChannelEdge(true);
      }
    }

    // Output nodes. These are also Dataflow graph channels.
    else if (line.find(".outputs") == 0) {
      std::string outputs = line.substr(9);
      std::istringstream iss(outputs);
      std::string output;
      while (iss >> output) {
        AigNode *outputNode = data->createNode(output);
        outputNode->setOutput(true);
        outputNode->setChannelEdge(true);
      }
    }

    // Latches.
    else if (line.find(".latch") == 0) {
      std::string latch = line.substr(7);
      std::istringstream iss(latch);
      std::string regInput;
      std::string regOutput;
      iss >> regInput;
      iss >> regOutput;

      AigNode *regInputNode = data->createNode(regInput);
      regInputNode->setLatchInput(true);

      AigNode *regOutputNode = data->createNode(regOutput);
      regOutputNode->setLatchOutput(true);

      data->addLatch(regInputNode, regOutputNode);
    }

    // .names stand for logic gates.
    else if (line.find(".names") == 0) {
      std::string function;
      std::vector<std::string> currNodes;

      // Extracts the substring from position 7 to the end of line. ".names "
      // consists of 7 characters, so we start from 7.
      // Example: .names a b c
      std::string names = line.substr(7);
      std::istringstream iss(names);

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

      // If there is only one node, it is a constant node.
      if (currNodes.size() == 1) {
        AigNode *fanOut = data->createNode(currNodes[0]);
        if (function == "0") {
          fanOut->setConstZero(true);
        } else if (function == "1") {
          fanOut->setConstOne(true);
        } else {
          llvm::errs() << "Unknown constant value: " << function << "\n";
        }
        fanOut->setFunction(function);
        // If there are two nodes, it is a wire.
      } else if (currNodes.size() == 2) {
        AigNode *fanout = data->createNode(currNodes.back());
        AigNode *fanin = data->createNode(currNodes.front());
        fanout->addFanin(fanin);
        fanin->addFanout(fanout);
        fanout->setFunction(line);
        // If there are three nodes, it is a logic gate. First the nodes are
        // fanins, and the last node is fanout.
      } else if (currNodes.size() == 3) {
        AigNode *fanin1 = data->createNode(currNodes[0]);
        AigNode *fanin2 = data->createNode(currNodes[1]);
        AigNode *fanout = data->createNode(currNodes.back());
        fanout->addFanin(fanin1);
        fanout->addFanin(fanin2);
        fanin1->addFanout(fanout);
        fanin2->addFanout(fanout);
        fanout->setFunction(line);
      } else {
        llvm::errs() << "Unknown number of nodes in .names: "
                     << currNodes.size() << "\n";
      }
    }

    // Subcircuits. not used for now.
    else if (line.find(".subckt") == 0) {
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

void BlifData::generateBlifFile(const std::string &filename) {
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

std::vector<AigNode *> BlifData::findPath(AigNode *start, AigNode *end) {
  // BFS search to find the shortest path from start to end.
  std::queue<AigNode *> queue;
  std::unordered_map<AigNode *, AigNode *, boost::hash<AigNode *>> parent;
  std::set<AigNode *> visited;

  queue.push(start);
  visited.insert(start);

  while (!queue.empty()) {
    AigNode *current = queue.front();
    queue.pop();

    if (current == end) {
      // Reconstruct the path
      std::vector<AigNode *> path;
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

std::set<AigNode *>
BlifData::findNodesWithLimitedWavyInputs(size_t limit,
                                         std::set<AigNode *> &wavyLine) {
  std::set<AigNode *> nodesWithLimitedPrimaryInputs;

  for (auto &node : nodesTopologicalOrder) {
    bool erased = false;
    if (node->isChannelEdge) {
      if (wavyLine.count(node) > 0) {
        wavyLine.erase(node);
        erased = true;
      }
    }
    std::set<AigNode *> wavyInputs = findWavyInputsOfNode(node, wavyLine);
    if (wavyInputs.size() <= limit) {
      nodesWithLimitedPrimaryInputs.insert(node);
    }

    if (erased) {
      wavyLine.insert(node);
    }
  }
  return nodesWithLimitedPrimaryInputs;
}

std::set<AigNode *>
BlifData::findWavyInputsOfNode(AigNode *node, std::set<AigNode *> &wavyLine) {
  std::set<AigNode *> primaryInputs;
  std::set<AigNode *> visited;
  std::function<void(AigNode *)> dfs = [&](AigNode *currentNode) {
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
