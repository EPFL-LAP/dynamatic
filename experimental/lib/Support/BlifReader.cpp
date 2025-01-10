//===-- BlifReader.cpp - Exp. support for MAPBUF buffer placement -----*-
// C++
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
//===----------------------------------------------------------------------===//

#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#include "experimental/Support/BlifReader.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

void Node::setName(const std::string &newName) {
  // If we do it like this, then the parent blif will not see the change
  this->name = newName;
}

void BlifData::traverseUtil(Node *node, std::set<Node *> &visitedNodes) {
  if (std::find(nodesTopologicalOrder.begin(), nodesTopologicalOrder.end(),
                node) != nodesTopologicalOrder.end()) {
    return;
  }

  visitedNodes.insert(node);
  // llvm::errs() << "Node: " << node << "\n";
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
  // traverse the nodes using DFS, sorting them into topological order
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

  for (auto *input : primaryInputs) {
    nodesTopologicalOrder.push_back(input);
  }

  std::set<Node *> visitedNodes;
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
  while (std::getline(file, line)) {
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
    }

    if (line.find(".model") == 0) {
      data->setModuleName(line.substr(7));
    }

    else if (line.find(".inputs") == 0) {
      std::string inputs = line.substr(8);
      std::istringstream iss(inputs);
      std::string input;
      while (iss >> input) {
        Node *inputNode = data->createNode(input);
        inputNode->setInput(true);
        inputNode->setChannelEdge(true);
      }
    }

    else if (line.find(".outputs") == 0) {
      std::string outputs = line.substr(9);
      std::istringstream iss(outputs);
      std::string output;
      while (iss >> output) {
        Node *outputNode = data->createNode(output);
        outputNode->setOutput(true);
        outputNode->setChannelEdge(true);
      }
    }

    else if (line.find(".latch") == 0) {
      std::string latch = line.substr(7);
      std::istringstream iss(latch);
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

    else if (line.find(".names") == 0) {
      std::string function;
      std::vector<std::string> currNodes;

      std::string names = line.substr(7);
      std::istringstream iss(names);

      std::string node;
      while (iss >> node) { // read the nodes
        currNodes.push_back(node);
      }

      std::getline(file, line); // read the function
      std::istringstream iss2(line);
      iss2 >> function;

      if (currNodes.empty()) {
        llvm::errs() << "No nodes found in .names\n";
      }

      if (currNodes.size() == 1) {
        Node *fanOut = data->createNode(currNodes[0]);
        if (function == "0") {
          fanOut->setConstZero(true);
        } else if (function == "1") {
          fanOut->setConstOne(true);
        } else {
          llvm::errs() << "Unknown constant value: " << function << "\n";
        }
        fanOut->setFunction(function);
      } else if (currNodes.size() == 2) {
        Node *fanout = data->createNode(currNodes.back());
        Node *fanin = data->createNode(currNodes.front());
        fanout->addFanin(fanin);
        fanin->addFanout(fanout);
        fanout->setFunction(line);
      } else if (currNodes.size() == 3) {
        Node *fanin1 = data->createNode(currNodes[0]);
        Node *fanin2 = data->createNode(currNodes[1]);
        Node *fanout = data->createNode(currNodes.back());
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

    else if (line.find(".subckt") == 0) {
      continue;
    }

    else if (line.find(".end") == 0) {
      break;
    }
  }

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

std::vector<Node *> BlifData::findPath(Node *start, Node *end) {
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
BlifData::findNodesWithLimitedWavyInputs(size_t limit,
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

std::set<Node *> BlifData::findWavyInputsOfNode(Node *node,
                                                std::set<Node *> &wavyLine) {
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

void BlifData::printModuleInfo() {
  llvm::errs() << "Module Name: " << moduleName << "\n";

  llvm::errs() << "Inputs: ";
  for (auto &input : getPrimaryInputs()) {
    llvm::errs() << input->str() << " ";
  }
  llvm::errs() << "\n";

  llvm::errs() << "Primary Outputs: ";
  for (auto &output : getPrimaryOutputs()) {
    llvm::errs() << output->str() << " ";
  }
  llvm::errs() << "\n";

  llvm::errs() << "Nodes: ";
  for (auto &node : nodesTopologicalOrder) {
    llvm::errs() << node->str() << " ";
  }
  llvm::errs() << "\n";
}
