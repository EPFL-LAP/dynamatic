//===- CutlessMapping.cpp - Exp. support for MAPBUF -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of cutless mapping algorithm and Cut
// class.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "experimental/Support/CutlessMapping.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

// Sorts the cuts based on the lexicographical order, and erases the duplicate
// cuts of a Node.
void sortAndEraseCuts(NodeToCuts &cuts) {
  for (auto &[node, cutVector] : cuts) {
    std::sort(cutVector.begin(), cutVector.end(), [](Cut &a, Cut &b) {
      const auto leavesA = a.getLeaves();
      const auto leavesB = b.getLeaves();

      if (leavesA.size() != leavesB.size()) {
        return leavesA.size() < leavesB.size();
      }

      return std::lexicographical_compare(
          leavesA.begin(), leavesA.end(), leavesB.begin(), leavesB.end(),
          [](const Node *nodeA, const Node *nodeB) {
            return nodeA->name < nodeB->name;
          });
    });

    cutVector.erase(std::unique(cutVector.begin(), cutVector.end(),
                                [](Cut &a, Cut &b) {
                                  const auto &leavesA = a.getLeaves();
                                  const auto &leavesB = b.getLeaves();

                                  // Compare the sizes first
                                  if (leavesA.size() != leavesB.size()) {
                                    return false;
                                  }

                                  // Compare elements in the set based on Node's
                                  // name strings
                                  return std::equal(
                                      leavesA.begin(), leavesA.end(),
                                      leavesB.begin(), leavesB.end(),
                                      [](const Node *nodeA, const Node *nodeB) {
                                        return nodeA->name == nodeB->name;
                                      });
                                }),
                    cutVector.end());
  }
}

// Finds the wavy inputs of a node. Wavy inputs are the nodes that can be used
// to implement the current node.
std::set<Node *> findWavyInputsOfNode(Node *node, std::set<Node *> &wavyLine) {
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

    for (const auto &fanin : currentNode->fanins) {
      dfs(fanin);
    }
  };

  // Erase a channel node from the wavyLine temporarily, so the search does
  // not end prematurely. This is is needed in the case when Channel Nodes are
  // included to the first wavy line.
  bool erased = false;
  if (node->isChannelEdge && wavyLine.count(node) > 0) {
    wavyLine.erase(node);
    erased = true;
  }

  dfs(node);

  if (erased) {
    wavyLine.insert(node);
  }

  return wavyInputs;
}

// Depth-oriented mapping algorithm. A wavy line is a set of nodes that can be
// implemented on an LUT in terms of Nodes that are below it. For example, third
// wavy line consists of the Nodes that can be implemented in terms of the first
// and second wavy line.
NodeToCuts cutAlgorithm(LogicNetwork *blif, int lutSize, bool includeChannels) {
  NodeToCuts cuts;
  // First wavy line consists of the Primary Inputs of the circuit.
  std::set<Node *> currentWavyLine = blif->getPrimaryInputs();

  // Add Channel Nodes to the first wavy line
  if (includeChannels) {
    for (auto *channel : blif->getChannels()) {
      currentWavyLine.insert(channel);
    }
  }

  int expansionCount = 0;
  int expansionWithChannels = 6; // The limit for the expansion of the algorithm

  // Keep expanding until we hit the expansion limit
  while (!(includeChannels && (expansionCount >= expansionWithChannels))) {
    std::set<Node *> nextWavyLine;

    for (auto &currentNode : blif->getNodesInOrder()) {
      // Find wavy inputs of the currentNode. Wavy inputs consists of the Nodes
      // that can be used to implement the currentNode.
      std::set<Node *> wavyInputs =
          findWavyInputsOfNode(currentNode, currentWavyLine);
      // if the number of wavy inputs is less than or equal to the limit (the
      // LUT size), add to the set of next wavy line
      if (wavyInputs.size() <= lutSize) {
        nextWavyLine.insert(currentNode);
        cuts[currentNode].emplace_back(currentNode, wavyInputs, expansionCount);
      }
    }

    // break if there are no changes to the wavy lines
    if ((nextWavyLine.size() == currentWavyLine.size())) {
      break;
    }

    expansionCount++;
    // Expand the wavy line by adding the next wavy line to the current wavy
    // line
    currentWavyLine.insert(nextWavyLine.begin(), nextWavyLine.end());
  }

  return cuts;
}

// Generates cuts for a given AIG and LUT size. Uses a modified version of the
// algorithm from the "Cutless FPGA Mapping" (Mishchenko et al., 2007).
NodeToCuts dynamatic::experimental::generateCuts(LogicNetwork *blif,
                                                 int lutSize) {
  // First, we generate cuts without any modifications
  auto cutsWithoutChannels = cutAlgorithm(blif, lutSize, false);
  // Then, we generate cuts with channel nodes marked as Primary Inputs of the
  // circuit
  auto cutsWithChannels = cutAlgorithm(blif, lutSize, true);

  NodeToCuts cuts;

  // Merge all cuts into a single map
  for (const auto &[node, cutVector] : cutsWithoutChannels) {
    cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
  }

  for (const auto &[node, cutVector] : cutsWithChannels) {
    cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
  }

  // Sort and erase duplicate cuts
  sortAndEraseCuts(cuts);

  return cuts;
}

void dynamatic::experimental::printCuts(NodeToCuts cuts,
                                        std::string &filename) {
  std::ofstream outFile("./" + filename);
  if (!outFile.is_open()) {
    llvm::errs() << "Error: Unable to open file for writing.\n";
    return;
  }

  for (auto &nodeCuts : cuts) {
    Node *node = nodeCuts.first;
    std::vector<Cut> &cutList = nodeCuts.second;
    std::size_t numCuts = cutList.size();

    outFile << node->str() << " " << numCuts << "\n";

    for (size_t i = 0; i < cutList.size(); ++i) {
      Cut &cut = cutList[i];
      std::set<Node *> leaves = cut.getLeaves();
      std::size_t cutSize = leaves.size();
      outFile << "Cut #" << i << ": " << cutSize << " depth: " << cut.getDepth()
              << "\n";

      for (auto *leaf : leaves) {
        outFile << '\t' << leaf->str() << "\n";
      }
    }
  }
  outFile.close();
}