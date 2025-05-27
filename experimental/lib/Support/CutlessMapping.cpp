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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

#include "experimental/Support/CutlessMapping.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

// Sorts the cuts based on the lexicographical order, and erases the duplicate
// cuts of a Node.
void sortAndEraseCuts(NodeToCuts &cuts) {
  for (auto &[node, cutVector] : cuts) {
    // Sort the cuts based on the lexicographical order of the names of the leaf
    // Nodes.
    std::sort(cutVector.begin(), cutVector.end(), [](Cut &a, Cut &b) {
      const auto leavesA = a.getLeaves();
      const auto leavesB = b.getLeaves();

      // Early exit if the sizes of the sets are different.
      if (leavesA.size() != leavesB.size()) {
        return leavesA.size() < leavesB.size();
      }

      return std::lexicographical_compare(
          leavesA.begin(), leavesA.end(), leavesB.begin(), leavesB.end(),
          [](const Node *nodeA, const Node *nodeB) {
            return nodeA->name < nodeB->name;
          });
    });

    // If there are duplicate cuts, erase one of them.
    cutVector.erase(std::unique(cutVector.begin(), cutVector.end(),
                                [](Cut &a, Cut &b) {
                                  const auto &leavesA = a.getLeaves();
                                  const auto &leavesB = b.getLeaves();

                                  // Early exit if the sizes of the sets are
                                  // different.
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

  // Insert the node back to the wavyLine if it was erased.
  if (erased) {
    wavyLine.insert(node);
  }

  return wavyInputs;
}

// Depth-oriented mapping algorithm. A wavy line represents a set of nodes in
// the LogicNetwork, grouped by depth. By definition, the nodes in the n-th wavy
// line can be implemented as a function of K (lutSize) number of nodes in any
// of the previous wavy lines, i.e., the (n−i)-th wavy line for any i such that
// 0 < i ≤ n. For example, nodes in the third wavy line may be implemented using
// nodes from the first or second wavy line.
NodeToCuts cutAlgorithm(LogicNetwork *blif, int lutSize, bool includeChannels) {
  NodeToCuts cuts;
  // First wavy line consists of the Primary Inputs of the circuit.
  std::set<Node *> currentWavyLine = blif->getPrimaryInputs();

  // Adds Channel Nodes to the first wavy line if the flag is set. In this case,
  // nodes in the AIG that corresponds to channels are considered as
  // primary inputs of the circuit. This means that for each node of a hardware
  // module, leaves of the cut will only consist of nodes up until the channel
  // of the module. This prevents finding the deepest cuts, however, it
  // ensures that there are some cuts that do not cover channels, so that
  // buffers can be placed in MapBuf without violating Cut Selection Conflict
  // Constraints.
  if (includeChannels) {
    for (auto *channel : blif->getChannels()) {
      currentWavyLine.insert(channel);
    }
  }

  // Variable to keep track of how many times the wavy line has been expanded.
  int expansionCount = 0;
  // The limit for the expansion of the algorithm when Channels are
  // included. Prevents infinite expansion of the wavy lines.
  int expansionWithChannels = 6;

  // Keep expanding until we hit the expansion limit.
  while (!includeChannels || (expansionCount < expansionWithChannels)) {
    std::set<Node *> nextWavyLine;

    for (auto &currentNode : blif->getNodesInTopologicalOrder()) {
      // Find wavy inputs of the currentNode. Wavy inputs consists of the Nodes
      // that can be used to implement the currentNode.
      std::set<Node *> wavyInputs =
          findWavyInputsOfNode(currentNode, currentWavyLine);
      // If the number of wavy inputs is less than or equal to the limit (the
      // LUT size), add to the set of nextWavyLine and update the cuts of the
      // currentNode
      if (wavyInputs.size() <= lutSize) {
        nextWavyLine.insert(currentNode);
        cuts[currentNode].emplace_back(currentNode, wavyInputs, expansionCount);
      }
    }

    // Break if the algorithm has converged.
    if ((nextWavyLine.size() == currentWavyLine.size())) {
      break;
    }

    expansionCount++;
    // Expand the wavy line by adding nextWavyLine to currentWavyLine
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

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
