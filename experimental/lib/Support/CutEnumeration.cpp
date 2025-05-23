//===-- CutEnumeration.cpp - Exp. support for MAPBUF buffer placement -----*-
// C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of cut enumeration algorithms and Cut
// class.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "experimental/Support/CutEnumeration.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

int expansionWithChannels = 6;

void sortAndEraseCuts(std::unordered_map<Node *, std::vector<Cut>, NodePtrHash,
                                         NodePtrEqual> &cuts) {
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
        }), cutVector.end());
  }
}

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
  // not end prematurely.
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

NodeToCuts cutAlgorithm(LogicNetwork *blif, int lutSize, bool includeChannels) {
  NodeToCuts cuts;
  
  std::set<Node *> currentWavyLine = blif->getPrimaryInputs();

  if (includeChannels) {
    for (auto *channel : blif->getChannels()) {
      currentWavyLine.insert(channel);
    }
  }

  int expansionCount = 0;
  
  // Keep expanding until we hit the expansion limit
  while (!(includeChannels && (expansionCount >= expansionWithChannels))) {
    std::set<Node *> nextWavyLine;

    for (auto &node : blif->getNodesInOrder()) {
      std::set<Node *> wavyInputs = findWavyInputsOfNode(node, currentWavyLine);
      // if the number of wavy inputs is less than or equal to the limit (less
      // than the LUT size), add to the set
      if (wavyInputs.size() <= lutSize) {
        nextWavyLine.insert(node);
        cuts[node].emplace_back(node, wavyInputs, expansionCount);
      }
    }

    // break if no more Nodes can be added to the wavy line
    if ((nextWavyLine.size() == currentWavyLine.size())) {
      break;
    }

    expansionCount++;
    currentWavyLine.insert(nextWavyLine.begin(), nextWavyLine.end());
  }

  return cuts;
}

NodeToCuts dynamatic::experimental::generateCuts(LogicNetwork *blif,
                                                 int lutSize) {
  auto cutsWithoutChannels = cutAlgorithm(blif, lutSize, false);
  auto cutsWithChannels = cutAlgorithm(blif, lutSize, true);

  // Merge cuts
  NodeToCuts cuts;

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

// Prints the cuts, can be used for debugging
void dynamatic::experimental::printCuts(NodeToCuts cuts, std::string &filename) {
  std::ofstream outFile("../mapbuf/" + filename);
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