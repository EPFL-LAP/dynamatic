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

void sortAndEraseCuts(std::unordered_map<Node *, std::vector<Cut>, NodePtrHash,
                                         NodePtrEqual> &cuts) {
  for (auto &[node, cutVector] : cuts) {
    std::sort(cutVector.begin(), cutVector.end(), [](Cut &a, Cut &b) {
      // Get the sets first
      const auto leavesA = a.getLeaves();
      const auto leavesB = b.getLeaves();

      // First compare by size
      if (leavesA.size() != leavesB.size()) {
        return leavesA.size() < leavesB.size();
      }

      // Compare elements using set's iterators
      auto itA = leavesA.begin();
      auto itB = leavesB.begin();
      while (itA != leavesA.end() && itB != leavesB.end()) {
        const Node *nodeA = *itA;
        const Node *nodeB = *itB;

        // Handle null pointers
        if (!nodeA || !nodeB) {
          if (nodeA == nodeB) {
            ++itA;
            ++itB;
            continue;
          }
          return !nodeA;
        }

        // Compare names
        if (nodeA->getName() != nodeB->getName()) {
          return nodeA->getName() < nodeB->getName();
        }

        ++itA;
        ++itB;
      }
      return false; // Sets are equal
    });

    // Remove duplicates
    cutVector.erase(std::unique(cutVector.begin(), cutVector.end(),
                                [](Cut &a, Cut &b) {
                                  const auto leavesA = a.getLeaves();
                                  const auto leavesB = b.getLeaves();

                                  if (leavesA.size() != leavesB.size()) {
                                    return false;
                                  }

                                  // Compare sets
                                  auto itA = leavesA.begin();
                                  auto itB = leavesB.begin();
                                  while (itA != leavesA.end() &&
                                         itB != leavesB.end()) {
                                    const Node *nodeA = *itA;
                                    const Node *nodeB = *itB;

                                    // Handle null pointers
                                    if (!nodeA || !nodeB) {
                                      if (nodeA != nodeB) {
                                        return false;
                                      }
                                      ++itA;
                                      ++itB;
                                      continue;
                                    }

                                    // Compare names
                                    if (nodeA->getName() != nodeB->getName()) {
                                      return false;
                                    }

                                    ++itA;
                                    ++itB;
                                  }
                                  return true; // Sets are equal
                                }),
                    cutVector.end());
  }
}

CutManager::CutManager(BlifData *blif, int lutSize)
    : lutSize(lutSize), blif(blif) {
  // Call the cutless algorithm, if argument is true, include channels in the
  // Primary Inputs set as well
  auto cutsWithoutChannels = cutless(false);
  auto cutsWithChannels = cutless(true);

  // Merge cuts
  cuts = std::move(cutsWithoutChannels);
  for (const auto &[node, cutVector] : cutsWithChannels) {
    cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
  }

  sortAndEraseCuts(cuts);
};

void CutManager::printCuts(const std::string &filename) {
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

NodeToCuts CutManager::cutless(bool includeChannels) {
  int n = 0;
  std::set<Node *> currentWavyLine = blif->getPrimaryInputs();
  std::set<Node *> nextWavyLine;
  NodeToCuts cutlessCuts;

  if (includeChannels) {
    for (auto *channel : blif->getChannels()) {
      currentWavyLine.insert(channel);
    }
  }

  while (true) {
    nextWavyLine =
        blif->findNodesWithLimitedWavyInputs(lutSize, currentWavyLine);
    if ((nextWavyLine.size() == currentWavyLine.size()) ||
        (includeChannels && (n >= expansionWithChannels))) {
      break;
    }

    for (auto *node : nextWavyLine) {
      cutlessCuts[node].emplace_back(
          node, blif->findWavyInputsOfNode(node, currentWavyLine), n);
    }

    n++;
    currentWavyLine.insert(nextWavyLine.begin(), nextWavyLine.end());
  }

  for (auto &[node, cuts] : cutlessCuts) {
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](Cut &a, Cut &b) {
                             const auto &leavesA = a.getLeaves();
                             const auto &leavesB = b.getLeaves();

                             // Compare the sizes first as an optimization
                             if (leavesA.size() != leavesB.size()) {
                               return false;
                             }

                             // Compare elements in the set based on Node's name
                             // strings
                             return std::equal(
                                 leavesA.begin(), leavesA.end(),
                                 leavesB.begin(), leavesB.end(),
                                 [](const Node *nodeA, const Node *nodeB) {
                                   return nodeA->getName() == nodeB->getName();
                                 });
                           }),
               cuts.end());
  }

  return cutlessCuts;
}
