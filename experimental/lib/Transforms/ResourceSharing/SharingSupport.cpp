//===- SharingSupport.cpp - Resource Sharing Support ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

using namespace dynamatic;
using namespace dynamatic::buffer;

void recursiveDfs(Operation *op, std::map<Operation *, bool> &visited,
                  std::vector<Operation *> &discoveredOps,
                  const std::set<Channel *> &cfChannels) {
  if (!visited[op]) {
    // 1. Mark op as visited
    visited[op] = true;

    // 2. For each of the outgoing channels of op, do a recursiveDfs call
    for (auto *ch : cfChannels) {
      if (ch->producer == op) {
        recursiveDfs(ch->consumer, visited, discoveredOps, cfChannels);
      }
    }

    // 3. Backtracking: Set the operation as discovered
    discoveredOps.insert(discoveredOps.begin(), op);
  }
}

void recursiveDfsAssignSCCId(Operation *op,
                             std::map<Operation *, size_t> &assigned,
                             const std::set<Channel *> &cfChannels,
                             size_t &currSCCId) {
  if (assigned[op] == 0) {
    // 1. If the node is not yet assigned: assign it to the current ID
    // currSCCId += 1;
    assigned[op] = currSCCId;

    // 2. For all the nodes discovered using backward DFS call, assign them with
    // the same SCC ID
    for (auto *ch : cfChannels) {
      if (ch->consumer == op) {
        recursiveDfsAssignSCCId(ch->producer, assigned, cfChannels, currSCCId);
      }
    }
  }
}

// for a CFC, find the list of SCCs
std::map<Operation *, size_t> dynamatic::experimental::sharing::getSccsInCfc(
    const std::set<Operation *> &cfUnits,
    const std::set<Channel *> &cfChannels) {
  std::map<Operation *, bool> visited;

  std::vector<Operation *> visitList;

  std::map<Operation *, size_t> assigned;

  // 1. For each unit in the CFC, mark u as unvisited.
  for (auto *op : cfUnits)
    visited[op] = false;

  // 2. For each unit in the CFC, do RecursiveBFS(u).
  for (auto *op : cfUnits)
    recursiveDfs(op, visited, visitList, cfChannels);

  // For each unit in the CFC, mark u as not assigned (0 means not assigned).
  for (auto *op : cfUnits)
    assigned[op] = 0;

  size_t currSCCId = 0;
  // 3. For each unit in the CFC, do recursiveBackBfs(u)
  for (auto *op : visitList) {
    if (assigned[op] == 0) {
      currSCCId += 1;
      recursiveDfsAssignSCCId(op, assigned, cfChannels, currSCCId);
      // llvm::errs() << "Current ID" << currSCCId << "\n";
    }
  }

  return assigned;
}
