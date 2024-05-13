//===- SharingSupport.cpp - Resource Sharing Support ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

void recursiveDfs(Operation *op, std::map<Operation *, bool> visited,
                  std::vector<Operation *> discoveredOps,
                  const std::set<Channel *> &cfChannels) {
  if (!visited[op]) {
    // 1. Mark op as visited
    visited[op] = true;

    // 2. For each of the outgoing channels of op, do a recursiveDfs call
    for (auto *ch : cfChannels) {
      if (ch->producer == op) {
        recursiveDfs(op, visited, discoveredOps, cfChannels);
      }
    }

    // 3. Backtracking: Set the operation as discovered
    discoveredOps.push_back(op);
  }
}

void recursiveDfsAssignSCCId(Operation *op,
                             std::map<Operation *, size_t> assigned,
                             const std::set<Channel *> &cfChannels,
                             size_t &currSCCId) {
  if (assigned[op] == 0) {
    // 1. If the node is not yet assigned: add a new component
    currSCCId += 1;
    assigned[op] = currSCCId;

    // 2. For all the nodes discovered using backward DFS call, assign them with
    // the same SCC ID
    for (auto *ch : cfChannels) {
      if (ch->consumer == op) {
        recursiveDfsAssignSCCId(op, assigned, cfChannels, currSCCId);
      }
    }
  }
}

// for a CFC, find the list of SCCs
std::map<Operation *, size_t> getSccsInCfc(handshake::FuncOp funcOp,
                                           std::set<Operation *> cfUnits,
                                           std::set<Channel *> cfChannels) {
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

  size_t currSCCId = 1;
  // 3. For each unit in the CFC, do recursiveBackBfs(u)
  for (auto *op : cfUnits) {
    recursiveDfsAssignSCCId(op, assigned, cfChannels, currSCCId);
  }
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic