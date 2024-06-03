//===- SharingSupport.cpp - Resource Sharing Support ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kosaraju's algorithm for generating SCCs in linear
// time.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

using namespace dynamatic;
using namespace dynamatic::buffer;

// This recursive function traverses the CFC and calculates a post-order, i.e.,
// if there is a path from node_1 to node_2, then in the returned relative
// order, node_1 < node_2.
void recursiveDfsComponentOrder(Operation *op,
                                std::map<Operation *, bool> &visited,
                                std::vector<Operation *> &dfsPostOrder,
                                const std::set<Channel *> &cfChannels) {
  if (!visited[op]) {
    // 1. Mark op as visited
    visited[op] = true;

    // 2. For each of the outgoing channels of op, do a
    // recursiveDfsComponentOrder call
    for (Channel *ch : cfChannels) {
      if (ch->producer == op) {
        recursiveDfsComponentOrder(ch->consumer, visited, dfsPostOrder,
                                   cfChannels);
      }
    }

    // 3. Backtracking: Set the operation as discovered
    dfsPostOrder.insert(dfsPostOrder.begin(), op);
  }
}

// This recursive function assigns a single ID to each operations in a SCC.
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
    for (Channel *ch : cfChannels) {
      if (ch->consumer == op) {
        recursiveDfsAssignSCCId(ch->producer, assigned, cfChannels, currSCCId);
      }
    }
  }
}

// For a given CFC (specified as the set of units and channels), find the list
// of SCCs.
std::map<Operation *, size_t> dynamatic::experimental::sharing::getSccsInCfc(
    const std::set<Operation *> &cfUnits,
    const std::set<Channel *> &cfChannels) {
  std::map<Operation *, bool> visited;

  // DFS post-order of the CFC (see description above).
  std::vector<Operation *> dfsPostOrder;

  std::map<Operation *, size_t> assigned;

  // 1. For each unit in the CFC, mark u as unvisited.
  for (Operation *op : cfUnits)
    visited[op] = false;

  // 2. For each unit in the CFC, do RecursiveBFS(u).
  for (Operation *op : cfUnits)
    recursiveDfsComponentOrder(op, visited, dfsPostOrder, cfChannels);

  // For each unit in the CFC, mark u as not assigned (0 means not assigned).
  for (Operation *op : cfUnits)
    assigned[op] = 0;

  size_t currSCCId = 0;
  // 3. For each unit in the CFC, do recursiveBackBfs(u)
  for (Operation *op : dfsPostOrder) {
    if (assigned[op] == 0) {
      currSCCId += 1;
      recursiveDfsAssignSCCId(op, assigned, cfChannels, currSCCId);
    }
  }

  return assigned;
}
