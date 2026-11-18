//===- DFCGraph.cpp - Dataflow circuit graph --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DFCGraph.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace dynamatic;

/// Materialized IR: a channel has at most one consumer.
[[maybe_unused]] static bool hasAtMostOneConsumer(Value channel) {
  return channel.use_empty() || channel.hasOneUse();
}

std::optional<DFCGraph::NodeId> DFCGraph::lookup(Operation *unit) const {
  auto it = unitToNode.find(unit);
  if (it == unitToNode.end())
    return std::nullopt;
  return it->second;
}

std::optional<DFCGraph::EdgeId> DFCGraph::lookup(Value channel) const {
  auto it = channelToEdge.find(channel);
  if (it == channelToEdge.end())
    return std::nullopt;
  return it->second;
}

void DFCGraph::addUnits(ArrayRef<Operation *> units) {
  nodes.reserve(units.size());
  for (Operation *unit : units) {
    auto [it, inserted] = unitToNode.try_emplace(unit, nodes.size());
    if (!inserted)
      continue;
    nodes.push_back(Node{unit, it->second, {}, {}, {}});
  }
}

void DFCGraph::addEdge(NodeId src, NodeId dst, Value channel) {
  EdgeId id = edges.size();
  [[maybe_unused]] bool inserted =
      channelToEdge.try_emplace(channel, id).second;
  assert(inserted && "channel added twice");
  edges.push_back(Edge{src, dst, channel});
  nodes[src].outEdges.push_back(id);
  nodes[dst].inEdges.push_back(id);
  nodes[src].succs.push_back(&nodes[dst]);
}

void DFCGraph::finalize() {
  root.succs.reserve(nodes.size());
  for (const Node &node : nodes)
    root.succs.push_back(&node);
  computeSCCs();
}

void DFCGraph::build(ArrayRef<Operation *> units, ArrayRef<Value> channels) {
  addUnits(units);
  for (Value channel : channels) {
    if (channelToEdge.contains(channel))
      continue;
    assert(hasAtMostOneConsumer(channel) &&
           "expected materialized IR (one consumer per channel)");
    std::optional<NodeId> src = lookup(channel.getDefiningOp());
    if (!src)
      continue;
    for (Operation *consumer : channel.getUsers()) {
      if (std::optional<NodeId> dst = lookup(consumer))
        addEdge(*src, *dst, channel);
    }
  }
  finalize();
}

void DFCGraph::computeSCCs() {
  sccOf.assign(nodes.size(), 0);
  unsigned sccId = 0;
  for (auto it = llvm::scc_begin(static_cast<const DFCGraph *>(this));
       !it.isAtEnd(); ++it) {
    const std::vector<const Node *> &scc = *it;
    // Skip the ghost root
    if (scc.size() == 1 && scc.front() == &root)
      continue;
    for (const Node *node : scc)
      sccOf[node->id] = sccId;
    ++sccId;
  }
}
