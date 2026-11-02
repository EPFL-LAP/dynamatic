//===- ControlNetworkAnalysis.cpp - Loops of the control network *- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the analysis recovering the loops of a Handshake function from its
// control network.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/ControlNetworkAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GenericLoopInfoImpl.h"

using namespace mlir;
using namespace dynamatic;

void ControlNode::printAsOperand(raw_ostream &os, bool /*printType*/) const {
  if (op)
    os << getUniqueName(op);
  else
    os << "<entry>";
}

ControlNode *ControlGraph::addNode(Operation *op) {
  ControlNode &node = nodes.emplace_back();
  node.op = op;
  node.graph = this;
  node.number = nodes.size() - 1;
  return &node;
}

void ControlGraph::addEdge(ControlNode *src, ControlNode *dst) {
  src->succs.push_back(dst);
  dst->preds.push_back(src);
}

ControlGraph::ControlGraph(handshake::FuncOp funcOp) {
  ControlNode *entry = addNode(nullptr);

  // Memory interfaces (shared by the whole function) are excluded from the
  // control graph so that loop detection reflects only the program's control
  // flow. Including these could otherwise run at the risk of creating
  // irreducible loops that are not detected.
  for (Operation &op : *funcOp.getBodyBlock())
    if (!isa<handshake::MemoryOpInterface>(op))
      nodeFor[&op] = addNode(&op);

  // Control-only edges between operations.
  for (Operation &op : *funcOp.getBodyBlock()) {
    ControlNode *src = getNode(&op);
    if (!src)
      continue;

    for (Value res : op.getResults()) {
      if (!isa<handshake::ControlType>(res.getType()))
        continue;

      for (Operation *user : res.getUsers())
        if (ControlNode *dst = getNode(user))
          addEdge(src, dst);
    }
  }

  // Connect all nodes without predecessors to the virtual entry node.
  // An init op starts a control network which is a closed cycle (its input
  // comes back from the network's control merge), so it is connected too:
  // otherwise the network is unreachable from the entry and its loops are
  // not found
  for (ControlNode &node : nodes)
    if (&node != entry &&
        (node.preds.empty() || isa_and_present<handshake::InitOp>(node.getOp())))
      addEdge(entry, &node);
}

unsigned ControlLoop::getMaxLoopDepth() const {
  unsigned maxDepth = getLoopDepth();
  for (const ControlLoop *subLoop : getSubLoops())
    maxDepth = std::max(maxDepth, subLoop->getMaxLoopDepth());
  return maxDepth;
}

ControlNetworkAnalysis::ControlNetworkAnalysis(handshake::FuncOp funcOp)
    : graph(funcOp) {
  domTree.recalculate(graph);
  loopInfo.analyze(domTree);
}

SmallVector<ControlLoop *> ControlNetworkAnalysis::getLoopsInPreorder() const {
  return loopInfo.getLoopsInPreorder();
}
