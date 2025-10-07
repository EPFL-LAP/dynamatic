//===- ReadOnceBDD.cpp - Read-Once BDD Implementation -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the construction and basic analysis of a **Read-Once
// Binary Decision Diagram (BDD)**. It provides:
//
//  * Building a read-once BDD from a minimized BoolExpression and a user-
//    defined variable order.
//  * Traversing a subgraph defined by a root and two designated sink nodes.
//  * Enumerating all 2-vertex cut pairs in the subgraph.
//
// Each internal node corresponds to a variable in the provided order. Two
// terminal nodes (0 and 1) are always appended at the end. The implementation
// assumes the boolean expression is already minimized and read-once compatible.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/ReadOnceBDD.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental::boolean;
using namespace llvm;
using namespace mlir;

ReadOnceBDD::ReadOnceBDD() {
  nodes.clear();
  order.clear();
  rootIndex = zeroIndex = oneIndex = 0;
}

LogicalResult
ReadOnceBDD::buildFromExpression(BoolExpression *expr,
                                 const std::vector<std::string> &varOrder) {
  nodes.clear();
  order.clear();
  rootIndex = zeroIndex = oneIndex = 0;

  if (!expr) {
    llvm::errs() << "ReadOnceBDD: null expression\n";
    return failure();
  }

  // Minimize the whole expression once before starting.
  BoolExpression *rootExpr = expr->boolMinimize();

  // If the expression itself is constant, build only two terminals.
  if (rootExpr->type == ExpressionType::Zero ||
      rootExpr->type == ExpressionType::One) {
    nodes.resize(2);
    zeroIndex = 0;
    oneIndex = 1;
    nodes[zeroIndex] = {"", zeroIndex, zeroIndex};
    nodes[oneIndex] = {"", oneIndex, oneIndex};
    rootIndex = (rootExpr->type == ExpressionType::One) ? oneIndex : zeroIndex;
    return success();
  }

  // Keep only variables that still appear after minimization and respect
  // the order provided by the user.
  {
    std::set<std::string> present = rootExpr->getVariables();
    for (const auto &v : varOrder)
      if (present.find(v) != present.end())
        order.push_back(v);
  }

  // Pre-allocate all internal nodes; initially connect them to the terminals.
  const unsigned n = (unsigned)order.size();
  nodes.resize(n + 2);
  zeroIndex = n;
  oneIndex = n + 1;
  for (unsigned i = 0; i < n; ++i)
    nodes[i] = BDDNode{order[i], zeroIndex, oneIndex};
  nodes[zeroIndex] = {"", zeroIndex, zeroIndex};
  nodes[oneIndex] = {"", oneIndex, oneIndex};

  // Root is always the first internal node (smallest variable index).
  rootIndex = 0;

  // Recursively expand edges from the root; mark each expanded node to
  // avoid rebuilding the same subgraph.
  std::vector<char> expanded(nodes.size(), 0);
  expandFrom(rootIndex, rootExpr, expanded);

  return success();
}

void ReadOnceBDD::expandFrom(unsigned idx, BoolExpression *residual,
                             std::vector<char> &expanded) {
  if (idx >= order.size() || expanded[idx])
    return;

  const std::string &var = order[idx];

  // Perform Shannon decomposition for the current variable.
  BoolExpression *f0 = residual->deepCopy();
  restrict(f0, var, false);
  f0 = f0->boolMinimize();
  BoolExpression *f1 = residual->deepCopy();
  restrict(f1, var, true);
  f1 = f1->boolMinimize();

  // Decide the next node index for each branch.
  auto decideNext = [&](BoolExpression *f, unsigned &succ) {
    if (!f) {
      succ = zeroIndex;
      return;
    }
    if (f->type == ExpressionType::Zero) {
      succ = zeroIndex;
      return;
    }
    if (f->type == ExpressionType::One) {
      succ = oneIndex;
      return;
    }

    // Find the earliest variable in the current order that appears in f.
    auto vars = f->getVariables();
    size_t vpos = order.size();
    for (size_t i = 0; i < order.size(); ++i)
      if (vars.find(order[i]) != vars.end()) {
        vpos = i;
        break;
      }

    succ = static_cast<unsigned>(vpos);
  };

  unsigned fSucc = zeroIndex, tSucc = oneIndex;
  decideNext(f0, fSucc);
  decideNext(f1, tSucc);

  // Connect edges for the current node.
  nodes[idx].falseSucc = fSucc;
  nodes[idx].trueSucc = tSucc;

  expanded[idx] = 1;

  // Recurse only on unexplored internal successors.
  if (fSucc < zeroIndex && !expanded[fSucc])
    expandFrom(fSucc, f0, expanded);
  if (tSucc < zeroIndex && !expanded[tSucc])
    expandFrom(tSucc, f1, expanded);
}

std::vector<unsigned> ReadOnceBDD::collectSubgraph(unsigned root, unsigned t1,
                                                   unsigned t0) const {
  std::vector<char> vis(nodes.size(), 0);
  std::vector<unsigned> st{root};
  std::vector<unsigned> subgraph;

  while (!st.empty()) {
    unsigned u = st.back();
    st.pop_back();
    if (u >= nodes.size() || vis[u])
      continue;
    vis[u] = 1;
    subgraph.push_back(u);

    // Stop expansion at the designated local sinks.
    if (u == t1 || u == t0)
      continue;

    // Abort if we accidentally reach the global terminals.
    if (u == zeroIndex || u == oneIndex) {
      llvm::errs() << "Illegal subgraph: reached global terminal\n";
      std::abort();
    }

    const auto &nd = nodes[u];
    st.push_back(nd.falseSucc);
    st.push_back(nd.trueSucc);
  }

  // Ensure both sinks appear in the final list.
  if (std::find(subgraph.begin(), subgraph.end(), t1) == subgraph.end())
    subgraph.push_back(t1);
  if (std::find(subgraph.begin(), subgraph.end(), t0) == subgraph.end())
    subgraph.push_back(t0);

  std::sort(subgraph.begin(), subgraph.end());
  return subgraph;
}

bool ReadOnceBDD::sinksUnreachableIfBan(unsigned root, unsigned t1, unsigned t0,
                                        unsigned a, unsigned b) const {
  std::vector<char> vis(nodes.size(), 0);
  std::vector<unsigned> st{root};

  auto push = [&](unsigned v) {
    if (v == a || v == b)
      return; // skip banned nodes
    if (v < nodes.size() && !vis[v])
      st.push_back(v);
  };

  while (!st.empty()) {
    unsigned u = st.back();
    st.pop_back();
    if (u >= nodes.size() || vis[u] || u == a || u == b)
      continue;
    vis[u] = 1;

    // Reaching either sink means the cut fails.
    if (u == t1 || u == t0)
      return false;

    const auto &nd = nodes[u];
    push(nd.falseSucc);
    push(nd.trueSucc);
  }
  // Neither sink reachable → valid cut.
  return true;
}

std::vector<std::pair<unsigned, unsigned>>
ReadOnceBDD::listTwoVertexCuts(unsigned root, unsigned t1, unsigned t0) const {
  // Collect and validate the subgraph (sorted, includes root/t1/t0).
  std::vector<unsigned> cand = collectSubgraph(root, t1, t0);
  std::vector<std::pair<unsigned, unsigned>> cuts;

  // Scan all pairs in ascending order.
  for (size_t i = 1; i < cand.size() - 2; ++i) {
    for (size_t j = i + 1; j < cand.size(); ++j) {
      if (sinksUnreachableIfBan(root, t1, t0, cand[i], cand[j])) {
        cuts.emplace_back(cand[i], cand[j]);
      }
    }
  }

  // Sort lexicographically by (first, second).
  std::sort(cuts.begin(), cuts.end(), [](const auto &a, const auto &b) {
    if (a.first != b.first)
      return a.first < b.first;
    return a.second < b.second;
  });

  return cuts;
}