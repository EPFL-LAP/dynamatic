//===- ROBDD.cpp - ROBDD construction and analysis --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements construction of a Reduced Ordered Binary Decision
// Diagram (ROBDD) from a BoolExpression and basic analysis utilities.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/BooleanLogic/ROBDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"

#include <algorithm>
#include <climits>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental::boolean;
using namespace llvm;
using namespace mlir;

void dynamatic::experimental::boolean::restrict(BoolExpression *exp,
                                                const std::string &var,
                                                bool expressionValue) {

  // If the input is a variable only, then possibly substitute the value with
  // the provided one. If the expression is a binary one, recursively call
  // `restrict` over the two inputs
  if (exp->type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(exp);
    if (singleCond->id == var) {
      // Invert the value if complemented
      if (singleCond->isNegated)
        exp->type =
            (expressionValue) ? ExpressionType::Zero : ExpressionType::One;
      else
        exp->type =
            (expressionValue) ? ExpressionType::One : ExpressionType::Zero;
    }

  } else if (exp->type == ExpressionType::And ||
             exp->type == ExpressionType::Or) {
    Operator *op = static_cast<Operator *>(exp);

    if (op->left)
      restrict(op->left, var, expressionValue);

    if (op->right)
      restrict(op->right, var, expressionValue);
  }
}

ROBDD::ROBDD() {
  nodes.clear();
  order.clear();
  rootIndex = zeroIndex = oneIndex = 0;
}

LogicalResult
ROBDD::buildROBDDFromExpression(BoolExpression *expr,
                                const std::vector<std::string> &varOrder) {
  nodes.clear();
  order.clear();
  rootIndex = zeroIndex = oneIndex = 0;

  if (!expr) {
    llvm::errs() << "ROBDD: null expression\n";
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

    // Terminals have no variables and no successors (indicated by specific
    // sentinel values like UINT_MAX). Note: We create BOTH terminals to
    // maintain valid zeroIndex/oneIndex invariants for the class, even if one
    // of them is unreachable in this specific trivial graph.
    nodes[zeroIndex] = {"", UINT_MAX, UINT_MAX, {}};
    nodes[oneIndex] = {"", UINT_MAX, UINT_MAX, {}};
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
    nodes[i] = ROBDDNode{order[i], zeroIndex, oneIndex, {}};
  nodes[zeroIndex] = {"", zeroIndex, zeroIndex, {}};
  nodes[oneIndex] = {"", oneIndex, oneIndex, {}};

  // Root is always the first internal node (smallest variable index).
  rootIndex = 0;

  // Recursively expand edges from the root; mark each expanded node to
  // avoid rebuilding the same subgraph.
  std::vector<char> expanded(nodes.size(), 0);
  expandFrom(rootIndex, rootExpr, expanded);

  // After the BDD is fully built, clean up each node's predecessor list:
  // sort in ascending order and remove any duplicates.
  for (auto &nd : nodes) {
    auto &ps = nd.preds;
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
  }

  return success();
}

void ROBDD::expandFrom(unsigned idx, BoolExpression *residual,
                       std::vector<char> &expanded) {
  if (idx >= order.size() || expanded[idx])
    return;

  const std::string &var = order[idx];

  // Perform Shannon expansion for the current variable.
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

  // While expanding the BDD, record the current node `idx`
  // as a predecessor of each of its false/true successors.
  nodes[fSucc].preds.push_back(idx);
  nodes[tSucc].preds.push_back(idx);

  expanded[idx] = 1;

  // Recurse only on unexplored internal successors.
  if (fSucc < zeroIndex && !expanded[fSucc])
    expandFrom(fSucc, f0, expanded);
  if (tSucc < zeroIndex && !expanded[tSucc])
    expandFrom(tSucc, f1, expanded);
}

std::vector<unsigned> ROBDD::collectSubgraph(unsigned root, unsigned t1,
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

bool ROBDD::doesPairCoverAllPaths(unsigned rootNode, unsigned trueTerminal,
                                  unsigned falseTerminal, unsigned coverNodeA,
                                  unsigned coverNodeB) const {
  // Use a visited array to avoid cycles and redundant processing.
  std::vector<char> visited(nodes.size(), 0);
  std::vector<unsigned> workStack{rootNode};

  // Helper to push valid successors to the stack.
  auto pushToStack = [&](unsigned v) {
    // If we hit either of the covering nodes, the path is "blocked" or
    // "covered" by them, so we stop traversing this path (effectively treating
    // them as sinks).
    if (v == coverNodeA || v == coverNodeB)
      return;

    if (v < nodes.size() && !visited[v])
      workStack.push_back(v);
  };

  while (!workStack.empty()) {
    unsigned u = workStack.back();
    workStack.pop_back();

    // Standard DFS checks: bounds, visited, or hitting the cover nodes (double
    // check).
    if (u >= nodes.size() || visited[u] || u == coverNodeA || u == coverNodeB)
      continue;
    visited[u] = 1;

    // If we managed to reach either terminal without hitting coverNodeA or
    // coverNodeB, then the pair does NOT cover all paths.
    if (u == trueTerminal || u == falseTerminal)
      return false;

    const auto &nd = nodes[u];
    pushToStack(nd.falseSucc);
    pushToStack(nd.trueSucc);
  }

  // If the stack is empty and we never reached a terminal,
  // all paths were covered by the pair.
  return true;
}

std::vector<std::pair<unsigned, unsigned>>
ROBDD::findPairsCoveringAllPaths(unsigned rootNode, unsigned trueTerminal,
                                 unsigned falseTerminal) const {
  // Collect and validate the subgraph (sorted, includes root, trueTerminal,
  // falseTerminal).
  std::vector<unsigned> candidates =
      collectSubgraph(rootNode, trueTerminal, falseTerminal);
  std::vector<std::pair<unsigned, unsigned>> coveringPairs;

  // Scan all pairs in ascending order.
  for (size_t i = 1; i < candidates.size() - 2; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (doesPairCoverAllPaths(rootNode, trueTerminal, falseTerminal,
                                candidates[i], candidates[j])) {
        coveringPairs.emplace_back(candidates[i], candidates[j]);
      }
    }
  }

  // Sort lexicographically by (first, second).
  std::sort(coveringPairs.begin(), coveringPairs.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first)
                return lhs.first < rhs.first;
              return lhs.second < rhs.second;
            });

  return coveringPairs;
}