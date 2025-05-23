//===- CutEnumeration.h - Exp. support for MAPBUF buffer placement -------*- C++
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
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
#define EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H

#include "BlifReader.h"
#include "gurobi_c++.h"
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class Cut {
public:
  Cut(Node *root, int depth = 0) : depth(depth), root(root) {};
  // for trivial cuts
  Cut(Node *root, Node *leaf, int depth = 0)
      : depth(depth), leaves({leaf}), root(root) {};
  Cut(Node *root, std::set<Node *> leaves, int depth = 0)
      : depth(depth), leaves({leaves}), root(root) {};

  int getDepth() { return depth; }

  GRBVar &getCutSelectionVariable() { return cutSelection; }

  Node *getNode() { return root; }

  std::set<Node *> &getLeaves() { return leaves; }

private:
  int depth;
  GRBVar cutSelection;
  Node *root;
  std::set<Node *> leaves;
};

struct NodePtrHash {
  std::size_t operator()(const Node *node) const {
    return std::hash<std::string>()(node->name); 
  }
};

struct NodePtrEqual {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->name == rhs->name;
  }
};

using NodeToCuts =
    std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>;

NodeToCuts generateCuts(LogicNetwork *blif, int lutSize);

void printCuts(NodeToCuts cuts, std::string &filename);
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H