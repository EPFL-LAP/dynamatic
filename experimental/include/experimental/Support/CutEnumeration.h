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
  Cut(Node *root, int depth = 0) : depth(depth), root(root){};
  // for trivial cuts
  Cut(Node *root, Node *leaf, int depth = 0)
      : depth(depth), leaves({leaf}), root(root){};
  Cut(Node *root, std::set<Node *> leaves, int depth = 0)
      : depth(depth), leaves({leaves}), root(root){};

  // Getters and Setters
  int getDepth() { return depth; }

  GRBVar &getCutSelectionVariable() { return cutSelection; }

  Node *getNode() { return root; }

  std::set<Node *> &getLeaves() { return leaves; }

  void addLeaves(Node *leaf) { this->leaves.insert(leaf); }

  void addLeaves(std::set<Node *> &leavesToAdd) {
    this->leaves.insert(leavesToAdd.begin(), leavesToAdd.end());
  }

  void setLeaves(std::set<Node *> &leavesToSet) { this->leaves = leavesToSet; }

private:
  // depth of the cut
  int depth;
  // Gurobi variable for cut selection
  GRBVar cutSelection;
  // leaves of the cut
  std::set<Node *> leaves;
  // root of the cut
  Node *root;
};

struct NodePtrHash {
  std::size_t operator()(const Node *node) const {
    return std::hash<std::string>()(node->getName()); // Hash the name
  }
};

struct NodePtrEqual {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->getName() == rhs->getName();
  }
};

using NodeToCuts =
    std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>;

class CutManager {
public:
  CutManager(BlifData *blif, int lutSize);

  // Node to Cuts Map
  static inline NodeToCuts cuts;
  // Prints cuts to a file
  static void printCuts(const std::string &filename);

private:
  // how many times to expand the cutless cuts when including channels
  const int expansionWithChannels = 6;
  // size of the LUT
  int lutSize{};
  // AIG that the cut enumeration is performed on
  experimental::BlifData *blif;

  // Cutless algorithm
  NodeToCuts cutless(bool includeChannels);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H