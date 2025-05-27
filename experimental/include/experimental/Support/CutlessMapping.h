//===- CutlessMapping.cpp - Exp. support for MAPBUF -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of cutless mapping algorithm and Cut
// class.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
#define EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

#include "BlifReader.h"
#include "gurobi_c++.h"
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

/// A cut C of node n is a set of nodes (called leaves) such that every
/// path from any combinational input to n traverses at least one
/// leaf of C. The cuts are used to map Subject Graphs to macro-cells. The Cut
/// class represents a single cut of node n.
class Cut {
public:
  // Constructor for trivial cuts, which only have itself as a leaf.
  Cut(Node *root, Node *leaf, int depth = 0)
      : depth(depth), root(root), leaves({leaf}){};
  // Constructor for non-trivial cuts
  Cut(Node *root, std::set<Node *> leaves, int depth = 0)
      : depth(depth), root(root), leaves({leaves}){};

  // Returns the depth of a cut, which is the number of wavy lines below the
  // root node
  int getDepth() { return depth; }

  // Returns the cut selection variable, which is a Gurobi variable used for
  // MapBuf formulation. This variable is unique for each cut.
  GRBVar &getCutSelectionVariable() { return cutSelection; }

  // Returns the root node of the cut.
  Node *getNode() { return root; }

  // Returns the leaves of the cut.
  std::set<Node *> &getLeaves() { return leaves; }

private:
  int depth;               // Depth of the cut
  GRBVar cutSelection;     // Cut selection variable for MILP of MapBuf
  Node *root;              // Root node of the cut
  std::set<Node *> leaves; // Set of leaves in the cut
};

// Maps each node to all cuts that have that node as their root node.
using NodeToCuts =
    std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>;

// Cut generation algorithm that finds the K-feasible (K given by lutSize) cuts
// for each node inside the given LogicNetwork object (in this case this
// corresponds to an AIG).
NodeToCuts generateCuts(LogicNetwork *blif, int lutSize);

// Prints the cuts, used for debugging
void printCuts(NodeToCuts cuts, std::string &filename);
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
