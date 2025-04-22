//===- OutOfOrderClustering.h Clustering Algorithm------*- C++ -*-===//
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the function required for the clustering algorithm
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_OUTOFORDEREXECUTION_OUTOFORDERCLUSTERING_H
#define EXPERIMENTAL_TRANSFORMS_OUTOFORDEREXECUTION_OUTOFORDERCLUSTERING_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include <utility>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace outoforder {

struct Cluster {
  llvm::DenseSet<Value> inputs;
  llvm::DenseSet<Value> outputs;
  llvm::DenseSet<Operation *> internalNodes;

  Cluster(llvm::DenseSet<Value> inputs, llvm::DenseSet<Value> outputs,
          llvm::DenseSet<Operation *> internalNodes)
      : inputs(std::move(inputs)), outputs(std::move(outputs)),
        internalNodes(std::move(internalNodes)) {}
  Cluster() = default;
  Cluster(const Cluster &other) = default;
  Cluster &operator=(const Cluster &other) = default;

  bool operator==(const Cluster &other) const {
    return inputs == other.inputs && outputs == other.outputs &&
           internalNodes == other.internalNodes;
  }

  // Checks if the operation is inside the cluster
  bool isInsideCluster(Operation *op) const {
    return internalNodes.contains(op);
  }

  // Checks if the operation is before the cluster
  // This is done by checking if starting from the operation, we can reach the
  // inputs of the cluster
  bool isBeforeCluster(Operation *op) const {
    llvm::SmallVector<Operation *, 8> worklist;
    llvm::DenseSet<Operation *> visited;

    worklist.push_back(op);

    while (!worklist.empty()) {
      Operation *current = worklist.pop_back_val();

      if (visited.contains(current))
        continue;

      visited.insert(current);

      for (Value result : current->getResults()) {
        if (inputs.contains(result))
          return true;

        for (Operation *user : result.getUsers()) {
          worklist.push_back(user);
        }
      }
    }

    return false;
  }

  void print(llvm::raw_ostream &os) const {
    os << "Cluster: \n";
    os << "Inputs: ";
    for (auto input : inputs) {
      os << input << "\n";
    }
    os << "\nOutputs: ";
    for (auto output : outputs) {
      os << output << "\n";
    }
    os << "\nInternal Nodes: ";
    for (auto *node : internalNodes) {
      os << *node << "\n";
    }
    os << "\n";
  }
};

// This is a tree strcuture to represent the hierarchy of clusters
// Any 2 clusters can only be related as follows:
// 1. Completely disjoint
// 2. One cluster is completely enclosing the other
struct ClusterHierarchyNode {
  Cluster cluster;
  ClusterHierarchyNode *parent = nullptr;
  std::vector<ClusterHierarchyNode *> children;

  ClusterHierarchyNode(const Cluster &c) : cluster(c) {}

  bool isLeaf() const { return children.empty(); }
};

// Identify the clusters in the graph
// This is done by
// 1. Identifying the MUXes with common condition, the searching for BRANCHes
// with the same
//    condition (if/else and loop clusters)
// 2. Identifying the Braches where the consumer eventually leads to a
// sink/store (if statement)
std::vector<Cluster> identifyClusters(handshake::FuncOp funcOp,
                                      MLIRContext *ctx);

// Two clusters can only be related as follows
// 1. Completely disjoint
// 2. One cluster is completely enclosing the other
LogicalResult verifyClusters(std::vector<Cluster> &clusters);

// Retrieves the innermost nodes from the cluster hierarchy, which are the nodes
// that have no children. These nodes represent the lowest level of nesting in
// the hierarchy.
// Returns a vector of dynamically allocated ClusterHierarchyNode pointers,
std::vector<ClusterHierarchyNode *>
getInnermostNodes(const std::vector<ClusterHierarchyNode *> &nodes);

/// Builds a hierarchy of nested clusters from a flat list of clusters.
///
/// Assumes that any two clusters are either completely disjoint or one is
/// entirely contained within the other. The resulting hierarchy captures
/// this nesting relationship in a tree-like structure, where each node
/// represents a cluster and links to its nested (child) clusters.
// The return of this function is the leaf nodes of the hierarchy, which are
// the innermost clusters.
// Returns a vector of dynamically allocated ClusterHierarchyNode pointers,
std::vector<ClusterHierarchyNode *>
buildClusterHierarchy(std::vector<Cluster> &clusters);

} // namespace outoforder
} // namespace experimental
} // namespace dynamatic

#endif
