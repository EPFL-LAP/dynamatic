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

// Enum representing the structural type of a cluster.
// - LoopCluster: A cluster representing a loop construct.
// - IfElseCluster: A cluster representing an if-else conditional structure.
// - IfCluster: A cluster representing a simple if (no else) structure.
// - GlobalCluster: A top-level cluster containing the entire function body
//   except memory and end operations.
enum ClusterType { LoopCluster, IfElseCluster, IfClsuter, GlobalCluster };

struct Cluster {
  ClusterType type;
  bool markedOutOfOrder;
  llvm::DenseSet<Value> inputs;
  llvm::DenseSet<Value> outputs;
  llvm::DenseSet<Operation *> internalOps;

  Cluster(ClusterType type, llvm::DenseSet<Value> inputs,
          llvm::DenseSet<Value> outputs,
          llvm::DenseSet<Operation *> internalOps)
      : type(type), inputs(std::move(inputs)), outputs(std::move(outputs)),
        internalOps(std::move(internalOps)) {}
  Cluster() = default;
  Cluster(const Cluster &other) = default;
  Cluster &operator=(const Cluster &other) = default;

  bool operator==(const Cluster &other) const {
    return inputs == other.inputs && outputs == other.outputs &&
           internalOps == other.internalOps;
  }

  // Checks if the operation is inside the cluster
  bool isInsideCluster(Operation *op) const { return internalOps.contains(op); }

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

  /// Checks whether a Mux operation lies on the boundary of the cluster.
  /// A Mux is considered at the boundary if any of its data operands
  /// are among the cluster's
  // 1- Loop Cluster: input values.
  // 1- If/else Cluster: output values.
  bool isMuxAtBoundary(Operation *op) {
    if (handshake::MuxOp muxOp = dyn_cast<handshake::MuxOp>(op)) {
      if (type == ClusterType::LoopCluster) {
        for (Value muxOperand : muxOp.getDataOperands()) {
          if (inputs.contains(muxOperand))
            return true;
        }
      }
      if (type == ClusterType::IfElseCluster)
        return outputs.contains(muxOp.getResult());
    }
    return false;
  }

  /// Returns the MUX operations at the boundary of the cluster.
  /// A Mux is considered at the boundary if any of its data operands
  /// are among the cluster's
  // 1- Loop Cluster: input values.
  // 1- If/else Cluster: output values.
  llvm::DenseSet<handshake::MuxOp> getMuxesAtBoundary() {
    llvm::DenseSet<handshake::MuxOp> boundaryMuxes;
    for (Operation *op : internalOps) {
      if (isMuxAtBoundary(op))
        boundaryMuxes.insert(dyn_cast<handshake::MuxOp>(op));
    }
    return boundaryMuxes;
  }

  // Adds an operation to the cluster by inserting it into its internal ops
  void addInternalOp(Operation *op) { internalOps.insert(op); }

  // Removes an operation to the cluster by erasing it from its internal ops
  void removeInternalOp(Operation *op) { internalOps.erase(op); }

  // Replaces an old input with a new one in the cluster's input set.
  void replaceInput(Value oldInput, Value newInput) {
    for (Value input : inputs) {
      if (input == oldInput) {
        inputs.erase(oldInput);
        inputs.insert(newInput);
        break;
      }
    }
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
    for (auto *node : internalOps) {
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

  // Adds an operation to this cluster and all ancestor clusters
  void addInternalOp(Operation *op) {
    cluster.addInternalOp(op);

    if (parent)
      parent->addInternalOp(op);
  }

  // Removes an operation from this cluster and all ancestor clusters
  void removeInternalOp(Operation *op) {
    cluster.removeInternalOp(op);

    if (parent)
      parent->removeInternalOp(op);
  }

  // Recursively replaces an input in the current cluster and all child
  // clusters.
  void replaceInputInChildren(Value oldInput, Value newInput) {
    cluster.replaceInput(oldInput, newInput);

    for (ClusterHierarchyNode *child : children)
      child->replaceInputInChildren(oldInput, newInput);
  }
};

// Gets the constant feeding the init
Value getInitConstantInput(handshake::MergeOp mergeOp);

// Analyzes the MUXes in a handshake function and groups them by their
// conditions.
llvm::DenseMap<Value, std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
analyzeMuxConditions(handshake::FuncOp funcOp);

// Analyzes the branches and their conditions in a handshake function.
llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
analyzeBranchesConditions(
    handshake::FuncOp funcOp,
    llvm::DenseMap<Value,
                   std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
        &condToMuxes,
    std::vector<Cluster> &clusters);

// Creates clusters based on the identified MUXes and branches.
void createClusters(
    llvm::DenseMap<Value,
                   std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
        &condToMuxes,
    llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
        &condToBranches,
    std::vector<Cluster> &clusters);

// Creates a global cluster spanning the function's entire graph(except for
// end).
void createGlobalCluster(handshake::FuncOp funcOp,
                         std::vector<Cluster> &clusters);

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
