//===- OutOfOrderClustering.cpp - Out-of-Order Clustering Algorithm -*-
// C++-*-===//
//
// Implements the out-of-order clustering methodology
// https://dl.acm.org/doi/10.1145/3626202.3637556
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderClustering.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::outoforder;

/**
 * @brief Identifies control-flow-based clusters in a handshake function.
 * This function analyzes the handshake IR to group Muxes and
 * ConditionalBranches into clusters based on shared control conditions.
 * Clusters can represent:
 * - Loops: Muxes selected by a Merge fed by a constant.
 * - If/Else blocks: branches and muxes sharing a condition.
 * - If-only statements: branches without corresponding muxes leading to
 * sinks/stores.
 *
 * Each cluster contains:
 * - Inputs: operands entering the region (excluding condition producers).
 * - Outputs: values exiting the region.
 * - Internal nodes: operations within the region, excluding condition
 * generators.
 *
 * Additionally, a global outer cluster covering the full function is added.
 *
 * @param funcOp The handshake function to analyze.
 * @param ctx The MLIR context.
 *
 * @return Vector of identified clusters.
 */
std::vector<Cluster> outoforder::identifyClusters(handshake::FuncOp funcOp,
                                                  MLIRContext *ctx) {
  std::vector<Cluster> clusters;

  // Step 1: Identify the MUXes and their conditions
  llvm::DenseMap<Value, std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
      condToMuxes = analyzeMuxConditions(funcOp);

  // Step 2: Find all the branches that are fed by the each condition
  llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
      condToBranches = analyzeBranchesConditions(funcOp, condToMuxes, clusters);

  // Step 3: Create clusters for each condition
  createClusters(condToMuxes, condToBranches, clusters);

  // Step 4: Define an outer cluster which is the entire graph
  // Inputs: start
  // Outputs: the inputs of EndOp
  // Internal nodes: all the operations in the graph except the EndOp and Memory
  // Ops(Memory Controller and LSQ)
  llvm::DenseSet<Operation *> globalOps;
  llvm::DenseSet<Value> inputs;
  llvm::DenseSet<Value> outputs;

  for (auto &op : funcOp.getBody().getOps()) {
    if (handshake::EndOp end = dyn_cast<handshake::EndOp>(op)) {
      outputs.insert(end.getOperands().begin(), end.getOperands().end());
    } else if (isa<handshake::MemoryControllerOp>(op) ||
               isa<handshake::LSQOp>(op)) {
      continue;
    } else {
      globalOps.insert(&op);
    }
  }
  inputs.insert((Value)funcOp.getArguments().back());
  Cluster graphCluster(inputs, outputs, globalOps);
  clusters.push_back(graphCluster);

  return clusters;
}

/**
 * @brief Checks if a Mux is a Shannon Mux. A Mux is considered a Shannon Mux if
 * its output drives the select line of another Mux or a ConditionalBranch.
 *
 * @param muxOp The Mux operation to check.
 *
 * @return True if it's a Shannon Mux, false otherwise.
 */

static bool isShannonMux(handshake::MuxOp muxOp) {
  Value result = muxOp.getResult();

  // If the MUX feeds the select of another MUX or a BRANCH, then it is a
  // Shannon MUX
  for (Operation *user : result.getUsers()) {
    if (auto nextMux = dyn_cast<handshake::MuxOp>(user)) {
      if (nextMux.getSelectOperand() == result)
        return true;
    } else if (auto branch = dyn_cast<handshake::ConditionalBranchOp>(user)) {
      if (branch.getConditionOperand() == result)
        return true;
    }
  }

  return false;
}

/**
 * @brief Analyzes the MUXes in a handshake function and groups them by their
 * conditions.
 *
 * @param funcOp The handshake function to analyze.
 *
 * @return A map of conditions to MUXes.
 */
llvm::DenseMap<Value, std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
outoforder::analyzeMuxConditions(handshake::FuncOp funcOp) {
  // Map each condition value to all the MUXes that it feeds as the select
  // Each value will then correspond to a cluster
  // All Muxes being fed by the same value or its negation will be in the
  // same cluster
  // The bool is used to differentiate between loop and if/else clusters
  // True: loop cluster
  // False: if/else cluster
  llvm::DenseMap<Value, std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
      condToMuxes;

  // Step 1: Identify the MUXes and their conditions
  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    Value cond = muxOp.getSelectOperand();

    // Skip Shannon MUXes
    if (isShannonMux(muxOp))
      continue;

    //  c1 and NOT c1 all should be in 1 cluster
    if (isa<handshake::NotOp>(cond.getDefiningOp()))
      cond = cond.getDefiningOp()->getOperand(0);

    // If a MUX is fed by a MergeOp(INIT) fed by condition c (or NOT c), then
    // the MUX is a loop header
    if (handshake::MergeOp init =
            dyn_cast<handshake::MergeOp>(cond.getDefiningOp())) {
      Value op1 = init->getOperand(0);
      Value op2 = init->getOperand(1);

      // The Merge Op is fed by a constant and the condition c (or NOT c)
      if (isa<handshake::ConstantOp>(op1.getDefiningOp())) {
        cond = op2;
      } else {
        assert((isa<handshake::ConstantOp>(op2.getDefiningOp())) &&
               "An in input to a MergeOp feeding the MUX loop header should be "
               "a constant");
        cond = op1;
      }
      if (handshake::NotOp notOp =
              dyn_cast<handshake::NotOp>(cond.getDefiningOp())) {
        cond = notOp.getOperand();
      }
      condToMuxes[cond].second = true;
    }

    condToMuxes[cond].first.push_back(muxOp);
  }
  return condToMuxes;
}

/**
 * @brief Checks if a value eventually leads to a SinkOp or StoreOp.
 *
 * @param value     The starting value.
 * @param visited   Tracks visited values to prevent cycles.
 *
 * @return True if a SinkOp or StoreOp is reachable, false otherwise.
 */
static bool leadsToSinkOrStore(Value value, llvm::DenseSet<Value> &visited) {
  if (visited.contains(value))
    return false;

  visited.insert(value);

  // Check if any of the users of this value is a SinkOp or StoreOp
  for (auto *user : value.getUsers()) {
    if (isa<handshake::SinkOp>(user) || isa<handshake::StoreOp>(user))
      return true;

    // Recursively check the users of the user's results (if needed)
    for (auto userResult : user->getResults()) {
      if (leadsToSinkOrStore(userResult, visited))
        return true;
    }
  }

  return false;
}

/**
 * @brief Recursively finds all operations reachable from a given operation.
 *
 * @param op       Starting operation.
 * @param visited  Set of already visited operations.
 */
static void findReachableOps(Operation *op,
                             llvm::DenseSet<Operation *> &visited) {
  if (visited.contains(op))
    return;

  visited.insert(op);

  for (auto result : op->getResults()) {
    for (auto *user : result.getUsers()) {
      findReachableOps(user, visited);
    }
  }
}

/**
 * @brief Analyzes the branches and their conditions in a handshake function.
 * For branches witout corresponding MUXes, this function immediately creates
 * the "if statement" cluster containing the branch and all the reachable
 * operations from it.
 *
 * @param funcOp The handshake function to analyze.
 * @param condToMuxes A map of conditions to MUXes.
 * @param clusters A vector to store the identified clusters.
 *
 * @return A map of conditions to branches.
 */
llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
outoforder::analyzeBranchesConditions(
    handshake::FuncOp funcOp,
    llvm::DenseMap<Value,
                   std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
        &condToMuxes,
    std::vector<Cluster> &clusters) {
  llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
      condToBranches;
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    Value cond = branchOp.getConditionOperand();

    //  c1 and NOT c1 all should be in 1 cluster
    if (isa<handshake::NotOp>(cond.getDefiningOp()))
      cond = cond.getDefiningOp()->getOperand(0);

    // Case 1: if/else or loop cluster
    // Check if this cond also feeds a MUX
    if (condToMuxes.contains(cond)) {
      condToBranches[cond].push_back(branchOp);
    } else {
      // Case 2: if statement
      //  For branches without corresponding Muxes, we must first verify that
      //  this branch unlitamtely feeds a sink/store
      llvm::DenseSet<Value> visitedTrue;
      llvm::DenseSet<Value> visitedFalse;

      bool trueLeadsToSink =
          leadsToSinkOrStore(branchOp.getTrueResult(), visitedTrue);
      bool falseLeadsToSink =
          leadsToSinkOrStore(branchOp.getFalseResult(), visitedFalse);

      // Assert that the outputs of the branch lead to a sink/store
      assert((trueLeadsToSink && falseLeadsToSink) &&
             "Branch without Mux should lead to sink/store");

      llvm::DenseSet<Operation *> reachableOps;
      findReachableOps(branchOp, reachableOps);

      // Create a cluster for the branch with:
      // Inputs: the data operand of the branch
      // Outputs: none
      // Internal nodes: all the reachable operations from the branch
      Cluster ifCluster(llvm::DenseSet<Value>{branchOp.getDataOperand()},
                        llvm::DenseSet<Value>(), reachableOps);
      clusters.push_back(ifCluster);
    }
  }
  return condToBranches;
}

/**
 * @brief Retrieves the constant input to a MergeOp.
 *
 * @param mergeOp The MergeOp to analyze.
 *
 * @return The constant input to the MergeOp.
 */
static Value getInitConstantInput(handshake::MergeOp mergeOp) {
  // Get the constant input to the MergeOp
  Value constantInput = mergeOp.getOperand(0);
  if (auto constantOp =
          dyn_cast<handshake::ConstantOp>(constantInput.getDefiningOp())) {
    return constantInput;
  }
  return mergeOp.getOperand(1);
}

/**
 * @brief Recursively finds all operations between a start operation and a set
 * of end operations.
 *
 * @param currentOp The operation to start from.
 * @param end       Set of end operations to stop traversal.
 * @param visited   Set to store the discovered intermediate operations.
 */
static void findOpsBetweenOpsRecursive(Operation *currentOp,
                                       const llvm::DenseSet<Operation *> &end,
                                       llvm::DenseSet<Operation *> &visited) {

  if (visited.contains(currentOp))
    return;

  // Memory operations are excluded from the search
  if (isa<handshake::MemoryControllerOp>(currentOp) ||
      isa<handshake::LSQOp>(currentOp)) {
    return;
  }

  visited.insert(currentOp);

  if (end.contains(currentOp))
    return;

  // Recurse through users of each result
  for (Value result : currentOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      findOpsBetweenOpsRecursive(user, end, visited);
    }
  }
}

/**
 * @brief Creates clusters based on the identified MUXes and branches.
 *
 * @param condToMuxes A map of conditions to MUXes.
 * @param condToBranches A map of conditions to branches.
 * @param clusters A vector to store the created clusters.
 */
void outoforder::createClusters(
    llvm::DenseMap<Value,
                   std::pair<llvm::SmallVector<handshake::MuxOp, 4>, bool>>
        &condToMuxes,
    llvm::DenseMap<Value, llvm::SmallVector<handshake::ConditionalBranchOp, 4>>
        &condToBranches,
    std::vector<Cluster> &clusters) {
  // IMPORTANT REMARK: The generators of the condition are considered as
  // external to the cluster and NOT inside it. This is to ensure that the
  // clusters relationship property (disjoint, completely nested) is not
  // violated
  for (auto &pair : condToMuxes) {
    Value cond = pair.first;
    auto muxOps = pair.second.first;

    assert(condToBranches.contains(cond) && "Condition should have branches");

    auto branches = condToBranches[cond];

    llvm::DenseSet<Value> inputs;
    llvm::DenseSet<Value> outputs;
    llvm::DenseSet<Operation *> internalNodes;

    // Case 1.1 : loop cluster
    // Muxes then branches
    if (pair.second.second) {
      // Create a cluster for the condition with:
      // Inputs: the data operands and the selects of the MUXes
      // Outputs: the true and false outputs of the BRANCHes
      // Internal nodes: all the muxes and branches that are fed by this
      // condition along with the operations between them

      for (auto muxOp : muxOps) {
        inputs.insert(muxOp.getDataOperands().begin(),
                      muxOp.getDataOperands().end());
        // inputs.insert(muxOp.getSelectOperand());

        handshake::MergeOp init = dyn_cast<handshake::MergeOp>(
            muxOp.getSelectOperand().getDefiningOp());

        assert(
            init &&
            "The select operand of the MUX in loop header should be a MergeOp");

        inputs.insert(getInitConstantInput(init));

        llvm::DenseSet<Operation *> visited;
        auto branchOps = condToBranches[cond];
        llvm::DenseSet<Operation *> branchOpPtrs;
        for (auto branchOp : condToBranches[cond]) {
          branchOpPtrs.insert(branchOp.getOperation());
        }
        findOpsBetweenOpsRecursive(muxOp.getOperation(), branchOpPtrs, visited);
        internalNodes.insert(visited.begin(), visited.end());
      }

      for (auto branchOp : condToBranches[cond]) {
        outputs.insert(branchOp.getTrueResult());
        outputs.insert(branchOp.getFalseResult());
      }

      // The bachwerd edges between the BRANCHes and the MUXes are now part of
      // the inputs and outputs, but this is wrong We need to remove them from
      // the inputs and outputs

      // 1. Find the backward edges aka the common values between the inputs and
      // outputs
      llvm::DenseSet<Value> backwardEdges;
      for (Value v : inputs) {
        if (outputs.contains(v))
          backwardEdges.insert(v);
      }

      // 2. Erase the backward edges from the cluster's inputs and outputs
      for (Value v : backwardEdges) {
        inputs.erase(v);
        outputs.erase(v);
      }

    } else {
      // Case 1.2 : if/else statement
      // BRANCHes then MUXes

      // Create a cluster for the condition with:
      // Inputs: the data operands and the select of the BRANCHes
      // Outputs: the outputs of the MUXes
      // Internal nodes: all the MUXes and BRANCHes that are fed by this
      // condition along with the operations between them

      for (auto branchOp : condToBranches[cond]) {
        inputs.insert(branchOp.getDataOperand());
        inputs.insert(branchOp.getConditionOperand());

        llvm::DenseSet<Operation *> visited;
        llvm::DenseSet<Operation *> muxOpPtrs;
        for (auto muxOp : muxOps) {
          muxOpPtrs.insert(muxOp.getOperation());
        }
        findOpsBetweenOpsRecursive(branchOp.getOperation(), muxOpPtrs, visited);
        internalNodes.insert(visited.begin(), visited.end());
      }

      for (auto muxOp : muxOps) {
        outputs.insert(muxOp.getResult());
      }
    }
    Cluster cluster(inputs, outputs, internalNodes);
    clusters.push_back(cluster);
  }
}

/**
 * @brief Verifies that all clusters are either disjoint or properly nested. Any
 * partial overlap (i.e., shared operations without full containment) is
 * considered invalid and results in failure.
 *
 * @param clusters Vector of clusters to verify.
 *
 * @return Success if all clusters are valid, failure otherwise.
 */
LogicalResult outoforder::verifyClusters(std::vector<Cluster> &clusters) {
  for (size_t i = 0; i < clusters.size(); ++i) {
    const auto &aOps = clusters[i].internalNodes;
    for (size_t j = i + 1; j < clusters.size(); ++j) {

      const auto &bOps = clusters[j].internalNodes;

      // Check intersection: if two clusters share any operations
      bool intersects = false;
      for (Operation *op : aOps) {
        if (bOps.contains(op)) {
          intersects = true;
          break;
        }
      }

      if (intersects) {
        // Check if one is fully contained in the other
        bool aInB = llvm::all_of(
            aOps, [&](Operation *op) { return bOps.contains(op); });

        bool bInA = llvm::all_of(
            bOps, [&](Operation *op) { return aOps.contains(op); });

        // Not nested: invalid overlap
        assert((aInB || bInA) && "Clusters are not nested: invalid overlap");
        if (!aInB && !bInA)
          return failure();
      }
    }
  }
  return success();
}

/**
 * @brief Builds a hierarchy of cluster nodes based on nesting relationships.
 *
 * @param clusters The list of disjoint or nested clusters.
 *
 * @return A list of ClusterHierarchyNode pointers representing the
 * hierarchy in increasing size of clusters(from innermost to outermost).
 * @note Assumes clusters are either disjoint or properly nested.
 */
std::vector<ClusterHierarchyNode *>
outoforder::buildClusterHierarchy(std::vector<Cluster> &clusters) {
  // Sort clusters by size (number of internal nodes)
  // This is important to ensure that the innermost clusters are processed first
  llvm::sort(clusters.begin(), clusters.end(),
             [](const Cluster &a, const Cluster &b) {
               return a.internalNodes.size() < b.internalNodes.size();
             });

  std::vector<ClusterHierarchyNode *> nodes;

  // Pre-allocate memory
  nodes.reserve(clusters.size());

  // Create node wrappers
  for (const auto &cluster : clusters)
    nodes.push_back(new ClusterHierarchyNode(cluster));

  // For each node, find its immediate parent
  // We know that since the clusters are sorted by size, then every 2
  // consecutive clusters Ci and Cj can be related as follows:
  // 1. Ci is completely disjoint from Cj
  // 2. Ci is completely enclosed inside Cj
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      // Check if Ci is completely enclosed inside Cj
      if (llvm::all_of(nodes[i]->cluster.internalNodes, [&](Operation *op) {
            return nodes[j]->cluster.internalNodes.contains(op);
          })) {

        // If so, set the parent of Ci to Cj
        // and add Ci as a child of Cj
        nodes[i]->parent = nodes[j];
        nodes[j]->children.push_back(nodes[i]);
        break; // Only one parent possible
      }
    }
  }

  return nodes;
}
