//===- OutOfOrderExecution.cpp - Out-of-Order Execution Algorithm -*-
// C++-*-===//
//
// Implements the out-of-order execution methodology
// https://dl.acm.org/doi/10.1145/3626202.3637556
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderExecution.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderClustering.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace dynamatic::experimental::outoforder;

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::outoforder;

namespace {
struct OutOfOrderExecutionPass
    : public dynamatic::experimental::outoforder::impl::OutOfOrderExecutionBase<
          OutOfOrderExecutionPass> {

  void runDynamaticPass() override;

private:
  // Removes any extra signals (tags, spec bits) from the select operand of each
  // MUX in the function.
  LogicalResult removeExtraSignalsFromMux(FuncOp funcOp, MLIRContext *ctx);

  // Step 1.1: Identify dirty nodes
  LogicalResult identifyDirtyNodes(
      const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
      const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
      ClusterHierarchyNode *clusterNode,
      llvm::DenseSet<Operation *> &dirtyNodes);

  // Step 1.2: Identfy unaligned edges
  LogicalResult
  identifyUnalignedEdges(const llvm::DenseSet<Value> &outOfOrderNodeInputs,
                         const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
                         ClusterHierarchyNode *clusterNode,
                         llvm::DenseSet<Operation *> &dirtyNodes,
                         llvm::DenseSet<Value> &unalignedEdges);

  // Applies out-of-order execution by tagging unaligned channels and
  // inserting sync ops, optionally recursing up the cluster hierarchy.
  void traverseGraph(OpOperand &opOperand, ClusterHierarchyNode *clusterNode,
                     llvm::DenseSet<Operation *> &dirtyNodes);

  // Step 1.3: Identify tagged edges; i.e. the edges that should receive a
  // tag
  LogicalResult
  identifyTaggedEdges(const llvm::DenseSet<Value> &outOfOrderNodeInputs,
                      const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
                      llvm::DenseSet<Value> &unalignedEdges,
                      llvm::DenseSet<Value> &taggedEdges);

  // Step 1.4: Add the tagger operations and connect them to the freeTagsFifo
  // and consumers Returns the output of teh fitrst tagger added to be used as
  // the select of the Aligner in case of a Controlled Aligner
  Value addTaggers(const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
                   OpBuilder builder, ClusterHierarchyNode *clusterNode,
                   FreeTagsFifoOp &freeTagsFifo,
                   llvm::DenseSet<Operation *> &dirtyNodes,
                   llvm::DenseSet<Value> &unalignedEdges,
                   llvm::DenseSet<Value> &taggedEdges);

  // Add the untagger operations and connect them to consumers
  // DEPRECATED (now part of addAligner)
  void addUntaggers(OpBuilder builder, ClusterHierarchyNode *clusterNode,
                    FreeTagsFifoOp &freeTagsFifo,
                    llvm::DenseSet<Value> &unalignedEdges,
                    llvm::DenseSet<Operation *> &untaggers);

  // Step 1.5: Add the aligner
  void addAligner(OpBuilder builder, ClusterHierarchyNode *clusterNode,
                  Value select, int numTags, FreeTagsFifoOp &freeTagsFifo,
                  llvm::DenseSet<Value> &unalignedEdges,
                  llvm::DenseSet<Operation *> &untaggers);

  // Step 1.6: Adds the FreeTagsFifo, Tagger and Untagger operations
  // Returns the FreeTagsFifo operation which uniquely identifies the tagged
  // region
  Operation *addTagOperations(
      const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
      const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
      OpBuilder builder, ClusterHierarchyNode *clusterNode, int numTags,
      bool controlled, llvm::DenseSet<Operation *> &dirtyNodes,
      llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges,
      llvm::DenseSet<Operation *> &untaggers);

  // Step 1.7: Tag the channels in the tagged region
  LogicalResult
  addTagSignalsToTaggedRegion(FuncOp funcOp, Operation *freeTagsFifo,
                              int numTags, const std::string &extraTag,
                              llvm::DenseSet<Operation *> &untaggers);

  // Recursively tags channels in the graph from a given operand,
  // stopping at control boundaries and handling special ops.
  LogicalResult addTagSignalsRecursive(OpOperand &opOperand,
                                       llvm::DenseSet<Operation *> &visited,
                                       const std::string &extraTag,
                                       Operation *freeTagsFifo,
                                       llvm::DenseSet<Operation *> &untaggers,
                                       int numTags);

  // Applies the out-of-order execution methodology to a single out-of-order
  // node
  LogicalResult applyOutOfOrderAlgorithm(
      FuncOp funcOp, MLIRContext *ctx,
      const llvm::DenseSet<Value> &outOfOrderNodeInputs,
      const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
      const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
      ClusterHierarchyNode *clusterNode, int &tagIndex, int numTags,
      bool controlled);

  // Applies the out-of-order execution methodology for a MUX
  LogicalResult applyMuxToCMerge(FuncOp funcOp, MLIRContext *ctx, MuxOp muxOp,
                                 ClusterHierarchyNode *clusterNode,
                                 int &tagIndex, int numTags, bool controlled);

  // Converts a loop header MUX to a CMERGE in a loop cluster
  LogicalResult applyMuxToCMergeLoopCluster(FuncOp funcOp, MLIRContext *ctx,
                                            MuxOp muxOp,
                                            ClusterHierarchyNode *clusterNode);

  // Converts all the MUXes to a MERGEs in an if/else cluster
  LogicalResult
  applyMuxToCMergeIfElseCluster(FuncOp funcOp, MLIRContext *ctx,
                                ClusterHierarchyNode *clusterNode);

  // Finds the innermost (smallest) cluster that contains an operation
  ClusterHierarchyNode *findInnermostClusterContainingOp(
      Operation *outOfOrderNode,
      const std::vector<ClusterHierarchyNode *> &hierarchyNodes);

  // MAIN: Applies the out-of-order execution methodology to all the
  // out-of-order nodes
  LogicalResult applyOutOfOrder(
      FuncOp funcOp, MLIRContext *ctx,
      llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes);

  // Reads the out-of-order nodes from a fle and find the corresponding
  // operations in the function
  LogicalResult readOutOfOrderNodes(
      FuncOp funcOp,
      llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes);
};
} // namespace

void OutOfOrderExecutionPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  for (auto funcOp : module.getOps<FuncOp>()) {
    // Each out of order node will have:
    // (1) numTags: int representing the number of tags
    // (2) controlled: bool representing wether the aligner is controlled
    llvm::DenseMap<Operation *, std::pair<int, bool>> outOfOrderNodes;

    // Step 1: Read the out-of-order nodes from the file and find the
    // corresponding operations in the function
    if (failed(readOutOfOrderNodes(funcOp, outOfOrderNodes)))
      signalPassFailure();

    // Step 2: Apply the out-of-order execution methodology to each out-of-order
    // node
    if (!outOfOrderNodes.empty()) {
      if (failed(applyOutOfOrder(funcOp, ctx, outOfOrderNodes)))
        signalPassFailure();
    }
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::outoforder::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}

/**
 * @brief Reads the out-of-order nodes from a file and finds the corresponding
 * operations in the function. The file should contain lines with the format:
 * <operation_name> <num_tags> <controlled>
 *
 * @param funcOp The function operation to which the out-of-order execution
 *        should be applied.
 * @param outOfOrderNodes A map of the out-of-order nodes, where the key is the
 *        operation and the value is a pair containing:
 *  - An integer specifying the number of tags for out-of-order processing.
 *  - A boolean flag indicating whether the aligner should be controlled
 *
 * @return Success if the out-of-order nodes are successfully read and
 *         corresponding operations are found, failure otherwise.
 */
LogicalResult OutOfOrderExecutionPass::readOutOfOrderNodes(
    FuncOp funcOp,
    llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes) {

  std::ifstream file("experimental/lib/Transforms/"
                     "OutOfOrderExecution/OutOfOrderOps.txt");
  if (!file.is_open()) {
    llvm::errs() << "Error opening file\n";
    return failure();
  }

  std::string outOfOrderOpName;
  int numTags;
  bool controlled;

  while (file >> outOfOrderOpName >> numTags >> controlled) {
    // Find the operation in the function by name
    Operation *outOfOrderOp = nullptr;
    funcOp.walk([&](Operation *op) -> WalkResult {
      // Get the name of the current operation
      StringAttr nameAttr = op->getAttrOfType<StringAttr>("handshake.name");

      // Continue walking if the name attribute
      // is missing
      if (!nameAttr)
        return WalkResult::advance();

      // Check if the operation name matches the out-of-order operation name
      if (nameAttr.getValue() == outOfOrderOpName) {
        outOfOrderOp = op;

        // Stop walking once the operation is found
        return WalkResult::interrupt();
      }

      // Continue walking otherwise
      return WalkResult::advance();
    });

    // If the operation was not found in the function, then there is an error
    if (!outOfOrderOp) {
      return failure();
    }

    outOfOrderNodes[outOfOrderOp] = {numTags, controlled};
  }

  if (file.bad()) {
    llvm::errs() << "I/O error while reading\n";
    return failure();
  }
  if (!file.eof()) {
    llvm::errs() << "Malformed line or wrong data format\n";
    return failure();
  }
  file.close();
  return success();
}

/**
 * @brief Applies out-of-order execution to a set of operations within a
 * function. This method identifies clusters of operations within the given
 * function, verifies the validity of these clusters, and then applies the
 * out-of-order execution algorithm to the operations that require it. The
 * out-of-order nodes are processed by first identifying the innermost cluster
 * containing each operation and then applying the out-of-order logic to each
 * operation with respect to its cluster.
 *
 * @param funcOp The function operation to which the out-of-order execution
 *        should be applied.
 * @param ctx The MLIR context used for building and manipulating operations.
 * @param outOfOrderNodes A map of operations that require out-of-order
 * execution.
 * The map’s key is an operation, and the value is a pair containing:
 *  - An integer specifying the number of tags for out-of-order processing.
 *  - A boolean flag indicating whether the aligner should be controlled
 *
 * @return Success if the out-of-order execution is successfully
 * applied to all operations, failure otherwise.
 */
LogicalResult OutOfOrderExecutionPass::applyOutOfOrder(
    FuncOp funcOp, MLIRContext *ctx,
    llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes) {

  // Step 1: Identify the clusters
  std::vector<Cluster> clusters = identifyClusters(funcOp, ctx);

  // llvm::errs() << "clusters: \n";
  // for (auto &clusters : clusters) {
  //   clusters.print(llvm::errs());
  //   llvm::errs() << "\n";
  // }
  // llvm::errs() << "Done printing clusters.\n";
  // llvm::errs() << "\n";
  // llvm::errs() << "\n";

  // Step 2: Check validity of the clusters
  if (failed(verifyClusters(clusters)))
    return failure();

  // Step 3: Build the cluster hierarchy, returning the all the nodes of the
  // tree from innermost to outermost
  std::vector<ClusterHierarchyNode *> hierarchyNodes =
      buildClusterHierarchy(clusters);

  llvm::errs() << "Hierarchy nodes: \n";
  for (auto &clusterNode : hierarchyNodes) {
    clusterNode->cluster.print(llvm::errs());
    llvm::errs() << "\n";
  }
  llvm::errs() << "Done printing hierarchy nodes.\n";
  llvm::errs() << "\n";
  llvm::errs() << "\n";

  // Step 4: Apply the out-of-order execution methodology to each ot-of-order
  // node with respect to each innermost cluster
  int tagIndex = 0;

  for (auto &[op, attributes] : outOfOrderNodes) {
    auto &[numTags, controlled] = attributes;

    ClusterHierarchyNode *innermostCluster =
        findInnermostClusterContainingOp(op, hierarchyNodes);

    // The entire graph is a cluster. So if the out-of-order node is not inside
    // any cluster, then it must not be inside the graph
    assert(innermostCluster &&
           "The out-of-order node should be inside the graph");

    if (MuxOp muxOp = dyn_cast<MuxOp>(op)) {
      if (failed(applyMuxToCMerge(funcOp, ctx, muxOp, innermostCluster,
                                  tagIndex, numTags, controlled)))
        return failure();
    } else {
      if (failed(applyOutOfOrderAlgorithm(
              funcOp, ctx,
              llvm::DenseSet<Value>(op->getOperands().begin(),
                                    op->getOperands().end()),
              llvm::DenseSet<Value>(op->getResults().begin(),
                                    op->getResults().end()),
              llvm::DenseSet<Operation *>({op}), innermostCluster, tagIndex,
              numTags, controlled)))
        return failure();
    }
  }

  // Free the cluster hierarchy nodes
  for (auto &clusterNode : hierarchyNodes) {
    delete clusterNode;
  }

  funcOp->print(llvm::errs());

  // Remove the extra signals from the select of the MUXes
  if (failed(removeExtraSignalsFromMux(funcOp, ctx)))
    return failure();

  llvm::errs() << "\n";
  llvm::errs() << "\n";
  funcOp->print(llvm::errs());

  return success();
}

/**
 * @brief Replaces a MuxOp with a ConditionalMerge (CMERGE) or a MERGE if it
 *        lies on the boundary of a loop or if/else cluster, enabling
 * out-of-order execution.
 *
 * This function determines whether the provided MuxOp lies at the boundary of a
 * LoopCluster or IfElseCluster and applies the appropriate transformation:
 * - LoopCluster: Replaces the MuxOp with a CMERGE and uses its index output to
 *   drive the rest of the Muxes at the cluster boundary.
 * - IfElseCluster: Converts all Muxes at the cluster boundary into MERGEs.
 *
 * If `controlled` is false, the function recursively invokes the out-of-order
 * execution algorithm on the parent cluster to propagate tagging and execution
 * adjustments.
 *
 * @param funcOp       The function being transformed.
 * @param ctx          The MLIR context.
 * @param muxOp        The MuxOp to be replaced.
 * @param clusterNode  The innermost cluster containing the MuxOp.
 * @param tagIndex     The current tag index (used for naming tags).
 * @param numTags      The total number of tags available for tagging
 * operations.
 * @param controlled   Whether the transformation is controlled or free-form.
 * @return LogicalResult indicating success or failure of the transformation.
 */
LogicalResult OutOfOrderExecutionPass::applyMuxToCMerge(
    FuncOp funcOp, MLIRContext *ctx, MuxOp muxOp,
    ClusterHierarchyNode *clusterNode, int &tagIndex, int numTags,
    bool controlled) {
  // If the MUX is at a boundary of a loop cluster, then we replace the MUX with
  // a CMERGE
  bool loopMux = clusterNode->cluster.type == ClusterType::LoopCluster &&
                 clusterNode->cluster.isMuxAtBoundary(muxOp);
  bool ifElseMux = clusterNode->cluster.type == ClusterType::IfElseCluster &&
                   clusterNode->cluster.isMuxAtBoundary(muxOp);
  if (loopMux || ifElseMux) {
    if (loopMux) {
      if (failed(applyMuxToCMergeLoopCluster(funcOp, ctx, muxOp, clusterNode)))
        return failure();
    } else {
      if (failed(applyMuxToCMergeIfElseCluster(funcOp, ctx, clusterNode)))
        return failure();
    }
    if (!controlled) {
      if (failed(applyOutOfOrderAlgorithm(
              funcOp, ctx, clusterNode->cluster.inputs,
              clusterNode->cluster.outputs, clusterNode->cluster.internalOps,
              clusterNode->parent, tagIndex, numTags, controlled)))
        return failure();
    }
  }
  return success();
}

/**
 * @brief Replaces a MuxOp with a ControlMerge (CMERGE) in a loop cluster.
 *
 * This function replaces a MuxOp at the boundary of a loop cluster with a
 * CMERGE, updating the select operands of other boundary MuxOps to the CMERGE’s
 * index output. It also removes the INIT operation and its constant input if
 * they are only used by the INIT, and deletes the MuxOp.
 *
 * @param funcOp       The function being transformed.
 * @param ctx          The MLIR context.
 * @param muxOp        The MuxOp to be replaced.
 * @param clusterNode  The cluster node containing the loop cluster.
 * @return LogicalResult indicating success or failure.
 *
 * @note This transformation enables out-of-order execution within the loop
 * cluster.
 */
LogicalResult OutOfOrderExecutionPass::applyMuxToCMergeLoopCluster(
    FuncOp funcOp, MLIRContext *ctx, MuxOp muxOp,
    ClusterHierarchyNode *clusterNode) {

  OpBuilder builder(ctx);
  builder.setInsertionPoint(muxOp);

  // Create a CMERGE that takes the same inputs as the MUX, and add it to the
  // cluster
  ControlMergeOp cmerge = builder.create<ControlMergeOp>(
      muxOp->getLoc(), muxOp.getDataResult().getType(),
      muxOp.getSelectOperand().getType(), muxOp.getDataOperands());
  clusterNode->cluster.internalOps.insert(cmerge);

  // Feed the index output of the CMERGE to the select of the rest of the
  // MUXes at the boundary of teh cluster
  llvm::DenseSet<handshake::MuxOp> boundaryMuxes =
      clusterNode->cluster.getMuxesAtBoundary();

  for (MuxOp mux : boundaryMuxes) {
    if (mux != muxOp) {
      mux.getOperation()->replaceUsesOfWith(mux.getSelectOperand(),
                                            cmerge.getIndex());
    }
  }

  // Get the INIT driving the select of the MUXes
  MergeOp init = dyn_cast<MergeOp>(muxOp.getSelectOperand().getDefiningOp());
  if (!init)
    return failure();

  // Now we replace the MUX's connections to the rest of teh circuit with the
  // output of the CMERGE and delet the MUX
  muxOp.getDataResult().replaceAllUsesWith(cmerge.getDataResult());
  muxOp.erase();

  // If the constant feeding the INIT is only used by this INIT and no other
  // operation, then we should delete it
  Value cnstFeedingInit = getInitConstantInput(init);
  size_t numUsers = std::distance(cnstFeedingInit.getUsers().begin(),
                                  cnstFeedingInit.getUsers().end());
  if (numUsers == 1) {
    Operation *constant = cnstFeedingInit.getDefiningOp();
    // Remove the constant from the parent clusters and rom the inputs of the
    // current node
    clusterNode->cluster.inputs.erase(cnstFeedingInit);
    clusterNode->removeInternalOp(constant);
    constant->erase();
  }

  // Now remove the INIT from the cluster and delete it
  clusterNode->removeInternalOp(init);
  init->erase();

  return success();
}

/**
 * @brief Replaces boundary MuxOps with MergeOps in an IfElse cluster.
 *
 * This function replaces all MuxOps at the boundary of an IfElse cluster with
 * corresponding MergeOps, enabling out-of-order execution. The MuxOps are
 * removed and replaced with the MergeOps within the cluster.
 *
 * @param funcOp       The function being transformed.
 * @param ctx          The MLIR context.
 * @param muxOp        The MuxOp to be replaced (though not directly used).
 * @param clusterNode  The cluster node containing the IfElse cluster.
 * @return LogicalResult indicating success or failure.
 */
LogicalResult OutOfOrderExecutionPass::applyMuxToCMergeIfElseCluster(
    FuncOp funcOp, MLIRContext *ctx, ClusterHierarchyNode *clusterNode) {
  ConversionPatternRewriter rewriter(ctx);

  // Convert all the MUXes at the boundaries of the cluster with a merge
  // And replace the MUXes with the merges in the cluster
  llvm::DenseSet<handshake::MuxOp> boundaryMuxes =
      clusterNode->cluster.getMuxesAtBoundary();
  rewriter.setInsertionPoint(*(boundaryMuxes.begin()));
  for (MuxOp mux : boundaryMuxes) {
    MergeOp merge =
        rewriter.create<MergeOp>(mux->getLoc(), mux.getDataOperands());
    clusterNode->addInternalOp(merge);
    clusterNode->removeInternalOp(mux);
    rewriter.replaceOp(mux, merge);
  }

  return success();
}

static bool isBackwardEdgeFromBranch(Value v) {
  if (auto branchOp = v.getDefiningOp<ConditionalBranchOp>()) {
    return true;
  }
  if (auto notOp = v.getDefiningOp<NotOp>()) {
    return isBackwardEdgeFromBranch(notOp.getOperand());
  }
  return false;
}

static bool isInit(Operation *op) {
  if (MergeOp mergeOp = dyn_cast<MergeOp>(op)) {
    Value op1 = op->getOperand(0);
    Value op2 = op->getOperand(1);
    bool ed1 =
        isa<ConstantOp>(op1.getDefiningOp()) && isBackwardEdgeFromBranch(op2);
    bool ed2 =
        isa<ConstantOp>(op2.getDefiningOp()) && isBackwardEdgeFromBranch(op1);
    if (ed1 && ed2) {
      for (auto *user : mergeOp.getResult().getUsers()) {
        if (isa<ConditionalBranchOp>(user)) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * @brief Applies the out-of-order execution algorithm within a cluster. The
 * algorithm identifies dirty nodes, unaligned edges, and tagged edges, then
 * adds tagger and untagger operations. The function operates recursively on the
 * cluster hierarchy.
 *
 * @param funcOp The function to which the algorithm is applied.
 * @param ctx The MLIR context.
 * @param outOfOrderNodeInputs The inputs of the out-of-order operation.
 * @param outOfOrderNodeOutputs The outputs of the out-of-order operation.
 * @param outOfOrderNodeInternalOps Internal operations within the out-of-order
 * node.
 * @param clusterNode The current cluster node in the hierarchy.
 * @param tagIndex The current tag index(tag name).
 * @param numTags The number of tags to be used.
 * @param controlled Flag indicating whether the execution is controlled.
 *
 * @return Success if out-of-order execution was applied correctly, failure
 * otherwise.
 */
LogicalResult OutOfOrderExecutionPass::applyOutOfOrderAlgorithm(
    FuncOp funcOp, MLIRContext *ctx,
    const llvm::DenseSet<Value> &outOfOrderNodeInputs,
    const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
    const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
    ClusterHierarchyNode *clusterNode, int &tagIndex, int numTags,
    bool controlled) {

  // We don't need to align the the main graph cluster (which has no cluster
  // enclosing it)
  if (!clusterNode)
    return success();

  // Step 1: Identify the dirty nodes
  llvm::DenseSet<Operation *> dirtyNodes;
  // llvm::DenseSet<Operation *> outOfOrderNodeOutputsOps;
  // for (Value output : outOfOrderNodeOutputs) {
  //   Operation *op = output.getDefiningOp();
  //   if (op)
  //     outOfOrderNodeOutputsOps.insert(op);
  // }
  if (!controlled) {
    if (failed(identifyDirtyNodes(outOfOrderNodeOutputs,
                                  outOfOrderNodeInternalOps, clusterNode,
                                  dirtyNodes)))
      return failure();
  }

  // llvm::errs() << "Dirty nodes: \n";
  // for (auto *node : dirtyNodes) {
  //   llvm::errs() << *node << "\n";
  // }
  // llvm::errs() << "Done printing dirty nodes.\n";

  // If there are no dirty nodes and there is free alignment, then we don't need
  // to do anything.
  // In the case of a controlled aligner, the dirty nodes will always be empty
  if ((!controlled && !dirtyNodes.empty()) || controlled) {

    // Step 2: Identify the unaligned edges
    llvm::DenseSet<Value> unalignedEdges;
    if (failed(identifyUnalignedEdges(outOfOrderNodeInputs,
                                      outOfOrderNodeOutputs, clusterNode,
                                      dirtyNodes, unalignedEdges)))
      return failure();

    // llvm::errs() << "Unaligned edges: \n";
    // for (auto edge : unalignedEdges) {
    //   llvm::errs() << edge << "\n";
    // }
    // llvm::errs() << "Done printing unaligned edges.\n";

    // If there are no unaligned edges, then we don't need to do
    // anything
    if ((!controlled && unalignedEdges.size() > 1) || controlled) {

      // Step 3: Identify the tagged edges
      llvm::DenseSet<Value> taggedEdges;
      if (failed(identifyTaggedEdges(outOfOrderNodeInputs,
                                     outOfOrderNodeOutputs, unalignedEdges,
                                     taggedEdges)))
        return failure();

      // llvm::errs() << "taggedEdges edges: \n";
      // for (auto edge : taggedEdges) {
      //   llvm::errs() << edge << "\n";
      // }
      // llvm::errs() << "Done printing taggedEdges edges.\n";
      // llvm::errs() << "\n";
      // llvm::errs() << "\n";

      // Step 4: Add the tagger and untagger operations and connect them to the
      // freeTagsFifo
      OpBuilder builder(ctx);
      Operation *op = *outOfOrderNodeInternalOps.begin();
      builder.setInsertionPoint(op);

      llvm::DenseSet<Operation *> untaggers;
      Operation *freeTagsFifo =
          addTagOperations(outOfOrderNodeOutputs, outOfOrderNodeInternalOps,
                           builder, clusterNode, numTags, controlled,
                           dirtyNodes, unalignedEdges, taggedEdges, untaggers);

      if (!freeTagsFifo)
        return failure();

      // Step 5: Tag the channels in the tagged region
      std::string extraTag = "tag" + std::to_string(tagIndex++);
      if (failed(addTagSignalsToTaggedRegion(funcOp, freeTagsFifo, numTags,
                                             extraTag, untaggers)))
        return failure();
    }
  }

  // Step 6: Recursively apply the out-of-order algorithm to the parent cluster
  // Now we consider the cluster as an out-of-order node inside its parent
  // cluster
  // If we are applying controlled alignmet, then we're following the program
  // order so no need to apply hierarchical alignment
  if (!controlled && !clusterNode->cluster.markedOutOfOrder) {
    clusterNode->cluster.markedOutOfOrder = true;
    if (failed(applyOutOfOrderAlgorithm(
            funcOp, ctx, clusterNode->cluster.inputs,
            clusterNode->cluster.outputs, clusterNode->cluster.internalOps,
            clusterNode->parent, tagIndex, numTags, controlled)))
      return failure();
  }

  return success();
}

/**
 * @brief Identifies dirty nodes in the graph relative to a cluster.
 * Starts from an out-of-order node's outputs and traverses the graph to mark
 * the reachable nodes, while respecting cluster boundaries.

 * @param outOfOrderNodeOutputs Set of output values from out-of-order node.
 * @param outOfOrderNodeInternalOps Set of internal operations from out-of-order
  * operation.
 * @param clusterNode Current cluster context.
 * @param dirtyNodes Set to populate with affected (dirty) operations.

 * @return Success result.
 * @note The starting operation itself is excluded from the dirty set.
*/
LogicalResult OutOfOrderExecutionPass::identifyDirtyNodes(
    const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
    const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
    ClusterHierarchyNode *clusterNode,
    llvm::DenseSet<Operation *> &dirtyNodes) {
  llvm::DenseSet<Value> visited;
  for (Value output : outOfOrderNodeOutputs) {
    for (auto &operand : output.getUses()) {
      traverseGraph(operand, clusterNode, dirtyNodes);
    }
  }

  for (Operation *op : outOfOrderNodeInternalOps) {
    dirtyNodes.erase(op);
  }
  return success();
}

/**
 * @brief Traverses the dataflow graph from a set of outputs to identify and
 mark
 * "dirty" operations within a cluster hierarchy (operations reachable from the
 * out-of-order operation).

 * @param outOfOrderNodeOutputs Set of output values from out-of-order node,
 * which form the starting point of the graph traversal.
 * @param clusterNode The current cluster node being analyzed; used to enforce
 * cluster boundary conditions during traversal.
 * @param dirtyNodes A set that is populated with operations deemed "dirty"
 during
 * the traversal (i.e., affected by the out-of-order operation).

 * @note Memory Operations (Memory Controller & LSQ) Control and End operations
 * should not be marked as a dirty node
*/
void OutOfOrderExecutionPass::traverseGraph(
    OpOperand &opOperand, ClusterHierarchyNode *clusterNode,
    llvm::DenseSet<Operation *> &dirtyNodes) {

  Operation *op = opOperand.getOwner();

  if (!op)
    return;

  // Memory Operations (Memory Controller & LSQ) Control and End operations
  // should not be marked as a dirty node
  // MUXes should also not be marked as dirty nodes because they naturally
  // allign their inputs according to their select irrespectively of the tags
  if (dirtyNodes.find(op) != dirtyNodes.end() || isa<MemoryControllerOp>(op) ||
      isa<LSQOp>(op) || isa<EndOp>(op) || isa<MuxOp>(op))
    return;

  // Case 1: We need to stop checking at the outputs of the cluster
  // Stop DFS at boundaries of teh cluster
  if (!clusterNode->cluster.outputs.contains(opOperand.get())) {
    // Case 2: If the cluster has any children clusters nested inside, we need
    // to keep these clusters closed. So whenever we encounter an input
    // to any of these clusters, we skip to their outputs
    bool inputToChildCluster = false;
    for (auto *childCluster : clusterNode->children) {
      if (childCluster->cluster.inputs.contains(opOperand.get())) {
        inputToChildCluster = true;

        dirtyNodes.insert(childCluster->cluster.internalOps.begin(),
                          childCluster->cluster.internalOps.end());

        for (Value output : childCluster->cluster.outputs) {
          for (auto &operand : output.getUses()) {
            traverseGraph(operand, clusterNode, dirtyNodes);
          }
        }
      }
    }

    if (!inputToChildCluster) {
      dirtyNodes.insert(op);

      // Case 3:Continue traversing the graph through the operations reachable
      // from the current value. This is done by traversing the users of the
      // value
      for (Value output : op->getResults()) {
        for (auto &operand : output.getUses()) {
          traverseGraph(operand, clusterNode, dirtyNodes);
        }
      }
    }
  }
}

/**
 * @brief Identifies unaligned edges between dirty and non-dirty nodes.
 * Marks operands of dirty nodes that are produced by non-dirty operations, as
 well
 * as outputs of the out-of-order node that are not consumed by memory
 operations.

 * @param outOfOrderNodeInputs Set of input values from out-of-order node.
 * @param outOfOrderNodeOutputs Set of output values from out-of-order node.
 * @param dirtyNodes Set of operations already marked as dirty.
 * @param unalignedEdges Set to be populated with values representing unaligned
 * edges.

 * @return Success result.
 * @note Edges to memory controllers and LSQs are excluded from unaligned edges.
*/

LogicalResult OutOfOrderExecutionPass::identifyUnalignedEdges(
    const llvm::DenseSet<Value> &outOfOrderNodeInputs,
    const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
    ClusterHierarchyNode *clusterNode, llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges) {
  // Backward edges from the dirty node
  for (auto *dirtyNode : dirtyNodes) {
    // Because INIT is currenly implemented as a MERGE, we should never consider
    // its 2 inputs as unaligned
    // Once the INIT becomes a single input component, this can be removed
    if (isInit(dirtyNode))
      continue;

    for (auto operand : dirtyNode->getOperands()) {
      // bool edgeFromCluster = false;
      // // Check if the operand is an output from one of the children clusters
      // for (auto *childCluster : clusterNode->children) {
      //   if (childCluster->cluster.outputs.contains(operand) &&
      //   childCluster->cluster.isBeforeCluster()) {
      //     llvm::errs() << "OPerand: " << operand << "\n";
      //     edgeFromCluster = true;
      //   }
      // }

      // // If the operand is an output from one of the children clusters, then
      // the
      // // children cluster is technically a dirtyNode, so this is a
      // // dirtyNode->dirtNode edge that is aligned
      // if (edgeFromCluster)
      //   continue;

      Operation *producer = operand.getDefiningOp();
      // Identify edges that connect a non-dirty node to a dirty node
      // Both the consumer and the producers shoud be internal to the cluster
      if (dirtyNodes.find(producer) == dirtyNodes.end() &&
          clusterNode->cluster.internalOps.find(producer) !=
              clusterNode->cluster.internalOps.end())
        unalignedEdges.insert(operand);
    }
  }

  // Add all the output edges of the out of order node, except those to a MC or
  // LSQ, to the unaligned edges
  for (auto res : outOfOrderNodeOutputs) {
    bool edgeToMC = false;
    for (auto *user : res.getUsers()) {
      if (isa<MemoryControllerOp>(user) || isa<LSQOp>(user))
        edgeToMC = true;
    }
    if (!edgeToMC)
      unalignedEdges.insert(res);
  }
  return success();
}

/**
 * @brief Identifies which unaligned edges should be tagged for synchronization.
 * Starts with unaligned edges, adds valid input edges of the out-of-order
 * node, and removes its outputs to isolate true tagged edges.

 * @param outOfOrderNodeInputs Set of input values from out-of-order node.
 * analyzed.
 * @param outOfOrderNodeOutputs Set of output values from out-of-order node.
 * analyzed.
 * @param unalignedEdges Set of previously identified unaligned edges.
 * @param taggedEdges Set to be populated with values that need tagging.

 * @return Success result.
 * @note Excludes outputs of the out-of-order op and edges to memory operations.
*/
LogicalResult OutOfOrderExecutionPass::identifyTaggedEdges(
    const llvm::DenseSet<Value> &outOfOrderNodeInputs,
    const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
    llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges) {
  // Step 1: Identify the Tagged Edges
  taggedEdges =
      llvm::DenseSet<Value>(unalignedEdges.begin(), unalignedEdges.end());

  // Insert input edges (operands of outOfOrderNode)
  for (Value operand : outOfOrderNodeInputs) {
    if (auto *prod = operand.getDefiningOp()) {
      if (!isa<MemoryControllerOp>(prod) && !isa<LSQOp>(prod))
        taggedEdges.insert(operand);
    } else {
      taggedEdges.insert(operand);
    }
  }

  // Remove output edges (results of outOfOrderNode)
  for (Value result : outOfOrderNodeOutputs) {
    taggedEdges.erase(result);
  }

  return success();
}

/**
 * @brief Returns a result from a set of operation results that is not connected
 * to memory operations.

 * @param opResults A set of result values from an operation.

 * @return A value from the set that is not used by a memory-related operation.
 * @note If all outputs are memory-related, returns the first available value
from the set.
*/
static Value returnNonMemoryOutput(const llvm::DenseSet<Value> &opResults) {
  auto result = llvm::find_if(opResults, [](auto res) {
    bool edgeToMem = llvm::any_of(res.getUsers(), [](auto *user) {
      return isa<MemoryControllerOp>(user) || isa<LSQOp>(user);
    });
    return !edgeToMem; // We want the result where there's no edge to memory
  });

  if (result != opResults.end())
    return *result;

  return *opResults.begin();
}

/**
 * @brief Adds tag-related operations around an out-of-order node to ensure
 * synchronized dataflow using tagging logic. Constructs the tag
 * generator(fifo), taggers, aligner, and untaggers as needed.

 * @param outOfOrderNodeOutputs Set of output values from out-of-order node.
 * @param outOfOrderNodeInternalOps Set of internal ops from out-of-order
 node.
 *
 * @param builder The builder used to insert Tagger operations.
 * @param clusterNode The current cluster node in the hierarchy.
 * @param numTags Number of unique tags available for synchronization.
 * @param controlled Whether the aligner should be controlled based on the
 progrem
 * order or free same as out-of-order order
 * @param unalignedEdges Set of values representing unaligned edges needing
 * tagging.
 * @param taggedEdges Set of edges determined to require tagging.
 * @param untaggers Set to be populated with created untagger operations.

 * @return The FreeTagsFifo operation used to generate and recycle tag values.
*/
Operation *OutOfOrderExecutionPass::addTagOperations(
    const llvm::DenseSet<Value> &outOfOrderNodeOutputs,
    const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
    OpBuilder builder, ClusterHierarchyNode *clusterNode, int numTags,
    bool controlled, llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges,
    llvm::DenseSet<Operation *> &untaggers) {

  // Step 1: Add the FreeTagsFifo that generates the tags
  // THe tag bitwidth must be atleast 1
  int tagBitWidth = ceil(log2(numTags));
  auto tagType = builder.getIntegerType(std::max(tagBitWidth, 1));

  // Create temporary condition that feeds the freeTagsFifo. This cond will
  // later be replaced by the tag output of the untagger
  BackedgeBuilder beb(builder, (*taggedEdges.begin()).getLoc());
  Backedge cond = beb.get(tagType);

  FreeTagsFifoOp freeTagsFifo = builder.create<FreeTagsFifoOp>(
      (*taggedEdges.begin()).getLoc(), ChannelType::get(tagType), cond);

  // The fifo is now internal to the cluster
  clusterNode->addInternalOp(freeTagsFifo.getOperation());
  inheritBBFromValue((*taggedEdges.begin()), freeTagsFifo);

  // Step 2: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FreeTagsFifo
  Value taggedOutput =
      addTaggers(outOfOrderNodeInternalOps, builder, clusterNode, freeTagsFifo,
                 dirtyNodes, unalignedEdges, taggedEdges);
  assert(taggedOutput && "At least one tagger should be created");

  // Step 3 (DEPRECATED): Add the Untagger Operations
  // addUntaggers(builder, clusterNode, freeTagsFifo, unalignedEdges,
  // untaggers);

  // The select of the Aligner is:
  // (1) Free Aligner: the output of the out-of-order node
  // (2) Controlled Aligner: the output of the first tagger
  Value select = *outOfOrderNodeOutputs.begin();
  if (!controlled) {
    select = returnNonMemoryOutput(outOfOrderNodeOutputs);
  } else {
    select = taggedOutput;
  }

  // Step 3: Add the Aligner Operation
  addAligner(builder, clusterNode, select, numTags, freeTagsFifo,
             unalignedEdges, untaggers);

  return freeTagsFifo.getOperation();
}

/**
 * @brief Creates Tagger operations for all tagged edges and connects them to
 their
 * consumers. Each Tagger pairs the original data with a tag from the
 FreeTagsFifo
 * and replaces direct data uses with the tagged output.

 * @param builder The builder used to insert Tagger operations.
 * @param clusterNode The current cluster node in the hierarchy.
 * @param freeTagsFifo The tag source providing tag values to taggers.
 * @param unalignedEdges Set of unaligned edges to be updated with tagged
 outputs.
 * @param taggedEdges Set of edges that require tagging.

 * @return The output of the first created Tagger operation.
*/
Value OutOfOrderExecutionPass::addTaggers(
    const llvm::DenseSet<Operation *> &outOfOrderNodeInternalOps,
    OpBuilder builder, ClusterHierarchyNode *clusterNode,
    FreeTagsFifoOp &freeTagsFifo, llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges) {

  // Step 3: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FreeTagsFifo
  Value firstTaggerOutput;
  for (auto edge : taggedEdges) {
    TaggerOp taggerOp = builder.create<TaggerOp>(
        edge.getLoc(), edge.getType(), edge, freeTagsFifo.getTagOut());

    // The tagger is now internal to the cluster
    clusterNode->addInternalOp(taggerOp.getOperation());

    if (!firstTaggerOutput) {
      firstTaggerOutput = taggerOp.getDataOut();
    }

    // Connect the tagger to the dirty/out-of-order consumer of the edge
    //  i.e., for each prod that feeds the tagger: replace all the edges
    //  producer->consumer to untagger->consumer, where the consumer is either a
    //  dirty node or an out-of-order node
    llvm::DenseSet<Operation *> usersToReplace;
    for (auto *user : edge.getUsers()) {
      if (dirtyNodes.contains(user) || outOfOrderNodeInternalOps.contains(user))
        usersToReplace.insert(user);
    }
    for (Operation *user : usersToReplace) {
      user->replaceUsesOfWith(edge, taggerOp.getDataOut());
    }

    // The output of the tagger now replaces the edge if it was considered as an
    // input to any of the children cluster of the current one.
    clusterNode->replaceInputInChildren(edge, taggerOp.getDataOut());

    // For the unaligner and untagger, replace all the edges that go through the
    // tagger from prod->cons to tagger->cons
    if (unalignedEdges.contains(edge)) {
      unalignedEdges.erase(edge);
      unalignedEdges.insert(taggerOp.getDataOut());
    }

    inheritBBFromValue(edge, taggerOp);
  }

  return firstTaggerOutput;
}

void OutOfOrderExecutionPass::addUntaggers(
    OpBuilder builder, ClusterHierarchyNode *clusterNode,
    FreeTagsFifoOp &freeTagsFifo, llvm::DenseSet<Value> &unalignedEdges,
    llvm::DenseSet<Operation *> &untaggers) {
  SmallVector<Value> joinOperands;
  Operation *consumer = (*unalignedEdges.begin()).getDefiningOp();

  for (auto edge : unalignedEdges) {
    // Start by untagging the edge
    UntaggerOp edgeUntagger =
        builder.create<UntaggerOp>(edge.getLoc(), edge.getType(),
                                   freeTagsFifo.getTagOut().getType(), edge);

    // Connect the mux to the users of the edge
    llvm::DenseSet<Operation *> users;
    for (Operation *user : edge.getUsers()) {
      if (user != edgeUntagger) {
        users.insert(user);
      }
    }
    for (auto *user : users) {
      consumer = user;
      inheritBB(user, edgeUntagger);
      user->replaceUsesOfWith(edge, edgeUntagger.getDataOut());
    }

    // The untagger, demux and mux are internal to the cluster
    clusterNode->addInternalOp(edgeUntagger.getOperation());

    joinOperands.push_back(edgeUntagger.getTagOut());
    untaggers.insert(edgeUntagger.getOperation());
  }

  // If more than on untagger was created, then join them and feed the
  // result of the join (the free tag) back into the freeTagsFifo. Else, feed
  // the free tag output of the single untagger into the freeTagsFifo.

  if (joinOperands.size() > 1) {
    JoinOp joinOp =
        builder.create<JoinOp>((*joinOperands.begin()).getLoc(), joinOperands);
    clusterNode->addInternalOp(joinOp.getOperation());
    inheritBB(consumer, joinOp);
    freeTagsFifo.getOperation()->replaceUsesOfWith(freeTagsFifo.getOperand(),
                                                   joinOp.getResult());
  } else {
    freeTagsFifo.getOperation()->replaceUsesOfWith(freeTagsFifo.getOperand(),
                                                   (*joinOperands.begin()));
  }
}

/**
 * @brief Adds an aligner to synchronize unaligned edges using tag-based
 routing.
 * Each edge is routed through a Demux-Mux pair controlled by tag signals
 derived
 * from the select value and the edge itself.

 * @param outOfOrderNode The operation being aligned.
 * @param builder The builder used to insert the aligner logic.
 * @param clusterNode The current cluster node in the hierarchy.
 * @param select The value used as the select signal for the aligner.
 * @param numTags The total number of tags used in the tagging scheme.
 * @param freeTagsFifo The FIFO providing tag types.
 * @param unalignedEdges The set of edges requiring alignment.
 * @param untaggers Collects all Untagger operations created during alignment.
*/
void OutOfOrderExecutionPass::addAligner(
    OpBuilder builder, ClusterHierarchyNode *clusterNode, Value select,
    int numTags, FreeTagsFifoOp &freeTagsFifo,
    llvm::DenseSet<Value> &unalignedEdges,
    llvm::DenseSet<Operation *> &untaggers) {

  // Untag the select of the aligner to get the tag flow
  UntaggerOp selectUntagger =
      builder.create<UntaggerOp>(select.getLoc(), select.getType(),
                                 freeTagsFifo.getTagOut().getType(), select);
  clusterNode->addInternalOp(selectUntagger.getOperation());
  untaggers.insert(selectUntagger.getOperation());

  Operation *consumer = selectUntagger.getOperation();
  SmallVector<Value> joinOperands;

  for (auto edge : unalignedEdges) {
    // Start by untagging the edge
    UntaggerOp edgeUntagger =
        builder.create<UntaggerOp>(edge.getLoc(), edge.getType(),
                                   freeTagsFifo.getTagOut().getType(), edge);

    // Feed the output of the edge untagger into a demux, with the select of
    // the demux being the tag of the edge. The number of ouputs of the demux is
    // equal to the number of tags
    llvm::SmallVector<Type> typeStorage(numTags,
                                        edgeUntagger.getDataOut().getType());
    TypeRange results(typeStorage);

    DemuxOp demux = builder.create<DemuxOp>(edge.getLoc(), results,
                                            edgeUntagger.getTagOut(),
                                            edgeUntagger.getDataOut());

    // Feed the output of demux into a mux, with the select of
    // the mux being the tag of the select of the aligner
    MuxOp mux = builder.create<MuxOp>(
        edge.getLoc(), demux->getResults().getType().front(),
        selectUntagger.getTagOut(), demux->getResults());

    // Connect the mux to the users of the edge
    llvm::DenseSet<Operation *> users;
    for (Operation *user : edge.getUsers()) {
      if (user != edgeUntagger && user != selectUntagger) {
        users.insert(user);
      }
    }
    for (auto *user : users) {
      consumer = user;
      inheritBB(user, selectUntagger);
      inheritBB(user, edgeUntagger);
      inheritBB(user, demux);
      inheritBB(user, mux);
      user->replaceUsesOfWith(edge, mux.getResult());
    }

    // The untagger, demux and mux are internal to the cluster
    clusterNode->addInternalOp(edgeUntagger.getOperation());
    clusterNode->addInternalOp(demux.getOperation());
    clusterNode->addInternalOp(mux.getOperation());

    joinOperands.push_back(edgeUntagger.getTagOut());
    untaggers.insert(edgeUntagger.getOperation());
  }

  // If more than on untagger was created, then join them and feed the
  // result of the join (the free tag) back into the freeTagsFifo. Else, feed
  // the free tag output of the single untagger into the freeTagsFifo.

  if (joinOperands.size() > 1) {
    JoinOp joinOp =
        builder.create<JoinOp>((*joinOperands.begin()).getLoc(), joinOperands);
    clusterNode->addInternalOp(joinOp.getOperation());
    inheritBB(consumer, joinOp);
    freeTagsFifo.getOperation()->replaceUsesOfWith(freeTagsFifo.getOperand(),
                                                   joinOp.getResult());
  } else {
    freeTagsFifo.getOperation()->replaceUsesOfWith(freeTagsFifo.getOperand(),
                                                   (*joinOperands.begin()));
  }
}

/**
 * @brief Adds a tag signal to a value's type if not already present.
 * The tag is appended as an extra signal (e.g., in ChannelType or ControlType).

 * @param value The value to which the tag signal is to be added.
 * @param extraTag The name of the tag signal.
 * @param numTags The number of distinct tags; used to compute tag bit width.

 * @return Success if the tag was added or already present, failure otherwise.
 * @note The value's type must implement ExtraSignalsTypeInterface.
*/
static LogicalResult addTagToValue(Value value, const std::string &extraTag,
                                   int numTags) {
  OpBuilder builder(value.getContext());

  // The value type must implement ExtraSignalsTypeInterface (e.g.,
  // ChannelType or ControlType).
  if (auto valueType = value.getType().dyn_cast<ExtraSignalsTypeInterface>()) {
    // Skip if the tag was already added during the algorithm.
    if (!valueType.hasExtraSignal(extraTag)) {
      llvm::SmallVector<ExtraSignal> newExtraSignals(
          valueType.getExtraSignals());
      int tagBitWidth = ceil(log2(numTags));

      // Ensure the tag bit width is at least 1
      newExtraSignals.emplace_back(
          extraTag, builder.getIntegerType(std::max(tagBitWidth, 1)));

      value.setType(valueType.copyWithExtraSignals(newExtraSignals));
    }
    return success();
  }
  value.getDefiningOp()->emitError("Unexpected type");
  return failure();
}

/**
 * @brief Adds tag signals to all operands within a tagged region starting from
 * a TaggerOp. The traversal continues until all operations
 * within the tagged region have been processed, and the tag is applied to their
 * operands' values.

 * @param funcOp The function containing the operations to be processed.
 * @param freeTagsFifo The `FreeTagsFifoOp` operation providing free tags used
 * for tagging.
 * @param numTags The total number of distinct tags; used to determine the size
 * of the tag signal.
 * @param extraTag The additional tag signal to be added to operands' values.
 * @param untaggers A set of untagger operations that should not be tagged.

 * @return Success if the tag signals were successfully added, failure
 * otherwise.
*/
LogicalResult OutOfOrderExecutionPass::addTagSignalsToTaggedRegion(
    FuncOp funcOp, Operation *freeTagsFifo, int numTags,
    const std::string &extraTag, llvm::DenseSet<Operation *> &untaggers) {
  llvm::DenseSet<Operation *> visited;
  // A TaggerOp marks the beginning of the tagged region, so we use it as a
  // starting point for tagging
  for (auto taggerOp : funcOp.getOps<TaggerOp>()) {
    // Check if this is a tagger corresponding to the current tagged region (by
    // identifying the freeTagsFifo it is fed by), then we start traversal
    if (taggerOp.getTagOperand() ==
        dyn_cast<FreeTagsFifoOp>(freeTagsFifo).getTagOut()) {
      visited.insert(taggerOp);

      Value taggerResult = taggerOp.getDataOut();
      for (OpOperand &opOperand : taggerResult.getUses()) {
        if (failed(addTagSignalsRecursive(opOperand, visited, extraTag,
                                          freeTagsFifo, untaggers, numTags)))
          return failure();
      }
    }
  }

  return success();
}

/**
 * @brief Recursively traverses operations starting from a given operand, adding
 * a tag signal to each operand's value along the way through.

 * @param opOperand The operand whose operation will be processed.
 * @param visited A set of visited operations to avoid reprocessing the same
 * operation.
 * @param extraTag The tag signal to be added to operands' values.
 * @param freeTagsFifo The operation that provides the free tags for tagging.
 * @param untaggers A set of untagger operations to stop the traversal if
 * encountered.
 * @param numTags The total number of distinct tags; used to determine the tag
 * signal size.

 * @return Success if the tag was added to operands recursively, failure
 * otherwise
 * @note The traversal stops if an `EndOp`, `FreeTagsFifoOp`, or an already
 * visited operation is encountered. Special handling is provided for specific
* operation types, including `UntaggerOp` and `LoadOp`.
*/
LogicalResult OutOfOrderExecutionPass::addTagSignalsRecursive(
    OpOperand &opOperand, llvm::DenseSet<Operation *> &visited,
    const std::string &extraTag, Operation *freeTagsFifo,
    llvm::DenseSet<Operation *> &untaggers, int numTags) {

  Operation *op;

  op = opOperand.getOwner();

  if (!op)
    // As long as the algorithm traverses inside the tagged region,
    // all operands should have an owner and defining operation.
    return failure();

  if (isa<EndOp>(op) || isa<StoreOp>(op))
    return success();

  // The tags coming in and out of the freeTagsFifo should never be tagged
  if (dyn_cast<FreeTagsFifoOp>(op))
    return failure();

  // Add the tag to the current operand
  if (failed(addTagToValue(opOperand.get(), extraTag, numTags)))
    return failure();

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  // UntaggerOp
  if (UntaggerOp untagger = dyn_cast<UntaggerOp>(op)) {
    // If this is the untagger corresponding to the current tagged region, then
    // we stop traversal
    if (untaggers.contains(op))
      return success();

    // Else this is an untagger in a nested region and we continue traversal
    // Special Case: For edges that feed nothing, we still want to tag their
    // values for type verification purposes
    if (untagger.getDataOut().getUses().empty()) {
      if (failed(addTagToValue(untagger.getDataOut(), extraTag, numTags)))
        return failure();
    }
    for (auto &operand : untagger.getDataOut().getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }
    return success();
  }

  // MemPortOp (Load and Store)
  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    // Continue traversal to dataOut, skipping ports connected to the memory
    // controller.
    // Special Case: For edges that feed nothing, we still want to tag their
    // values for type verification purposes
    if (loadOp->getOpResult(1).getUses().empty()) {
      if (failed(addTagToValue(loadOp->getOpResult(1), extraTag, numTags)))
        return failure();
    }
    for (auto &operand : loadOp->getOpResult(1).getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }

    return success();
  }

  // if (isa<ControlMergeOp>(op) || isa<MuxOp>(op)) {
  //   // Only perform traversal to the dataResult
  //   MergeLikeOpInterface mergeLikeOp = llvm::cast<MergeLikeOpInterface>(op);
  //   for (auto &operand : mergeLikeOp.getDataResult().getUses()) {
  //     if (failed(addTagSignalsRecursive(operand, visited, extraTag,
  //                                       freeTagsFifo, untaggers, numTags)))
  //       return failure();
  //   }
  //   return success();
  // }

  // Downstream traversal
  for (auto result : op->getResults()) {
    // Special Case: For edges that feed nothing, we still want to tag their
    // values for type verification purposes
    if (result.getUses().empty()) {
      if (failed(addTagToValue(result, extraTag, numTags)))
        return failure();
    }
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      if (operand.get() == opOperand.get())
        continue;

      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }
  }

  return success();
}

/**
 * @brief Finds the innermost cluster containing the given operation.
 *
 * @param outOfOrderNode The operation to search for in the clusters.
 * @param hierarchyNodes List of cluster hierarchy nodes in icreasing order of
 * size (from innermost to outermost).
 *
 * @return The innermost cluster node containing the operation
 * @note Returns nullptr if the operation is not found in any cluster.
 */
ClusterHierarchyNode *OutOfOrderExecutionPass::findInnermostClusterContainingOp(
    Operation *outOfOrderNode,
    const std::vector<ClusterHierarchyNode *> &hierarchyNodes) {
  for (auto &clusterNode : hierarchyNodes) {
    if (clusterNode->cluster.isInsideCluster(outOfOrderNode)) {
      return clusterNode;
    }
  }

  return nullptr;
}

/**
 * @brief Removes any extra signals (tags, spec bits) from the select operand of
 * each MUX in the function.
 *
 * @param funcOp The function containing the operations to be processed.
 * @param ctx The MLIR context.
 *
 * @return Success if extra signals are successfully removed from the MUX select
 * operands, failure otherwise.
 */
LogicalResult
OutOfOrderExecutionPass::removeExtraSignalsFromMux(FuncOp funcOp,
                                                   MLIRContext *ctx) {
  OpBuilder builder(ctx);
  // Loop over all the muxes in the graph
  for (MuxOp muxOp : funcOp.getOps<MuxOp>()) {
    // If the select of the MUX has any extra signals, then these extra signals
    // must be removed by an extract op.
    if (auto valueType = muxOp.getSelectOperand()
                             .getType()
                             .dyn_cast<ExtraSignalsTypeInterface>()) {
      if (valueType.getNumExtraSignals() > 0) {
        builder.setInsertionPoint(muxOp);
        ExtractOp extractor = builder.create<ExtractOp>(
            muxOp->getLoc(), muxOp.getSelectOperand());
        muxOp->replaceUsesOfWith(muxOp.getSelectOperand(),
                                 extractor.getResult());
      }
    }
  }
  return success();
}