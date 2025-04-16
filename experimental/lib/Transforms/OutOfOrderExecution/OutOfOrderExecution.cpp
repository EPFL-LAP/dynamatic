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
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <assert.h>
#include <cmath>
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

// Defines the postion of the out-of-order node in relation to the cluster
enum class ClusterPosition { NoCluster, BeforeCluster, InsideCluster };

namespace {
struct OutOfOrderExecutionPass
    : public dynamatic::experimental::outoforder::impl::OutOfOrderExecutionBase<
          OutOfOrderExecutionPass> {

  void runDynamaticPass() override;

private:
  // Step 1.1: Identify dirty nodes
  LogicalResult identifyDirtyNodes(Operation *outOfOrderOp,
                                   llvm::DenseSet<Operation *> &dirtyNodes,
                                   ClusterPosition position,
                                   ClusterHierarchyNode *clusterNode);

  // Step 1.2: Identfy unaligned edges
  LogicalResult identifyUnalignedEdges(Operation *outOfOrderOp,
                                       llvm::DenseSet<Operation *> &dirtyNodes,
                                       llvm::DenseSet<Value> &unalignedEdges);

  // Step 1.3: Identify tagged edges; i.e. the edges that should receive a tag
  LogicalResult identifyTaggedEdges(Operation *outOfOrderOp,
                                    llvm::DenseSet<Value> &unalignedEdges,
                                    llvm::DenseSet<Value> &taggedEdges);

  // Step 1.4: Add the tagger operations and connect them to the freeTagsFifo
  // and consumers Returns the output of teh fitrst tagger added to be used as
  // the select of the Aligner in case of a Controlled Aligner
  Value addTaggers(OpBuilder builder, Operation *outOfOrderOp,
                   llvm::DenseSet<Value> &unalignedEdges,
                   llvm::DenseSet<Value> &taggedEdges,
                   FreeTagsFifoOp &freeTagsFifo);

  // Add the untagger operations and connect them to consumers
  // DEPRECATED (now part of addAligner)
  void addUntaggers(OpBuilder builder, Operation *outOfOrderOp,
                    llvm::DenseSet<Value> &unalignedEdges,
                    SmallVector<Value> &joinOperands,
                    llvm::DenseSet<Operation *> &untaggers,
                    FreeTagsFifoOp &freeTagsFifo);

  // Step 1.6: Add the aligner
  void addAligner(OpBuilder builder, Operation *outOfOrderOp, Value select,
                  llvm::DenseSet<Value> &unalignedEdges,
                  SmallVector<Value> &joinOperands,
                  llvm::DenseSet<Operation *> &untaggers,
                  FreeTagsFifoOp &freeTagsFifo, int numTags);

  // Step 1.7: Adds the FreeTagsFifo, Tagger and Untagger operations
  // Returns the FreeTagsFifo operation which uniquely identifies the tagged
  // region
  Operation *addTagOperations(OpBuilder builder, Operation *outOfOrderOp,
                              llvm::DenseSet<Operation *> &dirtyNodes,
                              llvm::DenseSet<Value> &unalignedEdges,
                              llvm::DenseSet<Value> &taggedEdges,
                              llvm::DenseSet<Operation *> &untaggers,
                              int numTags, bool controlled);

  // Step 1.8: Tag the channels in the tagged region
  LogicalResult addTagSignalsToTaggedRegion(
      handshake::FuncOp funcOp, const std::string &extraTag,
      Operation *freeTagsFifo, llvm::DenseSet<Operation *> &untaggers,
      int numTags);

  // MAIN: Apply the out-of-order execution methodology to all the out-of-order
  // nodes
  LogicalResult applyOutOfOrder(
      handshake::FuncOp funcOp, MLIRContext *ctx,
      llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes);

  LogicalResult applyClusteringLogic(handshake::FuncOp funcOp, MLIRContext *ctx,
                                     Operation *outOfOrderNode,
                                     ClusterHierarchyNode *clusterNode,
                                     bool &tagged, int &tagIndex, int numTags,
                                     bool controlled);

  // Apply the out-of-order execution methodology according to the clustering
  // position NoCluster, BeforeCluster, InsideCluster
  LogicalResult applyOutOfOrderAlgorithm(
      handshake::FuncOp funcOp, MLIRContext *ctx, Operation *outOfOrderNode,
      ClusterHierarchyNode *clusterNode, int &tagIndex, int numTags,
      bool controlled, ClusterPosition position);

  LogicalResult applyHeirarchicalAlignment(handshake::FuncOp funcOp,
                                           MLIRContext *ctx,
                                           ClusterHierarchyNode *innerCluster,
                                           ClusterHierarchyNode *outerCluster,
                                           int &tagIndex, int numTags,
                                           bool controlled);
};
} // namespace

void OutOfOrderExecutionPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  for (auto funcOp : module.getOps<handshake::FuncOp>()) {

    // Each out of order node will have:
    // (1) numTags: int representing the number of tags
    // (2) controlled: bool representing wether the aligner is controlled
    llvm::DenseMap<Operation *, std::pair<int, bool>> outOfOrderNodes;
    for (auto loadOp : funcOp.getOps<handshake::LoadOp>()) {
      outOfOrderNodes.insert({loadOp, {4, false}});
    }

    // for (auto shli : funcOp.getOps<handshake::ShLIOp>()) {
    //   outOfOrderNodes.insert({shli, {8, false}});
    // }

    if (failed(applyOutOfOrder(funcOp, ctx, outOfOrderNodes)))
      signalPassFailure();
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::outoforder::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}

static void traverseGraph(Operation *op,
                          llvm::DenseSet<Operation *> &dirtyNodes,
                          ClusterPosition position,
                          ClusterHierarchyNode *clusterNode) {
  // Memory Operations (Memory Controller & LSQ) Control and End operations
  // should not be marked as a dirty node
  if (dirtyNodes.find(op) != dirtyNodes.end() ||
      isa<handshake::MemoryControllerOp>(op) || isa<handshake::LSQOp>(op) ||
      isa<handshake::EndOp>(op))
    return;

  dirtyNodes.insert(op);

  // Traverse the graph
  for (auto result : op->getResults()) {
    if (position == ClusterPosition::BeforeCluster) {
      // If the operation is before a cluster and we reach the inputs of the
      // cluster, then we keep the cluster closed and skip to the outputs of the
      // cluster
      if (clusterNode->cluster.inputs.contains(result)) {
        for (auto clusterOutput : clusterNode->cluster.outputs) {
          for (auto *user : clusterOutput.getUsers()) {
            traverseGraph(user, dirtyNodes, position, clusterNode);
          }
        }
      }
    } else {
      if (position == ClusterPosition::InsideCluster) {
        // If the operation is inside a cluster, we need to stop checking at the
        // outputs of the cluster
        if (clusterNode->cluster.outputs.contains(result)) {
          continue;
        }
        // If the cluster has any chilren cluster nested inside, we need to keep
        // these clusters closed. So whenever we wncounter an input to anuy of
        // these clusters, we skip to their outputs
        for (auto *nestedCluster : clusterNode->children) {
          if (nestedCluster->cluster.inputs.contains(result)) {
            for (auto clusterOutput : nestedCluster->cluster.outputs) {
              for (auto *user : clusterOutput.getUsers()) {
                traverseGraph(user, dirtyNodes, position, clusterNode);
              }
            }
          }
        }
      }
      for (auto *user : result.getUsers()) {
        traverseGraph(user, dirtyNodes, position, clusterNode);
      }
    }
  }
}

LogicalResult OutOfOrderExecutionPass::identifyDirtyNodes(
    Operation *outOfOrderOp, llvm::DenseSet<Operation *> &dirtyNodes,
    ClusterPosition position, ClusterHierarchyNode *clusterNode) {
  traverseGraph(outOfOrderOp, dirtyNodes, position, clusterNode);
  dirtyNodes.erase(outOfOrderOp);
  return success();
}

LogicalResult OutOfOrderExecutionPass::identifyUnalignedEdges(
    Operation *outOfOrderOp, llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges) {
  // Backward edges from the dirty node
  for (auto *dirtyNode : dirtyNodes) {
    for (auto operand : dirtyNode->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      // Identify edges that connect a non-dirty node to a dirty node
      if (dirtyNodes.find(producer) == dirtyNodes.end() &&
          !isa<handshake::MemoryControllerOp>(producer))
        unalignedEdges.insert(operand);
    }
  }

  // Add all the ouput edges of the out of order node, except those to a MC or
  // LSQ, to the unaligned edges
  for (auto res : outOfOrderOp->getResults()) {
    bool edgeToMC = false;
    for (auto *user : res.getUsers()) {
      if (isa<handshake::MemoryControllerOp>(user) ||
          isa<handshake::LSQOp>(user))
        edgeToMC = true;
    }
    if (!edgeToMC)
      unalignedEdges.insert(res);
  }
  return success();
}

LogicalResult OutOfOrderExecutionPass::identifyTaggedEdges(
    Operation *outOfOrderOp, llvm::DenseSet<Value> &unalignedEdges,
    llvm::DenseSet<Value> &taggedEdges) {
  // Step 1: Identify the Tagged Edges
  taggedEdges =
      llvm::DenseSet<Value>(unalignedEdges.begin(), unalignedEdges.end());

  // Insert input edges (operands of outOfOrderOp)
  for (Value operand : outOfOrderOp->getOperands()) {
    if (auto *prod = operand.getDefiningOp()) {
      if (!isa<handshake::MemoryControllerOp>(prod) &&
          !isa<handshake::LSQOp>(prod))
        taggedEdges.insert(operand);
    }
  }

  // Remove output edges (results of outOfOrderOp)
  for (Value result : outOfOrderOp->getResults()) {
    taggedEdges.erase(result);
  }

  return success();
}

// Returns the result of the op that does not have an edge to memory (Memory
// Controller & LSQ)
static Value returnNonMemoryOutput(Operation *op) {
  auto result = llvm::find_if(op->getResults(), [](auto res) {
    bool edgeToMem = llvm::any_of(res.getUsers(), [](auto *user) {
      return isa<handshake::MemoryControllerOp>(user) ||
             isa<handshake::LSQOp>(user);
    });
    return !edgeToMem; // We want the result where there's no edge to memory
  });

  if (result != op->getResults().end())
    return *result;

  return op->getResults().front();
}

Operation *OutOfOrderExecutionPass::addTagOperations(
    OpBuilder builder, Operation *outOfOrderOp,
    llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges,
    llvm::DenseSet<Operation *> &untaggers, int numTags, bool controlled) {

  // Step 1: Add the FreeTagsFifo that generates the tags

  auto tagType = builder.getIntegerType(ceil(log2(numTags)));
  // Create temporary condition that feeds the freeTagsFifo. This cond will
  // later be replaced by the tag output of the untagger
  BackedgeBuilder beb(builder, (*taggedEdges.begin()).getLoc());
  Backedge cond = beb.get(tagType);

  builder.setInsertionPoint(outOfOrderOp);
  FreeTagsFifoOp freeTagsFifo = builder.create<handshake::FreeTagsFifoOp>(
      (*taggedEdges.begin()).getLoc(), handshake::ChannelType::get(tagType),
      cond);
  inheritBB(outOfOrderOp, freeTagsFifo);

  // Step 2: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FreeTagsFifo
  Value taggedOutput = addTaggers(builder, outOfOrderOp, unalignedEdges,
                                  taggedEdges, freeTagsFifo);
  assert(taggedOutput && "At least one tagger should be created");

  SmallVector<Value> joinOperands;
  // Step 4: Add the Untagger Operations
  // addUntaggers(builder, outOfOrderOp, unalignedEdges, joinOperands,
  // untaggers,
  //             freeTagsFifo);

  // The select of the Aligner is:
  // (1) Free Aligner: the ouput of the out-of-order node
  // (2) Controlled Aligner: the output of the first tagger
  Value select = outOfOrderOp->getResults().front();
  if (!controlled) {
    select = returnNonMemoryOutput(outOfOrderOp);
  } else {
    select = taggedOutput;
  }

  // Step 3: Add the Aligner Operation
  addAligner(builder, outOfOrderOp, select, unalignedEdges, joinOperands,
             untaggers, freeTagsFifo, numTags);

  // If more than on untagger was created, then join them and feed the
  // result of the join (the free tag) back into the freeTagsFifo. Else, feed
  // the free tag output of the single untagger into the freeTagsFifo.

  if (joinOperands.size() > 1) {
    handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(
        (*joinOperands.begin()).getLoc(), joinOperands);
    inheritBB(outOfOrderOp, joinOp);
    freeTagsFifo.getOperation()->replaceUsesOfWith(cond, joinOp.getResult());
  } else {
    freeTagsFifo.getOperation()->replaceUsesOfWith(cond,
                                                   (*joinOperands.begin()));
  }

  return freeTagsFifo.getOperation();
}

Value OutOfOrderExecutionPass::addTaggers(OpBuilder builder,
                                          Operation *outOfOrderOp,
                                          llvm::DenseSet<Value> &unalignedEdges,
                                          llvm::DenseSet<Value> &taggedEdges,
                                          FreeTagsFifoOp &freeTagsFifo) {
  // Step 3: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FreeTagsFifo
  Value firstTaggerOutput;
  for (auto edge : taggedEdges) {
    handshake::TaggerOp taggerOp = builder.create<handshake::TaggerOp>(
        outOfOrderOp->getLoc(), edge.getType(), edge, freeTagsFifo.getTagOut());

    if (!firstTaggerOutput) {
      firstTaggerOutput = taggerOp.getDataOut();
    }

    // Connect the tagger to the consumer of the edge
    //  i.e., for each prod that feeds the tagger: replace all the edges
    //  producer->consumer to untagger->consumer
    edge.replaceAllUsesExcept(taggerOp.getDataOut(), taggerOp);

    // For the unaligner and untagger, replace all the edges that go through the
    // tagger from prod->cons to tagger->cons
    if (unalignedEdges.contains(edge)) {
      unalignedEdges.erase(edge);
      unalignedEdges.insert(taggerOp.getDataOut());
    }

    inheritBB(outOfOrderOp, taggerOp); // TODO: how to get the BB?
  }

  return firstTaggerOutput;
}

void OutOfOrderExecutionPass::addUntaggers(
    OpBuilder builder, Operation *outOfOrderOp,
    llvm::DenseSet<Value> &unalignedEdges, SmallVector<Value> &joinOperands,
    llvm::DenseSet<Operation *> &untaggers, FreeTagsFifoOp &freeTagsFifo) {
  for (auto edge : unalignedEdges) {
    UntaggerOp untaggerOp = builder.create<handshake::UntaggerOp>(
        edge.getLoc(), edge.getType(), freeTagsFifo.getTagOut().getType(),
        edge);

    // Connect the untagger to the consumer of the edge
    //  i.e., for each prod that feeds the untagger: replace all the edges
    //  producer->consumer to untagger->consumer
    edge.replaceAllUsesExcept(untaggerOp.getDataOut(), untaggerOp);

    inheritBB(outOfOrderOp, untaggerOp); // TODO: how to get the BB?

    joinOperands.push_back(untaggerOp.getTagOut());
    untaggers.insert(untaggerOp.getOperation());
  }
}

void OutOfOrderExecutionPass::addAligner(OpBuilder builder,
                                         Operation *outOfOrderOp, Value select,
                                         llvm::DenseSet<Value> &unalignedEdges,
                                         SmallVector<Value> &joinOperands,
                                         llvm::DenseSet<Operation *> &untaggers,
                                         FreeTagsFifoOp &freeTagsFifo,
                                         int numTags) {
  builder.setInsertionPoint(outOfOrderOp);
  // Untag the select of the aligner to get the tag flow
  UntaggerOp selectUntagger = builder.create<handshake::UntaggerOp>(
      select.getLoc(), select.getType(), freeTagsFifo.getTagOut().getType(),
      select);
  inheritBB(outOfOrderOp, selectUntagger);
  untaggers.insert(selectUntagger.getOperation());

  for (auto edge : unalignedEdges) {
    // Start by unatagging the edge
    UntaggerOp edgeUntagger = builder.create<handshake::UntaggerOp>(
        edge.getLoc(), edge.getType(), freeTagsFifo.getTagOut().getType(),
        edge);

    // Feed the output of the edge untagger into a demux, with the select of
    // the demux being the tag of the edge. The number of ouputs of the demux is
    // equal to the number of tags
    llvm::SmallVector<Type> typeStorage(numTags,
                                        edgeUntagger.getDataOut().getType());
    TypeRange results(typeStorage);

    DemuxOp demux = builder.create<handshake::DemuxOp>(
        edge.getLoc(), results, edgeUntagger.getTagOut(),
        edgeUntagger.getDataOut());

    // Feed the output of demux into a mux, with the select of
    // the mux being the tag of the select of the aligner
    MuxOp mux = builder.create<handshake::MuxOp>(
        edge.getLoc(), demux->getResults().getType().front(),
        selectUntagger.getTagOut(), demux->getResults());

    for (Operation *user : edge.getUsers()) {
      if (user != edgeUntagger && user != selectUntagger)
        user->replaceUsesOfWith(edge, mux.getResult());
    }

    inheritBB(outOfOrderOp, edgeUntagger);
    inheritBB(outOfOrderOp, demux);
    inheritBB(outOfOrderOp, mux);

    joinOperands.push_back(edgeUntagger.getTagOut());
    untaggers.insert(edgeUntagger.getOperation());
  }
}

static LogicalResult addTagToValue(Value value, const std::string &extraTag,
                                   int numTags) {
  OpBuilder builder(value.getContext());

  // The value type must implement ExtraSignalsTypeInterface (e.g.,
  // ChannelType or ControlType).
  if (auto valueType =
          value.getType().dyn_cast<handshake::ExtraSignalsTypeInterface>()) {
    // Skip if the tag was already added during the algorithm.
    if (!valueType.hasExtraSignal(extraTag)) {
      llvm::SmallVector<ExtraSignal> newExtraSignals(
          valueType.getExtraSignals());
      newExtraSignals.emplace_back(extraTag,
                                   builder.getIntegerType(ceil(log2(numTags))));
      value.setType(valueType.copyWithExtraSignals(newExtraSignals));
    }
    return success();
  }
  value.getDefiningOp()->emitError("Unexpected type");
  return failure();
}

static LogicalResult
addTagSignalsRecursive(OpOperand &opOperand,
                       llvm::DenseSet<Operation *> &visited,
                       const std::string &extraTag, Operation *freeTagsFifo,
                       llvm::DenseSet<Operation *> &untaggers, int numTags) {

  Operation *op;

  op = opOperand.getOwner();

  if (!op)
    // As long as the algorithm traverses inside the tagged region,
    // all operands should have an owner and defining operation.
    return failure();

  if (isa<handshake::EndOp>(op))
    return success();

  // Add the tag to the current operand
  if (failed(addTagToValue(opOperand.get(), extraTag, numTags)))
    return failure();

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  // UntaggerOp
  if (UntaggerOp untagger = dyn_cast<handshake::UntaggerOp>(op)) {
    // If this is the untagger corresponding to the current tagged region, then
    // we stop traversal
    if (untaggers.contains(op))
      return success();
    // Else this is an untagger in a nested region and we continue traversal
    for (auto &operand : untagger.getDataOut().getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }
    return success();
  }

  // MemPortOp (Load and Store)
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // Continue traversal to dataOut, skipping ports connected to the memory
    // controller.
    for (auto &operand : loadOp->getOpResult(1).getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }

    return success();
  }

  // TODO: Should storeOp be in tagged region?
  if (auto storeOp = dyn_cast<handshake::StoreOp>(op))
    return success();

  if (isa<handshake::ControlMergeOp>(op) || isa<handshake::MuxOp>(op)) {
    // Only perform traversal to the dataResult
    MergeLikeOpInterface mergeLikeOp = llvm::cast<MergeLikeOpInterface>(op);
    for (auto &operand : mergeLikeOp.getDataResult().getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }
    return success();
  }

  // The tags coming in and out of the freeTagsFifo should never be tagged
  if (dyn_cast<handshake::FreeTagsFifoOp>(op))
    return failure();

  // Downstream traversal
  for (auto result : op->getResults()) {
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      /*
      if (operand == &opOperand)
        continue;*/
      if (failed(addTagSignalsRecursive(operand, visited, extraTag,
                                        freeTagsFifo, untaggers, numTags)))
        return failure();
    }
  }

  return success();
}

LogicalResult OutOfOrderExecutionPass::addTagSignalsToTaggedRegion(
    handshake::FuncOp funcOp, const std::string &extraTag,
    Operation *freeTagsFifo, llvm::DenseSet<Operation *> &untaggers,
    int numTags) {
  llvm::DenseSet<Operation *> visited;
  // A TaggerOp marks the beginning of the tagged region, so we use it as a
  // starting point for tagging
  for (auto taggerOp : funcOp.getOps<handshake::TaggerOp>()) {
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

LogicalResult OutOfOrderExecutionPass::applyOutOfOrder(
    handshake::FuncOp funcOp, MLIRContext *ctx,
    llvm::DenseMap<Operation *, std::pair<int, bool>> &outOfOrderNodes) {

  // Step 1: Identify the clusters
  std::vector<Cluster> clusters = identifyClusters(funcOp, ctx);

  // Step 2: Check validity of the clusters
  if (failed(verifyClusters(clusters)))
    return failure();

  // Step 3: Build the cluster hierarchy, returning the leaf nodes (innermost
  // clusters)
  std::vector<ClusterHierarchyNode *> hierarchyNodes =
      buildClusterHierarchy(clusters);

  // Step 4: Apply the out-of-order execution methodology to each ot-of-order
  // node with respect to each innermost cluster
  int tagIndex = 0;
  for (auto &[op, attributes] : outOfOrderNodes) {
    auto &[numTags, controlled] = attributes;

    bool tagged = false;
    for (auto &clusterNode : hierarchyNodes) {
      if (failed(applyClusteringLogic(funcOp, ctx, op, clusterNode, tagged,
                                      tagIndex, numTags, controlled)))
        return failure();

      // TODO: check correctness
      if (tagged)
        break;
    }

    // If the out-of-order node is not inside/before any cluster, then we need
    // to apply the out-of-order algorithm as usual
    if (!tagged) {
      if (failed(applyOutOfOrderAlgorithm(funcOp, ctx, op, nullptr, tagIndex,
                                          numTags, controlled,
                                          ClusterPosition::NoCluster)))
        return failure();
    }
  }
  return success();
}

LogicalResult OutOfOrderExecutionPass::applyClusteringLogic(
    handshake::FuncOp funcOp, MLIRContext *ctx, Operation *outOfOrderNode,
    ClusterHierarchyNode *clusterNode, bool &tagged, int &tagIndex, int numTags,
    bool controlled) {
  Cluster &cluster = clusterNode->cluster;

  // Case 1: The out-of-order node is inside a cluster
  // If the out-of-order node is inside a cluster, then we need to apply the
  // out-of-order algorithm as usual but stop the dirty nodes identification
  // traversal at the boundries of the cluster
  if (cluster.isInsideCluster(outOfOrderNode)) {
    if (failed(applyOutOfOrderAlgorithm(
            funcOp, ctx, outOfOrderNode, clusterNode, tagIndex, numTags,
            controlled, ClusterPosition::InsideCluster)))
      return failure();
    tagged = true;

    // After taking care of the out-of-order node, we need to apply the
    // heirarchical alignment algorithm to the cluster by recrsively considering
    // the cluster as an out-of-order node inside its parent cluster. This is
    // done by calling the applyHeirarchicalAlignment
    if (failed(applyHeirarchicalAlignment(funcOp, ctx, clusterNode,
                                          clusterNode->parent, tagIndex,
                                          numTags, controlled)))
      return failure();
  }

  // Case 2: The out-of-order node is before a cluster
  // If the out-of-order node is before a cluster, then there are 2 cases
  // depending on whether the cluster is teh outermost one or no
  else if (cluster.isBeforeCluster(outOfOrderNode)) {
    // If the cluster has a parent, then the out-of-order node must be inside
    // this parent or one of its ancestor clusters up the hierarchy
    // In this case, we recursivly call the applyClusteringLogic function to the
    // out-of-order node and the parent cluster. The parent cluster will be
    // responsible of taking care of this operation as an out-of-order node
    // where it will fall into the inside cluster case
    if (clusterNode->parent) {
      if (failed(applyClusteringLogic(funcOp, ctx, outOfOrderNode,
                                      clusterNode->parent, tagged, tagIndex,
                                      numTags, controlled)))
        return failure();
    } else {
      // If the cluster is the outermost one, then we  need to apply
      // out-of-order algorithm as usual but whenever we encouneter the
      // inputs of the cluster, we skip to its outputs in order to keep the
      // cluster closed
      if (failed(applyOutOfOrderAlgorithm(
              funcOp, ctx, outOfOrderNode, clusterNode, tagIndex, numTags,
              controlled, ClusterPosition::BeforeCluster)))
        return failure();
      tagged = true;
    }

  }

  // Case 3: The out-of-order node is after a cluster
  // In this case, there is no relation bwetween the out-of-order node and the
  // cluster, so we go up the heirarchy to the parent cluster and apply the
  // clustering logic there
  else if (clusterNode->parent) {
    if (failed(applyClusteringLogic(funcOp, ctx, outOfOrderNode,
                                    clusterNode->parent, tagged, tagIndex,
                                    numTags, controlled)))
      return failure();
  }

  return success();
}

LogicalResult OutOfOrderExecutionPass::applyOutOfOrderAlgorithm(
    handshake::FuncOp funcOp, MLIRContext *ctx, Operation *outOfOrderNode,
    ClusterHierarchyNode *clusterNode, int &tagIndex, int numTags,
    bool controlled, ClusterPosition position) {

  // Step 1: Identify the dirty nodes
  llvm::DenseSet<Operation *> dirtyNodes;
  if (failed(identifyDirtyNodes(outOfOrderNode, dirtyNodes, position,
                                clusterNode)))
    return failure();

  // Step 2: Identify the unaligned edges
  llvm::DenseSet<Value> unalignedEdges;
  if (failed(
          identifyUnalignedEdges(outOfOrderNode, dirtyNodes, unalignedEdges)))
    return failure();

  // Step 3: Identify the tagged edges
  llvm::DenseSet<Value> taggedEdges;
  if (failed(identifyTaggedEdges(outOfOrderNode, unalignedEdges, taggedEdges)))
    return failure();

  // Step 4: Add the tagger and untagger operations and connect them to the
  // freeTagsFifo
  OpBuilder builder(ctx);
  llvm::DenseSet<Operation *> untaggers;
  Operation *freeTagsFifo =
      addTagOperations(builder, outOfOrderNode, dirtyNodes, unalignedEdges,
                       taggedEdges, untaggers, numTags, controlled);

  if (!freeTagsFifo)
    return failure();

  // Step 5: Tag the channels in the tagged region
  std::string extraTag = "tag" + std::to_string(tagIndex++);
  if (failed(addTagSignalsToTaggedRegion(funcOp, extraTag, freeTagsFifo,
                                         untaggers, numTags)))
    return failure();

  return success();
}

LogicalResult OutOfOrderExecutionPass::applyHeirarchicalAlignment(
    handshake::FuncOp funcOp, MLIRContext *ctx,
    ClusterHierarchyNode *innerCluster, ClusterHierarchyNode *outerCluster,
    int &tagIndex, int numTags, bool controlled) {
  llvm::DenseSet<Operation *> dirtyNodes;
  llvm::DenseSet<Value> unalignedEdges;
  llvm::DenseSet<Value> taggedEdges;

  // Base case: if the node is the root of the hierarchy, then we don't need to
  //  traverse the hierarchy
  if (!outerCluster || !innerCluster)
    return success();

  // To consider the cluster as an out-of-order node, we essentially need to
  // consider all the operations at its output edges as out-of-order node to
  // find the dirty nodes that are reachable from the cluster
  for (auto clusterOutput : innerCluster->cluster.outputs) {
    Operation *clusterOutputOp = clusterOutput.getDefiningOp();

    // Step 1: Identify the dirty nodes
    if (failed(identifyDirtyNodes(clusterOutputOp, dirtyNodes,
                                  ClusterPosition::InsideCluster,
                                  outerCluster)))
      return failure();

    // Step 2: Identify the unaligned edges
    if (failed(identifyUnalignedEdges(clusterOutputOp, dirtyNodes,
                                      unalignedEdges)))
      return failure();

    // Step 3: Identify the tagged edges
    if (failed(
            identifyTaggedEdges(clusterOutputOp, unalignedEdges, taggedEdges)))
      return failure();
  }

  // After identifying the tagged and unaligned edges of the cluster, we add the
  // out-of-order operations
  OpBuilder builder(ctx);
  llvm::DenseSet<Operation *> untaggers;

  // Step 4: Add the tagger and untagger operations and connect them to the
  // freeTagsFifo
  Operation *freeTagsFifo = addTagOperations(
      builder, innerCluster->cluster.outputs.begin()->getDefiningOp(),
      dirtyNodes, unalignedEdges, taggedEdges, untaggers, numTags, false);

  if (!freeTagsFifo)
    return failure();

  // Step 5: Tag the channels in the tagged region
  std::string extraTag = "tag" + std::to_string(tagIndex++);
  if (failed(addTagSignalsToTaggedRegion(funcOp, extraTag, freeTagsFifo,
                                         untaggers, numTags)))
    return failure();

  // Apply the hierearchical alignment recursively to the parent (outer) cluster
  return applyHeirarchicalAlignment(funcOp, ctx, outerCluster,
                                    outerCluster->parent, tagIndex, numTags,
                                    controlled);
}
