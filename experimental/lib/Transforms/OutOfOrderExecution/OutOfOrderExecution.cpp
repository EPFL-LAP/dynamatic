//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cmath>
#include <memory>
#include <string>

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
  // Step 1: Add the FIFO, Taggerand Untagger Operations
  LogicalResult createOutOfExecutionGraph(handshake::FuncOp funcOp,
                                          MLIRContext *ctx);

  // Step 2: Add the tag signals to the channels in the tagged region
  LogicalResult addTagSignals(handshake::FuncOp funcOp, MLIRContext *ctx);

  // Step 1.1: Identify dirty nodes
  LogicalResult identifyDirtyNodes(Operation *outOfOrderOp,
                                   llvm::DenseSet<Operation *> &dirtyNodes);

  // Step 1.2: Identfy unaligned edges
  LogicalResult identifyUnalignedEdges(Operation *outOfOrderOp,
                                       llvm::DenseSet<Operation *> &dirtyNodes,
                                       llvm::DenseSet<Value> &unalignedEdges);

  // Step 1.3: Identify tagged edges; i.e. the edges that should receive a tag
  LogicalResult identifyTaggedEdges(Operation *outOfOrderOp,
                                    llvm::DenseSet<Value> &unalignedEdges,
                                    llvm::DenseSet<Value> &taggedEdges);

  // Step 1.4: Add the tagger operations and connect them to the fifo and
  // consumers
  void addTaggers(OpBuilder builder, Operation *outOfOrderOp,
                  llvm::DenseSet<Value> &unalignedEdges,
                  llvm::DenseSet<Value> &taggedEdges, FreeTagsFifoOp &fifo);

  // Step 1.5: Add the untagger operations and connect them to consumers
  void addUntaggers(OpBuilder builder, Operation *outOfOrderOp,
                    llvm::DenseSet<Value> &unalignedEdges,
                    SmallVector<Value> &joinOperands, FreeTagsFifoOp &fifo);

  // Step 1.7: Adds the FIFO, Tagger and Untagger operations
  // Returns the FIFO operation which uniquely identifies the tagged region
  Operation *addTagOperations(handshake::FuncOp funcOp, OpBuilder builder,
                              Operation *outOfOrderOp,
                              llvm::DenseSet<Operation *> &dirtyNodes,
                              llvm::DenseSet<Value> &unalignedEdges,
                              llvm::DenseSet<Value> &taggedEdges, int numTags);

  // Step 1.8: TAg the channels in the tagged region
  LogicalResult addTagSignalsToTaggedRegion(handshake::FuncOp funcOp,
                                            const std::string &extraTag,
                                            Operation *fifo, int numTags);

  LogicalResult testNestedRegions(handshake::FuncOp funcOp, OpBuilder builder,
                                  Operation *outOfOrderOp, Operation *innerFifo,
                                  const std::string &extraTag);

  // MAIN: Apply the out-of-order execution methodology
  LogicalResult
  applyOutOfOrder(handshake::FuncOp funcOp, MLIRContext *ctx,
                  llvm::DenseMap<Operation *, int> &outOfOrderNodes);
};
} // namespace

LogicalResult
OutOfOrderExecutionPass::createOutOfExecutionGraph(handshake::FuncOp funcOp,
                                                   MLIRContext *ctx) {

  OpBuilder builder(ctx);
  for (auto loadOp : funcOp.getOps<handshake::LoadOp>()) {
    Value addrInput = loadOp.getAddressInput();
    builder.setInsertionPoint(loadOp);

    auto tagType = builder.getIntegerType(ceil(log2(4)));

    BackedgeBuilder beb(builder, loadOp.getLoc());
    Backedge cond = beb.get(tagType);
    FreeTagsFifoOp fifo = builder.create<handshake::FreeTagsFifoOp>(
        loadOp.getLoc(), handshake::ChannelType::get(tagType), cond);

    // Tag the address input of the load of
    handshake::TaggerOp taggerOp = builder.create<handshake::TaggerOp>(
        loadOp.getLoc(), addrInput.getType(), addrInput, fifo.getTagOut());

    // Connect the tagger to the load
    loadOp.getOperation()->replaceUsesOfWith(addrInput, taggerOp.getDataOut());

    // Create the untagegr and connect it to the load
    UntaggerOp untaggerOp = builder.create<handshake::UntaggerOp>(
        loadOp.getLoc(), loadOp.getDataResult().getType(),
        fifo.getTagOut().getType(), loadOp.getDataResult());

    // Replaces all the connections load->consumer to untagger->consumer
    Value loadOutput = loadOp.getDataOutput();
    loadOutput.replaceAllUsesExcept(untaggerOp.getDataOut(), untaggerOp);

    // Connet the free tag from the untagger to the fifo
    fifo.getOperation()->replaceUsesOfWith(cond, untaggerOp.getTagOut());

    inheritBB(loadOp, fifo);
    inheritBB(loadOp, taggerOp);
    inheritBB(loadOp, untaggerOp);
  }
  return success();
}

LogicalResult OutOfOrderExecutionPass::applyOutOfOrder(
    handshake::FuncOp funcOp, MLIRContext *ctx,
    llvm::DenseMap<Operation *, int> &outOfOrderNodes) {
  int tagIndex = 0;
  for (auto &[op, numTags] : outOfOrderNodes) {
    llvm::DenseSet<Operation *> dirtyNodes;
    if (failed(identifyDirtyNodes(op, dirtyNodes)))
      return failure();

    llvm::DenseSet<Value> unalignedEdges;
    if (failed(identifyUnalignedEdges(op, dirtyNodes, unalignedEdges)))
      return failure();

    llvm::DenseSet<Value> taggedEdges;
    if (failed(identifyTaggedEdges(op, unalignedEdges, taggedEdges)))
      return failure();

    // if (isa<handshake::ShLIOp>(op)) {
    //   llvm::errs() << "Printing unaligned edges: \n";
    //   for (Value edge : unalignedEdges) {
    //     edge.print(llvm::errs());
    //     llvm::errs() << "\n";
    //   }
    //   llvm::errs() << "Done printing unaligned edges\n";

    //   llvm::errs() << "Printing tagged edges: \n";
    //   for (Value edge : taggedEdges) {
    //     edge.print(llvm::errs());
    //     llvm::errs() << "\n";
    //   }
    //   llvm::errs() << "Done printing tagged edges\n";
    // }

    OpBuilder builder(ctx);

    Operation *fifo = addTagOperations(funcOp, builder, op, dirtyNodes,
                                       unalignedEdges, taggedEdges, numTags);

    if (!fifo)
      return failure();

    std::string extraTag = "tag" + std::to_string(tagIndex++);
    if (failed(addTagSignalsToTaggedRegion(funcOp, extraTag, fifo, numTags)))
      return failure();

    if (failed(testNestedRegions(funcOp, builder, op, fifo,
                                 (extraTag + "_outer"))))
      return failure();
  }
  return success();
}

static void tarverseGraph(Operation *op,
                          llvm::DenseSet<Operation *> &dirtyNodes) {
  // Memory Control and End operations should not be marked as illegal
  if (dirtyNodes.find(op) != dirtyNodes.end() ||
      isa<handshake::MemoryControllerOp>(op) || isa<handshake::EndOp>(op))
    return;

  dirtyNodes.insert(op);

  // Traverse the graph
  for (auto result : op->getResults()) {
    for (auto *user : result.getUsers()) {
      tarverseGraph(user, dirtyNodes);
    }
  }
}

LogicalResult OutOfOrderExecutionPass::identifyDirtyNodes(
    Operation *outOfOrderOp, llvm::DenseSet<Operation *> &dirtyNodes) {
  tarverseGraph(outOfOrderOp, dirtyNodes);
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
  // End, to the unaligned edges
  for (auto res : outOfOrderOp->getResults()) {
    bool edgeToMC = false;
    for (auto *user : res.getUsers()) {
      if (isa<handshake::MemoryControllerOp>(user))
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
      if (!isa<handshake::MemoryControllerOp>(prod))
        taggedEdges.insert(operand);
    }
  }

  // Remove output edges (results of outOfOrderOp)
  for (Value result : outOfOrderOp->getResults()) {
    taggedEdges.erase(result);
  }

  return success();
}

Operation *OutOfOrderExecutionPass::addTagOperations(
    handshake::FuncOp funcOp, OpBuilder builder, Operation *outOfOrderOp,
    llvm::DenseSet<Operation *> &dirtyNodes,
    llvm::DenseSet<Value> &unalignedEdges, llvm::DenseSet<Value> &taggedEdges,
    int numTags) {

  // Step 2: Add the FIFO that generates the tags

  auto tagType = builder.getIntegerType(ceil(log2(numTags)));
  // Create temporary condition that feeds the fifo. THis cond will later be
  // replaced by the tag output of the untagger
  BackedgeBuilder beb(builder, (*taggedEdges.begin()).getLoc());
  Backedge cond = beb.get(tagType);

  builder.setInsertionPoint(outOfOrderOp);
  FreeTagsFifoOp fifo = builder.create<handshake::FreeTagsFifoOp>(
      (*taggedEdges.begin()).getLoc(), handshake::ChannelType::get(tagType),
      cond);
  inheritBB(outOfOrderOp, fifo);

  // Step 3: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FIFO

  addTaggers(builder, outOfOrderOp, unalignedEdges, taggedEdges, fifo);

  SmallVector<Value> joinOperands;
  // Step 4: Add the Untagger Operations
  addUntaggers(builder, outOfOrderOp, unalignedEdges, joinOperands, fifo);

  // If more than on untagger was created, then join them and feed the result of
  // the join (the free tag) back into the fifo
  // Else, feed the free tag output of the single untagger into the fifo

  if (joinOperands.size() > 1) {
    handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(
        (*joinOperands.begin()).getLoc(), joinOperands);
    inheritBB(outOfOrderOp, joinOp);
    fifo.getOperation()->replaceUsesOfWith(cond, joinOp.getResult());
  } else {
    fifo.getOperation()->replaceUsesOfWith(cond, (*joinOperands.begin()));
  }

  return fifo.getOperation();
}

void OutOfOrderExecutionPass::addTaggers(OpBuilder builder,
                                         Operation *outOfOrderOp,
                                         llvm::DenseSet<Value> &unalignedEdges,
                                         llvm::DenseSet<Value> &taggedEdges,
                                         FreeTagsFifoOp &fifo) {
  // Step 3: Add the Tagger Operations
  // For each tagged edge, create a Tagger operation that is fed from this
  // tagged edge and the FIFO
  for (auto edge : taggedEdges) {
    handshake::TaggerOp taggerOp = builder.create<handshake::TaggerOp>(
        outOfOrderOp->getLoc(), edge.getType(), edge, fifo.getTagOut());

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
}

void OutOfOrderExecutionPass::addUntaggers(
    OpBuilder builder, Operation *outOfOrderOp,
    llvm::DenseSet<Value> &unalignedEdges, SmallVector<Value> &joinOperands,
    FreeTagsFifoOp &fifo) {
  for (auto edge : unalignedEdges) {
    UntaggerOp untaggerOp = builder.create<handshake::UntaggerOp>(
        edge.getLoc(), edge.getType(), fifo.getTagOut().getType(), edge);

    // Connect the untagger to the consumer of the edge
    //  i.e., for each prod that feeds the untagger: replace all the edges
    //  producer->consumer to untagger->consumer
    edge.replaceAllUsesExcept(untaggerOp.getDataOut(), untaggerOp);

    inheritBB(outOfOrderOp, untaggerOp); // TODO: how to get the BB?

    joinOperands.push_back(untaggerOp.getTagOut());
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

static LogicalResult addTagSignalsRecursive(
    OpOperand &opOperand, llvm::DenseSet<Operation *> &visited,
    const std::string &extraTag, Operation *fifo, int numTags) {

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

  if (isa<handshake::TaggerOp>(op))
    llvm::errs() << "Tagged tagger\n";

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  // UntaggerOp
  if (UntaggerOp untagger = dyn_cast<handshake::UntaggerOp>(op)) {
    // If this is the untagger corresponding to the current tagged region (by
    // identifying the fifo it feeds), then we stop traversal
    for (auto *user : untagger.getTagOut().getUsers()) {
      // Case 1: Untagger directly feeds fifo
      if (user == fifo)
        return success();

      // Case 2: Untagger feeds Join that feeds fifo
      if (JoinOp join = dyn_cast<handshake::JoinOp>(user)) {
        for (auto *user : join.getResult().getUsers()) {
          if (user == fifo)
            return success();
        }
      }
    }

    // Else this is an untagger in a nested region and we continue traversal
    for (auto &operand : untagger.getDataOut().getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag, fifo,
                                        numTags)))
        return failure();
    }
    return success();
  }

  // MemPortOp (Load and Store)
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {

    // Continue traversal to dataOut, skipping ports connected to the memory
    // controller.
    for (auto &operand : loadOp->getOpResult(1).getUses()) {
      if (failed(addTagSignalsRecursive(operand, visited, extraTag, fifo,
                                        numTags)))
        return failure();
    }

    return success();
  }

  // The tags coming in and out of teh fifo should never be tagged
  if (dyn_cast<handshake::FreeTagsFifoOp>(op))
    return failure();

  // Downstream traversal
  for (auto result : op->getResults()) {
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      /*
      if (operand == &opOperand)
        continue;*/
      if (failed(addTagSignalsRecursive(operand, visited, extraTag, fifo,
                                        numTags)))
        return failure();
    }
  }

  return success();
}

LogicalResult OutOfOrderExecutionPass::addTagSignalsToTaggedRegion(
    handshake::FuncOp funcOp, const std::string &extraTag, Operation *fifo,
    int numTags) {
  llvm::DenseSet<Operation *> visited;
  // A TaggerOp marks the beginning of the tagged region, so we use it as a
  // starting point for tagging
  for (auto taggerOp : funcOp.getOps<handshake::TaggerOp>()) {
    // Check if this is a tagger corresponding to the current tagged region (by
    // identifying the fifo it is fed by), then we start traversal
    if (taggerOp.getTagOperand() ==
        dyn_cast<FreeTagsFifoOp>(fifo).getTagOut()) {
      visited.insert(taggerOp);
      // For the speculator, perform downstream traversal to only dataOut,
      // skipping control signals. The upstream dataIn will be handled by the
      // recursive traversal.

      Value taggerResult = taggerOp.getDataOut();
      for (OpOperand &opOperand : taggerResult.getUses()) {
        if (failed(addTagSignalsRecursive(opOperand, visited, extraTag, fifo,
                                          numTags)))
          return failure();
      }
    }
  }

  return success();
}

// Nests the existing tagged region inside another tagged region
static void tarverseNestedGraph(Operation *op, OpBuilder builder,
                                llvm::DenseSet<Operation *> &visitedNodes,
                                FreeTagsFifoOp &fifo, Operation *innerFifo,
                                SmallVector<Value> &joinOperands) {
  // Skip Memory Control and End operations
  if (visitedNodes.find(op) != visitedNodes.end() ||
      isa<handshake::MemoryControllerOp>(op) || isa<handshake::EndOp>(op))
    return;

  visitedNodes.insert(op);

  // feed the tagger from a new outer tagger
  if (TaggerOp tagger = dyn_cast<TaggerOp>(op)) {
    if (tagger.getTagOperand() ==
        dyn_cast<FreeTagsFifoOp>(innerFifo).getTagOut()) {
      Value edge = tagger.getDataOperand();
      handshake::TaggerOp outerTagger = builder.create<handshake::TaggerOp>(
          op->getLoc(), edge.getType(), edge, fifo.getTagOut());
      edge.replaceAllUsesExcept(outerTagger.getDataOut(), outerTagger);
      inheritBB(tagger, outerTagger);
      visitedNodes.insert(outerTagger);
    }
  }

  // feed the untagger into a new outer untagger
  if (UntaggerOp untagger = dyn_cast<UntaggerOp>(op)) {
    bool untaggerForRegion = false;
    for (auto *user : untagger.getTagOut().getUsers()) {
      if (user == innerFifo)
        untaggerForRegion = true;
      if (JoinOp join = dyn_cast<handshake::JoinOp>(user)) {
        for (auto *user : join.getResult().getUsers()) {
          if (user == innerFifo)
            untaggerForRegion = true;
        }
      }
    }
    if (untaggerForRegion) {
      Value edge = untagger.getDataOut();
      handshake::UntaggerOp outerUntagger =
          builder.create<handshake::UntaggerOp>(
              op->getLoc(), edge.getType(), fifo.getTagOut().getType(), edge);
      edge.replaceAllUsesExcept(outerUntagger.getDataOut(), outerUntagger);
      inheritBB(untagger, outerUntagger);
      visitedNodes.insert(outerUntagger);
      joinOperands.push_back(outerUntagger.getTagOut());
      for (auto *user : outerUntagger.getDataOut().getUsers()) {
        tarverseNestedGraph(user, builder, visitedNodes, fifo, innerFifo,
                            joinOperands);
      }
    }
  }

  // Traverse the graph
  for (auto result : op->getResults()) {
    for (auto *user : result.getUsers()) {
      tarverseNestedGraph(user, builder, visitedNodes, fifo, innerFifo,
                          joinOperands);
    }
  }
}

LogicalResult OutOfOrderExecutionPass::testNestedRegions(
    handshake::FuncOp funcOp, OpBuilder builder, Operation *outOfOrderOp,
    Operation *innerFifo, const std::string &extraTag) {
  builder.setInsertionPoint(outOfOrderOp);
  auto tagType = builder.getIntegerType(ceil(log2(8)));

  // Add the fifo
  BackedgeBuilder beb(builder, outOfOrderOp->getLoc());
  Backedge cond = beb.get(tagType);
  FreeTagsFifoOp fifo = builder.create<handshake::FreeTagsFifoOp>(
      outOfOrderOp->getLoc(), handshake::ChannelType::get(tagType), cond);

  // Add the outer taggers+untaggers
  llvm::DenseSet<Operation *> visitedNodes;
  SmallVector<Value> joinOperands;
  tarverseNestedGraph(outOfOrderOp, builder, visitedNodes, fifo, innerFifo,
                      joinOperands);

  if (joinOperands.size() > 1) {
    handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(
        (*joinOperands.begin()).getLoc(), joinOperands);
    inheritBB(outOfOrderOp, joinOp);
    fifo.getOperation()->replaceUsesOfWith(cond, joinOp.getResult());
  } else {
    fifo.getOperation()->replaceUsesOfWith(cond, (*joinOperands.begin()));
  }

  // Tag the chennels of the outer region
  if (failed(addTagSignalsToTaggedRegion(funcOp, extraTag, fifo, 8)))
    return failure();

  return success();
}

void OutOfOrderExecutionPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  for (auto funcOp : module.getOps<handshake::FuncOp>()) {

    llvm::DenseMap<Operation *, int> outOfOrderNodes;
    for (auto loadOp : funcOp.getOps<handshake::LoadOp>()) {
      outOfOrderNodes.insert({loadOp, 4});
    }

    for (auto shli : funcOp.getOps<handshake::ShLIOp>()) {
      outOfOrderNodes.insert({shli, 8});
    }

    if (failed(applyOutOfOrder(funcOp, ctx, outOfOrderNodes)))
      signalPassFailure();
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::outoforder::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}