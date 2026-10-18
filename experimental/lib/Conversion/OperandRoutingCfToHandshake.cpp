//===- OperandRoutingCfToHandshake.cpp - cf to handshake, per-operand -----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --operand-routing-lower-cf-to-handshake pass with a plain
// builder, without the dialect conversion framework.
//
// Values are routed with one routine, `routeValue`, which places branches
// and muxes along a route graph, from the values entering the route to the
// nodes consuming it, wired value to value across blocks. The dominance this
// violates in the cf-level function is resolved when the blocks are spliced
// into one at the end. The nodes of a data route are the block arguments the
// value crosses, which maximised SSA gives one per live value per block, so
// a block holds as many nodes as it has live copies of the value. The nodes
// of a control route are blocks, since the start token is one per block.
//
// The lowering of a function goes in seven steps.
// 1. Build the handshake function.
// 2. Convert every operation: arith and math operations become the
//    handshake operation of the same name with channel results, constants
//    take a source or a placeholder control, loads and stores become ports
//    with a placeholder for a load's data. Operands stay as they are; the
//    routing replaces the block arguments among them later.
// 3. Build the backbone control network: start routed over the whole CFG,
//    which places a control merge in every block and a branch at every
//    terminator. The backbone links the route graphs to the real circuit:
//    it gives each block its real block and its real branch condition.
// 4. Route each consumer. For every block argument read in its block, mark
//    the paths from its producers on the CFG, fold it to the consumer's
//    block, and walk it once, building the control network and the data
//    together with the muxes indexed by the control merges.
// 5. Route start to each constant that needs it, filling its placeholder.
// 6. Build the memory interfaces, each with its own network of start.
// 7. Flatten: tag every operation with its block id, erase the terminators
//    and the block arguments, splice the blocks into one, build the end.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/OperandRoutingCfToHandshake.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

// [START Boilerplate code for the MLIR pass]
#include "experimental/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_OPERANDROUTINGCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::operand_routing;

// The helpers live in the pass's namespace rather than at file scope so that
// unqualified MLIR names resolve there first: the buffering support header
// pulls LLVM IR names of the same spelling into file scope
namespace dynamatic {
namespace experimental {
namespace operand_routing {

/// Returns the position of the block in its region, which is its bb id.
static unsigned getBlockNumber(Block *block) {
  return std::distance(block->getParent()->begin(), block->getIterator());
}

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

Value dynamatic::experimental::operand_routing::getBlockCondition(
    const BlockInfo &block) {
  if (block.backboneBranch) {
    handshake::ConditionalBranchOp branchOp = block.backboneBranch;
    return branchOp.getConditionOperand();
  }
  return block.condition;
}

void dynamatic::experimental::operand_routing::readCFG(
    Region &body, RoutingCFG &cfg, SmallVector<BlockInfo> &blocks) {
  // Edges are listed block by block, true side before false side
  for (Block &block : body) {
    unsigned bb = getBlockNumber(&block);
    cfg.nodeBlock.push_back(bb);
    Operation *termOp = block.getTerminator();
    Value condition;
    unsigned numSides = 0;
    if (auto condBrOp = dyn_cast<cf::CondBranchOp>(termOp)) {
      condition = condBrOp.getCondition();
      numSides = 2;
      cfg.edges.push_back(
          {bb, getBlockNumber(condBrOp.getTrueDest()), EdgeSide::TRUE, {}});
      cfg.edges.push_back(
          {bb, getBlockNumber(condBrOp.getFalseDest()), EdgeSide::FALSE, {}});
    } else if (auto brOp = dyn_cast<cf::BranchOp>(termOp)) {
      numSides = 1;
      cfg.edges.push_back(
          {bb, getBlockNumber(brOp.getDest()), EdgeSide::UNCONDITIONAL, {}});
    }
    blocks.push_back({&block, numSides, condition, nullptr});
  }
}

RoutingCFG dynamatic::experimental::operand_routing::markPaths(
    const RoutingCFG &cfg, ArrayRef<BlockArgument> consumerArgs) {
  RoutingCFG marked = cfg;
  llvm::DenseSet<BlockArgument> visited;
  SmallVector<BlockArgument> worklist;
  for (BlockArgument consumerArg : consumerArgs) {
    if (visited.insert(consumerArg).second)
      worklist.push_back(consumerArg);
  }

  while (!worklist.empty()) {
    BlockArgument arg = worklist.pop_back_val();
    Block *block = arg.getOwner();
    unsigned bb = getBlockNumber(block);

    // Every predecessor passes a value into the argument through its
    // terminator, on the side of its branch that leads here: that is what
    // the edge delivers to the argument. A value that is itself a block
    // argument of a non-entry predecessor came from further back and is
    // followed; anything else is a producer.
    for (auto [predBlock, isFirst] : getRealBlockPredecessors(block)) {
      Value passed = getMergeOperand(arg, predBlock, isFirst);
      assert(passed && "failed to find value passed into block argument");
      EdgeSide side = EdgeSide::UNCONDITIONAL;
      if (auto condBrOp =
              dyn_cast<cf::CondBranchOp>(predBlock->getTerminator())) {
        bool onTrueSide = condBrOp.getTrueDest() == block && isFirst;
        side = onTrueSide ? EdgeSide::TRUE : EdgeSide::FALSE;
      }
      unsigned pred = getBlockNumber(predBlock);
      for (RoutingEdge &edge : marked.edges) {
        if (edge.from == pred && edge.to == bb && edge.side == side)
          edge.values.push_back({arg, passed, nullptr});
      }
      auto passedArg = dyn_cast<BlockArgument>(passed);
      if (passedArg && passedArg.getOwner() == predBlock &&
          !predBlock->isEntryBlock()) {
        if (visited.insert(passedArg).second)
          worklist.push_back(passedArg);
      }
    }
  }
  return marked;
}

/// Whether a value is a block argument of the given block that data passes
/// through. The entry block's arguments are the function's arguments, which
/// are producers, not passed-through data.
static bool isArgumentOf(Value value, Block *block) {
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() == block && !block->isEntryBlock();
}

/// Whether an edge delivers a producer: a value that is not a block
/// argument of the edge's source.
static bool deliversProducer(const RoutingEdge &edge,
                             ArrayRef<BlockInfo> blocks) {
  for (const EdgeValue &delivered : edge.values) {
    if (!isArgumentOf(delivered.value, blocks[edge.from].block))
      return true;
    if (delivered.onFalse &&
        !isArgumentOf(delivered.onFalse, blocks[edge.from].block))
      return true;
  }
  return false;
}

/// Whether an edge delivers a select between its two sides' values.
static bool deliversSelect(const RoutingEdge &edge) {
  for (const EdgeValue &delivered : edge.values) {
    if (delivered.onFalse)
      return true;
  }
  return false;
}

RoutingCFG dynamatic::experimental::operand_routing::foldCFG(
    const RoutingCFG &cfg, ArrayRef<BlockInfo> blocks,
    ArrayRef<unsigned> consumerNodes) {
  // A node is on the way when a path leads from it to a consumer
  SmallVector<bool> onTheWay(cfg.numNodes(), false);
  for (unsigned node : consumerNodes)
    onTheWay[node] = true;
  bool grew = true;
  while (grew) {
    grew = false;
    for (const RoutingEdge &edge : cfg.edges) {
      if (!onTheWay[edge.from] && onTheWay[edge.to]) {
        onTheWay[edge.from] = true;
        grew = true;
      }
    }
  }

  // A node is needed when the control token starts there, a producer is
  // on one of its edges, or it consumes
  llvm::DenseSet<unsigned> needed(consumerNodes.begin(), consumerNodes.end());
  needed.insert(ENTRY_BB);
  for (const RoutingEdge &edge : cfg.edges) {
    if (deliversProducer(edge, blocks))
      needed.insert(edge.from);
  }

  // Keep the edges leading to a consumer only. A node whose block has a
  // conditional terminator and whose kept edges leave on fewer sides than
  // the terminator has, has a dropped side and must keep its branch.
  RoutingCFG folded;
  folded.nodeBlock = cfg.nodeBlock;
  for (const RoutingEdge &edge : cfg.edges) {
    if (onTheWay[edge.to])
      folded.edges.push_back(edge);
  }
  SmallVector<bool> hasDroppedSide(cfg.numNodes(), false);
  for (unsigned node = 0; node < cfg.numNodes(); ++node) {
    llvm::DenseSet<unsigned> sides;
    for (const RoutingEdge &edge : folded.edges) {
      if (edge.from == node)
        sides.insert(static_cast<unsigned>(edge.side));
    }
    hasDroppedSide[node] =
        !sides.empty() && sides.size() < blocks[cfg.nodeBlock[node]].numSides;
  }

  bool changed = true;
  while (changed) {
    changed = false;

    // Reconvergence: both sides of a node land on the same node. The single
    // edge delivers, to each block argument, the true side's value, with
    // the false side's kept where it differs so that routing selects
    // between them.
    for (unsigned node = 0; node < cfg.numNodes() && !changed; ++node) {
      SmallVector<unsigned> outgoing;
      for (auto [idx, edge] : llvm::enumerate(folded.edges))
        if (edge.from == node)
          outgoing.push_back(idx);
      if (outgoing.size() != 2)
        continue;
      RoutingEdge &first = folded.edges[outgoing[0]];
      RoutingEdge &second = folded.edges[outgoing[1]];
      if (first.to != second.to || first.side == EdgeSide::UNCONDITIONAL)
        continue;
      RoutingEdge &onTrue = first.side == EdgeSide::TRUE ? first : second;
      RoutingEdge &onFalse = first.side == EdgeSide::TRUE ? second : first;
      for (EdgeValue &delivered : onTrue.values) {
        Value falseValue;
        for (const EdgeValue &other : onFalse.values) {
          if (other.to == delivered.to)
            falseValue = other.value;
        }
        assert(falseValue && "false side delivers nothing to the argument");
        if (falseValue != delivered.value)
          delivered.onFalse = falseValue;
      }
      first.side = EdgeSide::UNCONDITIONAL;
      first.values = onTrue.values;
      folded.edges.erase(folded.edges.begin() + outgoing[1]);
      changed = true;
    }
    if (changed)
      continue;

    // An unneeded node with its single edge leading to itself is entered
    // once and left once through its other, dropped-free path: the loop on
    // it is not needed by this route
    for (unsigned node = 0; node < cfg.numNodes() && !changed; ++node) {
      if (needed.contains(node) || hasDroppedSide[node])
        continue;
      for (auto [idx, edge] : llvm::enumerate(folded.edges)) {
        if (edge.from == node && edge.to == node) {
          folded.edges.erase(folded.edges.begin() + idx);
          changed = true;
          break;
        }
      }
    }
    if (changed)
      continue;

    // Bypass: an unneeded node with one outgoing edge, no dropped side and
    // no select to place. Its incoming edges land on its successor and
    // deliver there what its own edge delivered, taken from what they
    // delivered to the node's own block arguments.
    for (unsigned node = 0; node < cfg.numNodes() && !changed; ++node) {
      if (needed.contains(node) || hasDroppedSide[node])
        continue;
      SmallVector<unsigned> outgoing;
      for (auto [idx, edge] : llvm::enumerate(folded.edges))
        if (edge.from == node)
          outgoing.push_back(idx);
      if (outgoing.size() != 1 || deliversSelect(folded.edges[outgoing[0]]))
        continue;
      RoutingEdge own = folded.edges[outgoing[0]];
      folded.edges.erase(folded.edges.begin() + outgoing[0]);
      for (RoutingEdge &edge : folded.edges) {
        if (edge.to != node)
          continue;
        SmallVector<EdgeValue> values;
        for (const EdgeValue &delivered : own.values) {
          assert(isArgumentOf(delivered.value, blocks[node].block) &&
                 "bypassed node delivers a producer");
          for (const EdgeValue &incoming : edge.values) {
            if (incoming.to == delivered.value)
              values.push_back({delivered.to, incoming.value, incoming.onFalse});
          }
        }
        edge.to = own.to;
        edge.values = values;
      }
      changed = true;
    }
  }
  return folded;
}

/// The data a value stands for at a node: for a block argument of the
/// node's block, the data arriving at that argument, which is its mux
/// output or, with a single incoming edge, its placeholder; for anything
/// else, a producer, the value itself.
static Value dataAtNode(Value value, Block *block, const RouteResult &result,
                        DenseMap<BlockArgument, SmallVector<Backedge>> &incomingData) {
  if (!isArgumentOf(value, block))
    return value;
  auto arg = cast<BlockArgument>(value);
  if (result.data.contains(arg))
    return result.data.lookup(arg);
  assert(incomingData[arg].size() == 1 && "argument has no data arriving");
  return incomingData[arg].front();
}

RouteResult dynamatic::experimental::operand_routing::routeValue(
    const RoutingCFG &cfg, ArrayRef<BlockInfo> blocks, Value start,
    ArrayRef<unsigned> tappedNodes, Attribute consumerId,
    OpBuilder &builder) {
  Location loc = blocks.front().block->getParent()->getLoc();
  Type controlType = start.getType();
  RouteResult result;
  llvm::DenseSet<unsigned> tapped(tappedNodes.begin(), tappedNodes.end());

  llvm::errs() << "[operand-routing] cfg";
  for (const RoutingEdge &edge : cfg.edges) {
    llvm::errs() << " " << edge.from
                 << (edge.side == EdgeSide::TRUE    ? "T"
                     : edge.side == EdgeSide::FALSE ? "F"
                                                    : "U")
                 << edge.to;
  }
  llvm::errs() << "\n";

  // The edges into each node, in edge order, and the block arguments of
  // each node that its edges deliver to
  DenseMap<unsigned, SmallVector<unsigned>> incomingEdges;
  DenseMap<unsigned, SmallVector<BlockArgument>> deliveredArgs;
  for (auto [idx, edge] : llvm::enumerate(cfg.edges)) {
    incomingEdges[edge.to].push_back(idx);
    for (const EdgeValue &delivered : edge.values) {
      if (!llvm::is_contained(deliveredArgs[edge.to], delivered.to))
        deliveredArgs[edge.to].push_back(delivered.to);
    }
  }

  // The branches at the source nodes feed the merges of the destination
  // nodes, and in a loop a merge feeds a branch that feeds it back. The
  // merges are therefore created first with placeholder operands, one per
  // incoming edge, which the branches fill in afterwards. Every edge into
  // a node delivers to every one of its block arguments, so a block
  // argument gets one placeholder per incoming edge.
  BackedgeBuilder edgeBuilder(builder, loc);
  DenseMap<unsigned, SmallVector<Backedge>> incomingControl;
  DenseMap<BlockArgument, SmallVector<Backedge>> incomingData;
  for (unsigned node = 0; node < cfg.numNodes(); ++node) {
    if (incomingEdges[node].empty())
      continue;
    builder.setInsertionPointToStart(blocks[cfg.nodeBlock[node]].block);
    SmallVector<Value> controlOperands;
    for (unsigned idx : incomingEdges[node]) {
      (void)idx;
      Backedge placeholder = edgeBuilder.get(controlType);
      incomingControl[node].push_back(placeholder);
      controlOperands.push_back(Value(placeholder));
    }
    DenseMap<BlockArgument, SmallVector<Value>> dataOperands;
    for (BlockArgument arg : deliveredArgs[node]) {
      for (unsigned idx : incomingEdges[node]) {
        (void)idx;
        Backedge placeholder = edgeBuilder.get(channelifyType(arg.getType()));
        incomingData[arg].push_back(placeholder);
        dataOperands[arg].push_back(Value(placeholder));
      }
    }

    // A node with a single incoming edge needs no merge: what arrives on
    // that edge is what is at the node, recorded once the edge is resolved
    // below
    if (controlOperands.size() == 1)
      continue;

    auto cmergeOp =
        builder.create<handshake::ControlMergeOp>(loc, controlOperands);
    if (consumerId)
      cmergeOp->setAttr(CONSUMER_ID_ATTR, consumerId);
    result.control[node] = cmergeOp.getResult();

    // Each block argument's mux takes its inputs on the same edges as the
    // control merge, in the same order, so the control merge's index picks
    // the input
    for (BlockArgument arg : deliveredArgs[node]) {
      auto muxOp = builder.create<handshake::MuxOp>(
          loc, channelifyType(arg.getType()), cmergeOp.getIndex(),
          dataOperands[arg]);
      if (consumerId)
        muxOp->setAttr(CONSUMER_ID_ATTR, consumerId);
      result.data[arg] = muxOp.getResult();
    }
  }

  // The continuation outputs of the lazy forks, which get their buffering
  // constraint once their consumers are final
  SmallVector<Value> continuations;

  // Branch the control token and the data out of every node along its
  // edges. The control token a node passes on is what arrives at it: for a
  // node with a single incoming edge its placeholder, resolved by the
  // predecessor's branch; for the entry block the start token. It gets one
  // branch when the node's sides are kept. The data leaving on an edge is
  // what the edge delivers, as it is at this node: for a block argument of
  // the node the data arriving at it, for a producer the value itself,
  // with a select on the node's condition where the edge was folded from
  // two sides delivering different values. Each distinct data value
  // leaving on a side gets its own branch, with only the sides carrying
  // that value connected.
  for (unsigned node = 0; node < cfg.numNodes(); ++node) {
    const BlockInfo &block = blocks[cfg.nodeBlock[node]];
    Value control;
    if (node == ENTRY_BB)
      control = start;
    else if (result.control.contains(node))
      control = result.control[node];
    else if (incomingControl[node].size() == 1)
      control = incomingControl[node].front();
    if (!control)
      continue;

    builder.setInsertionPoint(block.block->getTerminator());

    // A tapped node consumes the control token and passes it on: the lazy
    // fork hands the token to the node's consumer and produces the
    // continuation for the rest of the network together, so that the
    // network cannot run ahead of the consumer
    if (tapped.contains(node)) {
      auto forkOp = builder.create<handshake::LazyForkOp>(loc, control, 2);
      if (consumerId)
        forkOp->setAttr(CONSUMER_ID_ATTR, consumerId);
      result.control[node] = forkOp->getResult(0);
      control = forkOp->getResult(1);
      result.continuing[node] = control;
      continuations.push_back(control);
    }

    SmallVector<unsigned> outgoing;
    for (auto [idx, edge] : llvm::enumerate(cfg.edges)) {
      if (edge.from == node)
        outgoing.push_back(idx);
    }
    if (outgoing.empty())
      continue;

    // The control token leaves on one unconditional edge, which needs no
    // operation, or on sides, which need a branch
    Value controlOnTrue = control, controlOnFalse;
    if (cfg.edges[outgoing.front()].side != EdgeSide::UNCONDITIONAL) {
      auto branchOp = builder.create<handshake::ConditionalBranchOp>(
          loc, TypeRange{controlType, controlType},
          ValueRange{getBlockCondition(block), control});
      if (consumerId)
        branchOp->setAttr(CONSUMER_ID_ATTR, consumerId);
      controlOnTrue = branchOp.getTrueResult();
      controlOnFalse = branchOp.getFalseResult();
      result.conditionalBranches[node] = branchOp;
    }
    for (unsigned edgeIdx : outgoing) {
      const RoutingEdge &edge = cfg.edges[edgeIdx];
      const SmallVector<unsigned> &destinationEdges = incomingEdges[edge.to];
      unsigned position =
          llvm::find(destinationEdges, edgeIdx) - destinationEdges.begin();
      Value onEdge =
          edge.side == EdgeSide::FALSE ? controlOnFalse : controlOnTrue;
      incomingControl[edge.to][position].setValue(onEdge);
      if (incomingControl[edge.to].size() == 1 &&
          !result.control.contains(edge.to))
        result.control[edge.to] = onEdge;
    }

    // The data values leaving the node, each with the edges and the block
    // arguments it is delivered to, in edge order
    SmallVector<std::pair<Value, SmallVector<std::pair<unsigned, BlockArgument>>>>
        leaving;
    for (unsigned edgeIdx : outgoing) {
      const RoutingEdge &edge = cfg.edges[edgeIdx];
      for (const EdgeValue &delivered : edge.values) {
        Value value =
            dataAtNode(delivered.value, block.block, result, incomingData);
        if (delivered.onFalse) {
          Value onFalse = dataAtNode(delivered.onFalse, block.block, result,
                                     incomingData);
          OperationState state(loc, handshake::SelectOp::getOperationName(),
                               {getBlockCondition(block), value, onFalse},
                               channelifyType(delivered.to.getType()));
          Operation *selectOp = builder.create(state);
          if (consumerId)
            selectOp->setAttr(CONSUMER_ID_ATTR, consumerId);
          value = selectOp->getResult(0);
        }
        bool found = false;
        for (auto &[leavingValue, targets] : leaving) {
          if (leavingValue == value) {
            targets.push_back({edgeIdx, delivered.to});
            found = true;
          }
        }
        if (!found)
          leaving.push_back({value, {{edgeIdx, delivered.to}}});
      }
    }

    for (auto &[value, targets] : leaving) {
      // A value leaves a node either on one unconditional edge, which needs
      // no operation, or on sides, which need a branch whose results are
      // channels from the start and whose data operand is whatever the
      // producer is at this point, which may still be a cf-level value that
      // the operation conversion later replaces
      Type dataType = channelifyType(targets.front().second.getType());
      Value trueResult = value, falseResult;
      if (cfg.edges[targets.front().first].side != EdgeSide::UNCONDITIONAL) {
        auto branchOp = builder.create<handshake::ConditionalBranchOp>(
            loc, TypeRange{dataType, dataType},
            ValueRange{getBlockCondition(block), value});
        if (consumerId)
          branchOp->setAttr(CONSUMER_ID_ATTR, consumerId);
        trueResult = branchOp.getTrueResult();
        falseResult = branchOp.getFalseResult();
      }

      // Fill in the placeholder of each block argument the value is
      // delivered to. The placeholders of an argument were created in its
      // node's incoming edge order, so the edge's position in that order is
      // its placeholder.
      for (auto [edgeIdx, arg] : targets) {
        const RoutingEdge &edge = cfg.edges[edgeIdx];
        const SmallVector<unsigned> &destinationEdges = incomingEdges[edge.to];
        unsigned position =
            llvm::find(destinationEdges, edgeIdx) - destinationEdges.begin();
        Value onEdge =
            edge.side == EdgeSide::FALSE ? falseResult : trueResult;
        incomingData[arg][position].setValue(onEdge);
        if (destinationEdges.size() == 1 && !result.data.contains(arg))
          result.data[arg] = onEdge;
      }
    }
  }

  // The continuation of a tap gets a transparent buffer slot, cutting its
  // ready path, the same constraint the LSQ control path places on a lazy
  // fork output that does not feed the queue. The placeholders are resolved,
  // so each continuation now has its real single consumer, except the one
  // of the route's last node, whose consumer the caller connects later and
  // constrains itself.
  for (Value continuation : continuations) {
    if (continuation.use_empty())
      continue;
    buffer::Channel channel(continuation, true);
    channel.props->minTrans = std::max(channel.props->minTrans, 1U);
  }

  return result;
}

SmallVector<Consumer>
dynamatic::experimental::operand_routing::collectConsumers(Region &body) {
  SmallVector<Consumer> consumers;

  // Entry block arguments are function arguments, not merged values, so only
  // non-entry blocks can hold consumers
  for (Block &block : llvm::drop_begin(body)) {
    for (BlockArgument arg : block.getArguments()) {
      Consumer consumer{arg, {}};
      for (OpOperand &operand : arg.getUses()) {
        Operation *op = operand.getOwner();
        if (op->getBlock() != &block)
          continue;
        // A branch's successor operands pass values on rather than read
        // them; the condition of a conditional branch reads, and so do the
        // operands of a return, which the end takes over
        if (op->hasTrait<OpTrait::IsTerminator>() && !isa<func::ReturnOp>(op) &&
            !(isa<cf::CondBranchOp>(op) && operand.getOperandNumber() == 0))
          continue;
        consumer.operands.push_back(&operand);
      }
      if (!consumer.operands.empty())
        consumers.push_back(consumer);
    }
  }
  return consumers;
}

//===----------------------------------------------------------------------===//
// Function lowering
//===----------------------------------------------------------------------===//

/// Step 1a: creates the handshake function and moves the body into it. The
/// handshake function takes the original arguments with channel types, a
/// control argument per memory region and the start control, and returns the
/// original results, a control per memory region and the end control.
static handshake::FuncOp createHandshakeFunction(func::FuncOp funcOp,
                                                 OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  Type controlType = handshake::ControlType::get(ctx);

  // Argument and result names, with the memory region names reused for the
  // per-region start and end controls
  SmallVector<Attribute> argNames, resNames;
  SmallVector<std::string> memNames;
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    auto name = funcOp.getArgAttrOfType<StringAttr>(idx, "handshake.arg_name");
    assert(name && "function argument has no name");
    argNames.push_back(name);
    if (isa<MemRefType>(arg.getType()))
      memNames.push_back(name.str());
  }
  for (const std::string &memName : memNames)
    argNames.push_back(StringAttr::get(ctx, memName + "_start"));
  argNames.push_back(StringAttr::get(ctx, "start"));
  for (unsigned idx = 0; idx < funcOp.getNumResults(); ++idx)
    resNames.push_back(StringAttr::get(ctx, "out" + std::to_string(idx)));
  for (const std::string &memName : memNames)
    resNames.push_back(StringAttr::get(ctx, memName + "_end"));
  resNames.push_back(StringAttr::get(ctx, "end"));

  SmallVector<Type> argTypes, resTypes;
  for (Type type : funcOp.getArgumentTypes())
    argTypes.push_back(channelifyType(type));
  for (Type type : funcOp.getResultTypes())
    resTypes.push_back(channelifyType(type));
  argTypes.append(memNames.size() + 1, controlType);
  resTypes.append(memNames.size() + 1, controlType);

  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(StringAttr::get(ctx, "argNames"),
                     ArrayAttr::get(ctx, argNames));
  attrs.emplace_back(StringAttr::get(ctx, "resNames"),
                     ArrayAttr::get(ctx, resNames));

  builder.setInsertionPoint(funcOp);
  auto newFuncOp = builder.create<handshake::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      builder.getFunctionType(argTypes, resTypes), attrs);
  newFuncOp.getBody().takeBody(funcOp.getBody());

  // The entry block's arguments are the function's: give them their channel
  // types and add the control arguments
  Block &entry = newFuncOp.getBody().front();
  for (BlockArgument arg : entry.getArguments())
    arg.setType(channelifyType(arg.getType()));
  for (unsigned i = 0, e = memNames.size() + 1; i < e; ++i)
    entry.addArgument(controlType, funcOp.getLoc());

  // The other blocks' arguments are channels too, so that an operation
  // converted while it still reads one gets channel result types. Index
  // arguments have no channel type and keep theirs.
  for (Block &block : llvm::drop_begin(newFuncOp.getBody())) {
    for (BlockArgument arg : block.getArguments()) {
      if (Type channelType = channelifyType(arg.getType()))
        arg.setType(channelType);
    }
  }

  return newFuncOp;
}

/// Whether a constant can be triggered by an always-valid source. A constant
/// that a terminator passes to another block or returns, or that a load or
/// a store reads, must instead be triggered by its block's control, so that
/// it produces one token per execution of the block. This is asked before
/// any routing, while the function is still cf-level.
static bool isSourcable(Operation *op) {
  if (!isa<arith::ConstantOp>(op))
    return false;
  for (Operation *user : op->getUsers()) {
    if (isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp, memref::LoadOp,
            memref::StoreOp>(user))
      return false;
  }
  return true;
}

/// The data type of a value: the type a channel carries, or the value's own
/// type while it is still cf-level.
static Type getDataType(Type type) {
  if (auto channelType = dyn_cast<handshake::ChannelType>(type))
    return channelType.getDataType();
  return type;
}

/// Step 3: replaces one cf-level operation by its handshake counterpart, at
/// the same position, with channel result types and the same operands and
/// attributes. Uses of the old results move to the new ones, and so does the
/// name. Returns failure after emitting an error for an unsupported
/// operation.
static LogicalResult convertOperation(Operation *op, Value blockControl,
                                      NameAnalysis &namer, OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  Location loc = op->getLoc();
  builder.setInsertionPoint(op);

  // Every result becomes a channel
  SmallVector<Type> resultTypes;
  for (Type type : op->getResultTypes())
    resultTypes.push_back(channelifyType(type));

  Operation *newOp = nullptr;
  if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
    // Index constants become 32-bit integers
    Value control;
    if (isSourcable(op))
      control = builder.create<handshake::SourceOp>(loc).getResult();
    else
      control = blockControl;
    TypedAttr cstAttr = cstOp.getValue();
    if (isa<IndexType>(cstAttr.getType())) {
      cstAttr = IntegerAttr::get(
          builder.getIntegerType(32),
          cast<IntegerAttr>(cstAttr).getValue().trunc(32));
    }
    newOp = builder.create<handshake::ConstantOp>(loc, cstAttr, control);
    newOp->setDialectAttrs(op->getDialectAttrs());
  } else if (auto undefOp = dyn_cast<LLVM::UndefOp>(op)) {
    // An undefined value becomes a zero triggered by the block's control
    Type resType = getDataType(undefOp.getRes().getType());
    TypedAttr cstAttr;
    if (isa<IndexType>(resType))
      cstAttr = builder.getIntegerAttr(builder.getIntegerType(32), 0);
    else if (isa<IntegerType>(resType))
      cstAttr = builder.getIntegerAttr(resType, 0);
    else if (auto floatType = dyn_cast<FloatType>(resType))
      cstAttr = builder.getFloatAttr(floatType, 0.0);
    else
      return op->emitError() << "operation has unsupported result type";
    newOp = builder.create<handshake::ConstantOp>(loc, cstAttr, blockControl);
    newOp->setDialectAttrs(op->getAttrDictionary());
  } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
    // An index cast is an extension or a truncation depending on the widths,
    // with index counted as 32 bits
    Type srcType = getDataType(op->getOperand(0).getType());
    Type dstType = getDataType(op->getResult(0).getType());
    unsigned srcWidth = isa<IndexType>(srcType)
                            ? 32
                            : srcType.getIntOrFloatBitWidth();
    unsigned dstWidth = isa<IndexType>(dstType)
                            ? 32
                            : dstType.getIntOrFloatBitWidth();
    Type newType =
        handshake::ChannelType::get(builder.getIntegerType(dstWidth));
    StringRef newName;
    if (srcWidth >= dstWidth)
      newName = handshake::TruncIOp::getOperationName();
    else if (isa<arith::IndexCastOp>(op))
      newName = handshake::ExtSIOp::getOperationName();
    else
      newName = handshake::ExtUIOp::getOperationName();
    OperationState state(loc, newName, op->getOperands(), newType,
                         op->getAttrs());
    newOp = builder.create(state);
  } else if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    // Every other arith or math operation has a handshake operation of the
    // same name
    std::string newName =
        ("handshake." + op->getName().stripDialect()).str();
    if (!RegisteredOperationName::lookup(newName, ctx))
      return op->emitError() << "no handshake operation named " << newName;
    OperationState state(loc, newName, op->getOperands(), resultTypes,
                         op->getAttrs());
    newOp = builder.create(state);
  } else {
    return op->emitError() << "unsupported operation in operand routing";
  }

  namer.replaceOp(op, newOp);
  op->replaceAllUsesWith(newOp);
  op->erase();
  return success();
}

/// Merges conditional branches of a block that have the same condition and
/// the same data: the routes of different consumers often branch the same
/// value on the same condition. The first one is kept and takes over the
/// uses of the others' results; materialisation later forks its results. A
/// branch serving several consumers loses its consumer id.
static void mergeIdenticalBranches(Region &body) {
  for (Block &block : body) {
    SmallVector<handshake::ConditionalBranchOp> kept;
    for (auto branchOp : llvm::make_early_inc_range(
             block.getOps<handshake::ConditionalBranchOp>())) {
      handshake::ConditionalBranchOp same = nullptr;
      for (handshake::ConditionalBranchOp candidate : kept) {
        if (candidate.getConditionOperand() ==
                branchOp.getConditionOperand() &&
            candidate.getDataOperand() == branchOp.getDataOperand())
          same = candidate;
      }
      if (!same) {
        kept.push_back(branchOp);
        continue;
      }
      if (same->getAttr(CONSUMER_ID_ATTR) != branchOp->getAttr(CONSUMER_ID_ATTR))
        same->removeAttr(CONSUMER_ID_ATTR);
      branchOp.getTrueResult().replaceAllUsesWith(same.getTrueResult());
      branchOp.getFalseResult().replaceAllUsesWith(same.getFalseResult());
      branchOp.erase();
    }
  }
}

/// A cluster of consumers and constants routed on one folded CFG, sharing
/// one control network. Consumers and constants are indices into the
/// function's lists of them.
struct RouteCluster {
  SmallVector<unsigned> consumers;
  SmallVector<unsigned> constants;
  RoutingCFG folded;
};

/// Folds the CFG for a cluster: the paths of every member consumer on the
/// edges, folded towards every member's block.
static RoutingCFG
foldForCluster(const RoutingCFG &cfg, ArrayRef<BlockInfo> blocks,
               ArrayRef<Consumer> consumers,
               ArrayRef<std::pair<unsigned, Backedge>> constants,
               const RouteCluster &cluster) {
  SmallVector<BlockArgument> consumerArgs;
  SmallVector<unsigned> consumerBlocks;
  for (unsigned idx : cluster.consumers) {
    consumerArgs.push_back(consumers[idx].arg);
    consumerBlocks.push_back(getBlockNumber(consumers[idx].arg.getOwner()));
  }
  for (unsigned idx : cluster.constants)
    consumerBlocks.push_back(constants[idx].first);
  return foldCFG(markPaths(cfg, consumerArgs), blocks, consumerBlocks);
}

/// Whether two folded CFGs have the same edges, whatever the edges deliver.
static bool haveSameEdges(const RoutingCFG &a, const RoutingCFG &b) {
  if (a.edges.size() != b.edges.size())
    return false;
  for (auto [edgeA, edgeB] : llvm::zip(a.edges, b.edges)) {
    if (edgeA.from != edgeB.from || edgeA.to != edgeB.to ||
        edgeA.side != edgeB.side)
      return false;
  }
  return true;
}

/// Returns how many edges of the shared fold the member's own fold does not
/// have, by source, destination and side: the extra CFG the member's data
/// travels when routed on the shared fold.
static unsigned countExtraEdges(const RoutingCFG &shared,
                                const RoutingCFG &own) {
  unsigned extra = 0;
  for (const RoutingEdge &edge : shared.edges) {
    bool found = false;
    for (const RoutingEdge &ownEdge : own.edges) {
      if (ownEdge.from == edge.from && ownEdge.to == edge.to &&
          ownEdge.side == edge.side) {
        found = true;
        break;
      }
    }
    if (!found)
      ++extra;
  }
  return extra;
}

/// Returns the cluster holding the members of both.
static RouteCluster joinClusters(const RouteCluster &a, const RouteCluster &b) {
  RouteCluster joint;
  joint.consumers = a.consumers;
  joint.consumers.append(b.consumers);
  joint.constants = a.constants;
  joint.constants.append(b.constants);
  return joint;
}

handshake::FuncOp dynamatic::experimental::operand_routing::lowerFunction(
    func::FuncOp funcOp, NameAnalysis &namer, unsigned maxCfgs,
    unsigned mergeThreshold, OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  Type controlType = handshake::ControlType::get(ctx);
  Location loc = funcOp.getLoc();

  // The memory regions are the memref arguments, in order
  unsigned numArgs = funcOp.getNumArguments();
  SmallVector<unsigned> memrefArgs;
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      memrefArgs.push_back(idx);
  }

  // Step 1: the handshake function
  handshake::FuncOp newFuncOp = createHandshakeFunction(funcOp, builder);
  Region &body = newFuncOp.getBody();
  Value start = newFuncOp.getArguments().back();

  // Step 2: every operation becomes its handshake counterpart, so that
  // every value the routing delivers is a channel. An operation keeps its
  // operands as they are, block arguments included; the routing replaces
  // those later. The arith and math operations go first.
  for (Block &block : body) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<arith::ConstantOp, LLVM::UndefOp, memref::LoadOp,
              memref::StoreOp, UnrealizedConversionCastOp>(op) ||
          isa<handshake::HandshakeDialect>(op.getDialect()))
        continue;
      if (failed(convertOperation(&op, nullptr, namer, builder)))
        return nullptr;
    }
  }

  // A constant that cannot be sourced, and the zero standing in for an
  // undefined value, consume start, which step 5 routes to their block.
  // Until then a placeholder stands for that control.
  BackedgeBuilder constantControlEdges(builder, loc);
  SmallVector<std::pair<unsigned, Backedge>> pendingConstantControls;
  for (Block &block : body) {
    unsigned bb = getBlockNumber(&block);
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (!isa<arith::ConstantOp, LLVM::UndefOp>(op))
        continue;
      Value control;
      if (!isSourcable(&op)) {
        builder.setInsertionPointToStart(&body.front());
        Backedge placeholder = constantControlEdges.get(controlType);
        pendingConstantControls.emplace_back(bb, placeholder);
        control = placeholder;
      }
      if (failed(convertOperation(&op, control, namer, builder)))
        return nullptr;
    }
  }

  // Loads and stores become ports, recorded per memory region and per
  // block for the interfaces of step 6. A load's data comes from its
  // interface, so it gets a placeholder until then.
  BackedgeBuilder loadDataEdges(builder, loc);
  SmallVector<llvm::MapVector<Block *, SmallVector<handshake::MemPortOpInterface>>>
      portsPerRegion(memrefArgs.size());
  for (Block &block : body) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (!isa<memref::LoadOp, memref::StoreOp>(op))
        continue;
      auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(&op);
      if (!memAttr) {
        op.emitError() << "memory access must carry its interface label";
        return nullptr;
      }
      builder.setInsertionPoint(&op);
      handshake::MemPortOpInterface portOp;
      Value memref;
      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        memref = loadOp.getMemRef();
        Type dataType = cast<MemRefType>(memref.getType()).getElementType();
        Value data = loadDataEdges.get(channelifyType(dataType));
        auto newOp = builder.create<handshake::LoadOp>(
            op.getLoc(), loadOp.getIndices().front(), data);
        copyDialectAttr<handshake::MemDependenceArrayAttr>(&op, newOp);
        copyDialectAttr<handshake::MemInterfaceAttr>(&op, newOp);
        namer.replaceOp(&op, newOp);
        loadOp.getResult().replaceAllUsesWith(newOp.getDataResult());
        portOp = newOp;
      } else {
        auto storeOp = cast<memref::StoreOp>(op);
        memref = storeOp.getMemRef();
        auto newOp = builder.create<handshake::StoreOp>(
            op.getLoc(), storeOp.getIndices().front(),
            storeOp.getValueToStore());
        copyDialectAttr<handshake::MemDependenceArrayAttr>(&op, newOp);
        copyDialectAttr<handshake::MemInterfaceAttr>(&op, newOp);
        namer.replaceOp(&op, newOp);
        portOp = newOp;
      }
      op.erase();

      unsigned region = llvm::find(memrefArgs,
                                   cast<BlockArgument>(memref).getArgNumber()) -
                        memrefArgs.begin();
      portsPerRegion[region][&block].push_back(portOp);
    }
  }

  // Step 3: the block table and the backbone. The block graph has one node
  // per block; every control route is built on it.
  RoutingCFG cfg;
  SmallVector<BlockInfo> blocks;
  readCFG(body, cfg, blocks);

  // A block whose condition is a block argument has that condition routed
  // like any consumer. Every route through the block, the backbone included,
  // needs the routed condition before it is built, so the block gets a
  // placeholder condition until all routes exist.
  BackedgeBuilder conditionEdges(builder, loc);
  SmallVector<std::pair<unsigned, Backedge>> pendingConditions;
  builder.setInsertionPointToStart(&body.front());
  for (unsigned bb = 0; bb < blocks.size(); ++bb) {
    auto condArg = dyn_cast_or_null<BlockArgument>(blocks[bb].condition);
    if (!condArg || condArg.getOwner()->isEntryBlock())
      continue;
    Backedge placeholder = conditionEdges.get(channelifyType(condArg.getType()));
    pendingConditions.emplace_back(bb, placeholder);
    blocks[bb].condition = placeholder;
  }

  // The backbone, on the full CFG. Nothing consumes it; it is the link
  // between the route graphs and the circuit through its branches.
  RouteResult backbone = routeValue(cfg, blocks, start, {}, nullptr, builder);

  // From here on every route reads a block's condition off the backbone's
  // branch in that block
  for (auto [bb, branchOp] : backbone.conditionalBranches)
    blocks[bb].backboneBranch = branchOp;

  // Step 4: the consumers, collected now that their readers are the
  // handshake operations, and the constants that need a control. Each
  // starts as its own cluster, with the CFG folded towards it: for a
  // consumer with the paths from its producers on the edges, for a constant
  // with nothing on them.
  SmallVector<Consumer> consumers = collectConsumers(body);
  for (auto [consumerId, consumer] : llvm::enumerate(consumers)) {
    llvm::errs() << "[operand-routing] consumer " << consumerId << ": bb"
                 << getBlockNumber(consumer.arg.getOwner()) << " arg"
                 << consumer.arg.getArgNumber() << " " << consumer.arg.getType()
                 << " read by";
    for (OpOperand *operand : consumer.operands)
      llvm::errs() << " " << operand->getOwner()->getName();
    llvm::errs() << "\n";
  }
  // Every member's own fold is kept: the merge threshold measures a shared
  // fold against it
  std::vector<RouteCluster> clusters;
  std::vector<RoutingCFG> consumerFolds;
  std::vector<RoutingCFG> constantFolds;
  for (unsigned idx = 0; idx < consumers.size(); ++idx) {
    RouteCluster cluster;
    cluster.consumers.push_back(idx);
    cluster.folded = foldForCluster(cfg, blocks, consumers,
                                    pendingConstantControls, cluster);
    consumerFolds.push_back(cluster.folded);
    clusters.push_back(cluster);
  }
  for (unsigned idx = 0; idx < pendingConstantControls.size(); ++idx) {
    RouteCluster cluster;
    cluster.constants.push_back(idx);
    cluster.folded = foldForCluster(cfg, blocks, consumers,
                                    pendingConstantControls, cluster);
    constantFolds.push_back(cluster.folded);
    clusters.push_back(cluster);
  }

  // Clusters with the same folded CFG become one: routing them together
  // changes nothing in the data and shares the control network
  std::vector<RouteCluster> distinct;
  for (const RouteCluster &cluster : clusters) {
    bool joined = false;
    for (RouteCluster &kept : distinct) {
      if (haveSameEdges(kept.folded, cluster.folded)) {
        kept.consumers.append(cluster.consumers);
        kept.constants.append(cluster.constants);
        kept.folded = foldForCluster(cfg, blocks, consumers,
                                     pendingConstantControls, kept);
        joined = true;
        break;
      }
    }
    if (!joined)
      distinct.push_back(cluster);
  }
  clusters = distinct;

  // Up to the threshold: while two clusters can be joined with every member
  // of the joint fold having at most the threshold's extra edges in it over
  // its own fold, join the pair whose joint fold has the fewest edges. A
  // member is always measured against the current shared fold, so a cluster
  // cannot drift away from any member by repeated joins.
  while (mergeThreshold != 0) {
    unsigned bestFirst = 0, bestSecond = 1;
    RouteCluster best;
    bool haveBest = false;
    for (unsigned first = 0; first < clusters.size(); ++first) {
      for (unsigned second = first + 1; second < clusters.size(); ++second) {
        RouteCluster joint = joinClusters(clusters[first], clusters[second]);
        joint.folded = foldForCluster(cfg, blocks, consumers,
                                      pendingConstantControls, joint);
        bool admissible = true;
        for (unsigned idx : joint.consumers) {
          if (countExtraEdges(joint.folded, consumerFolds[idx]) > mergeThreshold)
            admissible = false;
        }
        for (unsigned idx : joint.constants) {
          if (countExtraEdges(joint.folded, constantFolds[idx]) > mergeThreshold)
            admissible = false;
        }
        if (!admissible)
          continue;
        if (!haveBest || joint.folded.edges.size() < best.folded.edges.size()) {
          best = joint;
          bestFirst = first;
          bestSecond = second;
          haveBest = true;
        }
      }
    }
    if (!haveBest)
      break;
    clusters[bestFirst] = best;
    clusters.erase(clusters.begin() + bestSecond);
  }

  // Down to the limit: join the two clusters whose joint fold has the
  // fewest edges, which is the fewest blocks a member passes through for
  // nothing
  while (maxCfgs != 0 && clusters.size() > maxCfgs) {
    unsigned bestFirst = 0, bestSecond = 1;
    RouteCluster best;
    bool haveBest = false;
    for (unsigned first = 0; first < clusters.size(); ++first) {
      for (unsigned second = first + 1; second < clusters.size(); ++second) {
        RouteCluster joint = joinClusters(clusters[first], clusters[second]);
        joint.folded = foldForCluster(cfg, blocks, consumers,
                                      pendingConstantControls, joint);
        if (!haveBest || joint.folded.edges.size() < best.folded.edges.size()) {
          best = joint;
          bestFirst = first;
          bestSecond = second;
          haveBest = true;
        }
      }
    }
    clusters[bestFirst] = best;
    clusters.erase(clusters.begin() + bestSecond);
  }

  // One walk per cluster builds its control network and every member's
  // data together. Every operand reading a member block argument gets the
  // one value arriving; materialisation forks it. A member constant takes
  // the control token at its block.
  for (auto [clusterId, cluster] : llvm::enumerate(clusters)) {
    IntegerAttr consumerIdAttr = builder.getUI32IntegerAttr(clusterId);
    llvm::errs() << "[operand-routing] cluster " << clusterId << ": consumers";
    for (unsigned idx : cluster.consumers)
      llvm::errs() << " " << idx;
    llvm::errs() << " constants";
    for (unsigned idx : cluster.constants)
      llvm::errs() << " " << idx;
    llvm::errs() << "\n";

    RouteResult route =
        routeValue(cluster.folded, blocks, start, {}, consumerIdAttr, builder);
    for (unsigned idx : cluster.consumers) {
      Value arriving = route.data.lookup(consumers[idx].arg);
      assert(arriving && "no data arrived at a member consumer");
      for (OpOperand *operand : consumers[idx].operands)
        operand->set(arriving);
    }
    for (unsigned idx : cluster.constants) {
      auto &[bb, placeholder] = pendingConstantControls[idx];
      Value control = bb == ENTRY_BB ? start : route.control.lookup(bb);
      assert(control && "no control token arrived at a member constant");
      placeholder.setValue(control);
    }
  }

  // The conditional branches that consumed a block argument now read their
  // routed condition, which is what the placeholders stood for
  for (auto &[bb, placeholder] : pendingConditions) {
    auto condBrOp = cast<cf::CondBranchOp>(blocks[bb].block->getTerminator());
    placeholder.setValue(condBrOp.getCondition());
  }

  // Step 6: the memory. Each memory interface gets its own network: start
  // routed through every block it takes a token from, in program order, to
  // the return block. For a memory controller those are the blocks storing to
  // it, directly or through an LSQ, and in each a lazy fork hands the token to
  // the count constant and on to the rest of the network, so the end token
  // reaches the controller only after every count it has to wait for. For an
  // LSQ those are the blocks with a load or store labelled for it, and the
  // tapped token is the block's allocation input.
  SmallVector<Block *> returnBlocks;
  for (Block &block : body) {
    if (isa<func::ReturnOp>(block.getTerminator()))
      returnBlocks.push_back(&block);
  }
  if (returnBlocks.size() != 1) {
    funcOp.emitError() << "operand routing supports a single return only";
    return nullptr;
  }
  unsigned returnBlock = getBlockNumber(returnBlocks.front());

  // The tapped blocks consume the token and produce it on for the rest of
  // the network; the return block consumes it as the end control, after its
  // own tap when it has one. The end token goes to the memory controller
  // when the region has one, else to the LSQ, so only that network reaches
  // the return block.
  SmallVector<DenseMap<unsigned, Value>> controllerControls(memrefArgs.size());
  SmallVector<DenseMap<unsigned, Value>> lsqControls(memrefArgs.size());
  SmallVector<Value> memoryEnds(memrefArgs.size());
  SmallVector<bool> hasController(memrefArgs.size(), false);
  for (auto [region, argIdx] : llvm::enumerate(memrefArgs)) {
    SmallVector<unsigned> storeBlocks;
    SmallVector<unsigned> lsqBlocks;
    for (auto &[block, ports] : portsPerRegion[region]) {
      bool hasStore = false;
      bool hasLSQPort = false;
      for (handshake::MemPortOpInterface portOp : ports) {
        auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(portOp);
        assert(memAttr && "memory port must carry its interface label");
        if (isa<handshake::StoreOp>(portOp.getOperation()))
          hasStore = true;
        if (memAttr.connectsToMC())
          hasController[region] = true;
        else
          hasLSQPort = true;
      }
      if (hasStore)
        storeBlocks.push_back(getBlockNumber(block));
      if (hasLSQPort)
        lsqBlocks.push_back(getBlockNumber(block));
    }

    // The memory controller's network
    if (hasController[region]) {
      SmallVector<unsigned> consumerBlocks(storeBlocks);
      consumerBlocks.push_back(returnBlock);
      RoutingCFG folded = foldCFG(cfg, blocks, consumerBlocks);
      RouteResult network =
          routeValue(folded, blocks, start, storeBlocks, nullptr, builder);

      for (unsigned bb : storeBlocks)
        controllerControls[region][bb] = network.control.lookup(bb);
      if (Value continuing = network.continuing.lookup(returnBlock))
        memoryEnds[region] = continuing;
      else
        memoryEnds[region] = network.control.lookup(returnBlock);
    }

    // The LSQ's network
    if (!lsqBlocks.empty()) {
      SmallVector<unsigned> consumerBlocks(lsqBlocks);
      if (!hasController[region])
        consumerBlocks.push_back(returnBlock);
      RoutingCFG folded = foldCFG(cfg, blocks, consumerBlocks);
      RouteResult network =
          routeValue(folded, blocks, start, lsqBlocks, nullptr, builder);

      for (unsigned bb : lsqBlocks)
        lsqControls[region][bb] = network.control.lookup(bb);
      if (!hasController[region]) {
        if (Value continuing = network.continuing.lookup(returnBlock))
          memoryEnds[region] = continuing;
        else
          memoryEnds[region] = network.control.lookup(returnBlock);
      }
    }

    if (!memoryEnds[region])
      memoryEnds[region] = start;
  }

  // The memory interfaces. Ports must carry their bb id before the
  // interfaces are built.
  for (Block &block : body) {
    unsigned bb = getBlockNumber(&block);
    for (Operation &op : block)
      op.setAttr(BB_ATTR_NAME, builder.getUI32IntegerAttr(bb));
  }

  SmallVector<Value> memEnds;
  for (auto [region, argIdx] : llvm::enumerate(memrefArgs)) {
    Value memref = newFuncOp.getArgument(argIdx);
    Value memStart = newFuncOp.getArgument(numArgs + region);
    MemoryInterfaceBuilder memBuilder(newFuncOp, memref, memStart,
                                      memoryEnds[region],
                                      controllerControls[region],
                                      lsqControls[region]);

    // Each port goes to the interface its label names; an LSQ port's group
    // is its block
    for (auto &[block, ports] : portsPerRegion[region]) {
      for (handshake::MemPortOpInterface portOp : ports) {
        auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(portOp);
        if (memAttr.connectsToMC())
          memBuilder.addMCPort(portOp);
        else
          memBuilder.addLSQPort(*memAttr.getLsqGroup(), portOp);
      }
    }
    handshake::MemoryControllerOp mcOp;
    handshake::LSQOp lsqOp;
    if (failed(memBuilder.instantiateInterfaces(builder, mcOp, lsqOp)))
      return nullptr;

    // The end token comes from the interface that took the end control: the
    // memory controller when there is one, else the LSQ. A region without
    // accesses gets an always-valid end token.
    if (mcOp || lsqOp) {
      if (mcOp)
        memEnds.push_back(mcOp.getMemEnd());
      else
        memEnds.push_back(lsqOp.getMemEnd());

      // When the end control is the continuation of the return block's
      // tap, it gets the same ready-cutting slot as every other continuation
      // now that the interface consumes it
      if (memoryEnds[region].getDefiningOp<handshake::LazyForkOp>()) {
        buffer::Channel channel(memoryEnds[region], true);
        channel.props->minTrans = std::max(channel.props->minTrans, 1U);
      }

      // The builder places each store-count constant right after the block
      // control it is triggered by, which for an always-reached block is
      // start, in the entry block. The constant belongs to the block whose
      // stores it counts.
      for (auto [bb, control] : controllerControls[region]) {
        for (Operation *user : control.getUsers()) {
          auto cstOp = dyn_cast<handshake::ConstantOp>(user);
          if (!cstOp || !mcOp || !cstOp.getResult().hasOneUse() ||
              *cstOp.getResult().getUsers().begin() != mcOp.getOperation())
            continue;
          cstOp->setAttr(BB_ATTR_NAME, builder.getUI32IntegerAttr(bb));
        }
      }
    } else {
      builder.setInsertionPoint(returnBlocks.front()->getTerminator());
      memEnds.push_back(builder.create<handshake::SourceOp>(loc).getResult());
    }
  }

  llvm::errs() << "[operand-routing] memory interfaces built\n";

  // Every route has been built, so the backbone has served as the condition
  // link and feeds nothing outside itself: its branches and control merges
  // only use each other, in cycles where the CFG has loops. Their uses are
  // dropped together before they are erased.
  llvm::DenseSet<Operation *> backboneOps;
  for (auto [bb, branchOp] : backbone.conditionalBranches)
    backboneOps.insert(branchOp);
  for (auto [bb, arriving] : backbone.control) {
    if (auto cmergeOp = arriving.getDefiningOp<handshake::ControlMergeOp>())
      backboneOps.insert(cmergeOp);
  }
  for (Operation *op : backboneOps) {
    for (Operation *user : op->getUsers())
      assert(backboneOps.contains(user) && "backbone used outside itself");
  }
  for (Operation *op : backboneOps)
    op->dropAllUses();
  for (Operation *op : backboneOps)
    op->erase();

  llvm::errs() << "[operand-routing] backbone removed\n";
  mergeIdenticalBranches(body);
  llvm::errs() << "[operand-routing] identical branches merged\n";

  // Step 7: flatten. The return's operands become the end's data operands.
  SmallVector<Value> endOperands;
  auto returnOp = cast<func::ReturnOp>(returnBlocks.front()->getTerminator());
  llvm::append_range(endOperands, returnOp.getOperands());
  llvm::append_range(endOperands, memEnds);
  endOperands.push_back(start);
  unsigned exitBlock = getBlockNumber(returnBlocks.front());

  for (Block &block : body)
    block.getTerminator()->erase();
  Block &entry = body.front();
  for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(body))) {
    for (BlockArgument arg : block.getArguments())
      assert(arg.use_empty() && "block argument still has a consumer");
    entry.getOperations().splice(entry.end(), block.getOperations());
    block.erase();
  }

  // Memory interfaces belong to no block
  for (Operation &op : entry) {
    if (isa<handshake::MemoryOpInterface>(op))
      op.removeAttr(BB_ATTR_NAME);
  }

  builder.setInsertionPointToEnd(&entry);
  auto endOp = builder.create<handshake::EndOp>(loc, endOperands);
  endOp->setAttr(BB_ATTR_NAME, builder.getUI32IntegerAttr(exitBlock));

  llvm::errs() << "[operand-routing] flattened\n";
  funcOp.erase();
  return newFuncOp;
}

} // namespace operand_routing
} // namespace experimental
} // namespace dynamatic

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

namespace {

struct OperandRoutingCfToHandshakePass
    : public dynamatic::experimental::impl::OperandRoutingCfToHandshakeBase<
          OperandRoutingCfToHandshakePass> {
  using OperandRoutingCfToHandshakeBase::OperandRoutingCfToHandshakeBase;

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    ModuleOp modOp = getOperation();
    OpBuilder builder(&getContext());

    // Globals are not supported; the ones left by the C testbench are dropped
    for (auto globalOp :
         llvm::make_early_inc_range(modOp.getOps<memref::GlobalOp>()))
      globalOp.erase();

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<func::FuncOp>())) {
      if (funcOp.isExternal())
        continue;

      // Every live-in of a non-entry block becomes a block argument, so that
      // all values crossing blocks are explicit
      FuncSSAStrategy strategy;
      if (failed(dynamatic::maximizeSSA(funcOp.getBody(), strategy)))
        return signalPassFailure();

      if (!lowerFunction(funcOp, namer, maxCfgs, mergeThreshold, builder))
        return signalPassFailure();
    }

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
