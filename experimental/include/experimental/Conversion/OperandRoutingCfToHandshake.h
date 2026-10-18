//===- OperandRoutingCfToHandshake.h - cf to handshake, per-operand -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --operand-routing-lower-cf-to-handshake pass along
// with the routing it is built on.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_OPERAND_ROUTING_CF_TO_HANDSHAKE_H
#define EXPERIMENTAL_CONVERSION_OPERAND_ROUTING_CF_TO_HANDSHAKE_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace operand_routing {

/// Operation attribute carrying the code of the consumer an operation was
/// created for. Operations without it belong to the backbone control network
/// or to the computation itself.
constexpr llvm::StringLiteral CONSUMER_ID_ATTR("consumer_id");

/// Which result of the source block's terminator an edge leaves on.
enum class EdgeSide { UNCONDITIONAL, TRUE, FALSE };

/// What an edge delivers to one block argument of its destination: the
/// value that argument takes when the edge is taken. A value that is a
/// block argument of the edge's source passes through; anything else is a
/// producer entering on this edge. On an edge folded from the two sides of
/// a block, `onFalse` is the false side's value when it differs from the
/// true side's `value`, and the delivered value is a select between them
/// on the block's condition.
struct EdgeValue {
  BlockArgument to;
  Value value;
  Value onFalse;
};

/// An edge of a route graph, between two nodes, leaving the source node's
/// block on a side of its terminator, with what it delivers to the block
/// arguments of its destination that lie on a path from a producer to the
/// consumer.
struct RoutingEdge {
  unsigned from;
  unsigned to;
  EdgeSide side;
  SmallVector<EdgeValue> values;
};

/// The graph a route is built on: the CFG, one node per block, numbered
/// like the blocks, possibly folded. A node's mux is placed in its block
/// and its branches use the block's condition. A folded edge may land
/// anywhere, but it starts at a real block on a real side of its
/// terminator.
struct RoutingCFG {
  SmallVector<unsigned> nodeBlock;
  SmallVector<RoutingEdge> edges;

  unsigned numNodes() const { return nodeBlock.size(); }
};

/// What the routing needs to know about a real block: where to insert, how
/// many sides its terminator has (two for a conditional branch, one for a
/// branch, none for a return), and where its branch condition comes from.
/// Before the backbone exists the condition is the cf-level value of the
/// block's terminator (null when the terminator is unconditional). Once the
/// backbone exists, its conditional branch in the block is the link to the
/// real circuit: its condition operand is read whenever a route needs the
/// block's condition, so that every replacement of the condition's producer
/// is seen.
struct BlockInfo {
  Block *block;
  unsigned numSides;
  Value condition;
  handshake::ConditionalBranchOp backboneBranch;
};

/// Returns the block's branch condition as it currently is in the circuit.
Value getBlockCondition(const BlockInfo &block);

/// What one walk built: per node the control token arriving there, the
/// control token's conditional branch in the node's block if one was
/// placed, and for a tapped node the continuation leaving it, which is what
/// the node passes on to the rest of the network; per block argument on a
/// path from a producer to the consumer the data arriving at it.
struct RouteResult {
  DenseMap<unsigned, Value> control;
  DenseMap<BlockArgument, Value> data;
  DenseMap<unsigned, handshake::ConditionalBranchOp> conditionalBranches;
  DenseMap<unsigned, Value> continuing;
};

/// Reads the function's CFG off the real blocks and terminators as a route
/// graph with one node per block, numbered like the blocks, along with the
/// per-block information.
void readCFG(Region &body, RoutingCFG &cfg, SmallVector<BlockInfo> &blocks);

/// Returns a copy of the CFG with the paths from the producers to the
/// consumers on its edges. The paths are found by following the block
/// arguments back from each consumer's: every edge into a block argument's
/// block delivers to it what the predecessor passes on that side. A value
/// passed that is itself a block argument of the predecessor is followed
/// further; anything else is a producer.
RoutingCFG markPaths(const RoutingCFG &cfg, ArrayRef<BlockArgument> consumerArgs);

/// Folds the CFG towards the consumer nodes. Edges that do not lead to a
/// consumer are dropped. Then, to a fixpoint: a node whose two sides land
/// on the same node gets one unconditional edge instead, which delivers a
/// select between the two sides' values where they differ; an unneeded
/// node, one that is not the entry block, has no producer and does not
/// consume, with a single edge to itself loses that edge; an unneeded node
/// with a single outgoing edge and no dropped side is bypassed, its
/// incoming edges landing on its successor with their own sides and
/// delivering what its own edge delivered. A node with a dropped side keeps
/// its branch, since tokens taking that side must be discarded. Node
/// numbers are unchanged.
RoutingCFG foldCFG(const RoutingCFG &cfg, ArrayRef<BlockInfo> blocks,
                   ArrayRef<unsigned> consumerNodes);

/// Walks the folded CFG once, building the control network and the data
/// on the paths its edges carry. The control token `start` enters at the
/// entry block and travels on every edge. Where several edges land on a
/// node, a control merge takes the control tokens on them, and every block
/// argument of the node on a path gets a mux taking what those edges
/// deliver to it, indexed by that control merge. Where a node's sides are kept, a
/// branch on the node's block condition for the control token and one per
/// distinct data value leaving on a side, with only the sides carrying that
/// value connected; an edge delivering a select gets the select placed in
/// the node's block. A tapped node consumes the control token and passes it
/// on: it goes through a lazy fork there, whose first result is what the
/// node consumes and whose second result goes on to the rest of the
/// network. Every created operation carries `consumerId` when it is not
/// null.
RouteResult routeValue(const RoutingCFG &cfg, ArrayRef<BlockInfo> blocks,
                       Value start, ArrayRef<unsigned> tappedNodes,
                       Attribute consumerId, OpBuilder &builder);

/// A consumer: a block argument of a non-entry block that is read in its
/// block, with the operands reading it. The successor operands of branches
/// pass block arguments on and do not read them; the condition of a
/// conditional branch and the operands of a return do.
struct Consumer {
  BlockArgument arg;
  SmallVector<mlir::OpOperand *> operands;
};

/// Returns the consumers of the function, in a deterministic order.
SmallVector<Consumer> collectConsumers(Region &body);

/// Lowers the function to a handshake function, routing the consumers and
/// the constants that need a control through control networks: one per
/// cluster, where consumers with identical folded CFGs form a cluster.
/// Then clusters are merged while every member of the joint fold has at
/// most `mergeThreshold` edges in it that its own fold does not have, and
/// then by fewest edges in the joint fold until there are at most
/// `maxCfgs` of them, 0 meaning no limit. Every replaced operation hands
/// its name to its replacement through the name analysis. Returns the
/// handshake function, or nullptr after emitting an error.
handshake::FuncOp lowerFunction(mlir::func::FuncOp funcOp, NameAnalysis &namer,
                                unsigned maxCfgs, unsigned mergeThreshold,
                                OpBuilder &builder);

} // namespace operand_routing
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_OPERAND_ROUTING_CF_TO_HANDSHAKE_H
