//===- AnnotatedCfg.cpp - Annotated CFGs for operand routing 2 --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/OperandRouting2/AnnotatedCfg.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::operand_routing2;

//===----------------------------------------------------------------------===//
// Block uses
//===----------------------------------------------------------------------===//

SetVector<BlockUse>
dynamatic::experimental::operand_routing2::collectBlockUses(Region &body) {
  // set vector gives uniqueness
  // and also iterates over in insertion order
  SetVector<BlockUse> uses;

  // for each basic block
  for (Block &block : body) {
    // the block args of the entry block are function arguments
    // we treat these as producers in the entry block for routing
    // not consumers which need an input
    if (block.isEntryBlock())
      continue;
    // for all other blocks, a subset of the SSA maximized block args
    // are the block uses we need
    for (BlockArgument arg : block.getArguments()) {
      // for each block arg, check what uses it in the basic block
      for (OpOperand &operand : arg.getUses()) {
        Operation *op = operand.getOwner();
        // A strange artifact of MLIR using block arguments instead of phi
        // operations means branch data operands
        // are not true uses belonging to this block
        // but the basic block forwarding the value
        //
        // The construction of BlockToCond,
        // which is how routing finds branch conditions,
        // handles the weirdness of "block uses and cf cond brs",
        // so we skip the cf cond brs here.
        // Branch condition block uses are detected
        // through the operands of the TempCons "condition readers"
        // placed by BlockToCond
        if (isa<cf::BranchOp, cf::CondBranchOp>(op))
          continue;
        // store block use in set
        uses.insert(arg);
        break;
      }
    }
  }
  // return a unique set of block uses
  return uses;
}

//===----------------------------------------------------------------------===//
// Annotating a CFG for block uses
//===----------------------------------------------------------------------===//

/// Checks whether the block produces the value
bool dynamatic::experimental::operand_routing2::isProducedIn(Value value,
                                                             Block *block) {
  // an operation result is produced in the operation's block:
  // check if the given block is equal to the operation's block
  if (Operation *producer = value.getDefiningOp())
    return producer->getBlock() == block;

  // if a value has no producing op,
  // it comes from a block arg

  // this convoluted check is
  // "is the block argument actually a function argument"
  if (value.getParentBlock()->isEntryBlock()) {
    // function arguments are "produced" in the entry block
    // so check if the given block is the entry block
    return block->isEntryBlock();
  }

  // If the value comes from a block arg
  // we don't actually have a real producer:
  // the value is a live-in of whatever block
  // this is a block arg of
  return false;
}

/// Converts from the block input of a block
/// to the live-out of its predecessor
static Value findLiveOutProducerFromBranch(const CfgEdge &edge,
                                           BlockInput blockInput) {
  // a block argument is filled in by the predecessor's branch,
  // with the operand it passes along this edge
  Operation *terminator = edge.from->getTerminator();

  // we need to find the branch operand to get the live-out
  // branch operands match block args by position
  unsigned argIdx = blockInput.getArgNumber();

  // unconditional branches have only one set of operands
  if (auto brOp = dyn_cast<cf::BranchOp>(terminator))
    return brOp.getDestOperands()[argIdx];

  // conditional branches have two sets of operands
  auto condBrOp = cast<cf::CondBranchOp>(terminator);

  // we pick the set of operands based on the edge condition
  if (*edge.condition)
    return condBrOp.getTrueOperand(argIdx);
  else
    return condBrOp.getFalseOperand(argIdx);
}

Cfg dynamatic::experimental::operand_routing2::buildCfg(
    const Cfg &base, const SetVector<BlockUse> &blockUses) {
  // Make a copy of the base cfg
  // to annotate which basic blocks are needed for these consumers
  // as well as the live-outs involved
  Cfg cfg = base;

  // we always mark the entry and exit blocks as needed,
  // to make the control network easier to construct,
  // not because routing needs them
  cfg.needed.insert(cfg.entry);
  cfg.needed.insert(cfg.exit);

  // We need every basic block of each block use
  // we want to route
  for (BlockUse blockUse : blockUses)
    cfg.needed.insert(blockUse.getOwner());

  // The routing algorithm processes many block args
  // but it does something different if the block arg
  // is a block use of the cfg it is routing
  // so we store a copy of the set of block uses on the cfg
  cfg.blockUses = blockUses;

  // We trace back block inputs
  // to find all producers of the block uses:
  //
  // If a block has multiple predecessors,
  // the block input may have multiple producers,
  // and the search splits.
  //
  // When the predecessor does not produce the block input,
  // we need to trace back to the predecessor's own block input.
  //
  // Otherwise, we have found a producer for the block use.
  //
  // We start with the block inputs of the block uses:
  // (because of SSA maximization,
  // the IR representation of both is the same:
  // the block arg of the basic block the use is in)
  SmallVector<BlockInput> blockInputsToTrace(blockUses.begin(),
                                             blockUses.end());
  DenseSet<BlockInput> blockInputsTraced;

  // while there is a block input to trace
  while (!blockInputsToTrace.empty()) {
    // get the next block input to trace
    BlockInput blockInput = blockInputsToTrace.pop_back_val();

    // skip a block input that has already been traced
    if (blockInputsTraced.contains(blockInput))
      continue;

    // mark it as traced
    blockInputsTraced.insert(blockInput);

    // walk the edges into the block of the block input we are tracing,
    // one from each predecessor
    for (CfgEdge *edge : cfg.getInEdges(blockInput.getOwner())) {
      // for this predecessor,
      // get the live-out for this block input
      Value liveOut = findLiveOutProducerFromBranch(*edge, blockInput);

      // annotate the live-out onto the CFG
      edge->liveOuts.push_back(
          {/*liveOut=*/liveOut, /*blockInput=*/blockInput});

      // if the predecessor produces it, we have found a producer,
      // and can mark the producer's block as needed.
      if (isProducedIn(liveOut, edge->from))
        cfg.needed.insert(edge->from);
      else {
        // Otherwise the live-out must be a block input of the predecessor,
        // to trace next.
        // the cast and asserts ensure that the live-out really
        // is a block input of the pred
        BlockInput predBlockInput = cast<BlockInput>(liveOut);
        assert(predBlockInput.getOwner() == edge->from &&
               "live-out is not a block input of the predecessor");
        blockInputsToTrace.push_back(predBlockInput);
      }
    }
  }

  return cfg;
}

//===----------------------------------------------------------------------===//
// Annotating a CFG for memory interfaces
//===----------------------------------------------------------------------===//

/// Builds the CFG for routing the control inputs
/// of one memory interface,
/// sequentialized by a single memory token:
/// a copy of the base CFG,
/// with every block that produces the token needed.
static Cfg buildMemoryInterfaceCfg(const Cfg &base, ArrayRef<Block *> blocks) {
  Cfg cfg = base;

  // we always mark the entry and exit blocks as needed,
  // to make the control network easier to construct,
  // not because routing needs them
  cfg.needed.insert(cfg.entry);
  cfg.needed.insert(cfg.exit);

  // every block the token is produced or consumed in is needed
  for (Block *block : blocks)
    cfg.needed.insert(block);

  return cfg;
}

MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg>
dynamatic::experimental::operand_routing2::collectMemoryInterfaceCfgs(
    const Cfg &base, const MapVector<MemrefValue, ControlValue> &memStarts) {
  // the memory's start produces the token in the entry block
  Block *entryBlock = base.entry;

  // the end control of a region comes from the return block
  Block *returnBlock = base.exit;

  MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg> cfgs;

  // each memory region is a memref with a memStart
  // (a memref argument of the function or a RAM op)
  for (auto [memref, memStart] : memStarts) {

    // collect the loads and stores to this region
    SmallVector<Operation *> accesses;
    for (Operation *user : memref.getUsers()) {
      if (isa<memref::LoadOp, memref::StoreOp>(user))
        accesses.push_back(user);
    }

    // if the region has a memory controller,
    // we must route its control inputs
    //
    // the region has a memory controller
    // if any access is labelled for it.
    // (sometimes a region just has an LSQ)
    bool buildRoutingForMemoryController = false;
    // iterate over the loads and stores of this region
    for (Operation *access : accesses) {
      // get their memory interface
      auto label = getDialectAttr<handshake::MemInterfaceAttr>(access);
      assert(label && "memory access must carry its interface label");
      // if any access needs an MC
      if (label.connectsToMC())
        // we must route for the MC
        buildRoutingForMemoryController = true;
    }

    // the MC's control inputs are the start, store counts and the end
    if (buildRoutingForMemoryController) {
      // collect all blocks with stores to the MC
      SetVector<Block *> storeBlocks;
      for (Operation *access : accesses) {
        if (isa<memref::StoreOp>(access))
          storeBlocks.insert(access->getBlock());
      }

      // start at the entry block, for the start input
      SetVector<Block *> mcBlocks;
      mcBlocks.insert(entryBlock);
      mcBlocks.insert(storeBlocks.begin(), storeBlocks.end());
      // and add the final block as well, for the end input
      mcBlocks.insert(returnBlock);
      // and build a CFG for it
      MemoryInterfaceCfg mcCfg;
      mcCfg.cfg = buildMemoryInterfaceCfg(base, mcBlocks.getArrayRef());
      mcCfg.interfacingBlocks = storeBlocks;
      mcCfg.takesEnd = true;
      cfgs[{memref, McOrLsq::MC}] = mcCfg;
    }

    // the LSQ's control inputs are group allocations:
    // one for every block with an access labelled for the LSQ
    SetVector<Block *> lsqBlocks;
    // iterate over the loads and stores
    for (Operation *access : accesses) {
      // check if they connect to an LSQ
      auto label = getDialectAttr<handshake::MemInterfaceAttr>(access);
      if (label.connectsToLSQ())
        // if so, add their block to the list
        lsqBlocks.insert(access->getBlock());
    }
    // if any basic block connects to the LSQ
    if (!lsqBlocks.empty()) {
      // the LSQ blocks take the token as their group allocation
      SetVector<Block *> groupBlocks = lsqBlocks;

      // start at the entry block, for the start input
      lsqBlocks.insert(entryBlock);

      // without a memory controller, the LSQ also takes the end
      if (!buildRoutingForMemoryController)
        lsqBlocks.insert(returnBlock);

      MemoryInterfaceCfg lsqCfg;
      lsqCfg.cfg = buildMemoryInterfaceCfg(base, lsqBlocks.getArrayRef());
      lsqCfg.interfacingBlocks = groupBlocks;
      lsqCfg.takesEnd = !buildRoutingForMemoryController;
      cfgs[{memref, McOrLsq::LSQ}] = lsqCfg;
    }
  }

  return cfgs;
}

//===----------------------------------------------------------------------===//
// Optimizing a CFG
//===----------------------------------------------------------------------===//

/// Whether an edge carries a live-out:
/// the same value, becoming the same block input.
static bool carriesLiveOut(const CfgEdge &edge, const LiveOut &liveOut) {
  // for each live-out
  for (const LiveOut &carried : edge.liveOuts) {
    // check if it matches the given live-out
    if (carried.liveOut == liveOut.liveOut &&
        carried.blockInput == liveOut.blockInput)
      return true;
  }
  return false;
}

/// Whether two edges carry the same live-outs, in any order.
static bool haveSameLiveOuts(const CfgEdge &first, const CfgEdge &second) {
  // edges carrying different numbers of live-outs differ
  if (first.liveOuts.size() != second.liveOuts.size())
    return false;

  // otherwise, every live-out of the first must be on the second.
  // an edge never carries the same live-out twice,
  // so this also covers the second's live-outs
  for (const LiveOut &liveOut : first.liveOuts) {
    if (!carriesLiveOut(second, liveOut))
      return false;
  }
  return true;
}

/// Merges the two sides of one branch that decides nothing:
/// both sides land on the same block, carrying the same live-outs,
/// so whichever side is taken,
/// the same values arrive at the same place.
/// Returns whether a branch was merged.
static bool mergeSides(Cfg &cfg) {
  // walk the basic blocks
  for (Block *block : cfg.blocks) {
    // if the basic block doesn't have two edges,
    // merging doesn't apply
    SmallVector<CfgEdge *> outEdges = cfg.getOutEdges(block);
    if (outEdges.size() != 2)
      continue;

    // get the two edges
    CfgEdge *first = outEdges[0];
    CfgEdge *second = outEdges[1];

    // can't merge if they don't go to the same block
    if (first->to != second->to)
      continue;

    // can't merge if the live-outs aren't the same
    if (!haveSameLiveOuts(*first, *second))
      continue;

    // if they are, both edges are equivalent
    // convert the first to unconditional and remove the second
    first->condition = std::nullopt;
    cfg.removeEdge(second);
    return true;
  }
  return false;
}

/// Removes one edge from an unneeded block to itself:
/// nothing is produced or used in the block,
/// so the values pass around the loop unchanged,
/// and how often it runs does not affect the routing.
/// Returns whether a loop was removed.
static bool removeSelfLoop(Cfg &cfg) {
  // walk the basic blocks
  for (Block *block : cfg.blocks) {
    // a needed block keeps its loop
    if (cfg.needed.contains(block))
      continue;

    // walk all the out edges of an unneeded block
    for (CfgEdge *edge : cfg.getOutEdges(block)) {
      // if the edge is a self loop
      if (edge->to == block) {
        // remove the self loop from the cfg
        cfg.removeEdge(edge);
        return true;
      }
    }
  }
  return false;
}

/// Bypasses one block that is not needed and has a single way out:
/// the edges into it land on its successor instead.
/// Returns whether a block was bypassed.
static bool bypassBlock(Cfg &cfg) {
  // a block can be bypassed if it is unneeded
  // and has a single outgoing edge
  SetVector<Block *> candidates;

  // walk the basic blocks
  for (Block *block : cfg.blocks) {
    // a needed block must stay
    if (cfg.needed.contains(block))
      continue;

    // only blocks with one successor can be bypassed
    if (cfg.getOutEdges(block).size() != 1)
      continue;

    candidates.insert(block);
  }

  // if there is nothing to bypass, nothing changes
  if (candidates.empty())
    return false;

  // bypass the first candidate
  Block *block = candidates.front();

  // remove its single outgoing edge, keeping a copy
  CfgEdge *outEdge = cfg.getOutEdges(block).front();
  CfgEdge bypassed = *outEdge;
  cfg.removeEdge(outEdge);

  // for bypass of A->B to (preds of A)->B
  // the edge A->B has been removed
  // now alter each edge from (pred of A)->A
  // to instead be from (pred of A)->B
  for (CfgEdge *inEdge : cfg.getInEdges(block)) {
    // change the successor of each edge
    inEdge->to = bypassed.to;

    // each cfg edge from (pred of A)
    // now arrives at B:
    // A may have more live-outs than each pred of A
    // if one block input of A goes to several live-outs
    //
    // so we have to build a new set of live-outs
    // for each pred of A
    SmallVector<LiveOut> updatedLiveOutsOfPredA;
    // iterate the live-outs of A
    for (const LiveOut &liveOutOfA : bypassed.liveOuts) {
      // since A is being bypassed
      // the producer of each live-out must be a block input
      BlockInput blockInputOfA = cast<BlockInput>(liveOutOfA.liveOut);

      // get the live-out of pred A->A
      Value liveOut = inEdge->getLiveOut(blockInputOfA)->liveOut;

      // Make the updated live-out
      updatedLiveOutsOfPredA.push_back({/*liveOut=*/liveOut,
                                        /*blockInput=*/liveOutOfA.blockInput});
    }

    // replace the live-outs of pred of A
    inEdge->liveOuts = updatedLiveOutsOfPredA;
  }

  // with no edges left, the block is no longer part of the CFG
  cfg.blocks.remove(block);

  return true;
}

void dynamatic::experimental::operand_routing2::optimizeCfg(Cfg &cfg) {
  bool changed = true;
  while (changed)
    changed = mergeSides(cfg) || removeSelfLoop(cfg) || bypassBlock(cfg);
}
