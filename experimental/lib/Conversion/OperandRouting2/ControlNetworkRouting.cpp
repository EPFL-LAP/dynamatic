//===- ControlNetworkRouting.cpp - Control-network routing ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/OperandRouting2/ControlNetworkRouting.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Conversion/OperandRouting2/TempOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::operand_routing2;

namespace {
/// Dictionary mapping from each basic block to its branch condition
class BlockToCond {
public:
  /// Place the cond readers on each CondBranchOp in the IR
  /// and store each under the basic block it comes from
  BlockToCond(Region &body, OpBuilder &builder) {
    // iterate over the cond br ops
    for (auto condBrOp : body.getOps<cf::CondBranchOp>()) {
      // add a cond reader and store it
      // under the basic block
      builder.setInsertionPoint(condBrOp);
      blockToCondReader[condBrOp->getBlock()] =
          TempCons(builder, condBrOp.getCondition());
    }
  }

  // remove the cond readers from the IR once routing is done
  ~BlockToCond() {
    // iterate over the basic blocks
    // and their cond readers
    for (auto &[block, condReader] : blockToCondReader)
      // and remove the cond reader from the IR
      condReader.erase();
  }

  /// index by basic block to get branch condition
  Value operator[](Block *block) const {
    assert(blockToCondReader.contains(block) &&
           "basic block has no branch condition");
    return blockToCondReader.lookup(block).getValue();
  }

private:
  // dense map internally
  DenseMap<Block *, TempCons> blockToCondReader;
};

/// Whether two edges are the same edge of the function's CFG,
/// even when they belong to different copies of it.
static bool isSameEdge(const CfgEdge &first, const CfgEdge &second) {
  return first.from == second.from && first.to == second.to &&
         first.condition == second.condition;
}

/// The outputs of a control network:
/// the select of each basic block with two or more predecessors,
/// and the edge each index of the select stands for
struct Selects {
  DenseMap<Block *, Value> select;
  // copies of the edges, since the control network's CFG is a copy
  DenseMap<Block *, SmallVector<CfgEdge>> edges;

  /// The select of a basic block,
  /// and the routing's edges into it, sorted in the order of the select
  std::pair<Value, SmallVector<const CfgEdge *>>
  getSelect(Block *block, ArrayRef<const CfgEdge *> inEdges) const {
    assert(select.contains(block) && "basic block has no control merge");
    SmallVector<const CfgEdge *> sortedInEdges;
    for (const CfgEdge &selectEdge : edges.lookup(block)) {
      for (const CfgEdge *inEdge : inEdges) {
        if (isSameEdge(selectEdge, *inEdge))
          sortedInEdges.push_back(inEdge);
      }
    }
    assert(sortedInEdges.size() == inEdges.size() &&
           "edges into the basic block differ from the control merge's");
    return {select.lookup(block), sortedInEdges};
  }
};

/// Dictionary mapping from each live-out of a control network (an edge)
/// to the value that produces it.
/// The value is read through a temporary consumer,
/// so it stays current as temporary producers are replaced:
/// a routed out can itself be a temporary producer,
/// passed straight through a basic block with one predecessor
class ControlLiveOutToProd {
public:
  /// record the producer of a live-out, at the builder's position
  void set(OpBuilder &builder, CfgEdge *liveOut, Value producer) {
    liveOutToReader[liveOut] = TempCons(builder, producer);
  }

  /// index by live-out to get its current producer
  Value operator[](CfgEdge *liveOut) const {
    assert(liveOutToReader.contains(liveOut) && "live-out has no producer");
    return liveOutToReader.lookup(liveOut).getValue();
  }

  // remove the readers from the IR once routing is done
  ~ControlLiveOutToProd() {
    for (auto &[liveOut, reader] : liveOutToReader)
      reader.erase();
  }

private:
  DenseMap<CfgEdge *, TempCons> liveOutToReader;
};

/// Dictionary mapping from each live-out of a routing
/// to the value that produces it.
/// The value is read through a temporary consumer,
/// so it stays current as temporary producers are replaced:
/// a routed out can itself be a temporary producer,
/// passed straight through a basic block with one predecessor
class LiveOutToProd {
public:
  /// record the producer of a live-out, at the builder's position
  void set(OpBuilder &builder, const LiveOut *liveOut, Value producer) {
    liveOutToReader[liveOut] = TempCons(builder, producer);
  }

  /// index by live-out to get its current producer
  Value operator[](const LiveOut *liveOut) const {
    assert(liveOutToReader.contains(liveOut) && "live-out has no producer");
    return liveOutToReader.lookup(liveOut).getValue();
  }

  // remove the readers from the IR once routing is done
  ~LiveOutToProd() {
    for (auto &[liveOut, reader] : liveOutToReader)
      reader.erase();
  }

private:
  DenseMap<const LiveOut *, TempCons> liveOutToReader;
};
} // namespace

/// Builds the control network of a CFG,
/// needed to provide the select input of multiplexers.
/// If the built control network would have no outputs,
/// the function returns without building anything
static Selects buildControlNetwork(const Cfg &cfg,
                                   const BlockToCond &blockToCond,
                                   OpBuilder &builder) {
  Selects selects;

  // control network doesn't correspond 
  // to any specific part of the IR
  Location loc = builder.getUnknownLoc();

  // MLIR type stuff handled here-
  // we need this to build dataless control units later
  Type controlType = handshake::ControlType::get(builder.getContext());

  // the control network is only needed 
  // if there will be multiplexers to control
  // multiplexers are placed when a basic block
  // has 2 or more predecessors, and block inputs arriving over them
  bool hasOutput = false;
  // for each basic block
  for (Block *block : cfg.blocks) {
    // check if it has 2 or more predecessors
    // whose edges carry live-outs into it
    SmallVector<const CfgEdge *> inEdges = cfg.getInEdges(block);
    if (inEdges.size() >= 2 && !inEdges.front()->liveOuts.empty())
      hasOutput = true;
  }

  // return early if the control network is not needed
  if (!hasOutput)
    return selects;

  Cfg controlNetworkCfg = cfg;

  // we add an additional edge to the cfg 
  // for the control network:
  // from the exit block to the entry block
  // which marks that the routing network only restarts
  // after it finishes
  controlNetworkCfg.edges.push_back({/*from=*/cfg.exit,
                                     /*to=*/cfg.entry,
                                     /*condition=*/std::nullopt,
                                     /*liveOuts=*/{}});

  // the control network carries no values,
  // so its live-out on an edge is just the edge
  using ControlLiveOut = CfgEdge *;

  // we place a temp for each live-out at the beginning
  // and work down from the live-outs
  DenseMap<ControlLiveOut, TempProd> temps;
  // at the end, we have all the real values of those live-outs
  // and will use them to replace the temp
  ControlLiveOutToProd routedOuts;

  // for each basic block
  for (Block *block : controlNetworkCfg.blocks) {
    // on the control network, there is only one input into the basic block,
    // and the incoming live-outs of that input are the edges
    // from each predecessor
    SmallVector<ControlLiveOut> incomingLiveOuts =
        controlNetworkCfg.getInEdges(block);

    // handle insertion point
    builder.setInsertionPointToStart(block);

    // on the control network, 
    // there is only one value into the basic block
    Value routedIn;

    if (incomingLiveOuts.size() == 1) {
      // if there is one predecessor

      // get the live-out from the predecessor
      // (the live-out is a CFG construct)
      ControlLiveOut liveOut = incomingLiveOuts.front();

      // and place a temporary producer for it in the IR
      temps[liveOut] = TempProd(builder, controlType);

      // the value in the basic block comes directly
      // from the producer
      routedIn = temps[liveOut].getValue();
    } else if (incomingLiveOuts.size() > 1) {
      // if there is more than 1 predecessor

      // we need to add a control merge
      SmallVector<Value> cmergeInputs;

      // the control merge needs a temp producer in the IR
      // one for each live-out of this basic block input
      // (each predecessor has one live-out that 
      // could become the input)
      for (ControlLiveOut liveOut : incomingLiveOuts) {
        temps[liveOut] = TempProd(builder, controlType);
        // we build the cmerge inputs in the order of incomingLiveOuts
        // will handle order matching later
        cmergeInputs.push_back(temps[liveOut].getValue());
      }

      // make the cmerge from the input list
      auto cmergeOp =
          builder.create<handshake::ControlMergeOp>(loc, cmergeInputs);

      // record the select, and the edge each of its indices stands for,
      // for the basic block's multiplexers
      selects.select[block] = cmergeOp.getIndex();
      for (ControlLiveOut liveOut : incomingLiveOuts)
        selects.edges[block].push_back(*liveOut);

      // the value in the basic block comes from the cmerge output
      routedIn = cmergeOp.getResult();
    } else{
      llvm_unreachable(
          "every basic block in the control network has a predecessor");
    }

    // the "straight data path" of the control network 
    // is almost always just a wire
    Value sdpOut = routedIn;
    // but in the entry block, it's an init op
    if (block == controlNetworkCfg.entry)
      sdpOut = builder.create<handshake::InitOp>(loc, controlType, sdpOut)
                   .getResult();

    // in the control network, a basic block has either one or two live-outs
    // because the basic block has one output to send (the control token)
    // but live-outs are per edge
    // so the number of live-outs is the number of outgoing edges
    SmallVector<ControlLiveOut> outgoingLiveOuts =
        controlNetworkCfg.getOutEdges(block);
    
    if (outgoingLiveOuts.size() == 1) {
      // if this basic block has one successor
      // the live-out comes directly from the straight data path
      routedOuts.set(builder, outgoingLiveOuts.front(), sdpOut);
    } else if (outgoingLiveOuts.size() == 2) {
      // if this basic block has two successors,
      // we must place a cond control branch

      // handle insertion point
      builder.setInsertionPoint(block->getTerminator());

      // add the cond control branch to the IR, using blockToCond
      // to get the branch condition of this basic block
      auto condBrOp = builder.create<handshake::ConditionalBranchOp>(
          loc, TypeRange{controlType, controlType},
          ValueRange{blockToCond[block], sdpOut});

      // then store the real value of the live-outs of this basic block
      // (the branch output for true or false)
      for (ControlLiveOut liveOut : outgoingLiveOuts) {
        routedOuts.set(builder, liveOut,
                       *liveOut->condition ? condBrOp.getTrueResult()
                                           : condBrOp.getFalseResult());
      }
    } else {
      llvm_unreachable("every basic block in the control network has one or "
                       "two successors");
    }
  }

  // Now that we have all the real values of the live-outs
  // we want to use them to replace the temps

  // for each basic block
  for (Block *block : controlNetworkCfg.blocks) {
    // get the live-outs of the basic block's one input
    for (ControlLiveOut liveOut : controlNetworkCfg.getInEdges(block)) {
      // replace the temp producer with the real value
      assert(routedOuts[liveOut] != temps[liveOut].getValue() &&
             "temporary producer would replace itself");
      temps[liveOut].replace(routedOuts[liveOut]);
    }
  }

  return selects;
}

/// Builds the routing of the live-outs of a CFG,
/// with multiplexers taking their select from the control network,
/// and feeds the routed values to the readers of the CFG's block uses.
static void buildRouting(const Cfg &cfg, const BlockToCond &blockToCond,
                         const Selects &selects, OpBuilder &builder) {
  // routing doesn't correspond
  // to any specific part of the IR
  Location loc = builder.getUnknownLoc();

  // we place a temp for each live-out at the beginning
  // and work down from the live-outs
  DenseMap<const LiveOut *, TempProd> temps;
  // at the end, we have all the real values of those live-outs
  // and will use them to replace the temp
  LiveOutToProd routedOuts;

  // the value into the basic block for each input
  DenseMap<BlockInput, Value> routedIns;

  // for each basic block
  for (Block *block : cfg.blocks) {
    // each edge into this basic block
    // contains the live-outs from that predecessor
    // but also the block inputs of this basic block
    // which are the values we must route to here
    SmallVector<const CfgEdge *> inEdges = cfg.getInEdges(block);

    // vector to collect the basic block inputs
    SmallVector<BlockInput> blockInputs;

    // entry block has no block inputs
    if (block == cfg.entry) {
      if (!inEdges.empty()){
        llvm_unreachable("the entry block has no predecessors");
      }
    } else {
      if (inEdges.empty()){
        llvm_unreachable("non entry blocks must have a predecessor");
      }
      // get the live-outs from the first incoming cfg edge
      const SmallVector<LiveOut> &liveOuts = inEdges.front()->liveOuts;
      // collect the block input of each live-out into inputs
      for (const LiveOut &liveOut : liveOuts) {
        BlockInput input = liveOut.blockInput;
        blockInputs.push_back(input);
      }

      // by construction, every incoming cfg edge should agree
      // on what block inputs this basic block has,
      // so assert this
      for (const CfgEdge *inEdge : inEdges) {
        assert(inEdge->liveOuts.size() == blockInputs.size() &&
               "incoming cfg edges have different block inputs");
        for (const LiveOut &liveOut : inEdge->liveOuts) {
          assert(llvm::is_contained(blockInputs, liveOut.blockInput) &&
                 "incoming cfg edges have different block inputs");
        }
      }
    }

    // handle insertion point
    builder.setInsertionPointToStart(block);

    // for each block input
    for (BlockInput blockInput : blockInputs) {
      if (inEdges.size() == 1) {
        // if there is one predecessor

        // get the one incoming edge
        const CfgEdge *inEdge = inEdges.front();

        // get the live-out for this blockInput for that edge
        const LiveOut *liveOut = inEdge->getLiveOut(blockInput);

        // and place a temporary producer for it in the IR
        temps[liveOut] = TempProd(builder, blockInput.getType());

        // the value in the basic block comes directly
        // from the producer
        routedIns[blockInput] = temps[liveOut].getValue();
      } else if (inEdges.size() > 1) {
        // if there is more than 1 predecessor

        // we need to add a multiplexer,
        // which must be controlled by the cmerge
        // the order of cfg edges into both must match
        // so when we ask for the cmerge select output
        // we also ask it to sort our cfg edges
        auto [select, sortedInEdges] = selects.getSelect(block, inEdges);

        // the multiplexer needs a temp producer in the IR
        // one for each live-out of this basic block input
        SmallVector<Value> muxInputs;
        for (const CfgEdge *inEdge : sortedInEdges) {
          const LiveOut *liveOut = inEdge->getLiveOut(blockInput);
          temps[liveOut] = TempProd(builder, blockInput.getType());
          muxInputs.push_back(temps[liveOut].getValue());
        }

        // make the mux from the input list
        auto muxOp = builder.create<handshake::MuxOp>(
            loc, blockInput.getType(), select, muxInputs);

        // the value in the basic block comes from the mux output
        routedIns[blockInput] = muxOp.getResult();
      } else {
        llvm_unreachable("a basic block with an input must have a predecessor");
      }
    }
  }

  // the straight data paths:
  // once all the block uses of this cfg have routed inputs
  // (coming from the temp live-outs)
  // we want the straight data paths
  // in that basic block to use the routed input
  for (BlockUse blockUse : cfg.blockUses)
    blockUse.replaceAllUsesWith(routedIns[blockUse]);

  // for each basic block
  for (Block *block : cfg.blocks) {
    // whether we need discarders (branch ops in practice)
    // depends on the number of successors this basic block has
    SmallVector<const CfgEdge *> outEdges = cfg.getOutEdges(block);

    // handle insertion point
    builder.setInsertionPoint(block->getTerminator());

    // for each live-out of this basic block
    for (const CfgEdge *outEdge : outEdges) {
      for (const LiveOut &liveOut : outEdge->liveOuts) {
        // the value the live-out leaves with:
        // from the straight data path if this basic block produces it,
        // otherwise the value routed in
        Value liveOutValue;

        // if a basic block produces its live-out,
        // it's the output of one of the straight data paths
        // and so routing hasn't affected it
        if (isProducedIn(liveOut.liveOut, block)) {
          liveOutValue = liveOut.liveOut;
        } else {
          // otherwise, it means the basic block's live-out is
          // a block input.
          // We need to use the routed block input instead of the original

          // first cast to block input (this asserts the cast succeeds)
          BlockInput blockInput = cast<BlockInput>(liveOut.liveOut);
          // the block input should belong to the
          // same basic block as the live-out
          assert(blockInput.getOwner() == block &&
                 "live-out is not a block input of this basic block");
          // and all the block inputs should be routed
          assert(routedIns.contains(blockInput) &&
                 "block input has no routed value");
          // then, get the routed block input
          liveOutValue = routedIns.lookup(blockInput);
        }

        if (outEdges.size() == 1) {
          // if this basic block has one successor
          // there is no branch added
          routedOuts.set(builder, &liveOut, liveOutValue);
        } else if (outEdges.size() == 2) {
          // if this basic block has two successors,
          // we must place a cond branch for each live-out

          // add the cond branch to the IR, using blockToCond
          // to get the branch condition of this basic block
          auto condBrOp = builder.create<handshake::ConditionalBranchOp>(
              loc, TypeRange{liveOutValue.getType(), liveOutValue.getType()},
              ValueRange{blockToCond[block], liveOutValue});

          // then store the real value of the live-out
          // (the branch output for true or false)
          routedOuts.set(builder, &liveOut,
                         *outEdge->condition ? condBrOp.getTrueResult()
                                             : condBrOp.getFalseResult());
        } else {
          llvm_unreachable("every basic block has at most two successors");
        }
      }
    }
  }

  // Now that we have all the real values of the live-outs
  // we want to use them to replace the temps

  // for each basic block
  for (Block *block : cfg.blocks) {
    // get the live-outs of each of the basic block's inputs
    for (const CfgEdge *inEdge : cfg.getInEdges(block)) {
      for (const LiveOut &liveOut : inEdge->liveOuts) {
        // replace the temp producer with the real value
        assert(routedOuts[&liveOut] != temps[&liveOut].getValue() &&
               "temporary producer would replace itself");
        temps[&liveOut].replace(routedOuts[&liveOut]);
      }
    }
  }
}

/// Routes the block uses of one optimized CFG.
static void route(const Cfg &cfg, const BlockToCond &blockToCond,
                  OpBuilder &builder) {
  Selects selects = buildControlNetwork(cfg, blockToCond, builder);
  buildRouting(cfg, blockToCond, selects, builder);
}

/// Routes the single memory token of a memory interface on its optimized CFG,
/// from memStart in the entry block to the exit block.
/// In each interfacing basic block, a lazy fork gives the token
/// to the interface's control input and passes it on,
/// so the interface's control inputs arrive in program order.
static void buildMemoryRouting(MemoryInterfaceCfg &memoryCfg, McOrLsq mcOrLsq,
                               MemoryInterfaceInputs &inputs,
                               const BlockToCond &blockToCond, Value memStart,
                               OpBuilder &builder) {
  // Pull the annotated CFG itself from the memory cfg struct
  Cfg &cfg = memoryCfg.cfg;

  // the controls of this interface, in the region's inputs
  DenseMap<unsigned, Value> &controls =
      mcOrLsq == McOrLsq::MC ? inputs.mcControls : inputs.lsqControls;

  // routing doesn't correspond
  // to any specific part of the IR
  Location loc = builder.getUnknownLoc();

  // MLIR type stuff handled here-
  // we need this to build dataless control units later
  Type controlType = handshake::ControlType::get(builder.getContext());

  // the memory token carries no values,
  // so its live-out on an edge is just the edge
  using ControlLiveOut = CfgEdge *;

  // we place a temp for each live-out at the beginning
  // and work down from the live-outs
  DenseMap<ControlLiveOut, TempProd> temps;
  // at the end, we have all the real values of those live-outs
  // and will use them to replace the temp
  ControlLiveOutToProd routedOuts;

  // We need to reliably find the producer of the end token
  // but when the end basic block is not an interfacing block,
  // the producer of the end token can be a TempProd for most of routing
  // Rather than trying to check when this happens or doesn't happen
  // We unconditionally place a TempCons after the end token producer
  // At the end, if there was a TempProd, it's been replaced
  // so the TempCons is guaranteed to point to the real producer
  TempCons endReader;

  // for each basic block
  for (Block *block : cfg.blocks) {
    // on the memory network, there is only one input into the basic block,
    // and the incoming live-outs of that input are the edges
    // from each predecessor
    SmallVector<ControlLiveOut> incomingLiveOuts = cfg.getInEdges(block);

    // handle insertion point
    builder.setInsertionPointToStart(block);

    // on the memory network,
    // there is only one value into the basic block
    Value routedIn;

    if (block == cfg.entry) {
      // the token enters the network from the "memStart" function arg
      routedIn = memStart;
    } else if (incomingLiveOuts.size() == 1) {
      // if there is one predecessor

      // get the live-out from the predecessor
      // (the live-out is a CFG construct)
      ControlLiveOut liveOut = incomingLiveOuts.front();

      // and place a temporary producer for it in the IR
      temps[liveOut] = TempProd(builder, controlType);

      // the value in the basic block comes directly
      // from the producer
      routedIn = temps[liveOut].getValue();
    } else if (incomingLiveOuts.size() > 1) {
      // if there is more than 1 predecessor

      // with a single token, a merge keeps the order,
      // and no multiplexer needs a select
      SmallVector<Value> mergeInputs;

      // the merge needs a temp producer in the IR
      // one for each live-out of this basic block input
      // (each predecessor has one live-out that
      // could become the input)
      for (ControlLiveOut liveOut : incomingLiveOuts) {
        temps[liveOut] = TempProd(builder, controlType);
        mergeInputs.push_back(temps[liveOut].getValue());
      }

      // make the merge from the input list
      // and the value in the basic block comes from the merge output
      routedIn =
          builder.create<handshake::MergeOp>(loc, controlType, mergeInputs)
              .getResult();
    } else {
      llvm_unreachable(
          "every basic block in the memory network has a predecessor");
    }

    // in non-interfacing blocks, the single memory token just passes through
    Value sdpOut = routedIn;

    // in an interfacing basic block, the straight data path is a lazy fork:
    // one output to the interface, one passing the token on
    if (memoryCfg.interfacingBlocks.contains(block)) {
      auto forkOp = builder.create<handshake::LazyForkOp>(loc, sdpOut, 2);
      controls[getBlockNumber(block)] = forkOp->getResult(0);
      sdpOut = forkOp->getResult(1);
    }

    // live-outs in a memory interface CFG are represented by CFG edges
    SmallVector<ControlLiveOut> outgoingLiveOuts = cfg.getOutEdges(block);

    if (outgoingLiveOuts.empty()) {
      // the token leaves the network at the exit, as the end control
      builder.setInsertionPoint(block->getTerminator());
      // use a TempCons to be able to reliably find the producer
      // no matter what it is or what happens to it later
      endReader = TempCons(builder, sdpOut);
    } else if (outgoingLiveOuts.size() == 1) {
      // if this basic block has one successor
      // the live-out comes directly from the straight data path
      routedOuts.set(builder, outgoingLiveOuts.front(), sdpOut);
    } else if (outgoingLiveOuts.size() == 2) {
      // if this basic block has two successors,
      // we must place a cond control branch

      // handle insertion point
      builder.setInsertionPoint(block->getTerminator());

      // add the cond control branch to the IR, using blockToCond
      // to get the branch condition of this basic block
      auto condBrOp = builder.create<handshake::ConditionalBranchOp>(
          loc, TypeRange{controlType, controlType},
          ValueRange{blockToCond[block], sdpOut});

      // then store the real value of the live-outs of this basic block
      // (the branch output for true or false)
      for (ControlLiveOut liveOut : outgoingLiveOuts) {
        routedOuts.set(builder, liveOut,
                       *liveOut->condition ? condBrOp.getTrueResult()
                                           : condBrOp.getFalseResult());
      }
    } else {
      llvm_unreachable("every basic block in the memory network has at most "
                       "two successors");
    }
  }

  // Now that we have all the real values of the live-outs
  // we want to use them to replace the temps

  // for each basic block
  for (Block *block : cfg.blocks) {
    // get the live-outs of the basic block's one input
    for (ControlLiveOut liveOut : cfg.getInEdges(block)) {
      // replace the temp producer with the real value
      assert(routedOuts[liveOut] != temps[liveOut].getValue() &&
             "temporary producer would replace itself");
      temps[liveOut].replace(routedOuts[liveOut]);
    }
  }

  // the producer of the end control is now final
  // so we get its output value from the end reader,
  // if this interface is the one taking the end
  if (memoryCfg.takesEnd) {
    inputs.ctrlEnd = endReader.getValue();
  }
  // and remove the end reader from the IR
  endReader.erase();
}

MapVector<MemrefValue, MemoryInterfaceInputs>
dynamatic::experimental::operand_routing2::routeCfgs(
    Region &body, const MapVector<MemrefValue, ControlValue> &memStarts,
    ArrayRef<Cfg> useCfgs,
    MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg> &memoryCfgs,
    OpBuilder &builder) {
  BlockToCond blockToCond(body, builder);

  for (const Cfg &cfg : useCfgs)
    route(cfg, blockToCond, builder);

  MapVector<MemrefValue, MemoryInterfaceInputs> memoryInputs;
  for (auto &[key, memoryCfg] : memoryCfgs) {
    auto [memref, mcOrLsq] = key;
    buildMemoryRouting(memoryCfg, mcOrLsq, memoryInputs[memref], blockToCond,
                       memStarts.lookup(memref), builder);
  }
  return memoryInputs;
}
