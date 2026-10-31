//===- BaseCfg.cpp - CFG representation -------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/CfToHandshake/BaseCfg.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::cf_to_handshake;

const LiveOut *CfgEdge::getLiveOut(BlockInput blockInput) const {
  for (const LiveOut &liveOut : liveOuts) {
    if (liveOut.blockInput == blockInput)
      return &liveOut;
  }
  llvm_unreachable("edge carries no live-out to this block input");
}

/// Get all CFG edges which come from this block
SmallVector<CfgEdge *> Cfg::getOutEdges(Block *block) {
  SmallVector<CfgEdge *> outEdges;
  for (CfgEdge &edge : edges) {
    if (edge.from == block)
      outEdges.push_back(&edge);
  }
  return outEdges;
}

SmallVector<const CfgEdge *> Cfg::getOutEdges(Block *block) const {
  SmallVector<const CfgEdge *> outEdges;
  for (const CfgEdge &edge : edges) {
    if (edge.from == block)
      outEdges.push_back(&edge);
  }
  return outEdges;
}

/// Get all CFG edges which go to this block
SmallVector<CfgEdge *> Cfg::getInEdges(Block *block) {
  SmallVector<CfgEdge *> inEdges;
  for (CfgEdge &edge : edges) {
    if (edge.to == block)
      inEdges.push_back(&edge);
  }
  return inEdges;
}

SmallVector<const CfgEdge *> Cfg::getInEdges(Block *block) const {
  SmallVector<const CfgEdge *> inEdges;
  for (const CfgEdge &edge : edges) {
    if (edge.to == block)
      inEdges.push_back(&edge);
  }
  return inEdges;
}

void Cfg::removeEdge(CfgEdge *edge) { edges.erase(edge); }

Cfg dynamatic::experimental::cf_to_handshake::getBaseCfg(Region &body) {
  // declare the CFG we will build and return
  Cfg cfg;

  cfg.entry = &body.front();

  // walk each block of the IR
  for (Block &block : body) {
    // record the block in the CFG
    cfg.blocks.insert(&block);

    // get the branch operation at the end of the block
    // (the return block ends in a return, and gets no edge)
    Operation *terminator = block.getTerminator();

    if (auto brOp = dyn_cast<cf::BranchOp>(terminator)) {
      // if it is an unconditional branch,
      // add an unconditional edge to the CFG
      cfg.edges.push_back({/*from=*/&block,
                           /*to=*/brOp.getDest(),
                           /*condition=*/std::nullopt,
                           /*liveOuts=*/{}});
    } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(terminator)) {
      // if it is a conditional branch,
      // add the true edge to the CFG
      cfg.edges.push_back({/*from=*/&block,
                           /*to=*/condBrOp.getTrueDest(),
                           /*condition=*/true,
                           /*liveOuts=*/{}});
      // and add the false edge to the CFG
      cfg.edges.push_back({/*from=*/&block,
                           /*to=*/condBrOp.getFalseDest(),
                           /*condition=*/false,
                           /*liveOuts=*/{}});
    } else if (isa<func::ReturnOp>(terminator)) {
      cfg.exit = &block;
    } else {
      llvm_unreachable("block terminator must be a cf.br, cf.cond_br or "
                       "func.return");
    }
  }

  // return the cfg
  return cfg;
}

unsigned
dynamatic::experimental::cf_to_handshake::getBlockNumber(Block *block) {
  return std::distance(block->getParent()->begin(), block->getIterator());
}
