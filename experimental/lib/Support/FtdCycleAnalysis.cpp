//===- FtdCycleAnalysis.cpp - Local Control Flow Graph Utils ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the core functions for analyzing the cyclic structure of Local
// Control Flow Graphs (LocalCFG) used in the Fast Token Delivery (FTD)
// algorithm.
//
//===----------------------------------------------------------------------===//
#include "experimental/Support/FtdCycleAnalysis.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::ftd;

CyclicGraphManager::CyclicGraphManager(LocalCFG &cfg)
    : lcfg(cfg), domInfo(cfg.containerOp),
      loopInfo(domInfo.getDomTree(cfg.region)) {
  analyzeTopology();
}

unsigned CyclicGraphManager::getNestingLevel(Block *bb) const {
  auto it = blockLevelMap.find(bb);
  if (it != blockLevelMap.end())
    return it->second;
  return 0;
}

std::unique_ptr<LoopScope>
CyclicGraphManager::buildScopeRecursive(mlir::CFGLoop *loop, unsigned level) {
  auto scope = std::make_unique<LoopScope>();
  scope->level = level;
  scope->loopInfo = loop;
  scope->header = loop->getHeader();

  SmallVector<Block *> latches;
  loop->getLoopLatches(latches);
  scope->latches = latches;

  for (Block *latch : latches) {
    scope->allBackEdges.insert({latch, scope->header});
  }

  auto loopBlocks = loop->getBlocks();
  for (Block *b : loopBlocks) {
    scope->allBlocksInclusive.insert(b);
    // Outer loops must be processed before inner loops for correct leveling.
    blockLevelMap[b] = level;
  }

  for (auto *subLoop : loop->getSubLoops()) {
    auto subScope = buildScopeRecursive(subLoop, level + 1);
    subScope->parent = scope.get();
    for (auto &edge : subScope->allBackEdges) {
      scope->allBackEdges.insert(edge);
    }
    scope->subLoops.push_back(std::move(subScope));
  }
  return scope;
}

void CyclicGraphManager::analyzeTopology() {
  blockLevelMap.clear();
  topLevelScope = std::make_unique<LoopScope>();
  topLevelScope->level = 0;
  topLevelScope->header = lcfg.newProd;
  topLevelScope->loopInfo = nullptr;

  for (Block &b : lcfg.region->getBlocks()) {
    topLevelScope->allBlocksInclusive.insert(&b);
    blockLevelMap[&b] = 0;
  }

  auto topLoops = loopInfo.getTopLevelLoops();
  for (auto *loop : topLoops) {
    auto subScope = buildScopeRecursive(loop, 1);
    subScope->parent = topLevelScope.get();
    for (auto &edge : subScope->allBackEdges) {
      topLevelScope->allBackEdges.insert(edge);
    }
    topLevelScope->subLoops.push_back(std::move(subScope));
  }
}

std::unique_ptr<LocalCFG>
CyclicGraphManager::extractLayeredCFG(const LoopScope *scope,
                                      OpBuilder &builder) {
  auto newGraph = std::make_unique<LocalCFG>();
  Location loc = builder.getUnknownLoc();

  OpBuilder::InsertionGuard guard(builder);
  auto funcType = builder.getFunctionType({}, {});
  auto dummyFunc =
      builder.create<func::FuncOp>(loc, "__ftd_layered_cfg__", funcType);
  Region &R = dummyFunc.getBody();
  newGraph->region = &R;
  newGraph->containerOp = dummyFunc;

  // Create the sink terminal (False).
  // For level > 0, back-edges to own header are redirected here.
  // All terminal paths eventually converge here.
  Block *falseTerm = new Block();
  R.push_back(falseTerm);
  newGraph->sinkBB = falseTerm;
  newGraph->origMap[falseTerm] = nullptr;

  // Create the consumer terminal (True).
  // For level > 0 this represents the loop exit.
  // For level 0 this maps to the actual consumer in the original CFG.
  Block *trueTerm = new Block();
  R.push_back(trueTerm);
  newGraph->newCons = trueTerm;

  if (scope->level == 0)
    newGraph->origMap[trueTerm] = lcfg.origMap.lookup(lcfg.newCons);
  else
    newGraph->origMap[trueTerm] = nullptr;

  // Clone the header block as the entry (Producer) of this layer.
  Block *clonedHeader = new Block();
  R.push_back(clonedHeader);
  newGraph->newProd = clonedHeader;
  newGraph->origMap[clonedHeader] = lcfg.origMap.lookup(scope->header);

  // Build a mapping from original blocks to their cloned counterparts.
  DenseMap<Block *, Block *> clonedMap;
  clonedMap[scope->header] = clonedHeader;

  // At level 0, map the original consumer and sink to the terminals
  // so that in-scope edges targeting them are resolved correctly.
  if (scope->level == 0) {
    clonedMap[lcfg.newCons] = trueTerm;
    clonedMap[lcfg.sinkBB] = falseTerm;
  }

  // Clone all in-scope blocks except those already handled above.
  for (Block *b : scope->allBlocksInclusive) {
    if (b == scope->header || b == lcfg.newCons || b == lcfg.sinkBB)
      continue;
    Block *nb = new Block();
    R.push_back(nb);
    clonedMap[b] = nb;
    newGraph->origMap[nb] = lcfg.origMap.lookup(b);
  }

  // Reconstruct edges and terminators for each cloned block.
  for (auto [origBlock, newBlock] : clonedMap) {
    // Skip terminal blocks; they receive their terminators later.
    if (origBlock == lcfg.newCons || origBlock == lcfg.sinkBB)
      continue;

    builder.setInsertionPointToEnd(newBlock);
    Operation *origTerm = origBlock->getTerminator();

    if (!origTerm) {
      llvm::errs() << "Warning: Block without terminator found in LocalCFG "
                      "extraction.\n";
      continue;
    }

    SmallVector<Block *, 2> validSuccessors;
    SmallVector<bool, 2> keepSuccessor;

    for (unsigned i = 0; i < origTerm->getNumSuccessors(); ++i) {
      Block *origSucc = origTerm->getSuccessor(i);

      if (origSucc == scope->header) {
        // Back-edge to this scope's own header.
        // Level 0: header is dummystart; no real back-edge should exist,
        // but cut defensively if encountered.
        // Level > 0: redirect to the sink terminal.
        if (scope->level == 0) {
          validSuccessors.push_back(nullptr);
          keepSuccessor.push_back(false);
        } else {
          validSuccessors.push_back(falseTerm);
          keepSuccessor.push_back(true);
        }
      } else if (scope->allBackEdges.contains({origBlock, origSucc})) {
        // Deep back-edge belonging to an inner loop.
        // Always cut; the inner loop's own layer handles it.
        validSuccessors.push_back(nullptr);
        keepSuccessor.push_back(false);
      } else if (clonedMap.count(origSucc)) {
        // Standard in-scope edge.
        validSuccessors.push_back(clonedMap[origSucc]);
        keepSuccessor.push_back(true);
      } else {
        // Edge leaving the current scope.
        if (scope->level == 0) {
          llvm::errs()
              << "[FTD Warning] Block outside Level 0 scope encountered.\n";
        }
        // Redirect to the consumer terminal (represents loop exit).
        validSuccessors.push_back(trueTerm);
        keepSuccessor.push_back(true);
      }
    }

    // Rebuild the terminator based on surviving successors.
    if (auto cbr = dyn_cast<cf::CondBranchOp>(origTerm)) {
      if (keepSuccessor[0] && keepSuccessor[1]) {
        // Both paths survived; keep conditional branch.
        builder.create<cf::CondBranchOp>(
            loc, cbr.getCondition(), validSuccessors[0], validSuccessors[1]);
      } else if (keepSuccessor[0] && !keepSuccessor[1]) {
        // Only the true path survived.
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
      } else if (!keepSuccessor[0] && keepSuccessor[1]) {
        // Only the false path survived.
        builder.create<cf::BranchOp>(loc, validSuccessors[1]);
      }
      // If neither survived the block is left without a terminator (dead).
    } else if (auto br = dyn_cast<cf::BranchOp>(origTerm)) {
      if (keepSuccessor[0]) {
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
      }
      // If the sole successor was cut, no terminator is created (dead).
    }
  }

  // Graph simplification.
  //
  // Level 0: only remove dead blocks (those without a terminator that are
  // not terminals). This eliminates unreachable subgraphs created by
  // cutting deep back-edges. No duplicate-successor merging and no path
  // compression are performed; those are deferred to the caller.
  //
  // Level > 0: full canonicalization. After this pass every non-terminal
  // block should carry a conditional branch, which is what the BDD builder
  // expects.
  bool changed = true;
  while (changed) {
    changed = false;

    for (Block &block : llvm::make_early_inc_range(R)) {
      // Never touch the entry or the two terminal blocks.
      if (&block == newGraph->newProd || &block == newGraph->newCons ||
          &block == newGraph->sinkBB)
        continue;

      Operation *term = block.getTerminator();

      // Dead block removal (applies to all levels).
      // A block without a terminator has no outgoing edges and is dead.
      // Disconnect its predecessors so the removal can propagate.
      if (!term) {
        SmallVector<Operation *, 4> predTerms;
        for (auto &use : block.getUses())
          predTerms.push_back(use.getOwner());

        for (Operation *predTerm : predTerms) {
          OpBuilder localBuilder(predTerm->getContext());
          localBuilder.setInsertionPoint(predTerm);

          if (auto br = dyn_cast<cf::BranchOp>(predTerm)) {
            // Predecessor unconditionally jumps here; it becomes dead too.
            predTerm->erase();
          } else if (auto cbr = dyn_cast<cf::CondBranchOp>(predTerm)) {
            Block *trueDest = cbr.getTrueDest();
            Block *falseDest = cbr.getFalseDest();

            if (trueDest == &block && falseDest == &block) {
              // Both legs land on this dead block; predecessor dies.
              predTerm->erase();
            } else if (trueDest == &block) {
              // True leg is dead; degrade to unconditional false branch.
              localBuilder.create<cf::BranchOp>(loc, falseDest);
              predTerm->erase();
            } else if (falseDest == &block) {
              // False leg is dead; degrade to unconditional true branch.
              localBuilder.create<cf::BranchOp>(loc, trueDest);
              predTerm->erase();
            }
          }
        }

        block.erase();
        changed = true;
        continue;
      }

      // The remaining optimizations only apply to level > 0.
      if (scope->level == 0)
        continue;

      // Merge duplicate successors: CondBranch(A, A) becomes Branch(A).
      if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          OpBuilder localBuilder(term->getContext());
          localBuilder.setInsertionPoint(term);
          localBuilder.create<cf::BranchOp>(loc, condBr.getTrueDest());
          term->erase();
          term = block.getTerminator();
          changed = true;
        }
      }

      // Path compression: if a block unconditionally jumps to a single
      // destination, redirect all predecessors directly to that destination
      // and remove the block.
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        Block *dest = br.getDest();
        block.replaceAllUsesWith(dest);
        block.erase();
        changed = true;
        continue;
      }
    }
  }

  // Finalize the terminal blocks with proper terminators.
  // The consumer terminal (True) falls through to the sink (False).
  builder.setInsertionPointToEnd(trueTerm);
  builder.create<cf::BranchOp>(loc, falseTerm);
  // The sink terminal ends the region.
  builder.setInsertionPointToEnd(falseTerm);
  builder.create<func::ReturnOp>(loc);

  // Compute topological order starting from the producer.
  DenseSet<Block *> visited;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visited.contains(u))
      return;
    visited.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(); it != term->successor_end(); ++it)
        topo(*it);
    newGraph->topoOrder.push_back(u);
  };
  topo(newGraph->newProd);
  std::reverse(newGraph->topoOrder.begin(), newGraph->topoOrder.end());

  return newGraph;
}