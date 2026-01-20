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
  // TODO: print

  for (Block *latch : latches) {
    scope->allBackEdges.insert({latch, scope->header});
  }

  auto loopBlocks = loop->getBlocks();
  for (Block *b : loopBlocks) {
    scope->allBlocksInclusive.insert(b);
    // This requires outer loops to be executed prior to inner loops.
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

  // Create the False terminal representing loop back-edges or sink.
  Block *falseTerm = new Block();
  R.push_back(falseTerm);
  newGraph->sinkBB = falseTerm;
  newGraph->origMap[falseTerm] = nullptr;

  // Create the True terminal representing the loop exit or consumer.
  Block *trueTerm = new Block();
  R.push_back(trueTerm);
  newGraph->newCons = trueTerm;

  // For Level 0, terminals map to the actual Consumer in the original CFG.
  if (scope->level == 0)
    newGraph->origMap[trueTerm] = lcfg.origMap.lookup(lcfg.newCons);
  else
    newGraph->origMap[trueTerm] = nullptr;

  // Clone the header block which acts as the entry (Producer) for this layer.
  Block *clonedHeader = new Block();
  R.push_back(clonedHeader);
  newGraph->newProd = clonedHeader;

  // Map the new header directly to the block in the original CFG.
  newGraph->origMap[clonedHeader] = lcfg.origMap.lookup(scope->header);

  // Initialize the mapping from the current Scope to the new blocks.
  DenseMap<Block *, Block *> clonedMap;
  clonedMap[scope->header] = clonedHeader;

  // Pre-fill terminals for Level 0.
  if (scope->level == 0) {
    clonedMap[lcfg.newCons] = trueTerm;
    clonedMap[lcfg.sinkBB] = falseTerm;
  }

  // Clone all blocks within the scope, excluding special ones handled above.
  for (Block *b : scope->allBlocksInclusive) {
    if (b == scope->header || b == lcfg.newCons || b == lcfg.sinkBB)
      continue;
    Block *nb = new Block();
    R.push_back(nb);
    clonedMap[b] = nb;
    newGraph->origMap[nb] = lcfg.origMap.lookup(b);
  }

  // Reconstruct edges and terminators.
  for (auto [origBlock, newBlock] : clonedMap) {
    // Skip special blocks handled above.
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

    // Analyze successors to reconstruct paths.
    for (unsigned i = 0; i < origTerm->getNumSuccessors(); ++i) {
      Block *origSucc = origTerm->getSuccessor(i);

      if (origSucc == scope->header) {
        // Current level back-edge redirects to Sink (False).
        if (scope->level == 0)
          llvm::errs() << "[FTD Warning] Level 0 scope has back-edge to its "
                          "own header.\n";
        validSuccessors.push_back(falseTerm);
        keepSuccessor.push_back(true);
      } else if (scope->allBackEdges.contains({origBlock, origSucc})) {
        // Deep back-edge (from inner loops) is pruned/dropped.
        validSuccessors.push_back(nullptr);
        keepSuccessor.push_back(false);
      } else if (clonedMap.count(origSucc)) {
        // Standard in-scope jump.
        validSuccessors.push_back(clonedMap[origSucc]);
        keepSuccessor.push_back(true);
      } else {
        // Jump outside scope redirects to True terminal.
        if (scope->level == 0)
          llvm::errs()
              << "[FTD Warning] Block outside Level 0 scope encountered.\n";
        validSuccessors.push_back(trueTerm);
        keepSuccessor.push_back(true);
      }
    }

    // Rebuild terminator based on valid successors.
    if (auto cbr = dyn_cast<cf::CondBranchOp>(origTerm)) {
      if (keepSuccessor[0] && keepSuccessor[1]) {
        // Both paths are valid, keep condition.
        builder.create<cf::CondBranchOp>(
            loc, cbr.getCondition(), validSuccessors[0], validSuccessors[1]);
      } else if (keepSuccessor[0] && !keepSuccessor[1]) {
        // Only true path is valid, degrade to unconditional branch.
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
      } else if (!keepSuccessor[0] && keepSuccessor[1]) {
        // Only false path is valid, degrade to unconditional branch.
        builder.create<cf::BranchOp>(loc, validSuccessors[1]);
      }
    } else if (auto br = dyn_cast<cf::BranchOp>(origTerm)) {
      if (keepSuccessor[0])
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
    }
  }

  // Iterative Graph Optimization (Canonicalization)
  // 1. Remove blocks with no terminator due to invalid successors.
  //    - Explicitly update predecessors to disconnect them.
  //    - Erase the dead block.
  // 2. Merge identical successors (CondBranch -> Branch).
  // 3. Inline single-successor blocks (Path Compression).
  // 4. Repeat until fixed point.
  bool changed = true;
  while (changed) {
    changed = false;

    // Use early_inc_range to safely erase blocks during iteration.
    for (Block &block : llvm::make_early_inc_range(R)) {
      // Do not optimize/remove terminal blocks or the entry.
      if (&block == newGraph->newProd || &block == newGraph->newCons ||
          &block == newGraph->sinkBB)
        continue;

      Operation *term = block.getTerminator();

      // Remove blocks with no terminator.
      if (!term) {
        // Collect all users (Terminators of predecessor blocks) first.
        // We cannot iterate predecessors while modifying them.
        SmallVector<Operation *, 4> predTerms;
        for (auto &use : block.getUses())
          predTerms.push_back(use.getOwner());

        for (Operation *predTerm : predTerms) {
          OpBuilder localBuilder(predTerm->getContext());
          localBuilder.setInsertionPoint(predTerm);

          if (auto br = dyn_cast<cf::BranchOp>(predTerm)) {
            // Predecessor unconditionally jumps to this dead block.
            // Remove the jump, making the predecessor dead as well.
            predTerm->erase();
          } else if (auto cbr = dyn_cast<cf::CondBranchOp>(predTerm)) {
            Block *trueDest = cbr.getTrueDest();
            Block *falseDest = cbr.getFalseDest();

            if (trueDest == &block && falseDest == &block) {
              // Both legs go to dead block. Predecessor becomes dead.
              predTerm->erase();
            } else if (trueDest == &block) {
              // True leg is dead, simplify to unconditional jump to False.
              localBuilder.create<cf::BranchOp>(loc, falseDest);
              predTerm->erase();
            } else if (falseDest == &block) {
              // False leg is dead, simplify to unconditional jump to True.
              localBuilder.create<cf::BranchOp>(loc, trueDest);
              predTerm->erase();
            }
          }
        }

        // Now that all edges pointing to this block are removed/redirected,
        // erase it.
        block.erase();
        changed = true;
        continue;
      }

      // Optimization: Merge Duplicate Successors.
      // If CondBranch jumps to A and A, replace with Branch(A).
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

      // Optimization: Single Successor Inlining (Path Compression).
      // If Block A simply jumps to Block B, replace all jumps to A with jumps
      // to B, then remove A. This implicitly handles "Dead Ends" (nodes that
      // only point to Sink).
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        Block *dest = br.getDest();
        // Redirect all predecessors of 'block' to 'dest'.
        block.replaceAllUsesWith(dest);
        block.erase();
        changed = true;
        continue;
      }
    }
  }

  // Finalize terminals.
  builder.setInsertionPointToEnd(trueTerm);
  builder.create<cf::BranchOp>(loc, falseTerm);
  builder.setInsertionPointToEnd(falseTerm);
  builder.create<func::ReturnOp>(loc);

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
