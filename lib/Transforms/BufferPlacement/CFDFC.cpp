//===- CFDFC.cpp - Control-Free DataFlow Circuit ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions related to CFDFC extraction and creation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <fstream>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace {
/// Helper data structure to hold mappings between each arch/basic block and the
/// Gurobi variable that corresponds to it.
struct MILPVars {
  /// Mapping between each arch and Gurobi variable.
  std::map<ArchBB *, GRBVar> archs;
  /// Mapping between each basic block and Gurobi variable.
  std::map<unsigned, GRBVar> bbs;
  /// Holds the maximum number of executions achievable.
  GRBVar numExecs;
};
} // namespace

/// Initializes all variables in the MILP, one per arch and per basic block.
/// Fills in the last argument with mappings between archs/BBs and their
/// associated Gurobi variable.
static void initMILPVariables(GRBModel &model, ArchSet &archs, BBSet &bbs,
                              MILPVars &vars) {
  // Keep track of the maximum number of transitions in any arch
  unsigned maxTrans = 0;

  // Create a variable for each basic block
  for (unsigned bb : bbs)
    vars.bbs[bb] =
        model.addVar(0.0, 1, 0.0, GRB_BINARY, "sBB_" + std::to_string(bb));

  // Create a variable for each arch
  for (ArchBB *arch : archs) {
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                          std::to_string(arch->dstBB);
    vars.archs[arch] = model.addVar(0.0, 1, 0.0, GRB_BINARY, arcName);
    maxTrans = std::max(maxTrans, arch->numTrans);
  }

  // Create a variable to hold the maximum number of CFDFC executions
  vars.numExecs = model.addVar(0, maxTrans, 0.0, GRB_INTEGER, "varMaxExecs");

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

/// Sets the MILP objective, which is to maximize the sum over all archs of
/// sArc_<srcBB>_<dstBB> * varMaxTrans.
static void setObjective(GRBModel &model, MILPVars &vars) {
  GRBQuadExpr objExpr;
  for (auto &[_, var] : vars.archs)
    objExpr += vars.numExecs * var;
  model.setObjective(objExpr, GRB_MAXIMIZE);
}

/// Sets the MILP's edge constraints, one per arch to limit the number of times
/// it can be taken plus one to force the number of selected backedges to be 1.
static void setEdgeConstraints(GRBModel &model, MILPVars &vars) {
  GRBLinExpr backedgeConstraint;

  // Add a constraint for each arch
  for (auto &[arch, var] : vars.archs) {
    // For each arch, limit the output number of transitions to, if it is
    // selected, the arch's number of transitions, or, if it is not selected, to
    // the maximum number of transitions
    unsigned maxExecsUB =
        static_cast<unsigned>(vars.numExecs.get(GRB_DoubleAttr_UB));
    std::string name = "arch_" + std::to_string(arch->srcBB) + "_" +
                       std::to_string(arch->dstBB);
    model.addConstr(
        vars.numExecs <= var * arch->numTrans + (1 - var) * maxExecsUB, name);

    // Only select one backedge
    if (arch->isBackEdge)
      backedgeConstraint += var;
  }

  // Finally, the backedge constraint
  model.addConstr(backedgeConstraint == 1, "oneBackedge");
}

/// Get all variables corresponding to "predecessor archs" i.e., archs from
/// predecessor blocks to the given block.
static SmallVector<GRBVar> getPredArchVars(unsigned bb, MILPVars &vars) {
  SmallVector<GRBVar> predVars;
  for (auto &[arch, var] : vars.archs)
    if (arch->dstBB == bb)
      predVars.push_back(var);
  return predVars;
}

/// Get all variables corresponding to "successor archs" i.e., archs from the
/// given block to its successor blocks.
static SmallVector<GRBVar> getSuccArchvars(unsigned bb, MILPVars &vars) {
  SmallVector<GRBVar> succVars;
  for (auto &[arch, var] : vars.archs)
    if (arch->srcBB == bb)
      succVars.push_back(var);
  return succVars;
}

/// Sets the MILP's basic block constraints, two per block to force the
/// selection of (1) exactly one predecessor and successor arch if it is
/// selected or (2) exactly zero predecessor and successor arch if it is not
/// selected.
static void setBBConstraints(GRBModel &model, MILPVars &vars) {
  // Add two constraints for each arch
  for (auto &[bb, varBB] : vars.bbs) {
    // Set constraint for predecessor archs
    GRBLinExpr predArchsConstr;
    for (GRBVar &var : getPredArchVars(bb, vars))
      predArchsConstr += var;
    model.addConstr(predArchsConstr == varBB, "in" + std::to_string(bb));

    // Set constraint for successor archs
    GRBLinExpr succArchsConstr;
    for (GRBVar &var : getSuccArchvars(bb, vars))
      succArchsConstr += var;
    model.addConstr(succArchsConstr == varBB, "out" + std::to_string(bb));
  }
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

CFDFC::CFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec)
    : numExecs(numExec) {

  // Identify the block that starts the CFDFC; it's the only one that is both
  // the source of an arch and the destination of another
  std::optional<unsigned> startBB;
  llvm::SmallSet<unsigned, 4> uniqueBlocks;
  for (ArchBB *arch : archs) {
    if (auto [_, inserted] = uniqueBlocks.insert(arch->srcBB); !inserted) {
      startBB = arch->srcBB;
      break;
    }
    if (auto [_, inserted] = uniqueBlocks.insert(arch->dstBB); !inserted) {
      startBB = arch->dstBB;
      break;
    }
  }
  assert(startBB.has_value() && "failed to identify start of CFDFC");

  // Form the cycle by stupidly iterating over the archs
  cycle.insert(*startBB);
  unsigned currentBB = *startBB;
  for (size_t i = 0, e = archs.size() - 1; i < e; ++i) {
    for (ArchBB *arch : archs) {
      if (arch->srcBB == currentBB) {
        currentBB = arch->dstBB;
        cycle.insert(currentBB);
        break;
      }
    }
  }
  assert(cycle.size() == archs.size() && "failed to construct cycle");

  for (Operation &op : funcOp.getOps()) {
    // Get operation's basic block
    unsigned srcBB;
    if (auto optBB = getLogicBB(&op); !optBB.has_value())
      continue;
    else
      srcBB = *optBB;

    // The basic block the operation belongs to must be selected
    if (!cycle.contains(srcBB))
      continue;

    // Add the unit and valid outgoing channels to the CFDFC
    units.insert(&op);
    for (OpResult res : op.getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      // Get the value's unique user and its basic block
      Operation *user = *res.getUsers().begin();
      unsigned dstBB;
      if (std::optional<unsigned> optBB = getLogicBB(user); !optBB.has_value())
        continue;
      else
        dstBB = *optBB;

      if (srcBB != dstBB) {
        // The channel is in the CFDFC if it belongs belong to a selected arch
        // between two basic blocks
        for (size_t i = 0; i < cycle.size(); ++i) {
          unsigned nextBB = i == cycle.size() - 1 ? 0 : i + 1;
          if (srcBB == cycle[i] && dstBB == cycle[nextBB]) {
            channels.insert(res);
            if (isCFDFCBackedge(res))
              backedges.insert(res);
            break;
          }
        }
      } else if (cycle.size() == 1) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the CFDFC is just a block looping to itself
        channels.insert(res);
        if (isCFDFCBackedge(res))
          backedges.insert(res);
      } else if (!isBackedge(res)) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the channel is not a backedge
        channels.insert(res);
      }
    }
  }
}

bool CFDFC::isCFDFCBackedge(Value val) {
  // A CFDFC backedge is a backedge
  if (!isBackedge(val))
    return false;

  Operation *defOp = val.getDefiningOp();
  std::optional<unsigned> srcBB = getLogicBB(defOp);
  std::optional<unsigned> dstBB = getLogicBB(*val.getUsers().begin());

  // If the edge is between operations in the same block, the source operation
  // must be a conditional branch
  if (srcBB.has_value() && dstBB.has_value() && *srcBB == *dstBB)
    return isa<handshake::ConditionalBranchOp>(defOp);

  // Otherwise the edge must be between different blocks, where the destination
  // can be out of all blocks
  return srcBB.has_value() && (!dstBB.has_value() || *srcBB != *dstBB);
}

CFDFCUnion::CFDFCUnion(ArrayRef<CFDFC *> cfdfcs) {
  // Just do the union of everything
  for (CFDFC *cf : cfdfcs) {
    // Blocks
    for (unsigned bb : cf->cycle)
      blocks.insert(bb);
    // Units
    for (Operation *op : cf->units)
      units.insert(op);
    // Channels
    for (Value val : cf->channels)
      channels.insert(val);
    // Backedges
    for (Value val : cf->backedges)
      backedges.insert(val);

    this->cfdfcs.push_back(cf);
  }
}

namespace {
/// Disjoint-set data structure for storing information about a disjoint set.
struct DisjointSetInfo {
  /// The set's parent.
  size_t parent;
  /// The set's rank.
  size_t rank;
};
} // namespace

void dynamatic::buffer::getDisjointBlockUnions(
    ArrayRef<CFDFC *> cfdfcs, std::vector<CFDFCUnion> &unions) {

  SmallVector<DisjointSetInfo> sets;
  size_t numCFDFC = cfdfcs.size();
  for (size_t idx = 0; idx < numCFDFC; ++idx)
    sets.emplace_back(DisjointSetInfo{idx, 0});

  auto find = [&](size_t i) -> size_t {
    while (sets[i].parent != i) {
      size_t &parent = sets[i].parent;
      parent = sets[parent].parent;
      i = parent;
    }
    return i;
  };

  auto setUnion = [&](size_t i, size_t j) -> void {
    // Replace nodes by roots
    i = find(i);
    j = find(j);
    if (i == j)
      // Already in the same set
      return;

    // Check that i has the bigger rank
    size_t &iRank = sets[i].rank, &jRank = sets[j].rank;
    if (iRank < jRank)
      std::swap(i, j);

    sets[j].parent = i;
    // If both sets have the same rank, increment one
    if (iRank == jRank)
      ++iRank;
  };

  for (size_t i = 0; i < numCFDFC; ++i) {
    for (size_t j = 0; j < i; ++j) {
      std::set<unsigned> cycle, cycleRef;
      for (unsigned bb : cfdfcs[i]->cycle)
        cycle.insert(bb);
      for (unsigned bb : cfdfcs[j]->cycle)
        cycleRef.insert(bb);
      llvm::set_intersect(cycle, cycleRef);
      if (!cycle.empty())
        // Non-empty intersection
        setUnion(i, j);
    }
  }

  // Collect all set indices
  std::set<size_t> setIndices;
  for (size_t idx = 0; idx < numCFDFC; ++idx)
    setIndices.insert(find(idx));

  // For each index, collect CFDFCs and create the union
  for (size_t setIdx : setIndices) {
    SmallVector<CFDFC *> cfUnion;
    for (auto [idx, cf] : llvm::enumerate(cfdfcs))
      if (find(idx) == setIdx)
        cfUnion.push_back(cf);
    unions.emplace_back(cfUnion);
  }
}

LogicalResult dynamatic::buffer::extractCFDFC(handshake::FuncOp funcOp,
                                              ArchSet &archs, BBSet &bbs,
                                              ArchSet &selectedArchs,
                                              unsigned &numExecs,
                                              const std::string &logPath,
                                              int *milpStat) {
#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  return funcOp->emitError() << "Project was built without Gurobi, can't run "
                                "CFDFC extraction";
#else
  // Create Gurobi MILP model for CFDFC extraction, suppressing stdout
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);

  // Create all MILP variables we need
  MILPVars vars;
  initMILPVariables(model, archs, bbs, vars);

  // Set up the MILP and solve it
  setObjective(model, vars);
  setEdgeConstraints(model, vars);
  setBBConstraints(model, vars);

  if (!logPath.empty())
    model.write(logPath + "_model.lp");
  model.optimize();
  if (!logPath.empty())
    model.write(logPath + "_solutions.json");

  int stat = model.get(GRB_IntAttr_Status);
  if (milpStat)
    *milpStat = stat;
  if (int status = model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
    return failure();

  // Retrieve the maximum number of transitions identified by the MILP
  // solution
  numExecs = static_cast<unsigned>(vars.numExecs.get(GRB_DoubleAttr_X));
  if (numExecs == 0)
    return success();

  // Fill in the set of selected archs and decrease their associated number of
  // transitions
  for (auto &[arch, var] : vars.archs) {
    if (archs.count(arch) > 0 && var.get(GRB_DoubleAttr_X) > 0) {
      selectedArchs.insert(arch);
      arch->numTrans -= numExecs;
    }
  }

  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}
