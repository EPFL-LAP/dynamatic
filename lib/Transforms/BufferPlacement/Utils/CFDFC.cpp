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

#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

namespace {
/// Helper data structure to hold mappings between each arch/basic block and the
/// Gurobi variable that corresponds to it.
struct MILPVars {
  /// Mapping between each arch and Gurobi variable.
  std::map<ArchBB *, CPVar> archs;
  /// Mapping between each basic block and Gurobi variable.
  std::map<unsigned, CPVar> bbs;
  /// Holds the maximum number of executions achievable.
  CPVar numExecs;
};
} // namespace

/// Initializes all variables in the MILP, one per arch and per basic block.
/// Fills in the last argument with mappings between archs/BBs and their
/// associated Gurobi variable.
static void initMILPVariables(std::unique_ptr<CPSolver> &model, ArchSet &archs,
                              BBSet &bbs, MILPVars &vars) {
  // Keep track of the maximum number of transitions in any arch
  unsigned maxTrans = 0;

  // Create a variable for each basic block
  for (unsigned bb : bbs)
    vars.bbs[bb] = model->addVar("sBB_" + std::to_string(bb), BOOLEAN, 0, 1);

  // Create a variable for each arch
  for (ArchBB *arch : archs) {
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                          std::to_string(arch->dstBB);
    vars.archs[arch] = model->addVar(arcName, BOOLEAN, 0, 1);
    maxTrans = std::max(maxTrans, arch->numTrans);
  }

  // Create a variable to hold the maximum number of CFDFC executions
  vars.numExecs = model->addVar("varMaxExecs", INTEGER, 0, maxTrans);
}

/// Sets the MILP objective, which is to maximize the sum over all archs of
/// sArc_<srcBB>_<dstBB> * varMaxTrans.
static void setObjective(std::unique_ptr<CPSolver> &model, MILPVars &vars) {
  LinExpr objExpr = 0;
  for (auto &[_, var] : vars.archs) {

    // vars.numExecs * var is not a linear term, we should linearize it
    auto numExecsTimesVar = model->addVar("numExec_times_" + var.getName(),
                                          INTEGER, 0, std::nullopt);
    constexpr double bigM = 1e4;

    // - If var == 0: 0 <= w <= 0
    // - If var == 1: numExecs <= w <= numExecs
    model->addConstr(numExecsTimesVar <= vars.numExecs);
    model->addConstr(numExecsTimesVar >= vars.numExecs - bigM * (1 - var));
    model->addConstr(numExecsTimesVar <= bigM * var);
    objExpr += numExecsTimesVar;
  }
  model->setMaximizeObjective(objExpr);
}

/// Sets the MILP's edge constraints, one per arch to limit the number of times
/// it can be taken plus one to force the number of selected backedges to be 1.
static void setEdgeConstraints(std::unique_ptr<CPSolver> &model, MILPVars &vars,
                               unsigned maxExecs) {
  LinExpr backedgeConstraint;

  // Add a constraint for each arch
  for (auto &[arch, var] : vars.archs) {
    // For each arch, limit the output number of transitions to, if it is
    // selected, the arch's number of transitions, or, if it is not selected, to
    // the maximum number of transitions
    std::string name = "arch_" + std::to_string(arch->srcBB) + "_" +
                       std::to_string(arch->dstBB);
    model->addConstr(
        vars.numExecs <= var * arch->numTrans + (1 - var) * maxExecs, name);

    // Only select one backedge
    if (arch->isBackEdge)
      backedgeConstraint += var;
  }

  // Finally, the backedge constraint
  model->addConstr(backedgeConstraint == 1, "oneBackedge");
}

/// Get all variables corresponding to "predecessor archs" i.e., archs from
/// predecessor blocks to the given block.
static SmallVector<CPVar> getPredArchVars(unsigned bb, MILPVars &vars) {
  SmallVector<CPVar> predVars;
  for (auto &[arch, var] : vars.archs)
    if (arch->dstBB == bb)
      predVars.push_back(var);
  return predVars;
}

/// Get all variables corresponding to "successor archs" i.e., archs from the
/// given block to its successor blocks.
static SmallVector<CPVar> getSuccArchvars(unsigned bb, MILPVars &vars) {
  SmallVector<CPVar> succVars;
  for (auto &[arch, var] : vars.archs)
    if (arch->srcBB == bb)
      succVars.push_back(var);
  return succVars;
}

/// Sets the MILP's basic block constraints, two per block to force the
/// selection of (1) exactly one predecessor and successor arch if it is
/// selected or (2) exactly zero predecessor and successor arch if it is not
/// selected.
static void setBBConstraints(std::unique_ptr<CPSolver> &model, MILPVars &vars) {
  // Add two constraints for each arch
  for (auto &[bb, varBB] : vars.bbs) {
    // Set constraint for predecessor archs
    LinExpr predArchsConstr;
    for (CPVar &var : getPredArchVars(bb, vars))
      predArchsConstr += var;
    model->addConstr(predArchsConstr == varBB, "in" + std::to_string(bb));

    // Set constraint for successor archs
    LinExpr succArchsConstr;
    for (CPVar &var : getSuccArchvars(bb, vars))
      succArchsConstr += var;
    model->addConstr(succArchsConstr == varBB, "out" + std::to_string(bb));
  }
}

CFDFC CFDFC::create(handshake::FuncOp funcOp, ArchSet &archs,
                    unsigned numExec) {
  CFDFC cfdfc(funcOp, archs, numExec);
  cfdfc.collectChannels();
  return cfdfc;
}

CFDFC CFDFC::createFTD(handshake::FuncOp funcOp, ArchSet &archs,
                       unsigned numExec,
                       const DenseSet<Value> &backwardChannels) {
  CFDFC cfdfc(funcOp, archs, numExec);
  cfdfc.collectChannelsFTD(backwardChannels);
  return cfdfc;
}

CFDFC::CFDFC(handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec)
    : CFDFC() {
  numExecs = numExec;

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

    // Add the unit to the CFDFC
    addUnit(&op);
  }
}

void CFDFC::collectChannels() {
  // Iterate over the pre-collected units which have been filtered to just
  // this CFDFC
  for (Operation *op : getUnits()) {
    unsigned srcBB = *getLogicBB(op);

    // Add each handshake channel to the CFDFC, and additionally to backedges
    // if the handshake channel is part of a backedge
    for (OpResult res : op->getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      // Get the value's unique user and its basic block
      Operation *user = *res.getUsers().begin();
      std::optional<unsigned> optDstBB = getLogicBB(user);

      // Channels to users without a basic block are not added to the CFDFC
      if (optDstBB.has_value()) {
        unsigned dstBB = *optDstBB;

        if (srcBB != dstBB) {
          // The channel is in the CFDFC if it belongs belong to a selected arch
          // between two basic blocks

          // check each arch in the CFDFC
          for (size_t i = 0; i < cycle.size(); ++i) {
            unsigned nextBB = i == cycle.size() - 1 ? 0 : i + 1;

            // check if a matching arch is found
            if (srcBB == cycle[i] && dstBB == cycle[nextBB]) {
              addChannel(res, isCFDFCBackedge(res));

              // stop if a matching arch is found
              break;
            }
          }
        } else if (cycle.size() == 1) {
          // The channel is in the CFDFC if its producer/consumer belong to the
          // same basic block and the CFDFC is just a block looping to itself
          addChannel(res, isCFDFCBackedge(res));
        } else if (!dynamatic::isBackedge(res)) {
          // The channel is in the CFDFC if its producer/consumer belong to the
          // same basic block and the channel is not a backedge
          addChannel(res, /*isBackedge=*/false);
        }
      }
    }
  }
}

void CFDFC::collectChannelsFTD(const DenseSet<Value> &backwardChannels) {
  for (Operation *op : getUnits()) {
    unsigned srcBB = *getLogicBB(op);

    // Add each handshake channel to the CFDFC, and additionally to backedges
    // if the handshake channel is part of a backedge
    for (OpResult res : op->getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");
      Operation *user = *res.getUsers().begin();
      std::optional<unsigned> optDstBB = getLogicBB(user);
      if (optDstBB.has_value()) {
        unsigned dstBB = *optDstBB;
        if (cycle.contains(dstBB)) {
          // FTD circuits may contain channels between blocks without a matching
          // CFG edge. Include every non-backward channel whose endpoints belong
          // to the cycle. Include backward channels only when they correspond
          // to an edge in the CFG cycle currently being modeled.
          if (!backwardChannels.contains(res)) {
            addChannel(res, /*isBackedge=*/false);
          } else if (isCFGCompliant(srcBB, dstBB)) {
            // Backward channels that are compliant with the CFG are CFDFC
            // channels and backedges
            addChannel(res, /*isBackedge=*/true);
          } else {
            // Other backward channels are recorded as FTD backedges, but are
            // not CFDFC channels
            ftdBackedges.push_back(res);
          }
        }
      }
    }
  }
}

bool CFDFC::isCFGCompliant(unsigned srcBB, unsigned dstBB) const {
  for (size_t i = 0; i < cycle.size(); ++i) {
    unsigned nextBB = i == cycle.size() - 1 ? 0 : i + 1;
    if (srcBB == cycle[i] && dstBB == cycle[nextBB])
      return true;
  }
  return false;
}

bool CFDFC::isCFDFCBackedge(Value val) {
  // A CFDFC backedge is a backedge
  if (!dynamatic::isBackedge(val))
    return false;

  Operation *defOp = val.getDefiningOp();
  std::optional<unsigned> srcBB = getLogicBB(defOp);
  std::optional<unsigned> dstBB = getLogicBB(*val.getUsers().begin());

  // If the edge is between operations in the same block, the source operation
  // must be a conditional branch
  if (srcBB.has_value() && dstBB.has_value() && *srcBB == *dstBB)
    return isa<handshake::ConditionalBranchOp>(defOp);

  // Otherwise the edge must be between different blocks, where the
  // destination can be out of all blocks
  return srcBB.has_value() && (!dstBB.has_value() || *srcBB != *dstBB);
}

CFDFC::CFDFC() : entry(new CFUnit(/*op=*/nullptr)) {}

CFDFC CFDFC::clone() const {
  CFDFC copy;
  static_cast<CFDFCInfo &>(copy) = *this;
  for (Operation *op : getUnits())
    copy.addUnit(op);
  for (const CFChannel *channel : getCFChannels())
    copy.addChannel(channel->getValue(), channel->isBackedge());
  copy.ftdBackedges = ftdBackedges;
  return copy;
}

CFUnit *CFDFC::addUnit(Operation *op) {
  auto [it, inserted] =
      units.try_emplace(op, std::unique_ptr<CFUnit>(new CFUnit(op)));
  assert(inserted && "unit added twice to the CFDFC");
  CFUnit *unit = it->second.get();
  entry->succs.push_back(unit);
  return unit;
}

void CFDFC::addChannel(Value value, bool isBackedge) {
  auto srcIt = units.find(value.getDefiningOp());
  auto dstIt = units.find(*value.getUsers().begin());
  assert(srcIt != units.end() && dstIt != units.end() &&
         "channel endpoints must be units of the CFDFC");
  CFUnit *src = srcIt->second.get();
  CFUnit *dst = dstIt->second.get();

  auto [it, inserted] = channels.try_emplace(
      value,
      std::unique_ptr<CFChannel>(new CFChannel(value, src, dst, isBackedge)));
  assert(inserted && "channel added twice to the CFDFC");
  CFChannel *channel = it->second.get();
  src->outChannels.push_back(channel);
  src->succs.push_back(dst);
  dst->inChannels.push_back(channel);
  dst->preds.push_back(src);
}

size_t CFDFC::getNumBackedges() const {
  return llvm::count_if(getCFChannels(), [](const CFChannel *channel) {
    return channel->isBackedge();
  });
}

const CFUnit *CFDFC::lookup(Operation *op) const {
  auto it = units.find(op);
  return it == units.end() ? nullptr : it->second.get();
}

const CFChannel *CFDFC::lookup(Value value) const {
  auto it = channels.find(value);
  return it == channels.end() ? nullptr : it->second.get();
}

bool CFDFC::isBackedge(Value value) const {
  const CFChannel *channel = lookup(value);
  return channel && channel->isBackedge();
}

std::vector<SmallVector<Operation *>> CFDFC::getSCCs() const {
  std::vector<SmallVector<Operation *>> sccs;
  for (auto it = llvm::scc_begin(this); !it.isAtEnd(); ++it) {
    if (it->front() == getEntryNode())
      continue;
    SmallVector<Operation *> &scc = sccs.emplace_back();
    for (const CFUnit *unit : *it)
      scc.push_back(unit->getOp());
  }
  return sccs;
}

/// Removes `channel` from `channels`, and the unit at the same position from
/// `units`, which lists the other endpoint of each of the channels.
static void unlink(SmallVectorImpl<CFChannel *> &channels,
                   SmallVectorImpl<CFUnit *> &units, CFChannel *channel) {
  auto *it = llvm::find(channels, channel);
  assert(it != channels.end() && "channel not linked to the unit");
  units.erase(units.begin() + (it - channels.begin()));
  channels.erase(it);
}

void CFDFC::insertUnitsOnChannel(Value value, ArrayRef<Operation *> newOps) {
  if (newOps.empty())
    return;
  auto channelIt = channels.find(value);
  assert(channelIt != channels.end() && "channel must be in the CFDFC");
  assert(*value.getUsers().begin() == newOps.front() &&
         "the IR must already be rewired through the new units");
  CFChannel *channel = channelIt->second.get();
  CFUnit *src = channel->src;
  CFUnit *originalDst = channel->dst;

  CFUnit *first = addUnit(newOps.front());
  for (Operation *op : newOps.drop_front())
    addUnit(op);

  auto *outIt = llvm::find(src->outChannels, channel);
  src->succs[outIt - src->outChannels.begin()] = first;
  unlink(originalDst->inChannels, originalDst->preds, channel);
  channel->dst = first;
  first->inChannels.push_back(channel);
  first->preds.push_back(src);

  for (Operation *op : newOps)
    addChannel(op->getResult(0), /*isBackedge=*/false);
}

void CFDFC::removeChannel(Value value) {
  auto it = channels.find(value);
  assert(it != channels.end() && "channel must be in the CFDFC");
  CFChannel *channel = it->second.get();

  unlink(channel->src->outChannels, channel->src->succs, channel);
  unlink(channel->dst->inChannels, channel->dst->preds, channel);
  channels.erase(it);
}

CFDFCUnion::CFDFCUnion(ArrayRef<CFDFC *> cfdfcs) {
  // Just do the union of everything
  for (CFDFC *cf : cfdfcs) {
    // Blocks
    for (unsigned bb : cf->cycle)
      blocks.insert(bb);
    // Units
    for (Operation *op : cf->getUnits())
      units.insert(op);
    // Channels
    for (Value val : cf->getChannels())
      channels.insert(val);
    // Backedges
    for (Value val : cf->getBackedges())
      backedges.insert(val);
    for (Value val : cf->getFTDBackedges())
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

LogicalResult dynamatic::buffer::extractCFDFC(
    handshake::FuncOp funcOp, ArchSet &archs, BBSet &bbs,
    ArchSet &selectedArchs, unsigned &numExecs, CPSolver::SolverKind solverKind,
    const std::string &logPath, int *milpStat) {
  // Create an MILP model for CFDFC extraction (NOTE: since the problem is
  // small, we could use CBC here).
  std::unique_ptr<CPSolver> model;

  switch (solverKind) {
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
  case CPSolver::GUROBI:
    model = std::make_unique<GurobiSolver>();
    break;
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
#ifdef DYNAMATIC_ENABLE_CBC
  case CPSolver::CBC:
    model = std::make_unique<CbcSolver>();
    break;
#endif // DYNAMATIC_ENABLE_CBC
  case CPSolver::DEFAULT:
    llvm_unreachable("Default option should not be used! Check if you have "
                     "correctly installed one of CBC or GUROBI milp solver!");
  }

  // Create all MILP variables we need
  MILPVars vars;
  initMILPVariables(model, archs, bbs, vars);

  unsigned maxExec = 0;

  for (auto *arch : archs) {
    maxExec = std::max(arch->numTrans, maxExec);
  }

  // Set up the MILP and solve it
  setObjective(model, vars);
  setEdgeConstraints(model, vars, maxExec);
  setBBConstraints(model, vars);

  if (!logPath.empty())
    model->write(logPath + "_model.lp");
  model->optimize();

  if (model->status != CPSolver::OPTIMAL)
    return failure();

  // Retrieve the maximum number of transitions identified by the MILP
  // solution
  numExecs = static_cast<unsigned>(model->getValue(vars.numExecs));
  if (numExecs == 0)
    return success();

  // Fill in the set of selected archs and decrease their associated number of
  // transitions
  for (auto &[arch, var] : vars.archs) {
    if (archs.count(arch) > 0 && model->getValue(var) > 0) {
      selectedArchs.insert(arch);
      arch->numTrans -= numExecs;
    }
  }

  return success();
}
