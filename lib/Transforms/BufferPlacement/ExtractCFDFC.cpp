//===- ExtractCFDFC.cpp - Extract CFDFCs from dataflow circuits -*- C++ -*-===//
//
// This file implements functions related to CFDFC extraction and creation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ExtractCFDFC.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
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
};
} // namespace

/// Initializes all variables in the MILP, one per arch and per basic block.
/// Fills in the last argument with mappings between archs/BBs and their
/// associated Gurobi variable.
static unsigned initMILPVariables(GRBModel &model, ArchSet &archs, BBSet &bbs,
                                  MILPVars &vars) {
  // Keep track of the maximum number of transitions in any arch
  unsigned maxNumTrans = 0;

  // Create a variable for each basic block
  for (unsigned bb : bbs)
    vars.bbs[bb] =
        model.addVar(0.0, 1, 0.0, GRB_BINARY, "sBB_" + std::to_string(bb));

  // Create a variable for each arch
  for (ArchBB *arch : archs) {
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                          std::to_string(arch->dstBB);
    vars.archs[arch] = model.addVar(0.0, 1, 0.0, GRB_BINARY, arcName);
    maxNumTrans = std::max(maxNumTrans, arch->numTrans);
  }

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
  return maxNumTrans;
}

/// Sets the MILP objective, which is to maximize the sum over all archs of
/// sArc_<srcBB>_<dstBB> * varMaxTrans.
static void setObjective(GRBModel &model, MILPVars &vars, GRBVar &varMaxTrans) {
  GRBQuadExpr objExpr;
  for (auto &[_, var] : vars.archs)
    objExpr += varMaxTrans * var;
  model.setObjective(objExpr, GRB_MAXIMIZE);
}

/// Sets the MILP's edge constraints, one per arch to limit the number of times
/// it can be taken plus one to force the number of selected backedges to be 1.
static void setEdgeConstraints(GRBModel &model, MILPVars &vars,
                               GRBVar &varMaxTrans) {
  GRBLinExpr backEdgeConstr;

  // Add a constraint for each arch
  for (auto [constrInd, archVar] : llvm::enumerate(vars.archs)) {
    auto &[arch, var] = archVar;
    // For each arch, limit the output number of transitions to, if it is
    // selected, the arch's number of transitions, or, if it is not selected, to
    // the maximum number of transitions
    model.addConstr(varMaxTrans <= var * arch->numTrans +
                                       (1 - var) * (unsigned)varMaxTrans.get(
                                                       GRB_DoubleAttr_UB),
                    "cN" + std::to_string(constrInd));

    // Only select one backedge
    if (arch->isBackEdge)
      backEdgeConstr += var;
  }

  // Finally, the backedge constraint
  model.addConstr(backEdgeConstr == 1, "cBack");
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
  for (auto &[bbInd, varBB] : vars.bbs) {
    // Set constraint for predecessor archs
    GRBLinExpr predArchsConstr;
    for (GRBVar &var : getPredArchVars(bbInd, vars))
      predArchsConstr += var;
    model.addConstr(predArchsConstr == varBB, "cIn" + std::to_string(bbInd));

    // Set constraint for successor archs
    GRBLinExpr succArchsConstr;
    for (GRBVar &var : getSuccArchvars(bbInd, vars))
      succArchsConstr += var;
    model.addConstr(succArchsConstr == varBB, "cOut" + std::to_string(bbInd));
  }
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

CFDFC::CFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs, BBSet &bbs,
             unsigned numExec)
    : numExec(numExec) {
  for (Operation &op : funcOp.getOps()) {
    // Get operation's basic block
    unsigned srcBB;
    if (auto optBB = getLogicBB(&op); !optBB.has_value())
      continue;
    else
      srcBB = optBB.value();

    // The basic block the operation belongs to must be selected
    if (!bbs.contains(srcBB))
      continue;

    // Add the unit and valid outgoing channels to the CFDFC
    units.insert(&op);
    for (OpResult channel : op.getResults()) {
      assert(std::distance(channel.getUsers().begin(),
                           channel.getUsers().end()) == 1 &&
             "value must have unique user");

      // Get the value's unique user and its basic block
      Operation *user = *channel.getUsers().begin();
      unsigned dstBB;
      if (auto optBB = getLogicBB(user); !optBB.has_value())
        continue;
      else
        dstBB = optBB.value();

      // The channel is in the CFDFC if its producer/consumer belong to the same
      // basic block and the channel isn't a backedge
      if (srcBB == dstBB && !isBackedge(&op, user))
        channels.insert(channel);

      // The channel is in the CFDFC if its producer/consumer belong to a
      // selected arch between two basic blocks
      for (ArchBB *arch : archs) {
        if (arch->srcBB == srcBB && arch->dstBB == dstBB) {
          channels.insert(channel);
          break;
        }
      }
    }
  }
}

bool dynamatic::buffer::isBackedge(Operation *src, Operation *dst) {
  if (dst->isProperAncestor(src))
    return true;
  if (isa<BranchOp, ConditionalBranchOp>(src) &&
      isa<MergeLikeOpInterface>(dst)) {
    std::optional<unsigned> srcBB = getLogicBB(src);
    std::optional<unsigned> dstBB = getLogicBB(dst);
    assert(srcBB.has_value() && dstBB.has_value() &&
           "cannot determine backedge on operations that do not belong to any "
           "block");
    return srcBB.value() >= dstBB.value();
  }
  return false;
}

LogicalResult dynamatic::buffer::extractCFDFC(handshake::FuncOp funcOp,
                                              ArchSet &archs, BBSet &bbs,
                                              ArchSet &selectedArchs,
                                              BBSet &selectedBBs,
                                              unsigned &numExec) {
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
  unsigned maxTrans = initMILPVariables(model, archs, bbs, vars);

  // Define variable to hold the maximum number of transitions achievable
  GRBVar varMaxTrans =
      model.addVar(0, maxTrans, 0.0, GRB_INTEGER, "varMaxTrans");

  // Set up the MILP and solve it
  setObjective(model, vars, varMaxTrans);
  setEdgeConstraints(model, vars, varMaxTrans);
  setBBConstraints(model, vars);
  model.optimize();
  if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      varMaxTrans.get(GRB_DoubleAttr_X) < 0)
    return funcOp.emitError()
           << "Gurobi failed to find optimal solution to CFDFC extraction MILP";

  // Retrieve the maximum number of transitions identified by the MILP solution
  numExec = static_cast<unsigned>(varMaxTrans.get(GRB_DoubleAttr_X));
  if (numExec == 0)
    return success();

  // Fill in the set of selected archs and decrease their associated number of
  // transitions
  for (auto &[arch, var] : vars.archs) {
    if (archs.count(arch) > 0 && var.get(GRB_DoubleAttr_X) > 0) {
      archs.insert(arch);
      arch->numTrans -= numExec;
    }
  }

  // Fill in the set of selected basic blocks
  for (auto &[bb, var] : vars.bbs)
    if (bbs.count(bb) > 0 && var.get(GRB_DoubleAttr_X) > 0)
      selectedBBs.insert(bb);

  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}
