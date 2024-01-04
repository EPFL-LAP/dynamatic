//===- FPGA20Buffers.cpp - FPGA'20 buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPGA'20 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga20;

FPGA20Buffers::FPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, bool legacyPlacement)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod),
      legacyPlacement(legacyPlacement) {
  if (!unsatisfiable)
    setup();
}

FPGA20Buffers::FPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, bool legacyPlacement,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName),
      legacyPlacement(legacyPlacement) {
  if (!unsatisfiable)
    setup();
}

void FPGA20Buffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto &[channel, channelVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool placeOpaque = channelVars.signalVars[SignalType::DATA].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;

    ChannelBufProps &props = channelProps[channel];

    PlacementResult result;
    if (placeOpaque) {
      if (legacyPlacement) {
        // Satisfy the transparent slots requirement, all other slots are opaque
        result.numTrans = props.minTrans;
        result.numOpaque = numSlotsToPlace - props.minTrans;
      } else {
        // We want as many slots as possible to be transparent and at least one
        // opaque slot, while satisfying all buffering constraints
        unsigned actualMinOpaque = std::max(1U, props.minOpaque);
        if (props.maxTrans.has_value() &&
            (props.maxTrans.value() < numSlotsToPlace - actualMinOpaque)) {
          result.numTrans = props.maxTrans.value();
          result.numOpaque = numSlotsToPlace - result.numTrans;
        } else {
          result.numOpaque = actualMinOpaque;
          result.numTrans = numSlotsToPlace - result.numOpaque;
        }
      }
    } else {
      // All slots should be transparent
      result.numTrans = numSlotsToPlace;
    }

    deductInternalBuffers(channel, result);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void FPGA20Buffers::addCustomChannelConstraints(Value channel) {
  ChannelVars &chVars = vars.channelVars[channel];
  ChannelBufProps &props = channelProps[channel];
  GRBVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;

  if (props.minOpaque > 0) {
    // Force the MILP to use opaque slots
    model.addConstr(dataBuf == 1, "custom_forceOpaque");
    if (props.minTrans > 0) {
      // If the properties ask for both opaque and transparent slots, let
      // opaque slots take over. Transparents slots will be placed "manually"
      // from the total number of slots indicated by the MILP's result
      unsigned minTotalSlots = props.minOpaque + props.minTrans;
      model.addConstr(chVars.bufNumSlots >= minTotalSlots,
                      "custom_minOpaqueAndTrans");
    } else {
      // Force the MILP to place a minimum number of opaque slots
      model.addConstr(chVars.bufNumSlots >= props.minOpaque,
                      "custom_minOpaque");
    }
  } else if (props.minTrans > 0) {
    // Force the MILP to place a minimum number of transparent slots
    model.addConstr(chVars.bufNumSlots >= props.minTrans + dataBuf,
                    "custom_minTrans");
  }
  if (props.minOpaque + props.minTrans > 0)
    model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  // Set a maximum number of slots to be placed
  if (props.maxOpaque.has_value()) {
    if (*props.maxOpaque == 0) {
      // Force the MILP to use transparent slots
      model.addConstr(dataBuf == 0, "custom_forceTransparent");
    }
    if (props.maxTrans.has_value()) {
      // Force the MILP to use a maximum number of slots
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        model.addConstr(chVars.bufPresent == 0, "custom_noBuffers");
        model.addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
      } else {
        model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }
}

void FPGA20Buffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 1> signals;
  signals.push_back(SignalType::DATA);

  // Create buffering groups. In this MILP we only care for the data signal
  SmallVector<BufferingGroup> bufGroups;
  OperationName bufName = OperationName(handshake::BufferOp::getOperationName(),
                                        funcInfo.funcOp->getContext());
  const TimingModel *dataBufModel = timingDB.getModel(bufName);
  bufGroups.emplace_back(ArrayRef<SignalType>{SignalType::DATA}, dataBufModel);

  // Create channel variables and constraints
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    // Add path and elasticity constraints over all channels in the function
    // that are not adjacent to a memory interface
    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addChannelPathConstraints(channel, SignalType::DATA, dataBufModel);
      addChannelElasticityConstraints(channel, bufGroups);
    }
  }

  // Add path and elasticity constraints over all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitPathConstraints(&op, SignalType::DATA);
    addUnitElasticityConstraints(&op);
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC that
  // was marked to be optimized
  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    addChannelThroughputConstraints(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  addObjective(allChannels, cfdfcs);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
