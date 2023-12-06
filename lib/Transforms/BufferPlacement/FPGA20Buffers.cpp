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
#include "dynamatic/Support/LogicBB.h"
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
                             Logger &logger)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          "placement"),
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

void FPGA20Buffers::addChannelPathConstraints(Value channel) {
  // Manually get the timing model for buffers
  const TimingModel *bufModel = timingDB.getModel(OperationName(
      handshake::BufferOp::getOperationName(), funcInfo.funcOp->getContext()));
  double bigCst = targetPeriod * 10;

  // Get delays for a buffer that would be placed on this channel
  double inBufDelay = 0.0, outBufDelay = 0.0, dataBufDelay = 0.0;
  if (bufModel) {
    Type channelType = channel.getType();
    unsigned bitwidth = 0;
    if (isa<IntegerType, FloatType>(channelType))
      bitwidth = channelType.getIntOrFloatBitWidth();
    /// TODO: It's bad to discard these results, needs a safer way of querying
    /// for these delays
    (void)bufModel->inputModel.dataDelay.getCeilMetric(bitwidth, inBufDelay);
    (void)bufModel->outputModel.dataDelay.getCeilMetric(bitwidth, outBufDelay);
    (void)bufModel->dataDelay.getCeilMetric(bitwidth, dataBufDelay);
    // Add the input and output port delays to the total buffer delay
    dataBufDelay += inBufDelay + outBufDelay;
  }

  ChannelVars &chVars = vars.channelVars[channel];
  ChannelBufProps &props = channelProps[channel];
  ChannelSignalVars &dataVars = chVars.signalVars[SignalType::DATA];
  GRBVar &t1 = dataVars.path.tIn;
  GRBVar &t2 = dataVars.path.tOut;
  GRBVar &present = chVars.bufPresent;
  GRBVar &dataBuf = dataVars.bufPresent;

  // Arrival time at channel's input must be lower than target clock period
  double inToBufDelay = props.inDelay + inBufDelay;
  model.addConstr(t1 + present * inToBufDelay <= targetPeriod,
                  "path_channelInPeriod");

  // Arrival time at channel's output must be lower than target clock period
  model.addConstr(t2 <= targetPeriod, "path_channelOutPeriod");

  // If there is an opaque buffer, arrival time at channel's output must be
  // greater than the delay between the buffer's internal register and the
  // post-buffer channel delay
  double bufToOutDelay = outBufDelay + props.outDelay;
  if (bufToOutDelay > 0)
    model.addConstr(dataBuf * bufToOutDelay <= t2, "path_opaqueChannel");

  // If there is a transparent buffer, arrival time at channel's output must
  // be greater than at channel's input (+ whole channel and buffer delay)
  double inToOutDelay = props.inDelay + dataBufDelay + props.outDelay;
  model.addConstr(t1 + inToOutDelay - bigCst * (dataBuf - present + 1) <= t2,
                  "path_transparentChannel");

  // If there are no buffers, arrival time at channel's output must be greater
  // than at channel's input (+ channel delay)
  model.addConstr(t1 + props.delay - bigCst * present <= t2,
                  "path_unbufferedChannel");
}

void FPGA20Buffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 1> signals;
  signals.push_back(SignalType::DATA);

  // Group signals by matching buffer type for elasticty constraints
  SmallVector<ArrayRef<SignalType>> signalGroups;
  SmallVector<SignalType> dataGroup{SignalType::DATA};
  signalGroups.push_back(dataGroup);

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
      addChannelPathConstraints(channel);
      addChannelElasticityConstraints(channel, signalGroups);
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
