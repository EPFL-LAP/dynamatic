//===- FPL22Buffers.cpp - FPL'22 buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPL'22 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <optional>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl22;

void FPL22BuffersBase::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto [channel, channelVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool placeOpaque = channelVars.signalVars[SignalType::DATA].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;
    bool placeTransparent =
        channelVars.signalVars[SignalType::READY].bufPresent.get(
            GRB_DoubleAttr_X) > 0;

    ChannelBufProps &props = channelProps[channel];
    PlacementResult result;
    if (placeOpaque && placeTransparent) {
      // Place at least one opaque slot and satisfy the opaque slot requirement,
      // all other slots are transparent
      result.numOpaque = std::max(props.minOpaque, 1U);
      result.numTrans = numSlotsToPlace - result.numOpaque;
    } else if (placeOpaque) {
      // Satisfy the transparent slots requirement, all other slots are opaque
      result.numTrans = props.minTrans;
      result.numOpaque = numSlotsToPlace - props.minTrans;
    } else {
      // All slots transparent
      result.numTrans = numSlotsToPlace;
    }

    result.deductInternalBuffers(Channel(channel), timingDB);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void FPL22BuffersBase::addCustomChannelConstraints(Value channel) {
  // Get channel-specific buffering properties and channel's variables
  ChannelBufProps &props = channelProps[channel];
  ChannelVars &chVars = vars.channelVars[channel];

  // Force buffer presence if at least one slot is requested
  unsigned minSlots = props.minOpaque + props.minTrans;
  if (minSlots > 0) {
    model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");
    model.addConstr(chVars.bufNumSlots >= minSlots, "custom_minSlots");
  }

  // Set constraints based on minimum number of buffer slots
  GRBVar &bufData = chVars.signalVars[SignalType::DATA].bufPresent;
  GRBVar &bufReady = chVars.signalVars[SignalType::READY].bufPresent;
  if (props.minOpaque > 0) {
    // Force the MILP to place at least one opaque slot
    model.addConstr(bufData == 1, "custom_forceData");
    // If the MILP decides to also place a ready buffer, then we must reserve
    // an extra slot for it
    model.addConstr(chVars.bufNumSlots >= props.minOpaque + bufReady,
                    "custom_minData");
  }
  if (props.minTrans > 0) {
    // Force the MILP to place at least one transparent slot
    model.addConstr(bufReady == 1, "custom_forceReady");
    // If the MILP decides to also place a data buffer, then we must reserve
    // an extra slot for it
    model.addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                    "custom_minReady");
  }

  // Set constraints based on maximum number of buffer slots
  if (props.maxOpaque && props.maxTrans) {
    unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
    // Forbid buffer placement on the channel entirely when no slots are allowed
    if (maxSlots == 0)
      model.addConstr(chVars.bufPresent == 0, "custom_noBuffer");
    // Restrict the maximum number of slots allowed
    model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
  }

  // Forbid placement of some buffer type based on maximum number of allowed
  // slots on each signal
  if (props.maxOpaque && *props.maxOpaque == 0) {
    // Force the MILP to use transparent slots only
    model.addConstr(bufData == 0, "custom_noData");
  }
  if (props.maxTrans && *props.maxTrans == 0) {
    // Force the MILP to use opaque slots only
    model.addConstr(bufReady == 0, "custom_noReady");
  }
}

namespace {

/// Represents a specific pin of a unit's input or output port. Used internally
/// by the mixed-domain unit path constraints logic.
struct Pin {
  /// The channel connected to the unit's port.
  Value channel;
  /// The pin's timing domain, denoted by a signal type.
  SignalType type;

  /// Simple member-by-member constructor.
  Pin(Value channel, SignalType type) : channel(channel), type(type) {};
};

/// Represents a mixed domain constraint between an input pin and an output pin,
/// with a combinational delay between the two.
struct MixedDomainConstraint {
  /// The input pin.
  Pin input;
  /// The output pin.
  Pin output;
  /// Combinational delay (in ns) on the path.
  double internalDelay;

  /// Simple member-by-member constructor.
  MixedDomainConstraint(Pin input, Pin output, double internalDelay)
      : input(input), output(output), internalDelay(internalDelay) {};
};

} // namespace

void FPL22BuffersBase::addUnitMixedPathConstraints(Operation *unit,
                                                   ChannelFilter filter) {
  std::vector<MixedDomainConstraint> constraints;
  const TimingModel *unitModel = timingDB.getModel(unit);

  // Adds constraints between the input ports' valid and ready pins of a unit
  // with two operands.
  auto addJoinedOprdConstraints = [&]() -> void {
    double vr = unitModel ? unitModel->validToReady : 0.0;
    Value oprd0 = unit->getOperand(0), oprd1 = unit->getOperand(1);
    constraints.emplace_back(Pin(oprd0, SignalType::VALID),
                             Pin(oprd1, SignalType::READY), vr);
    constraints.emplace_back(Pin(oprd1, SignalType::VALID),
                             Pin(oprd0, SignalType::READY), vr);
  };

  // Adds constraints between the data pin of the provided input channel and all
  // valid/ready output pins.
  auto addDataToAllValidReadyConstraints = [&](Value inputChannel) -> void {
    Pin input(inputChannel, SignalType::DATA);
    double cv = unitModel ? unitModel->condToValid : 0.0;
    for (OpResult res : unit->getResults())
      constraints.emplace_back(input, Pin(res, SignalType::VALID), cv);
    double cr = unitModel ? unitModel->condToReady : 0.0;
    for (Value oprd : unit->getOperands())
      constraints.emplace_back(input, Pin(oprd, SignalType::READY), cr);
  };

  llvm::TypeSwitch<Operation *, void>(unit)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // There is a path between the data pin of the condition operand and
            // every valid/ready output pin
            addDataToAllValidReadyConstraints(condBrOp.getConditionOperand());

            // The two branch inputs are joined therefore there are cross
            // connections between the valid and ready pins
            addJoinedOprdConstraints();
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        // There is a path between the valid pin of the first operand and the
        // data pin of the index result
        Pin input(cmergeOp.getOperand(0), SignalType::VALID);
        Pin output(cmergeOp.getIndex(), SignalType::DATA);
        double vc = unitModel ? unitModel->validToCond : 0.0;
        constraints.emplace_back(input, output, vc);
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        // There is a path between every valid input pin and the data output
        // pin
        double vd = unitModel ? unitModel->validToData : 0.0;
        Pin output(mergeOp.getResult(), SignalType::DATA);
        for (Value oprd : mergeOp->getOperands())
          constraints.emplace_back(Pin(oprd, SignalType::VALID), output, vd);
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // There is a path between the data pin of the select operand and every
        // valid/ready output pin
        addDataToAllValidReadyConstraints(muxOp.getSelectOperand());

        // There is a path between every valid input pin and every data/ready
        // output pin
        double vd = unitModel ? unitModel->validToData : 0.0;
        double vr = unitModel ? unitModel->validToReady : 0.0;
        for (Value oprd : muxOp->getOperands()) {
          for (OpResult res : muxOp->getResults()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(res, SignalType::DATA), vd);
          }
          for (Value readyOprd : muxOp->getOperands()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(readyOprd, SignalType::READY), vr);
          }
        }
      })
      .Case<handshake::MCLoadOp, handshake::LSQLoadOp, handshake::MCStoreOp,
            handshake::LSQStoreOp, handshake::AddIOp, handshake::AddFOp,
            handshake::SubIOp, handshake::SubFOp, handshake::AndIOp,
            handshake::OrIOp, handshake::XOrIOp, handshake::MulIOp,
            handshake::MulFOp, handshake::DivUIOp, handshake::DivSIOp,
            handshake::DivFOp, handshake::ShRSIOp, handshake::ShLIOp,
            handshake::CmpIOp, handshake::CmpFOp>(
          [&](auto) { addJoinedOprdConstraints(); });

  StringRef unitName = getUniqueName(unit);
  unsigned idx = 0;
  for (MixedDomainConstraint &cons : constraints) {
    // The input/output channels must both be inside the CFDFC union
    if (!filter(cons.input.channel) || !filter(cons.output.channel))
      return;

    // Find variables for arrival time at input/output pin
    GRBVar &tPinIn = vars.channelVars[cons.input.channel]
                         .signalVars[cons.input.type]
                         .path.tOut;
    GRBVar &tPinOut = vars.channelVars[cons.output.channel]
                          .signalVars[cons.output.type]
                          .path.tIn;

    // Arrival time at unit's output pin must be greater than arrival time at
    // unit's input pin plus the unit's internal delay on the path
    std::string consName =
        "path_mixed_" + unitName.str() + "_" + std::to_string(idx++);
    model.addConstr(tPinIn + cons.internalDelay <= tPinOut, consName);
  }
}

CFDFCUnionBuffers::CFDFCUnionBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                     const TimingDatabase &timingDB,
                                     double targetPeriod, CFDFCUnion &cfUnion)
    : FPL22BuffersBase(env, funcInfo, timingDB, targetPeriod),
      cfUnion(cfUnion) {
  if (!unsatisfiable)
    setup();
}

CFDFCUnionBuffers::CFDFCUnionBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                     const TimingDatabase &timingDB,
                                     double targetPeriod, CFDFCUnion &cfUnion,
                                     Logger &logger, StringRef milpName)
    : FPL22BuffersBase(env, funcInfo, timingDB, targetPeriod, logger, milpName),
      cfUnion(cfUnion) {
  if (!unsatisfiable)
    setup();
}

void CFDFCUnionBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  // Create buffering groups. In this MILP we care for all signals, but the data
  // and valid paths are always cut together.
  OperationName oehbName = OperationName(handshake::OEHBOp::getOperationName(),
                                         funcInfo.funcOp->getContext());
  const TimingModel *oehbModel = timingDB.getModel(oehbName);
  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                oehbModel);

  OperationName tehbName = OperationName(handshake::TEHBOp::getOperationName(),
                                         funcInfo.funcOp->getContext());
  const TimingModel *tehbModel = timingDB.getModel(tehbName);
  BufferingGroup readyGroup({SignalType::READY}, tehbModel);

  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  // Group signals by matching buffer type for elasticty constraints
  SmallVector<ArrayRef<SignalType>> signalGroups;
  SmallVector<SignalType> opaqueGroup{SignalType::DATA, SignalType::VALID};
  SmallVector<SignalType> transparentGroup{SignalType::READY};
  signalGroups.push_back(opaqueGroup);
  signalGroups.push_back(transparentGroup);

  // Create channel variables and add custom, path, and elasticity contraints
  // over all channels in the CFDFC union
  for (Value channel : cfUnion.channels) {
    // Create variables and add custom channel constraints
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    // Add single-domain path constraints
    addChannelPathConstraints(channel, SignalType::DATA, oehbModel, {},
                              readyGroup);
    addChannelPathConstraints(channel, SignalType::VALID, oehbModel, {},
                              readyGroup);
    addChannelPathConstraints(channel, SignalType::READY, tehbModel,
                              dataValidGroup, {});

    // Elasticity constraints
    addChannelElasticityConstraints(channel, bufGroups);
  }

  // For unit constraints, filter out ports that are not part of the CFDFC union
  ChannelFilter channelFilter = [&](Value channel) -> bool {
    return cfUnion.channels.contains(channel);
  };

  // Add single-domain and mixed-domain path constraints as well as elasticity
  // constraints over all units in the CFDFC union
  for (Operation *unit : cfUnion.units) {
    addUnitPathConstraints(unit, SignalType::DATA, channelFilter);
    addUnitPathConstraints(unit, SignalType::VALID, channelFilter);
    addUnitPathConstraints(unit, SignalType::READY, channelFilter);
    addUnitMixedPathConstraints(unit, channelFilter);
    addUnitElasticityConstraints(unit, channelFilter);
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC in the
  // union which was marked for optimization
  for (CFDFC *cfdfc : cfUnion.cfdfcs) {
    assert(funcInfo.cfdfcs.contains(cfdfc) && "unknown CFDFC");
    if (!funcInfo.cfdfcs[cfdfc])
      continue;
    addCFDFCVars(*cfdfc);
    addChannelThroughputConstraints(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  std::vector<Value> allChannels;
  llvm::copy(cfUnion.channels, std::back_inserter(allChannels));
  addObjective(allChannels, cfUnion.cfdfcs);
  markReadyToOptimize();
}

OutOfCycleBuffers::OutOfCycleBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                     const TimingDatabase &timingDB,
                                     double targetPeriod)
    : FPL22BuffersBase(env, funcInfo, timingDB, targetPeriod) {
  if (!unsatisfiable)
    setup();
}

OutOfCycleBuffers::OutOfCycleBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                     const TimingDatabase &timingDB,
                                     double targetPeriod, Logger &logger,
                                     StringRef milpName)
    : FPL22BuffersBase(env, funcInfo, timingDB, targetPeriod, logger,
                       milpName) {
  if (!unsatisfiable)
    setup();
}

void OutOfCycleBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  // Create buffering groups. In this MILP we care for all signals, but the data
  // and valid paths are always cut together.
  OperationName oehbName = OperationName(handshake::OEHBOp::getOperationName(),
                                         funcInfo.funcOp->getContext());
  const TimingModel *oehbModel = timingDB.getModel(oehbName);
  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                oehbModel);

  OperationName tehbName = OperationName(handshake::TEHBOp::getOperationName(),
                                         funcInfo.funcOp->getContext());
  const TimingModel *tehbModel = timingDB.getModel(tehbName);
  BufferingGroup readyGroup({SignalType::READY}, tehbModel);

  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // Create a CFDFC union from all CFDFCs in the function so that we can make
  // very fast queries of the kind: is this channel part of any CFDFC?
  SmallVector<CFDFC *> allCFDFCs;
  for (auto [cfdfc, _] : funcInfo.cfdfcs)
    allCFDFCs.push_back(cfdfc);
  CFDFCUnion cfUnion(allCFDFCs);

  // Filter out channels part of any CFDFC or adjacent to a memory interface
  ChannelFilter channelFilter = [&](Value channel) -> bool {
    if (cfUnion.channels.contains(channel))
      return false;

    Operation *defOp = channel.getDefiningOp();
    return !isa_and_present<handshake::MemoryOpInterface>(defOp) &&
           !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
  };

  // Create variables and  add path and elasticity constraints for all channels
  // covered by the MILP. These are the channels that are not part of any CFDFC
  // identified in the Handshake function under consideration
  for (auto [channel, _] : channelProps) {
    if (!channelFilter(channel))
      continue;

    // Create channel variables and add custom constraints for the channel
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    // Add single-domain path constraints
    addChannelPathConstraints(channel, SignalType::DATA, oehbModel, {},
                              readyGroup);
    addChannelPathConstraints(channel, SignalType::VALID, oehbModel, {},
                              readyGroup);
    addChannelPathConstraints(channel, SignalType::READY, tehbModel,
                              dataValidGroup, {});

    // Add elasticity constraints
    addChannelElasticityConstraints(channel, bufGroups);

    // Add negative terms to MILP objective, penalizing placement of buffers
    ChannelVars &channelVars = vars.channelVars[channel];
    GRBVar &dataBuf = channelVars.signalVars[SignalType::DATA].bufPresent;
    GRBVar &readyBuf = channelVars.signalVars[SignalType::READY].bufPresent;
    objective -= dataBuf;
    objective -= readyBuf;
    objective -= 0.1 * channelVars.bufNumSlots;
  }

  // Add single-domain and mixed-domain path constraints as well as elasticity
  // constraints over all units that are not part of any CFDFC
  for (Operation &unit : funcInfo.funcOp.getOps()) {
    if (cfUnion.units.contains(&unit))
      continue;

    addUnitPathConstraints(&unit, SignalType::DATA, channelFilter);
    addUnitPathConstraints(&unit, SignalType::VALID, channelFilter);
    addUnitPathConstraints(&unit, SignalType::READY, channelFilter);
    addUnitMixedPathConstraints(&unit, channelFilter);
    addUnitElasticityConstraints(&unit, channelFilter);
  }

  // Set MILP objective and mark it ready to be optimized
  model.setObjective(objective, GRB_MAXIMIZE);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
