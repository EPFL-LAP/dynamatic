//===- MixedBuffers.cpp - FPGA'20/FPL'22 buffer placement -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements mixed buffer placement, merging strategies from FPGA20 and FPL22
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/MixedBuffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode, and
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "mixed-buffers"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mixedbuffers;

//===----------------------------------------------------------------------===//
// Construction and scope
//===----------------------------------------------------------------------===//

MixedBuffers::MixedBuffers(
      CPSolver::SolverKind solverKind, int timeout, FuncInfo &funcInfo,
      const TimingDatabase &timingDB, double targetPeriod,
      SignalModel signalModel, MILPMode mode, CFDFCUnion *cfUnion,
      StringRef writeTo, bool ftd
    ): BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          getAlgorithm(signalModel), writeTo, ftd),
      signalModel(signalModel), mode(mode), cfUnion(cfUnion),
      bufGroups(getBufferingGroups(signalModel)) 
    {
  // Sanity check :)
  assert((mode == MILPMode::CFDFC_UNION) == (cfUnion != nullptr) &&
         "a CFDFC union must be provided iff the scope is a CFDFC union");

  for (const BufferingGroup &group : bufGroups)
    llvm::append_range(signalTypes, group.signalTypes);

  switch (mode) {
    case MILPMode::ONE_MILP:
      channelFilter = nullFilter;
      break;
    case MILPMode::CFDFC_UNION:
      channelFilter = [this](Value channel) -> bool {
        return this->cfUnion->channels.contains(channel);
      };
      break;
    case MILPMode::OUT_OF_CYCLE: {
      SmallVector<CFDFC *> cfdfcs;
      for (auto &[cfdfc, _] : funcInfo.cfdfcs)
        cfdfcs.push_back(cfdfc);
      allCFDFCs.emplace(cfdfcs);
      channelFilter = [this](Value channel) -> bool {
        return isChannelInScope(channel);
      };
      break;
    }
  }

  if (!unsatisfiable)
    setup();
}

SmallVector<BufferPlacementMILP::BufferingGroup>
MixedBuffers::getBufferingGroups(SignalModel model) {
  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  // Groups are ordered as buffers are placed on a channel: the data/valid
  // breaking slot comes before the ready breaking slot
  SmallVector<BufferingGroup> groups;
  switch (model) {
  case SignalModel::DATA_ONLY:
    groups.emplace_back(ArrayRef<SignalType>{SignalType::DATA}, bufModel);
    break;
  case SignalModel::DATA_VALID_READY:
    groups.emplace_back(
        ArrayRef<SignalType>{SignalType::DATA, SignalType::VALID}, bufModel);
    groups.emplace_back(ArrayRef<SignalType>{SignalType::READY}, bufModel);
    break;
  }
  return groups;
}

// // FIXME: [amos] This, at the moment, is very stupid; we should not be passing an 
// //        algorithm as argument AT ALL imo.
BufferPlacementMILP::Algorithm
MixedBuffers::getAlgorithm(SignalModel model) {
  switch (model) {
  case SignalModel::DATA_ONLY:
    return Algorithm::FPGA20;
  case SignalModel::DATA_VALID_READY:
    return Algorithm::FPL22;
  }
  llvm_unreachable("unknown signal model");
}

bool MixedBuffers::isMemoryAdjacent(Value channel) {
  return isa_and_present<handshake::MemoryOpInterface>(
             channel.getDefiningOp()) ||
         isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
}

bool MixedBuffers::isChannelInScope(Value channel) const {
  switch (mode) {
    case MILPMode::ONE_MILP:
      return true;
    case MILPMode::CFDFC_UNION:
      return cfUnion->channels.contains(channel);
    case MILPMode::OUT_OF_CYCLE:
      return !allCFDFCs->channels.contains(channel) && !isMemoryAdjacent(channel);
  }
  llvm_unreachable("unknown MILP scope");
}

bool MixedBuffers::hasTimingConstraints(Value channel) const {
  // In the one MILP mode, memory-adjacent channels get variables and
  // custom constraints, but no path or elasticity constraints. The CFDFC
  // union MILP does not treat them specially.
  // TODO: [amos] decide whether these three behaviors should be made consistent
  // (this would change placement results).
  if (mode == MILPMode::ONE_MILP)
    return !isMemoryAdjacent(channel);
  return true;
}

bool MixedBuffers::isUnitInScope(Operation *unit) const {
  // TODO: [amos] Same as above (hasTimingConstraints)
  switch (mode) {
    case MILPMode::ONE_MILP:
      return true;
    case MILPMode::CFDFC_UNION:
      return cfUnion->units.contains(unit);
    case MILPMode::OUT_OF_CYCLE:
      return !allCFDFCs->units.contains(unit);
  }
  llvm_unreachable("unknown MILP scope");
}

// //===----------------------------------------------------------------------===//
// // Result extraction
// //===----------------------------------------------------------------------===//

void MixedBuffers::extractResult(BufferPlacement &placement) {
  for (auto &[channel, chVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution
    unsigned numSlotsToPlace =
        static_cast<unsigned>(model->getValue(chVars.bufNumSlots) + 0.5);
    bool forceBreakDV =
        model->getValue(chVars.signalVars[SignalType::DATA].bufPresent) > 0;

    PlacementResult result;
    if (signalModel == SignalModel::DATA_ONLY) {
      // 1. If breaking DV:
      // Map to ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.
      //
      // 2. If breaking none:
      // Map to numslot * FIFO_BREAK_NONE.
      if (numSlotsToPlace >= 1) {
        if (forceBreakDV) {
          result.numOneSlotDV = 1;
          result.numFifoNone = numSlotsToPlace - 1;
        } else {
          result.numFifoNone = numSlotsToPlace;
        }
      }

      // See docs/Specs/Buffering.md
      // The MILP only breaks data and valid paths. We insert TEHBs after
      // Merge-like and Init operations to break ready paths. We only break the
      // ready path if the channel is on cycle. Note that this slot comes on
      // top of the slots decided by the MILP.
      Operation *srcOp = channel.getDefiningOp();
      bool needsReadyPathBuffer =
          isa_and_nonnull<handshake::InitOp>(srcOp) ||
          (isa_and_nonnull<handshake::MuxOp, handshake::MergeOp>(srcOp) &&
           srcOp->getNumOperands() > 1);
      if (needsReadyPathBuffer && isChannelOnCycle(channel))
        result.numOneSlotR = 1;
    } else {
      if (numSlotsToPlace == 0)
        continue;
      bool forceBreakR =
          model->getValue(chVars.signalVars[SignalType::READY].bufPresent) > 0;

      // 1. If breaking DV & R:
      // When numslot = 1, map to ONE_SLOT_BREAK_DVR;
      // When numslot > 1, map to ONE_SLOT_BREAK_DV + (numslot - 2) *
      //                            FIFO_BREAK_NONE + ONE_SLOT_BREAK_R.
      //
      // 2. If only breaking DV:
      // Map to ONE_SLOT_BREAK_DV + (numslot - 1) * FIFO_BREAK_NONE.
      //
      // 3. If only breaking R:
      // Map to ONE_SLOT_BREAK_R + (numslot - 1) * FIFO_BREAK_NONE.
      //
      // 4. If breaking none:
      // Map to numslot * FIFO_BREAK_NONE.
      if (forceBreakDV && forceBreakR) {
        if (numSlotsToPlace == 1) {
          result.numOneSlotDVR = 1;
        } else {
          result.numOneSlotDV = 1;
          result.numFifoNone = numSlotsToPlace - 2;
          result.numOneSlotR = 1;
        }
      } else if (forceBreakDV) {
        result.numOneSlotDV = 1;
        result.numFifoNone = numSlotsToPlace - 1;
      } else if (forceBreakR) {
        result.numOneSlotR = 1;
        result.numFifoNone = numSlotsToPlace - 1;
      } else {
        result.numFifoNone = numSlotsToPlace;
      }
    }

    placement[channel] = result;
  }

  LLVM_DEBUG(logResults(placement));

  // Store throughputs and occupancies in the CFDFCs themselves. The
  // `handshake.tp` attribute is set once all MILPs have been solved, see
  // `setCFDFCThroughputAttr`
  populateCFDFCThroughputAndOccupancy();
}

void mixedbuffers::setCFDFCThroughputAttr(
    FuncInfo &funcInfo) {
  // Keys are indices in `funcInfo.cfdfcs`, matching the `handshake.cfdfc`
  // attribute set by the buffer placement pass
  llvm::MapVector<size_t, double> cfdfcTPResult;
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(funcInfo.cfdfcs)) {
    auto &[cfdfc, optimize] = cfdfcAndOpt;
    if (optimize)
      cfdfcTPResult[idx] = cfdfc->throughput;
  }
  auto cfdfcTPMap = handshake::CFDFCThroughputAttr::get(
      funcInfo.funcOp.getContext(), cfdfcTPResult);
  setDialectAttr(funcInfo.funcOp, cfdfcTPMap);
}

// //===----------------------------------------------------------------------===//
// // Custom channel constraints
// //===----------------------------------------------------------------------===//

void MixedBuffers::addCustomChannelConstraints(Value channel) {
  // "Transparent slot" meant "slot not breaking data/valid" for FPGA20 
  // but "slot breaking ready" for FPL22, 
  // so the FPGA20 encoding is NOT the FPL22 one with no ready buffer
  switch (signalModel) {
  case SignalModel::DATA_ONLY:
    addDataOnlyCustomChannelConstraints(channel);
    return;
  case SignalModel::DATA_VALID_READY:
    addDataValidReadyCustomChannelConstraints(channel);
    return;
  }
}

void MixedBuffers::addDataOnlyCustomChannelConstraints(Value channel) {
  ChannelVars &chVars = vars.channelVars[channel];
  handshake::ChannelBufProps &props = channelProps[channel];
  CPVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;

  if (props.minOpaque > 0) {
    // Force the MILP to use opaque slots
    model->addConstr(dataBuf == 1, "custom_forceOpaque");
    if (props.minTrans > 0) {
      // If the properties ask for both opaque and transparent slots, let
      // opaque slots take over. Transparents slots will be placed "manually"
      // from the total number of slots indicated by the MILP's result
      unsigned minTotalSlots = props.minOpaque + props.minTrans;
      model->addConstr(chVars.bufNumSlots >= minTotalSlots,
                       "custom_minOpaqueAndTrans");
    } else {
      // Force the MILP to place a minimum number of opaque slots
      model->addConstr(chVars.bufNumSlots >= props.minOpaque,
                       "custom_minOpaque");
    }
  } else if (props.minTrans > 0) {
    // Force the MILP to place a minimum number of transparent slots
    model->addConstr(chVars.bufNumSlots >= props.minTrans + dataBuf,
                     "custom_minTrans");
  } else if (props.minSlots > 0) {
    // Force the MILP to place a minimum number of slots
    model->addConstr(chVars.bufNumSlots >= props.minSlots, "custom_minSlots");
  }
  if (props.minOpaque + props.minTrans + props.minSlots > 0)
    model->addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  // Set a maximum number of slots to be placed
  if (props.maxOpaque.has_value()) {
    if (*props.maxOpaque == 0) {
      // Force the MILP to use transparent slots
      model->addConstr(dataBuf == 0, "custom_forceTransparent");
    }
    if (props.maxTrans.has_value()) {
      // Force the MILP to use a maximum number of slots
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        model->addConstr(chVars.bufPresent == 0, "custom_noBuffers");
        model->addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
      } else {
        model->addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }
}

void MixedBuffers::addDataValidReadyCustomChannelConstraints(
    Value channel) {
  // Get channel-specific buffering properties and channel's variables
  handshake::ChannelBufProps &props = channelProps[channel];
  ChannelVars &chVars = vars.channelVars[channel];

  // Force buffer presence if at least one slot is requested
  unsigned minSlots =
      std::max(props.minOpaque + props.minTrans, props.minSlots);
  if (minSlots > 0) {
    model->addConstr(chVars.bufPresent == 1, "custom_forceBuffers");
    model->addConstr(chVars.bufNumSlots >= minSlots, "custom_minSlots");
  }

  // Set constraints based on minimum number of buffer slots
  CPVar &bufData = chVars.signalVars[SignalType::DATA].bufPresent;
  CPVar &bufReady = chVars.signalVars[SignalType::READY].bufPresent;
  if (props.minOpaque > 0) {
    // Force the MILP to place at least one opaque slot
    model->addConstr(bufData == 1, "custom_forceData");
    // If the MILP decides to also place a ready buffer, then we must reserve
    // an extra slot for it
    model->addConstr(chVars.bufNumSlots >= props.minOpaque + bufReady,
                     "custom_minData");
  }
  if (props.minTrans > 0) {
    // If the MILP decides to also place a data buffer, then we must reserve
    // an extra slot for it
    model->addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                     "custom_minReady");
  }

  // Set constraints based on maximum number of buffer slots
  if (props.maxOpaque && props.maxTrans) {
    unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
    if (maxSlots == 0) {
      // Forbid buffer placement on the channel entirely when no slots are
      // allowed
      model->addConstr(chVars.bufPresent == 0, "custom_noBuffer");
      model->addConstr(chVars.bufNumSlots == 0, "custom_maxSlots");
    } else {
      // Restrict the maximum number of slots allowed. If both types are allowed
      // but the MILP decides to only place one type, then the maximum allowed
      // number is the maximum number of slots we can place for that type
      model->addConstr(chVars.bufNumSlots <=
                           maxSlots - *props.maxOpaque * (1 - bufData) -
                               *props.maxTrans * (1 - bufReady),
                       "custom_maxSlots");
    }
  }

  // Forbid placement of some buffer type based on maximum number of allowed
  // slots on each signal
  if (props.maxOpaque && *props.maxOpaque == 0) {
    // Force the MILP to use transparent slots only
    model->addConstr(bufData == 0, "custom_noData");
  }
  if (props.maxTrans && *props.maxTrans == 0) {
    // Force the MILP to use opaque slots only
    model->addConstr(bufReady == 0, "custom_noReady");
  }
}

// //===----------------------------------------------------------------------===//
// // Mixed-domain path constraints (data/valid/ready model only)
// //===----------------------------------------------------------------------===//

namespace {

/// Represents a specific pin of a unit's input or output port. Used internally
/// by the mixed-domain unit path constraints logic.
struct Pin {
  /// The channel connected to the unit's port.
  Value channel;
  /// The pin's timing domain, denoted by a signal type.
  SignalType signalType;

  /// Simple member-by-member constructor.
  Pin(Value channel, SignalType signalType)
      : channel(channel), signalType(signalType) {};
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

void MixedBuffers::addUnitMixedPathConstraints(Operation *unit,
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
      .Case<handshake::LoadOp, handshake::StoreOp, handshake::AddIOp,
            handshake::AddFOp, handshake::SubIOp, handshake::SubFOp,
            handshake::AndIOp, handshake::OrIOp, handshake::XOrIOp,
            handshake::MulIOp, handshake::MulFOp, handshake::DivUIOp,
            handshake::DivSIOp, handshake::DivFOp, handshake::ShRSIOp,
            handshake::ShLIOp, handshake::CmpIOp, handshake::CmpFOp>(
          [&](auto) { addJoinedOprdConstraints(); });

  StringRef unitName = getUniqueName(unit);
  unsigned idx = 0;
  for (MixedDomainConstraint &cons : constraints) {
    // The input/output channels must both be inside the MILP's scope
    if (!filter(cons.input.channel) || !filter(cons.output.channel))
      continue;

    // Find variables for arrival time at input/output pin
    CPVar &tPinIn = vars.channelVars[cons.input.channel]
                        .signalVars[cons.input.signalType]
                        .path.tOut;
    CPVar &tPinOut = vars.channelVars[cons.output.channel]
                         .signalVars[cons.output.signalType]
                         .path.tIn;

    // Arrival time at unit's output pin must be greater than arrival time at
    // unit's input pin plus the unit's internal delay on the path
    std::string consName =
        "path_mixed_" + unitName.str() + "_" + std::to_string(idx++);
    model->addConstr(tPinIn + cons.internalDelay <= tPinOut, consName);
  }
}

// //===----------------------------------------------------------------------===//
// // MILP setup
// //===----------------------------------------------------------------------===//

void MixedBuffers::addChannelConstraints(Value channel) {
  addChannelVars(channel, signalTypes);
  addCustomChannelConstraints(channel);
  if (!hasTimingConstraints(channel))
    return;

  // Path constraints for each modelled signal. A signal's path may go through
  // the buffers of the groups placed before and after its own group on the
  // channel
  ArrayRef<BufferingGroup> groups(bufGroups);
  for (auto [idx, group] : llvm::enumerate(groups)) {
    // These will be empty for ONE_MILP since groups.size() = 1,
    // and will correctly set before = {} / after = readyGroup for DV and 
    //                        before = dataValidGroup / after = {}
    // for CDFDC_UNION
    ArrayRef<BufferingGroup> before = groups.take_front(idx);
    ArrayRef<BufferingGroup> after = groups.drop_front(idx + 1);
    for (SignalType signalType : group.signalTypes) {
      addChannelTimingConstraints(channel, signalType, group.bufModel, before,
                                  after);
    }
  }

  // Elasticity constraints
  addBufferPresenceConstraints(channel);
  addBufferingGroupConstraints(channel, bufGroups);
}

void MixedBuffers::addUnitConstraints(Operation *unit) {
  if (isa<handshake::SpeculatorOp, handshake::SpecSaveCommitOp>(unit)) {
    addSpecUnitConstraints(unit, signalTypes, channelFilter);
    return;
  }
  for (SignalType signalType : signalTypes)
    addUnitTimingConstraints(unit, signalType, channelFilter);
  if (signalModel == SignalModel::DATA_VALID_READY)
    addUnitMixedPathConstraints(unit, channelFilter);
}

SmallVector<CFDFC *> MixedBuffers::getCFDFCsToOptimize() const {
  SmallVector<CFDFC *> cfdfcs;
  switch (mode) {
    case MILPMode::ONE_MILP:
      for (auto &[cfdfc, optimize] : funcInfo.cfdfcs) {
        if (optimize)
          cfdfcs.push_back(cfdfc);
      }
      break;
    case MILPMode::CFDFC_UNION:
      for (CFDFC *cfdfc : cfUnion->cfdfcs) {
        assert(funcInfo.cfdfcs.contains(cfdfc) && "unknown CFDFC");
        if (funcInfo.cfdfcs.lookup(cfdfc))
          cfdfcs.push_back(cfdfc);
      }
      break;
    case MILPMode::OUT_OF_CYCLE:
      // Channels outside of all CFDFCs have no throughput to optimize
      break;
  }
  return cfdfcs;
}

void MixedBuffers::addMinBuffersObjective(ArrayRef<Value> channels) {
  // Penalize the placement of buffers of each group and of each slot
  LinExpr objective;
  for (Value channel : channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    for (const BufferingGroup &group : bufGroups)
      objective -= chVars.signalVars[group.getRefSignal()].bufPresent;
    objective -= 0.1 * chVars.bufNumSlots;
  }
  model->setMaximizeObjective(objective);
}

void MixedBuffers::setup() {
  // Collect the channels covered by the MILP before creating any variable
  SmallVector<Value> channels;
  if (mode == MILPMode::CFDFC_UNION) {
    llvm::append_range(channels, cfUnion->channels);
  } else {
    for (auto &[channel, _] : channelProps) {
      if (isChannelInScope(channel))
        channels.push_back(channel);
    }
  }

  // Create channel variables and add custom, path, and elasticity constraints
  for (Value channel : channels)
    addChannelConstraints(channel);

  // Add path and elasticity constraints over all units in scope
  if (mode == MILPMode::CFDFC_UNION) {
    for (Operation *unit : cfUnion->units)
      addUnitConstraints(unit);
  } else {
    for (Operation &unit : funcInfo.funcOp.getOps()) {
      if (isUnitInScope(&unit))
        addUnitConstraints(&unit);
    }
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC in
  // scope that was marked to be optimized
  SmallVector<CFDFC *> cfdfcs = getCFDFCsToOptimize();
  for (CFDFC *cfdfc : cfdfcs) {
    addCFDFCVars(*cfdfc);
    addSteadyStateReachabilityConstraints(*cfdfc);
    addChannelThroughputConstraintsForBinaryLatencyChannel(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  if (mode == MILPMode::OUT_OF_CYCLE)
    addMinBuffersObjective(channels);
  else
  addMaxThroughputObjective(channels, cfdfcs);
  markReadyToOptimize();
}