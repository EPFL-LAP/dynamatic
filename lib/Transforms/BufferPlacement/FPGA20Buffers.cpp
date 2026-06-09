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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "fpga20-buffers"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga20;

FPGA20Buffers::FPGA20Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod, StringRef writeTo)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          writeTo) {
  if (!unsatisfiable)
    setup();
}

void FPGA20Buffers::addSpecUnitDataPathConstraints(Operation *unit) {
  const SpecTimingModel *specModel = timingDB.getSpecModel(unit);
  if (!specModel) {
    unit->emitError() << "addSpecUnitDataPathConstraints: no spec timing model "
                      << "loaded for op '" << unit->getName().getStringRef()
                      << "'";
    return;
  }

  auto namedIO = dyn_cast<handshake::NamedIOInterface>(unit);
  if (!namedIO) {
    unit->emitError() << "addSpecUnitDataPathConstraints: op does not "
                      << "implement NamedIOInterface";
    return;
  }

  llvm::StringMap<Value> portToValue;
  for (unsigned i = 0, e = unit->getNumOperands(); i < e; ++i)
    portToValue[namedIO.getOperandName(i)] = unit->getOperand(i);
  for (unsigned i = 0, e = unit->getNumResults(); i < e; ++i)
    portToValue[namedIO.getResultName(i)] = unit->getResult(i);

  Value bitwidthChannel =
      llvm::TypeSwitch<Operation *, Value>(unit)
          .Case<handshake::SpeculatorOp>(
              [](handshake::SpeculatorOp op) { return op.getDataIn(); })
          .Case<handshake::SpecSaveCommitOp>(
              [](handshake::SpecSaveCommitOp op) { return op.getDataIn(); })
          .Default([&](Operation *op) {
            op->emitError() << "addSpecUnitDataPathConstraints called on op "
                            << "type with no bitwidth accessor registered";
            return Value();
          });
  if (!bitwidthChannel)
    return;

  llvm::StringMap<int64_t> currentParams;
  if (auto chTy = dyn_cast<handshake::ChannelType>(bitwidthChannel.getType()))
    currentParams["BITWIDTH"] = chTy.getDataBitWidth();
  else
    currentParams["BITWIDTH"] = 0;

  auto pickClosest =
      [&](const std::vector<SpecTimingEdge::Sample> &samples) -> double {
    assert(!samples.empty() && "spec timing edge has no samples");
    const SpecTimingEdge::Sample *best = &samples.front();
    int64_t bestDist = std::numeric_limits<int64_t>::max();
    for (const auto &s : samples) {
      int64_t dist = 0;
      for (const auto &targetKV : currentParams) {
        auto it = s.params.find(targetKV.first());
        if (it == s.params.end())
          continue;
        int64_t d = it->second - targetKV.second;
        dist += d * d;
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = &s;
      }
    }
    return best->delay;
  };

  StringRef unitName = getUniqueName(unit);
  unsigned idx = 0;
  for (const SpecTimingEdge &edge : specModel->pin2pin) {
    if (edge.from.signal != "data" || edge.to.signal != "data")
      continue;
    auto fromIt = portToValue.find(edge.from.port);
    auto toIt = portToValue.find(edge.to.port);
    if (fromIt == portToValue.end()) {
      unit->emitError() << "spec timing edge references unknown port '"
                        << edge.from.port << "'";
      return;
    }
    if (toIt == portToValue.end()) {
      unit->emitError() << "spec timing edge references unknown port '"
                        << edge.to.port << "'";
      return;
    }
    Value fromVal = fromIt->second;
    Value toVal = toIt->second;
    if (vars.channelVars.find(fromVal) == vars.channelVars.end() ||
        vars.channelVars.find(toVal) == vars.channelVars.end())
      continue;
    if (edge.samples.empty()) {
      unit->emitError() << "spec timing edge " << edge.from.port << "."
                        << edge.from.signal << " -> " << edge.to.port << "."
                        << edge.to.signal << " has no samples";
      return;
    }
    double delay = pickClosest(edge.samples);
    CPVar &tFrom =
        vars.channelVars[fromVal].signalVars[SignalType::DATA].path.tOut;
    CPVar &tTo = vars.channelVars[toVal].signalVars[SignalType::DATA].path.tIn;
    std::string consName =
        "path_spec_p2p_" + unitName.str() + "_" + std::to_string(idx++);
    model->addConstr(tFrom + delay <= tTo, consName);
  }

  auto unitSideDataVar = [&](Value v) -> CPVar & {
    bool unitIsProducer = (v.getDefiningOp() == unit);
    if (unitIsProducer)
      return vars.channelVars[v].signalVars[SignalType::DATA].path.tIn;
    return vars.channelVars[v].signalVars[SignalType::DATA].path.tOut;
  };

  for (const SpecTimingPortDelay &pd : specModel->pin2reg) {
    if (pd.port.signal != "data")
      continue;
    auto it = portToValue.find(pd.port.port);
    if (it == portToValue.end()) {
      unit->emitError() << "spec pin2reg references unknown port '"
                        << pd.port.port << "'";
      return;
    }
    Value v = it->second;
    if (vars.channelVars.find(v) == vars.channelVars.end())
      continue;
    if (pd.samples.empty()) {
      unit->emitError() << "spec pin2reg " << pd.port.port << "."
                        << pd.port.signal << " has no samples";
      return;
    }
    double delay = pickClosest(pd.samples);
    CPVar &tArr = unitSideDataVar(v);
    std::string consName =
        "path_spec_p2r_" + unitName.str() + "_" + std::to_string(idx++);
    model->addConstr(tArr + delay <= targetPeriod, consName);
  }

  for (const SpecTimingPortDelay &pd : specModel->reg2pin) {
    if (pd.port.signal != "data")
      continue;
    auto it = portToValue.find(pd.port.port);
    if (it == portToValue.end()) {
      unit->emitError() << "spec reg2pin references unknown port '"
                        << pd.port.port << "'";
      return;
    }
    Value v = it->second;
    if (vars.channelVars.find(v) == vars.channelVars.end())
      continue;
    if (pd.samples.empty()) {
      unit->emitError() << "spec reg2pin " << pd.port.port << "."
                        << pd.port.signal << " has no samples";
      return;
    }
    double delay = pickClosest(pd.samples);
    CPVar &tDep = unitSideDataVar(v);
    std::string consName =
        "path_spec_r2p_" + unitName.str() + "_" + std::to_string(idx++);
    model->addConstr(tDep >= delay, consName);
  }
}

void FPGA20Buffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto &[channel, chVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace =
        static_cast<unsigned>(model->getValue(chVars.bufNumSlots) + 0.5);

    // forceBreakDV == 1 means break D, V; forceBreakDV == 0 means break
    // nothing.
    bool forceBreakDV =
        model->getValue(chVars.signalVars[SignalType::DATA].bufPresent) > 0;

    PlacementResult result;
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
    // In FPGA20, buffers only break the data and valid paths.
    // We insert TEHBs after all Merge-like operations to break the ready paths.
    // We only break the ready path if the channel is on cycle.
    Operation *srcOp = channel.getDefiningOp();
    if (isa_and_nonnull<handshake::MuxOp, handshake::MergeOp>(srcOp) &&
        srcOp->getNumOperands() > 1 && isChannelOnCycle(channel)) {
      result.numOneSlotR = 1;
    }

    placement[channel] = result;
  }

  LLVM_DEBUG(logResults(placement););

  llvm::MapVector<size_t, double> cfdfcTPResult;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double tmpThroughput = model->getValue(cfVars.throughput);

    cfdfcTPResult[idx] = tmpThroughput;
  }

  // Create and add the handshake.tp attribute
  auto cfdfcTPMap = handshake::CFDFCThroughputAttr::get(
      funcInfo.funcOp.getContext(), cfdfcTPResult);
  setDialectAttr(funcInfo.funcOp, cfdfcTPMap);

  populateCFDFCThroughputAndOccupancy();
}

void FPGA20Buffers::addCustomChannelConstraints(Value channel) {
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

void FPGA20Buffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 1> signalTypes;
  signalTypes.push_back(SignalType::DATA);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  // Create buffering groups. In this MILP we only care for the data signal
  SmallVector<BufferingGroup> bufGroups;
  bufGroups.emplace_back(ArrayRef<SignalType>{SignalType::DATA}, bufModel);

  // Create channel variables and constraints
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signalTypes);
    addCustomChannelConstraints(channel);

    // Add path and elasticity constraints over all channels in the function
    // that are not adjacent to a memory interface
    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addChannelTimingConstraints(channel, SignalType::DATA, bufModel);
      addBufferPresenceConstraints(channel);
      addBufferingGroupConstraints(channel, bufGroups);
    }
  }

  // Add path and elasticity constraints over all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    if (isa<handshake::SpeculatorOp, handshake::SpecSaveCommitOp>(&op)) {
      addSpecUnitDataPathConstraints(&op);
      continue;
    }
    addUnitTimingConstraints(&op, SignalType::DATA);
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC that
  // was marked to be optimized
  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    addSteadyStateReachabilityConstraints(*cfdfc);
    addChannelThroughputConstraintsForBinaryLatencyChannel(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  addMaxThroughputObjective(allChannels, cfdfcs);
  markReadyToOptimize();
}
