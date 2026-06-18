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
#include <map>

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
                          Algorithm::FPGA20, writeTo) {
  if (!unsatisfiable)
    setup();
}

void FPGA20Buffers::addSpecUnitDataPathConstraints(Operation *op) {
  // get the timing model for this operation
  const SpecTimingModel *specModel = timingDB.getSpecModel(op);
  assert(specModel &&
         "addSpecUnitDataPathConstraints: no spec timing model loaded for op");

  // get the operand/result names of this operation
  auto namedIO = dyn_cast<handshake::NamedIOInterface>(op);
  assert(namedIO && "addSpecUnitDataPathConstraints: op does not implement "
                    "NamedIOInterface");

  // store mapping from operand/result name to operand/result
  // since we don't have a native function to go name -> value
  // only operand/result index -> name
  std::map<StringRef, Value> portNameToValue;
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i)
    portNameToValue[namedIO.getOperandName(i)] = op->getOperand(i);
  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
    portNameToValue[namedIO.getResultName(i)] = op->getResult(i);

  // we need specific op objects to get bitwidths
  // this is a bit hacky, probably an interface/trait would be better
  // if there were more ops
  Value bitwidthChannel =
      llvm::TypeSwitch<Operation *, Value>(op)
          .Case<handshake::SpeculatorOp>(
              [](handshake::SpeculatorOp op) { return op.getDataIn(); })
          .Case<handshake::SpecSaveCommitOp>(
              [](handshake::SpecSaveCommitOp op) { return op.getDataIn(); })
          .Default([](Operation *) -> Value {
            llvm_unreachable("addSpecUnitDataPathConstraints called on op type "
                             "with no bitwidth accessor registered");
          });

  // get the bitwidth of the operation
  unsigned currentBitwidth = 0;
  if (auto chTy = dyn_cast<handshake::ChannelType>(bitwidthChannel.getType()))
    currentBitwidth = chTy.getDataBitWidth();

  // get the name of the operation
  StringRef opName = getUniqueName(op);
  // index to make constraint names unique
  unsigned idx = 0;
  for (const SpecTimingPort2Port &edgeDelay : specModel->port2port) {

    // only adding data to data delays
    if (edgeDelay.from.signal != SignalType::DATA ||
        edgeDelay.to.signal != SignalType::DATA)
      continue;

    // beginning of path
    Value fromVal = portNameToValue.at(edgeDelay.from.name);
    // end of path
    Value toVal = portNameToValue.at(edgeDelay.to.name);

    // we need to have real channel variables for these values
    assert(vars.channelVars.find(fromVal) != vars.channelVars.end() &&
           "value has no corresponding channel variable");
    assert(vars.channelVars.find(toVal) != vars.channelVars.end() &&
           "value has no corresponding channel variable");

    // get the delay based on the bitwidth
    double delay = edgeDelay.delayList.selectDelay(currentBitwidth);

    // get the actual CPVars
    CPVar &tFrom =
        vars.channelVars[fromVal].signalVars[SignalType::DATA].path.tOut;
    CPVar &tTo = vars.channelVars[toVal].signalVars[SignalType::DATA].path.tIn;

    // make constraint name
    std::string consName =
        "path_spec_p2p_" + opName.str() + "_" + std::to_string(idx++);

    // add constraint
    model->addConstr(tFrom + delay <= tTo, consName);
  }

  for (const SpecTimingPort2Reg &portDelay : specModel->port2reg) {

    // only adding data signal delays
    if (portDelay.port.signal != SignalType::DATA)
      continue;

    // the port of this delay
    Value v = portNameToValue.at(portDelay.port.name);

    // we need to have a real channel variable for this value
    assert(vars.channelVars.find(v) != vars.channelVars.end() &&
           "value has no corresponding channel variable");

    // get the delay based on the bitwidth
    double delay = portDelay.delayList.selectDelay(currentBitwidth);

    // get the actual CPVar
    CPVar &tArr = vars.channelVars[v].signalVars[SignalType::DATA].path.tOut;

    // make constraint name
    std::string consName =
        "path_spec_p2r_" + opName.str() + "_" + std::to_string(idx++);

    // add constraint
    model->addConstr(tArr + delay <= targetPeriod, consName);
  }

  for (const SpecTimingReg2Port &portDelay : specModel->reg2port) {

    // only adding data signal delays
    if (portDelay.port.signal != SignalType::DATA)
      continue;

    // the port of this delay
    Value v = portNameToValue.at(portDelay.port.name);

    // we need to have a real channel variable for this value
    assert(vars.channelVars.find(v) != vars.channelVars.end() &&
           "value has no corresponding channel variable");

    // get the delay based on the bitwidth
    double delay = portDelay.delayList.selectDelay(currentBitwidth);

    // get the actual CPVar
    CPVar &tDep = vars.channelVars[v].signalVars[SignalType::DATA].path.tIn;

    // make constraint name
    std::string consName =
        "path_spec_r2p_" + opName.str() + "_" + std::to_string(idx++);

    // add constraint
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
