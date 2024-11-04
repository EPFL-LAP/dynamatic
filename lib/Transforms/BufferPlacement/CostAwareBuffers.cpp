//===- CostAwareBuffers.cpp - Cost-aware buffer placement -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements cost-aware smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/CostAwareBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/IR/Value.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::costaware;

/// Returns a textual name for a signal type.
static StringRef getSignalName(SignalType type) {
  switch (type) {
  case SignalType::DATA:
    return "data";
  case SignalType::VALID:
    return "valid";
  case SignalType::READY:
    return "ready";
  }
}

/// Returns a textual name for a buffer type.
static StringRef getBufferName(BufferType type) {
  switch (type) {
  case BufferType::OB:
    return "oehb";
  case BufferType::TB:
    return "tehb";
  case BufferType::FT:
    return "tfifo";
  case BufferType::SE:
    return "dvse";
  case BufferType::DR:
    return "dvr";
  }
}

/// Returns penalty weight for buffer existence and each buffertype.
double getpenaltycoef(const llvm::StringRef &type) {
    static const std::unordered_map<std::string, double> lookupTable = {
        {"oehb", 2},
        {"tehb", 20},
        {"tfifo", 25},
        {"dvse", 0.01},
        {"dvr", 2.1},
        {"seExist", 0.01},
        {"bufExist", 16},
    };

    auto it = lookupTable.find(type.str());
    if (it != lookupTable.end()){
        return it->second;
    } else{
      llvm::errs() << "Unknown buffer type: " << type.str();
      return 0;
    }
}

CostAwareBuffers::CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod){
  if (!unsatisfiable)
    setup();
}

CostAwareBuffers::CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, Logger &logger,
                            StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName){
  if (!unsatisfiable)
    setup();
}

void CostAwareBuffers::addChannelVars(Value channel, ArrayRef<BufferType> buffertypes,
                                         ArrayRef<SignalType> signals) {

  // Default-initialize channel variables and retrieve a reference
  ChannelVars &channelVars = vars.channelVars[channel];
  std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
  
  // Create a Gurobi variable of the given name and type for the channel
  auto createVar = [&](const llvm::Twine &name, char type) {
    return model.addVar(0, GRB_INFINITY, 0.0, type, (name + suffix).str());
  };

  // Signal-specific variables
  for (SignalType sig : signals) {
    ChannelSignalVars &signalVars = channelVars.signalVars[sig];
    StringRef name = getSignalName(sig);
    signalVars.path.tIn = createVar(name + "PathIn", GRB_CONTINUOUS);
    signalVars.path.tOut = createVar(name + "PathOut", GRB_CONTINUOUS);
    signalVars.bufPresent = createVar(name + "BufPresent", GRB_BINARY);
  }

  channelVars.bufPresent = createVar("bufPresent", GRB_BINARY);
  channelVars.sePresent = createVar("sePresent", GRB_BINARY);

  for (BufferType buffertype : buffertypes) {
    GRBVar &bufnumSlots = channelVars.bufNumSlots[buffertype];
    StringRef name = getBufferName(buffertype);
    bufnumSlots = createVar(name + "BufNumSlots", GRB_INTEGER);
  }
  
  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void CostAwareBuffers::addCFDFCVars(CFDFC &cfdfc) {
  // Create a set of variables for each CFDFC
  std::string prefix = "cfdfc" + std::to_string(vars.cfVars.size()) + "_";
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];

  // Create a Gurobi variable of the given name (prefixed by the CFDFC index)
  auto createVar = [&](const llvm::Twine &name) {
    return model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        (prefix + name).str());
  };

  // Create a set of variables for each unit in the CFDFC
  for (Operation *unit : cfdfc.units) {
    std::string suffix = "_" + getUniqueName(unit).str();
    // Default-initialize unit variables and retrieve a reference
    UnitVars &unitVars = cfVars.unitVars[unit];
    unitVars.retIn = createVar("retIn" + suffix);

    // If the component is combinational (i.e., 0 latency) its output fluid
    // retiming equals its input fluid retiming, otherwise it is different
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)))
      latency = 0.0;
    if (latency == 0.0)
      unitVars.retOut = unitVars.retIn;
    else
      unitVars.retOut = createVar("retOut" + suffix);
  }

  // Create a variable to represent the throughput of each CFDFC channel
  for (Value channel : cfdfc.channels) {
    cfVars.channelThroughputs[channel] =
        createVar("throughput_" + getUniqueName(*channel.getUses().begin()));
  }

  // Create a variable for the CFDFC's throughput
  cfVars.throughput = model.addVar(0, 1, 0.0, GRB_CONTINUOUS,
                        "throughput");

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void CostAwareBuffers::addChannelPathConstraints(
    Value channel, SignalType signal, const TimingModel *bufModel) {
  
  ChannelVars &channelVars = vars.channelVars[channel];
  double bigCst = targetPeriod * 10;

  ChannelSignalVars &signalVars = channelVars.signalVars[signal];
  GRBVar &t1 = signalVars.path.tIn;
  GRBVar &t2 = signalVars.path.tOut;
  GRBVar &bufPresent = signalVars.bufPresent;

  std::string pathbufferedChannelInName = "path_bufferedChannelIn_" 
                                          + getUniqueName(*channel.getUses().begin()) 
                                          + '_' + getSignalName(signal).str();
  model.addConstr(t1 <= targetPeriod,
                  pathbufferedChannelInName);

  std::string pathunbufferedChannelName = "path_unbufferedChannel_" 
                                          + getUniqueName(*channel.getUses().begin()) 
                                          + '_' + getSignalName(signal).str();
  model.addConstr(t1 - bigCst * bufPresent <= t2,
                  pathunbufferedChannelName);

  model.update();
}

void CostAwareBuffers::addUnitPathConstraints(Operation *unit,
                                                 SignalType type,
                                                 ChannelFilter filter) {
  // Add path constraints for units
  double latency;
  if (failed(timingDB.getLatency(unit, type, latency)))
    latency = 0.0;

  if (latency == 0.0) {
    double delay;
    if (failed(timingDB.getTotalDelay(unit, type, delay)))
      delay = 0.0;

    // The unit is not pipelined, add a path constraint for each input/output
    // port pair in the unit
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!filter(in) || !filter(out))
        return;

      // Flip channels on ready path which goes upstream
      if (type == SignalType::READY)
        std::swap(in, out);

      GRBVar &tInPort = vars.channelVars[in].signalVars[type].path.tOut;
      GRBVar &tOutPort = vars.channelVars[out].signalVars[type].path.tIn;
      // Arrival time at unit's output port must be greater than arrival
      // time at unit's input port + the unit's combinational data delay
      model.addConstr(tOutPort >= tInPort + delay, "path_combDelay");
    });
    model.update();
    return;
  }

  // The unit is pipelined, add a constraint for every of the unit's inputs
  // and every of the unit's output ports

  // Input port constraints
  for (Value in : unit->getOperands()) {
    if (!filter(in))
      continue;

    double inPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::IN, inPortDelay)))
      inPortDelay = 0.0;

    TimeVars &path = vars.channelVars[in].signalVars[type].path;
    GRBVar &tInPort = path.tOut;
    // Arrival time at unit's input port + input port delay must be less
    // than the target clock period
    model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
  }

  // Output port constraints
  for (OpResult out : unit->getResults()) {
    if (!filter(out))
      continue;

    double outPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::OUT, outPortDelay)))
      outPortDelay = 0.0;

    TimeVars &path = vars.channelVars[out].signalVars[type].path;
    GRBVar &tOutPort = path.tIn;
    // Arrival time at unit's output port is equal to the output port delay
    model.addConstr(tOutPort >= outPortDelay, "path_outDelay");
  }

  model.update();
}

void CostAwareBuffers::addChannelCustomConstraints(
    Value channel) {
  
  ChannelVars &chVars = vars.channelVars[channel];
  GRBVar &bufPresent = chVars.bufPresent;

  std::string ChannelName = getUniqueName(*channel.getUses().begin());
  // Decide whether there is at least a type of buffer present on a signal
  for (auto &[sig, signalVars] : chVars.signalVars) {
    GRBLinExpr signalbufSlots = 0;
    for (auto &[buffertype, buffernumSlots] : chVars.bufNumSlots) {
      if (signalBufferMatrix[sig][buffertype]) {
        signalbufSlots += buffernumSlots;
      }
    }
    model.addConstr(0.001 * signalbufSlots <= signalVars.bufPresent,
                    getSignalName(sig).str() 
                    + '_' + ChannelName
                    + "_PresenceLower");
    model.addConstr(signalbufSlots >= signalVars.bufPresent,
                    getSignalName(sig).str() 
                    + '_' + ChannelName
                    + "_PresenceUpper");
  }

  // If there is at least one slot, there must be a buffer
  GRBLinExpr TotalSlotNum = 0;
  for (auto &[buffertype, buffernumSlots] : chVars.bufNumSlots) {
    TotalSlotNum += buffernumSlots;
    // Temporary setting since the current model is imprecise.
    // More than one TBs bring higher cost but lower elasticity
    // compared with FTs.
    if (buffertype == BufferType::TB){
      model.addConstr(buffernumSlots <= 1, ChannelName + "_oneTBslotatMost");
    }
  }
  model.addConstr(0.001 * TotalSlotNum <= bufPresent, ChannelName + "bufPresence");
  // Whether a DVSE is on the channel.
  model.addConstr(1000 * chVars.sePresent >= chVars.bufNumSlots[BufferType::SE], ChannelName + "sePresentConstrlb");
  model.addConstr(chVars.sePresent <= chVars.bufNumSlots[BufferType::SE], ChannelName + "sePresentConstrub");
  model.update();
}

void CostAwareBuffers::addChannelThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];
  for (Value channel : cfdfc.channels) {
    // Get the ports the channels connect and their retiming MILP variables
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();

    // No throughput constraints on channels going to LSQ stores
    if (isa<handshake::LSQStoreOp>(dstOp))
      continue;

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (auto selOp = dyn_cast<handshake::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // The channel must have variables for the data signal
    ChannelVars &chVars = vars.channelVars[channel];
    GRBVar &chThroughput = cfVars.channelThroughputs[channel];
    GRBVar &retSrc = cfVars.unitVars[srcOp].retOut;
    GRBVar &retDst = cfVars.unitVars[dstOp].retIn;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    std::string ChannelName = getUniqueName(*channel.getUses().begin());
    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    std::string RetimingConstrName = "throughput_channelRetiming_" + ChannelName;
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    RetimingConstrName);

    // The Lowerbound of channel Throuhgput
    GRBQuadExpr chThroughputLower = chVars.bufNumSlots[BufferType::OB] * cfVars.throughput +
                                chVars.bufNumSlots[BufferType::SE] * cfVars.throughput +
                                chVars.bufNumSlots[BufferType::DR] * cfVars.throughput;
    std::string throughputLowerConstrName = "throughput_channel_lower_" + ChannelName;
    model.addQConstr(chThroughput >= chThroughputLower, throughputLowerConstrName);

    // Set a ceiling funciton for SE type
    std::string ceilingVarName = "ceiling_" + ChannelName;
    GRBVar Ceiling = model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, ceilingVarName);
    std::string ceilingConstrName = "throughput_channel_seCeiling_" + ChannelName;
    model.addQConstr(Ceiling <= chVars.bufNumSlots[BufferType::SE] * cfVars.throughput + 0.99, ceilingConstrName);
    // The Upperbound of channel Throughput
    GRBQuadExpr chThroughputUpper = chVars.bufNumSlots[BufferType::OB] +
                                chVars.bufNumSlots[BufferType::TB] * (1 - cfVars.throughput) +
                                chVars.bufNumSlots[BufferType::FT] +
                                Ceiling +
                                chVars.bufNumSlots[BufferType::DR] * (1 - cfVars.throughput);
    std::string throughputUpperConstrName = "throughput_channel_upper_" + ChannelName;
    model.addQConstr(chThroughput <= chThroughputUpper, throughputUpperConstrName);
    
    
    // // The Lowerbound of channel Throuhgput
    // double temp = 0.5;
    // GRBLinExpr chThroughputLower = chVars.bufNumSlots[BufferType::OB] * temp +
    //                             chVars.bufNumSlots[BufferType::SE] * temp +
    //                             chVars.bufNumSlots[BufferType::DR] * temp;
    // std::string throughputLowerConstrName = "throughput_channel_lower_" + ChannelName;
    // model.addConstr(chThroughput >= chThroughputLower, throughputLowerConstrName);

    // // Set a ceiling funciton for SE type
    // std::string ceilingVarName = "ceiling_" + ChannelName;
    // GRBVar Ceiling = model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, ceilingVarName);
    // std::string ceilingConstrName = "throughput_channel_seCeiling_" + ChannelName;
    // model.addConstr(Ceiling <= chVars.bufNumSlots[BufferType::SE] * temp + 0.99, ceilingConstrName);
    // temp = 0;
    // The Upperbound of channel Throughput
    // GRBLinExpr chThroughputUpper = chVars.bufNumSlots[BufferType::OB] +
    //                             chVars.bufNumSlots[BufferType::TB] * (1 - temp) +
    //                             chVars.bufNumSlots[BufferType::FT] +
    //                             Ceiling +
    //                             chVars.bufNumSlots[BufferType::DR] * (1 - temp);
    // std::string throughputUpperConstrName = "throughput_channel_upper_" + ChannelName;
    // model.addConstr(chThroughput <= chThroughputUpper, throughputUpperConstrName);
  }
  model.update();
}

void CostAwareBuffers::addUnitThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];
  for (Operation *unit : cfdfc.units) {
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)) ||
        latency == 0.0)
      continue;

    // Retrieve the MILP variables corresponding to the unit's fluid retiming
    UnitVars &unitVars = cfVars.unitVars[unit];
    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;

    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC's throughput
    model.addConstr(cfVars.throughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
  model.update();
}

void CostAwareBuffers::addObjective(ValueRange channels, ArrayRef<BufferType> buffertypes,
                                       ArrayRef<CFDFC *> cfdfcs) {
  // Compute the total number of executions over channels that are part of any
  // CFDFC
  unsigned totalExecs = 0;
  for (Value channel : channels) {
    totalExecs += BufferPlacementMILP::getChannelNumExecs(channel);
  }

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC, only if coef > 0.15
  double totalNormalizedCoef = 0.0;
  double fTotalExecs = static_cast<double>(totalExecs);
  std::vector<std::pair<CFDFC*, double>> validCFDFCs; 

  if (totalExecs != 0) {
    for (CFDFC* cfdfc : cfdfcs) {
      double coef = (cfdfc->channels.size() * cfdfc->numExecs) / fTotalExecs;

      // Collect only significant coefs and their corresponding CFDFCs
      if (coef > 0.15) {
        validCFDFCs.emplace_back(cfdfc, coef);
        totalNormalizedCoef += coef;
      }
    }

    // Normalize the coefficients if there are any significant CFDFCs
    if (totalNormalizedCoef > 0) {
      for (auto& cfdfc_pair : validCFDFCs) {
        double normalizedCoef = cfdfc_pair.second / totalNormalizedCoef;
        CFDFC* cfdfc = cfdfc_pair.first;
        objective += normalizedCoef * vars.cfVars[cfdfc].throughput;
      }
    }
  }

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double slotPenaltycoef = 3e-5;
  for (Value channel : channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    for (BufferType buffertype : buffertypes) {
      objective -= getpenaltycoef(getBufferName(buffertype)) * slotPenaltycoef * chVars.bufNumSlots[buffertype];
    }
    objective -= getpenaltycoef("bufExist") * slotPenaltycoef * chVars.bufPresent;
    objective -= getpenaltycoef("seExist") * slotPenaltycoef * chVars.sePresent; 
  }
  
  model.update();
  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
}


void CostAwareBuffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto &[channel, chVars] : vars.channelVars) {
    
    PlacementResult result;
    
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        chVars.bufNumSlots[BufferType::OB].get(GRB_DoubleAttr_X) + 0.5);
      result.numSlotOB = numSlotsToPlace;
    numSlotsToPlace = static_cast<unsigned>(
        chVars.bufNumSlots[BufferType::TB].get(GRB_DoubleAttr_X) + 0.5);
      result.numSlotTB = numSlotsToPlace;
    numSlotsToPlace = static_cast<unsigned>(
        chVars.bufNumSlots[BufferType::FT].get(GRB_DoubleAttr_X) + 0.5);
      result.numTFIFO = numSlotsToPlace;
    numSlotsToPlace = static_cast<unsigned>(
        chVars.bufNumSlots[BufferType::SE].get(GRB_DoubleAttr_X) + 0.5);
      result.numDVSE = numSlotsToPlace;
    numSlotsToPlace = static_cast<unsigned>(
        chVars.bufNumSlots[BufferType::DR].get(GRB_DoubleAttr_X) + 0.5);
      result.numDVR = numSlotsToPlace;
  
    // This is an optimization to combine one slot OB and a n-slot TranspFIFO to
    // a (n+1)-slot DVFIFO. This optimization saves the area cost.
    if (result.numSlotOB && result.numTFIFO) {
      result.numSlotOB -= 1;
      result.numDVFIFO = result.numTFIFO + 1;
      result.numTFIFO = 0;
    }

    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void CostAwareBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  SmallVector<BufferType, 8> buffertypes;
  buffertypes.push_back(BufferType::OB);
  buffertypes.push_back(BufferType::TB);
  buffertypes.push_back(BufferType::FT);
  buffertypes.push_back(BufferType::SE);
  buffertypes.push_back(BufferType::DR);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  // Create channel variables and constraints
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    // std::string find_mem = getUniqueName(*channel.getUses().begin());
    // if (find_mem.find("mem_controller") == std::string::npos)
    //   continue;
    allChannels.push_back(channel);
    addChannelVars(channel, buffertypes, signals);

    // Add single-domain path constraints
    addChannelPathConstraints(channel, SignalType::DATA, bufModel);
    addChannelPathConstraints(channel, SignalType::VALID, bufModel);
    addChannelPathConstraints(channel, SignalType::READY, bufModel);

    addChannelCustomConstraints(channel);
  }

  // Add path and elasticity constraints over all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitPathConstraints(&op, SignalType::DATA);
    addUnitPathConstraints(&op, SignalType::VALID);
    addUnitPathConstraints(&op, SignalType::READY);
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
  addObjective(allChannels, buffertypes, cfdfcs);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
