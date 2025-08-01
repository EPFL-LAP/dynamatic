//===- MAPBUFBuffers.cpp - MAPBUF buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements MAPBUF smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/MAPBUFBuffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/BlifReader.h"
#include "experimental/Support/CutlessMapping.h"
#include "experimental/Support/SubjectGraph.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include <string>
#include <unordered_map>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mapbuf;

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, StringRef blifFiles,
                             double lutDelay, int lutSize, bool acyclicType)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod),
      acyclicType(acyclicType), lutSize(lutSize), lutDelay(lutDelay),
      blifFiles(blifFiles) {
  if (!unsatisfiable)
    setup();
}

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, StringRef blifFiles,
                             double lutDelay, int lutSize, bool acyclicType,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName),
      acyclicType(acyclicType), lutSize(lutSize), lutDelay(lutDelay),
      blifFiles(blifFiles) {
  if (!unsatisfiable)
    setup();
}

void MAPBUFBuffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto [channel, chVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace =
        static_cast<unsigned>(chVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool forceBreakDV = chVars.signalVars[SignalType::DATA].bufPresent.get(
                            GRB_DoubleAttr_X) > 0;
    bool forceBreakR = chVars.signalVars[SignalType::READY].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;

    PlacementResult result;
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

    placement[channel] = result;
  }

  if (logger)
    logResults(placement);

  llvm::MapVector<size_t, double> cfdfcTPResult;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double tmpThroughput = cfVars.throughput.get(GRB_DoubleAttr_X);

    cfdfcTPResult[idx] = tmpThroughput;
  }

  // Create and add the handshake.tp attribute
  auto cfdfcTPMap = handshake::CFDFCThroughputAttr::get(
      funcInfo.funcOp.getContext(), cfdfcTPResult);
  setDialectAttr(funcInfo.funcOp, cfdfcTPMap);
}

// Check if the path from leaf to root has already been computed, if so then
// return it. If not, return the shortest path by running BFS.
static std::vector<experimental::Node *>
getPath(experimental::Node *key, experimental::Node *leaf,
        pathMap &leafToRootPaths, experimental::LogicNetwork *blif) {
  // Check if this leaf/key pair is already computed
  auto leafKeyPair = std::make_pair(leaf, key);
  if (leafToRootPaths.find(leafKeyPair) != leafToRootPaths.end()) {
    return leafToRootPaths[leafKeyPair];
  }

  // Run BFS and compute the path
  auto path = blif->findPath(leaf, key);

  if (!path.empty()) {
    // remove the starting node and the root node, as we should be able to place
    // buffers on channels adjacent to these nodes
    path.pop_back();
    path.erase(path.begin());
  }

  // Save the path for quick lookups in the future
  leafToRootPaths[leafKeyPair] = path;
  return path;
}

const std::map<unsigned int, double> ADD_SUB_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.6}, {16, 0.7}, {32, 1.0}};

const std::map<unsigned int, double> COMPARATOR_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.8}, {16, 1.0}, {32, 1.2}};

void MAPBUFBuffers::addCustomChannelConstraints(Value channel) {
  // Get channel-specific buffering properties and channel's variables
  handshake::ChannelBufProps &props = channelProps[channel];
  ChannelVars &chVars = vars.channelVars[channel];

  // Force buffer presence if at least one slot is requested
  unsigned minSlots =
      std::max(props.minOpaque + props.minTrans, props.minSlots);
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
    // If the MILP decides to also place a data buffer, then we must reserve
    // an extra slot for it
    model.addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                    "custom_minReady");
  }

  // Set constraints based on maximum number of buffer slots
  if (props.maxOpaque && props.maxTrans) {
    unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
    if (maxSlots == 0) {
      // Forbid buffer placement on the channel entirely when no slots are
      // allowed
      model.addConstr(chVars.bufPresent == 0, "custom_noBuffer");
      model.addConstr(chVars.bufNumSlots == 0, "custom_maxSlots");
    } else {
      // Restrict the maximum number of slots allowed. If both types are allowed
      // but the MILP decides to only place one type, then the maximum allowed
      // number is the maximum number of slots we can place for that type
      model.addConstr(chVars.bufNumSlots <=
                          maxSlots - *props.maxOpaque * (1 - bufData) -
                              *props.maxTrans * (1 - bufReady),
                      "custom_maxSlots");
    }
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

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group.
  /// We don't have models for these buffers at the moment therefore we
  /// provide a null-model to each group, but this hurts our placement's
  /// accuracy.
  const TimingModel *bufModel = nullptr;
  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                bufModel);
  BufferingGroup readyGroup({SignalType::READY}, bufModel);

  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  std::vector<Value> allChannels;
  std::vector<Value> backedges;
  for (auto &[channel, _] : channelProps) {
    // Create channel variables and constraints
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);
    addBlackboxConstraints(channel, ADD_SUB_DELAYS, COMPARATOR_DELAYS);

    if (isBackedge(channel))
      backedges.push_back(channel);

    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addBufferPresenceConstraints(channel);
      addBufferingGroupConstraints(channel, bufGroups);
    }
  }

  // Generate Subject Graphs
  experimental::subjectGraphGenerator(funcInfo.funcOp, blifFiles);

  std::vector<Value> channelsToBuffer;
  if (!acyclicType) {
    channelsToBuffer = backedges;
  } else {
    channelsToBuffer = findMinimumFeedbackArcSet();
  }

  for (auto &result : channelsToBuffer) {
    cutGraphEdges(result);
  }

  // Connect input/output nodes of Subject Graphs
  blifData = experimental::connectSubjectGraphs();

  // Generate cuts of the circuit.
  auto cuts = experimental::generateCuts(blifData, lutSize);

  addNodeVars(blifData);
  addClockPeriodConstraintsNodes(blifData);

  for (auto &[rootNode, cutVector] : cuts) {
    addCutSelectionConstraints(cutVector);
    addDelayAndCutConflictConstraints(rootNode, cutVector, blifData, lutDelay);
    for (auto &cut : cutVector) {
      auto &leaves = cut.getLeaves();
      GRBVar &cutSelectionVar = cut.getCutSelectionVariable();
      for (auto *leaf : leaves) {
        if ((leaves.size() == 1) && (*leaves.begin() == rootNode)) {
          continue;
        }
        // Get the path from the leaf to the root
        std::vector<experimental::Node *> path;
        path = getPath(rootNode, leaf, leafToRootPaths, blifData);
        // Add cut selection conflict constraints for the root
        addCutSelectionConflicts(rootNode, leaf, cutSelectionVar, blifData,
                                 path);
      }
    }
  }

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

  addMaxThroughputObjective(allChannels, cfdfcs);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
