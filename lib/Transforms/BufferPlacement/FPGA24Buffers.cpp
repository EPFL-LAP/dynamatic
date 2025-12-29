//===- FPGA24Buffers.cpp - FPGA'24 buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a stall-eliminating buffer placement algorithm for dataflow
// circuits.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA24Buffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Support/DataflowGraph/ReconvergentPathFinder.h"
#include "dynamatic/Support/DataflowGraph/SynchronizingCyclesFinder.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/IR/Value.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga24;

FPGA24Buffers::FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod) {
  if (!unsatisfiable)
    setup();
}

FPGA24Buffers::FPGA24Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod, Logger &logger,
                             StringRef milpName)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          logger, milpName) {
  if (!unsatisfiable)
    setup();
}

// NOTE: This contains the same logic as FPGA20Buffers::extractResult, this is
// temporary.
void FPGA24Buffers::extractResult(BufferPlacement &placement) {
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

  if (logger)
    logResults(placement);

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

// TODO: Same channel constraints as FPGA20Buffers::addCustomChannelConstraints,
// this is temporary.
void FPGA24Buffers::addCustomChannelConstraints(Value channel) {
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

// TODO: Same setup as FPGA20Buffers::setup, this is temporary.
void FPGA24Buffers::setup() {
  for (auto &transition : funcInfo.archs) {
    llvm::errs() << transition.srcBB << "->" << transition.dstBB << "\n";
  }

  // --- Reconvergent Path Analysis ---
  // Convert SmallVector to std::vector for the enumeration function
  // std::vector<experimental::ArchBB> transitions(funcInfo.archs.begin(),
  //                                               funcInfo.archs.end());

  // unsigned sequenceLength = 2;
  // auto allSequences = enumerateTransitionSequences(transitions, sequenceLength);

  // // Build graphs for all sequences
  // std::vector<ReconvergentPathFinderGraph> allGraphs;
  // for (const auto & sequence : allSequences) {
  //   allGraphs.emplace_back();
  //   allGraphs.back().buildGraphFromSequence(funcInfo.funcOp, sequence);
  // }

  // // Find reconvergent paths for each graph (after all graphs are built)
  // std::vector<std::pair<size_t, std::pair<const ReconvergentPathFinderGraph *,
  //                                         std::vector<ReconvergentPath>>>>
  //     allReconvergentPaths;

  // for (size_t seqIdx = 0; seqIdx < allGraphs.size(); ++seqIdx) {
  //   std::vector<ReconvergentPath> reconvergentPaths =
  //       allGraphs[seqIdx].findReconvergentPaths();
  //   if (!reconvergentPaths.empty()) {
  //     allReconvergentPaths.emplace_back(
  //         seqIdx,
  //         std::make_pair(&allGraphs[seqIdx], std::move(reconvergentPaths)));
  //   }
  // }

  // // Dump all graphs to a single file
  // ReconvergentPathFinderGraph::dumpAllGraphs(allGraphs, "dataflow_graphs.dot");

  // // Dump all reconvergent paths to a single file
  // if (!allReconvergentPaths.empty()) {
  //   ReconvergentPathFinderGraph::dumpAllReconvergentPaths(allReconvergentPaths,
  //                                                        "reconvergent_paths.dot");
  // }
  // --- End Reconvergent Path Analysis ---

  // --- Synchronizing Cycles Analysis ---

  size_t cfdfcIdx = 0;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs) {

    SynchronizingCyclesFinderGraph graph;
    graph.buildFromCFDFC(funcInfo.funcOp, *cfdfc);

    auto pairs = graph.findSynchronizingCyclePairs();
    
    llvm::errs() << "Found " << pairs.size() 
                 << " synchronizing cycle pairs in CFDFC " << cfdfcIdx << "\n";
    
    for (const auto &pair : pairs) {
      llvm::errs() << "  Pair: cycle with " << pair.cycleOne.nodes.size()
                   << " nodes <-> cycle with " << pair.cycleTwo.nodes.size()
                   << " nodes, " << pair.pathsToJoins.size() 
                   << " common joins\n";
    }

    // Dump to GraphViz for visualization
    if (!pairs.empty()) {
      std::string filename = "synchronizing_cycle_pairs_cfdfc" + 
                             std::to_string(cfdfcIdx) + ".dot";
      graph.dumpAllSynchronizingCyclePairs(pairs, filename);
    }
    ++cfdfcIdx;
  }

  // --- End Synchronizing Cycles Analysis ---

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
