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
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <string>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga20;

FPGA20Buffers::FPGA20Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB,
                          targetPeriod) {
  if (!unsatisfiable)
    setup();
}

FPGA20Buffers::FPGA20Buffers(CPSolver::SolverKind solverKind, int timeout,
                             FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             double targetPeriod, Logger &logger,
                             StringRef milpName)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          logger, milpName) {
  if (!unsatisfiable)
    setup();
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

void FPGA20Buffers::addCustomChannelConstraints(Value channel) {
  // ChannelVars &chVars = vars.channelVars[channel];
  // handshake::ChannelBufProps &props = channelProps[channel];
  // CPVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;

  // if (props.minOpaque > 0) {
  //   // Force the MILP to use opaque slots
  //   model->addConstr(dataBuf == 1, "custom_forceOpaque");
  //   if (props.minTrans > 0) {
  //     // If the properties ask for both opaque and transparent slots, let
  //     // opaque slots take over. Transparents slots will be placed "manually"
  //     // from the total number of slots indicated by the MILP's result
  //     unsigned minTotalSlots = props.minOpaque + props.minTrans;
  //     model->addConstr(chVars.bufNumSlots >= minTotalSlots,
  //                      "custom_minOpaqueAndTrans");
  //   } else {
  //     // Force the MILP to place a minimum number of opaque slots
  //     model->addConstr(chVars.bufNumSlots >= props.minOpaque,
  //                      "custom_minOpaque");
  //   }
  // } else if (props.minTrans > 0) {
  //   // Force the MILP to place a minimum number of transparent slots
  //   model->addConstr(chVars.bufNumSlots >= props.minTrans + dataBuf,
  //                    "custom_minTrans");
  // } else if (props.minSlots > 0) {
  //   // Force the MILP to place a minimum number of slots
  //   model->addConstr(chVars.bufNumSlots >= props.minSlots,
  //   "custom_minSlots");
  // }
  // if (props.minOpaque + props.minTrans + props.minSlots > 0)
  //   model->addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  // // Set a maximum number of slots to be placed
  // if (props.maxOpaque.has_value()) {
  //   if (*props.maxOpaque == 0) {
  //     // Force the MILP to use transparent slots
  //     model->addConstr(dataBuf == 0, "custom_forceTransparent");
  //   }
  //   if (props.maxTrans.has_value()) {
  //     // Force the MILP to use a maximum number of slots
  //     unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
  //     if (maxSlots == 0) {
  //       model->addConstr(chVars.bufPresent == 0, "custom_noBuffers");
  //       model->addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
  //     } else {
  //       model->addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
  //     }
  //   }
  // }
}

/*
  AYA: List of Ambiguities and Problems
  - Problem 1: What is the objective function? It cannot be only maximize the
  global return output (and cannot count on memory returns because they are
  currently pointless) because in FTD, if the function is void, this return is
  simply START.

  - Problem 2: Is it okay that my objective function is only about maximizing 1
  channel throughput? The point is that I constraint all throughputs to not be >
  1, then depending on the loop latency, I have multiple inequality constraints.
    The other problem is that I have no notion of buffer slots, so how to
  incorporate them to see their effect on throughputs and how to add them to the
  objective function

  URGENT!!! About incorporating number of slots in the story, I may want to
  revisit the constrain of equating Join input throughputs because they are the
  main source of deadlock (?) and by adding N slots, we can reduce this deadlock
  in different ways...

  - Make sure to figure out how to use `extractResult` function with your new
  fields...

 */

using Cycle = std::vector<Operation *>;
using CycleList = std::vector<Cycle>;

static void printCycles(const CycleList &cycles) {
  llvm::errs() << "=== Circuit Cycles ===\n";
  int cycleIdx = 0;

  for (const auto &cycle : cycles) {
    llvm::errs() << "Cycle " << cycleIdx++ << " (" << cycle.size()
                 << " ops):\n";

    for (Operation *op : cycle) {
      llvm::errs() << "  - ";
      op->print(llvm::errs());
      llvm::errs() << "\n";
    }

    llvm::errs() << "\n";
  }
}

static Cycle normalizeCycle(const Cycle &cycle) {
  if (cycle.empty())
    return cycle;
  auto minIt = std::min_element(cycle.begin(), cycle.end());
  Cycle rotated;
  rotated.insert(rotated.end(), minIt, cycle.end());
  rotated.insert(rotated.end(), cycle.begin(), minIt);
  return rotated;
}

static std::string hashCycle(const Cycle &cycle) {
  std::string repr;
  llvm::raw_string_ostream rso(repr);

  for (auto *op : cycle) {
    op->print(rso);
    rso << ";"; // separator between ops
  }
  rso.flush();

  const std::string key = "handshake.name = \"";
  size_t start = repr.find(key);
  if (start != std::string::npos) {
    start += key.size(); // move right after the opening quote
    size_t end = repr.find("\"", start);
    if (end != std::string::npos) {
      return repr.substr(start, end - start);
    }
  }

  return "";
}

// DFS cycle detection
void findCyclesFrom(Operation *op, llvm::SmallVectorImpl<Operation *> &stack,
                    llvm::SmallPtrSetImpl<Operation *> &recursionStack,
                    llvm::SmallPtrSetImpl<Operation *> &visited,
                    CycleList &cycles, llvm::StringSet<> &seenCycleHashes) {
  // If already fully processed, skip
  if (visited.contains(op))
    return;

  recursionStack.insert(op);
  stack.push_back(op);

  for (auto result : op->getResults()) {
    for (auto &use : result.getUses()) {
      Operation *nextOp = use.getOwner();

      if (isa<handshake::MemoryControllerOp>(nextOp) ||
          isa<handshake::LSQOp>(nextOp))
        continue;

      if (!recursionStack.contains(nextOp))
        findCyclesFrom(nextOp, stack, recursionStack, visited, cycles,
                       seenCycleHashes);
      else {
        // nextOp is already visited indicating a cycle
        auto it = std::find(stack.begin(), stack.end(), nextOp);
        if (it != stack.end()) {
          Cycle rawCycle(it, stack.end());
          if (rawCycle.front() != nextOp)
            rawCycle.push_back(nextOp);

          Cycle normalized = normalizeCycle(rawCycle);
          std::string hash = hashCycle(normalized);

          if (!seenCycleHashes.contains(hash)) {
            seenCycleHashes.insert(hash);
            cycles.push_back(normalized);
          }
        }
      }
    }
  }

  recursionStack.erase(op);
  stack.pop_back();
  visited.insert(op);
}

CycleList findAllCycles(handshake::FuncOp funcOp) {
  CycleList cycles;
  llvm::SmallPtrSet<Operation *, 32> recursionStack;
  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallVector<Operation *, 32> stack;
  llvm::StringSet<> seenCycleHashes;

  for (Operation &op : funcOp.getOps()) {
    // we do not care of cycles created around mcs and lsqs
    if (isa<handshake::MemoryControllerOp>(op) || isa<handshake::LSQOp>(op))
      continue;
    findCyclesFrom(&op, stack, recursionStack, visited, cycles,
                   seenCycleHashes);
  }

  return cycles;
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

  bool aya = true;
  if (aya) {
    // (1) For every channel, create a throughput variable
    for (auto &[channel, _] : channelProps) {
      vars.ayaChannelThroughputs[channel].throughput = model->addVar(
          "throughput_" + getUniqueName(*channel.getUses().begin()),
          CPVar::REAL, 0, std::nullopt);

      // (2) Constrain all those throughputs to not be more than 1
      // New constraint to ensure that the throughput cannot be more than 1,
      // especially when bc ends up being > 1
      model->addConstr(vars.ayaChannelThroughputs[channel].throughput <= 1,
                       "throughput_upper_bound_" +
                           getUniqueName(*channel.getUses().begin()));
    }

    // (3) Loop on every operation and add a constraint for each
    for (Operation &op : funcInfo.funcOp.getOps()) {
      if (isa_and_nonnull<handshake::ConditionalBranchOp>(op)) {
        // TODO: FIND A WAY TO SYSTEMATICALLY CALCULATE PROBABILITIES, E.G., IN
        // SHANNON, IT CAN BE PRODUCT WHERE THE BB IS IDENTIFIED BY THE BRANCH
        // COND??
      } else if (isa_and_nonnull<handshake::MuxOp>(op)) {

      } else if (isa_and_nonnull<handshake::MergeOp>(op)) {
      } else if (isa_and_nonnull<handshake::ControlMergeOp>(op)) {

      } else {
        // everything else including Forks and all arithmetic units add a
        // constraint equating their input and output channels

        // All inputs = to each other
        // All outputs = to each other
        // One input is = to one output
        // Effectively, the above means that the rates of all inputs and outputs
        // should match

        for (auto operand : op.getOperands()) {
          model->addConstr(
              vars.ayaChannelThroughputs[operand].throughput ==
                  vars.ayaChannelThroughputs[op.getOperands()[0]].throughput,
              getUniqueName(*operand.getUses().begin()) + "&" +
                  getUniqueName(*op.getOperands()[0].getUses().begin()));
        }

        for (auto res : op.getResults()) {
          model->addConstr(
              vars.ayaChannelThroughputs[res].throughput ==
                  vars.ayaChannelThroughputs[op.getResults()[0]].throughput,
              getUniqueName(*res.getUses().begin()) + "&" +
                  getUniqueName(*op.getResults()[0].getUses().begin()));
        }

        model->addConstr(
            vars.ayaChannelThroughputs[op.getOperands()[0]].throughput ==
                vars.ayaChannelThroughputs[op.getResults()[0]].throughput,
            getUniqueName(*op.getOperands()[0].getUses().begin()) + "&" +
                getUniqueName(*op.getResults()[0].getUses().begin()));
      }
    }

    // TODO: AS AN INITIAL TEST, I NEED A WAY TO PRINT ALL OF THE CONSTRAINTS
    // AND POSSIBLY THE VARIABLES OF THE MODEL!!!

    // (4) Enumerate cycles
    CycleList circuitCycles = findAllCycles(funcInfo.funcOp);

    // (5) Loop on every cycle and calculate the latency and let it constrain a
    // particular channel throughput as a new constraint

    // (6) Define an objective function
    // Identify the global output channel that represents the return of the
    // function because the throughput of it is what will go in the objective
    // function
    Value globalOutputChannel;
    for (Operation &op : funcInfo.funcOp.getOps()) {
      if (isa_and_nonnull<handshake::EndOp>(op))
        globalOutputChannel = op.getOperands()[0];
    }
    // TODO: Add Objective

  } else {
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
  }

  markReadyToOptimize();
}
