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
#include "gurobi_c.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include <boost/functional/hash/extensions.hpp>
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

const std::map<unsigned int, double> ADD_SUB_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.6}, {16, 0.7}, {32, 1.0}};

const std::map<unsigned int, double> COMPARATOR_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.8}, {16, 1.0}, {32, 1.2}};

static double getDelay(const std::map<unsigned int, double> &delayTable,
                       unsigned int bitwidth) {
  auto it = delayTable.lower_bound(bitwidth);
  if (it == delayTable.end() || it->first != bitwidth) {
    it = delayTable.upper_bound(bitwidth);
  }
  return it != delayTable.end() ? it->second : 0.0;
}

void MAPBUFBuffers::addBlackboxConstraints(Value channel) {
  Operation *definingOp = channel.getDefiningOp();
  std::map<unsigned int, double> delays;

  // Blackbox constraints are only added for ADDI, SUBI and CMPI operations
  if (isa_and_present<handshake::AddIOp>(definingOp) ||
      isa_and_present<handshake::SubIOp>(definingOp)) {
    delays = ADD_SUB_DELAYS;
  } else if (isa_and_present<handshake::CmpIOp>(definingOp)) {
    delays = COMPARATOR_DELAYS;
  } else {
    return;
  }

  auto bitwidth =
      handshake::getHandshakeTypeBitWidth(definingOp->getOperand(0).getType());

  // Components are blackboxed only if their bitwidth is more than 4
  if (bitwidth <= 4)
    return;

  double delay = getDelay(delays, bitwidth);

  for (unsigned int i = 0; i < definingOp->getNumOperands(); i++) {
    // Looping over the input channels of the blackbox operation
    Value inputChannel = definingOp->getOperand(i);

    // Path In variable of the channel that comes after blackbox module (output
    // of blackbox)
    GRBVar &outputPathIn =
        vars.channelVars[channel].signalVars[SignalType::DATA].path.tIn;

    // Path Out variable of the channel that comes before blackbox module (input
    // of blackbox)
    GRBVar &inputPathOut =
        vars.channelVars[inputChannel].signalVars[SignalType::DATA].path.tOut;

    // Delay propagation constraint for blackbox nodes. Delay propagates through
    // input edges to output edges, increasing by delay variable.
    model.addConstr(inputPathOut + delay == outputPathIn,
                    "blackbox_constraint_" + std::to_string(bitwidth));
  }
  model.update();
}

void MAPBUFBuffers::addCustomChannelConstraints(Value channel) {
  // Get channel-specific buffering properties and channel's variables
  handshake::ChannelBufProps &props = channelProps[channel];
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

void MAPBUFBuffers::addCutSelectionConstraints(
    std::vector<experimental::Cut> &cutVector) {
  GRBLinExpr cutSelectionSum = 0;
  for (size_t i = 0; i < cutVector.size(); ++i) {
    // Loop over cuts of the node
    auto &cut = cutVector[i];
    // Add cut selection variable to the Gurobi model
    GRBVar &cutSelection = cut.getCutSelectionVariable();
    cutSelection = model.addVar(
        0, GRB_INFINITY, 0, GRB_BINARY,
        (cut.getNode()->str() + "__CutSelection_" + std::to_string(i)));
    cutSelectionSum += cutSelection;
  }
  model.update();
  // Cut Selection Constraint. Only a single cut of a node can be selected.
  // This affects delay propagation, as delay will propagate through the chosen
  // cut.
  model.addConstr(cutSelectionSum == 1, "cut_selection_constraint");
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

void MAPBUFBuffers::addCutSelectionConflicts(experimental::Node *root,
                                             experimental::Node *leaf,
                                             GRBVar &cutSelectionVar) {
  // Get the path from the leaf to the root
  std::vector<experimental::Node *> path;
  path = getPath(root, leaf, leafToRootPaths, blifData);
  // Loop over edges in the path from the leaf to the root.
  for (auto &nodePath : path) {
    if (nodePath->value) {
      // Add the Cut Selection Conflict Constraints. An LUT cannot cover an edge
      // if it is cut by a buffer, as LUTs cannot cover multiple sequential
      // stages. This constraint ensures an edge is either covered by a LUT, or
      // a buffer is inserted on the edge.
      model.addConstr(1 >= nodePath->gurobiVars->bufferVar + cutSelectionVar,
                      "cut_selection_conflict");
    }
  }
}

void MAPBUFBuffers::addCutLoopbackBuffers() {
  auto funcOp = funcInfo.funcOp;
  // Loop over all of the Channels
  funcOp.walk([&](mlir::Operation *op) {
    for (Value channel : op->getResults()) {
      for (Operation *user : channel.getUsers()) {
        if (isBackedge(channel, user)) {
          // Add buffer insertion constraints to the Gurobi model
          handshake::ChannelBufProps &resProps = channelProps[channel];
          if (resProps.maxTrans.value_or(1) >= 1) {
            GRBVar &bufVar = vars.channelVars[channel]
                                 .signalVars[SignalType::READY]
                                 .bufPresent;
            model.addConstr(bufVar == 1, "backedge_ready");
          }
          if (resProps.maxOpaque.value_or(1) >= 1) {
            GRBVar &bufVar = vars.channelVars[channel]
                                 .signalVars[SignalType::DATA]
                                 .bufPresent;
            model.addConstr(bufVar == 1, "backedge_data");
          }

          // Insert buffers in the Subject Graph
          experimental::BufferSubjectGraph::createAndInsertNewBuffer(
              op, user, "one_slot_break_dvr");
        }
      }
    }
  });
}

void MAPBUFBuffers::findMinimumFeedbackArcSet() {
  // Create a new Gurobi Model
  GRBEnv envFeedback = GRBEnv(true);
  envFeedback.set(GRB_IntParam_OutputFlag, 0);
  envFeedback.start();
  GRBModel modelFeedback = GRBModel(envFeedback);

  int numOps = 0;

  // Maps operations to GRBVars that holds the topological order index of MLIR
  // Operations
  DenseMap<Operation *, GRBVar> opToGRB;

  funcInfo.funcOp.walk([&](Operation *op) {
    // Create a Gurobi variable for each operation, which will hold the order of
    // the Operation in the topological ordering
    ++numOps;
    StringRef uniqueName = getUniqueName(op);
    GRBVar operationVariable = modelFeedback.addVar(
        0, GRB_INFINITY, 0.0, GRB_INTEGER, uniqueName.str());
    opToGRB[op] = operationVariable;
  });

  modelFeedback.update();

  DenseMap<std::pair<Operation *, Operation *>, GRBVar> edgeToOps;

  funcInfo.funcOp.walk([&](Operation *op) {
    // Add the constraint that forces topological ordering among adjacent
    // operations
    for (Operation *user : op->getUsers()) {
      GRBVar currentOpVar = opToGRB[op];
      GRBVar userOpVar = opToGRB[user];
      GRBVar edge = modelFeedback.addVar(
          0, 1, 0.0, GRB_BINARY,
          (getUniqueName(op) + "_" + getUniqueName(user)).str());
      edgeToOps[std::make_pair(op, user)] = edge;
      model.update();
      // This constraint enforces topological order, by forcing successor
      // operations to have a bigger larger index in the topological order than
      // their predecessors. It such an order cannot be satisfied with the given
      // set of nodes, "edge" variable is set to 1, which means the edge needs
      // to be cut to have an acyclic graph.
      modelFeedback.addConstr(userOpVar - currentOpVar + bigConstant * edge >=
                                  1,
                              "operation_order");
    }
  });

  modelFeedback.update();

  // Minimize the number of edges that needs to be removed to
  GRBLinExpr obj = 0;
  for (const auto &entry : edgeToOps) {
    obj += entry.second;
  }

  modelFeedback.setObjective(obj, GRB_MINIMIZE);

  modelFeedback.update();

  // Solve the model
  modelFeedback.optimize();

  // Loop over Gurobi Variables (edgeVar) to see which Channels are chosen
  // to be cut with buffers.
  for (const auto &entry : edgeToOps) {
    auto ops = entry.first;
    auto edgeVar = entry.second;
    auto *inputOp = ops.first;
    auto *outputOp = ops.second;
    if (edgeVar.get(GRB_DoubleAttr_X) > 0) {
      for (Value channel : outputOp->getOperands()) {
        // no buffers on MCs because they form self loops
        if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
            !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
          if (channel.getDefiningOp() == inputOp) {
            handshake::ChannelBufProps &resProps = channelProps[channel];
            // If the channel is chosen to be cut by MFAS, add buffer placement
            // constraints to the original MAPBUF Buffer placement Gurobi Model
            // as well
            if (resProps.maxTrans.value_or(1) >= 1) {
              GRBVar &bufVar = vars.channelVars[channel]
                                   .signalVars[SignalType::READY]
                                   .bufPresent;
              model.addConstr(bufVar == 1, "backedge_ready");
            }
            if (resProps.maxOpaque.value_or(1) >= 1) {
              GRBVar &bufVar = vars.channelVars[channel]
                                   .signalVars[SignalType::DATA]
                                   .bufPresent;
              model.addConstr(bufVar == 1, "backedge_data");
            }

            // Insert buffers in the Subject Graph
            experimental::BufferSubjectGraph::createAndInsertNewBuffer(
                inputOp, outputOp, "one_slot_break_dvr");
          }
        }
      }
    }
  }
}

void MAPBUFBuffers::addClockPeriodConstraintsNodes() {
  for (auto *node : blifData->getNodesInTopologicalOrder()) {
    GRBVar &nodeVarIn = node->gurobiVars->tIn;
    GRBVar &nodeVarOut = node->gurobiVars->tOut;
    GRBVar &bufVarSignal = node->gurobiVars->bufferVar;

    if (Value nodeChannel = node->value) {
      std::string nodeName = node->str();
      SignalType signalType =
          nodeName.find("ready") != std::string::npos   ? SignalType::READY
          : nodeName.find("valid") != std::string::npos ? SignalType::VALID
                                                        : SignalType::DATA;

      ChannelSignalVars &signalVars =
          vars.channelVars[nodeChannel].signalVars[signalType];

      nodeVarIn = signalVars.path.tIn;
      nodeVarOut = signalVars.path.tOut;
      bufVarSignal = signalVars.bufPresent;

      model.addConstr(nodeVarIn <= targetPeriod, "pathIn_period");
      model.addConstr(nodeVarOut <= targetPeriod, "pathOut_period");
      model.addConstr(nodeVarOut - nodeVarIn + bigConstant * bufVarSignal >= 0,
                      "buf_delay");
    } else {
      // Create the timing variable for Subject Graph Node. If the node is a
      // Primary Input, the delay is 0. Also, there is only one timing variable
      // is needed for these Nodes, therefore nodeVarOut and nodeVarIn are the
      // same.
      nodeVarIn =
          model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node->str());
      nodeVarOut = nodeVarIn;
      model.addConstr(nodeVarIn <= (node->isPrimaryInput() ? 0 : targetPeriod),
                      node->isPrimaryInput() ? "input_delay"
                                             : "clock_period_constraint");
    }
  }
  model.update();
}

void MAPBUFBuffers::addDelayAndCutConflictConstraints(
    experimental::Node *root, std::vector<experimental::Cut> &cutVector) {
  // Using cuts map to loop over subject graph edges, and adds delay
  // propagation constraints to the nodes that have cuts
  GRBVar &nodeVar = root->gurobiVars->tIn;
  std::set<experimental::Node *> fanIns = root->fanins;

  if (fanIns.size() == 1) {
    // If a node has single fanin, then it is not mapped to LUT. The
    // delay of the node is simply equal to the delay of the fanin.
    GRBVar &faninVar = (*fanIns.begin())->gurobiVars->tOut;
    model.addConstr(nodeVar == faninVar, "single_fanin_delay");
    return;
  }

  for (auto &cut : cutVector) {
    // Loop over the cuts of the subject graph edge
    GRBVar &cutSelectionVar = cut.getCutSelectionVariable();
    auto addDelayPropagationConstraint = [&](experimental::Node *leaf,
                                             const char *name) {
      GRBVar &leafVar = leaf->gurobiVars->tOut;
      // Add delay propagation constraint
      model.addConstr(nodeVar + (1 - cutSelectionVar) * bigConstant >=
                          leafVar + lutDelay,
                      name);
    };

    auto &leaves = cut.getLeaves();

    // Trivial cut. Delay is propagated from the fanins of the root
    if ((leaves.size() == 1) && (*leaves.begin() == root)) {
      for (auto &fanIn : fanIns) {
        addDelayPropagationConstraint(fanIn, "trivial_cut_delay");
      }
      continue;
    }

    // Loop over leaves of the cut
    for (auto *leaf : leaves) {
      addDelayPropagationConstraint(leaf, "delay_propagation");
      // Add cut selection conflict constraints for the root
      addCutSelectionConflicts(root, leaf, cutSelectionVar);
    }
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
  for (auto &[channel, _] : channelProps) {
    // Create channel variables and constraints
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);
    addBlackboxConstraints(channel);

    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addBufferPresenceConstraints(channel);
      addBufferingGroupConstraints(channel, bufGroups);
    }
  }

  // Generates Subject Graphs
  experimental::subjectGraphGenerator(funcInfo.funcOp, blifFiles);

  if (!acyclicType) {
    addCutLoopbackBuffers();
  } else {
    findMinimumFeedbackArcSet();
  }

  blifData = experimental::connectSubjectGraphs();

  auto cuts = experimental::generateCuts(blifData, lutSize);

  addClockPeriodConstraintsNodes();

  for (auto &[rootNode, cutVector] : cuts) {
    addCutSelectionConstraints(cutVector);
    addDelayAndCutConflictConstraints(rootNode, cutVector);
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
