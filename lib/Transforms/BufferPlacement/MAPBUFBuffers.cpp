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
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/BlifReader.h"
#include "experimental/Support/CutEnumeration.h"
#include "experimental/Support/SubjectGraph.h"
#include "gurobi_c.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include <boost/functional/hash/extensions.hpp>
#include <omp.h>
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
                             double targetPeriod, StringRef blifFiles)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod),
      blifFiles(blifFiles) {
  if (!unsatisfiable)
    setup();
}

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, StringRef blifFiles,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName),
      blifFiles(blifFiles) {
  if (!unsatisfiable)
    setup();
}

void MAPBUFBuffers::extractResult(BufferPlacement &placement) {
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

    handshake::ChannelBufProps &props = channelProps[channel];
    PlacementResult result;
    if (placeOpaque && placeTransparent) {
      // Place the minumum number of opaque slots; at least one and enough to
      // satisfy all our opaque/transparent requirements
      if (props.maxTrans) {
        // We must place enough opaque slots as to not exceed the maximum number
        // of transparent slots
        result.numOpaque =
            std::max(props.minOpaque, numSlotsToPlace - *props.maxTrans);
      } else {
        // At least one slot, but no more than necessary
        result.numOpaque = std::max(props.minOpaque, 1U);
      }
      // All remaining slots are transparent
      result.numTrans = numSlotsToPlace - result.numOpaque;
    } else if (placeOpaque) {
      // Place the minimum number of transparent slots; at least the expected
      // minimum and enough to satisfy all our opaque/transparent requirements
      if (props.maxOpaque) {
        result.numTrans =
            std::max(props.minTrans, numSlotsToPlace - *props.maxOpaque);
      } else {
        result.numTrans = props.minTrans;
      }
      // All remaining slots are opaque
      result.numOpaque = numSlotsToPlace - result.numTrans;
    } else {
      // placeOpaque == 0 --> props.minOpaque == 0 so all slots can be
      // transparent
      result.numTrans = numSlotsToPlace;
    }

    result.deductInternalBuffers(Channel(channel), timingDB);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

bool isOperationType(Operation *op, std::string_view type) {
  return getUniqueName(op).find(type) != std::string::npos;
}

const std::map<unsigned int, double> ADD_SUB_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.6}, {16, 0.7}, {32, 1.0}};

const std::map<unsigned int, double> COMPARATOR_DELAYS = {
    {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.8}, {16, 1.0}, {32, 1.2}};

double getDelay(const std::map<unsigned int, double> &delayTable,
                unsigned int bitwidth) {
  auto it = delayTable.lower_bound(bitwidth);
  if (it == delayTable.end() || it->first != bitwidth) {
    it = delayTable.upper_bound(bitwidth);
  }
  return it != delayTable.end() ? it->second : 0.0;
}

void MAPBUFBuffers::addBlackboxConstraints(Value channel) {
  Operation *definingOp = channel.getDefiningOp();
  bool isCmpi = false;

  if (!definingOp) {
    return;
  }

  // Blackbox constraints are only added for ADDI, SUBI and CMPI operations
  // Need a bool for CMPI as it has a different delay than ADDI and SUBI
  if (isa<handshake::AddIOp>(definingOp) ||
      isa<handshake::SubIOp>(definingOp)) {
    // Do nothing for AddIOp and SubIOp
  } else if (isa<handshake::CmpIOp>(definingOp)) {
    isCmpi = true;
  } else {
    return;
  }

  for (unsigned int i = 0; i < definingOp->getNumOperands(); i++) {
    // Looping over the input channels of the blackbox operation
    Value inputChannel = definingOp->getOperand(i);

    // Skip mapping to blackboxes for operations with bitwidth <= 4.
    unsigned int bitwidth =
        handshake::getHandshakeTypeBitWidth(inputChannel.getType());
    if (bitwidth <= 4) {
      break;
    }

    ChannelVars &inputChannelVars = vars.channelVars[inputChannel];
    ChannelVars &outputChannelVars = vars.channelVars[channel];

    // Path In variable of the channel that comes after blackbox module (output
    // of blackbox)
    GRBVar &outputPathIn =
        outputChannelVars.signalVars[SignalType::DATA].path.tIn;

    // Path Out variable of the channel that comes before blackbox module (input
    // of blackbox)
    GRBVar &inputPathOut =
        inputChannelVars.signalVars[SignalType::DATA].path.tOut;

    if (isCmpi) {
      double delay = getDelay(COMPARATOR_DELAYS, bitwidth);
      model.addConstr(inputPathOut + delay == outputPathIn,
                      "cmpi_constraint_" + std::to_string(bitwidth));
    } else { // addi or subi
      double delay = getDelay(ADD_SUB_DELAYS, bitwidth);
      model.addConstr(inputPathOut + delay == outputPathIn,
                      "add_sub_constraint_" + std::to_string(bitwidth));
    }
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
    // Initialize the Gurobi variable for the cut selection
    GRBVar &cutSelection = cut.getCutSelectionVariable();
    cutSelection = model.addVar(
        0, GRB_INFINITY, 0, GRB_BINARY,
        (cut.getNode()->str() + "__CutSelection_" + std::to_string(i)));
    cutSelectionSum += cutSelection;
  }
  model.update();
  model.addConstr(cutSelectionSum == 1, "cut_selection_constraint");
}

std::vector<experimental::Node *>
getOrCreateLeafToRootPath(experimental::Node *key, experimental::Node *leaf,
                          pathMap &leafToRootPaths,
                          experimental::BlifData *blif) {

  // Check if the path from leaf to root has already been computed, if so then
  // returns it. If not, returns the shortest path by running BFS.
  auto leafKeyPair = std::make_pair(leaf, key);
  if (leafToRootPaths.find(leafKeyPair) != leafToRootPaths.end()) {
    return leafToRootPaths[leafKeyPair];
  }

  auto path = blif->findPath(leaf, key);
  if (!path.empty()) {
    // remove the starting node and the root node, as we should be able to place
    // buffers on channels adjacent to these nodes
    path.pop_back();
    path.erase(path.begin());
  }

  leafToRootPaths[leafKeyPair] = path;
  return path;
}

void MAPBUFBuffers::addCutSelectionConflicts(experimental::Node *root,
                                             experimental::Node *leaf,
                                             GRBVar &cutSelectionVar) {
  // Get the path from the leaf to the root
  std::vector<experimental::Node *> path;
  path = getOrCreateLeafToRootPath(root, leaf, leafToRootPaths, blifData);
  for (auto &nodePath : path) {
    // Loop over edges in the path from the leaf to the root. Add cut
    // selection conflict constraints for channels that are on the
    // path.
    if (nodePath->gurobiVars->bufferVar.has_value()) {
      model.addConstr(1 >= nodePath->gurobiVars->bufferVar.value() +
                               cutSelectionVar,
                      "cut_selection_conflict");
    }
  }
}

bool isChannelVar(const std::string &node) {
  // a hacky way to determine if a variable is a channel variable.
  // if it includes "new", "." and does not include "_", it is not a channel
  // variable
  return node.find("new") == std::string::npos &&
         node.find('.') == std::string::npos &&
         node.find('_') != std::string::npos;
}

std::ostringstream retrieveChannelName(const std::string &node,
                                       const std::string &variableType) {
  if (!isChannelVar(node)) {
    return std::ostringstream{node};
  }

  std::string variableTypeName;
  if (variableType == "buffer")
    variableTypeName = "BufPresent_";
  else if (variableType == "pathIn")
    variableTypeName = "PathIn_";
  else if (variableType == "pathOut")
    variableTypeName = "PathOut_";

  std::stringstream ss(node);
  std::string token;
  std::vector<std::string> result;

  while (std::getline(ss, token, '_')) {
    result.emplace_back(token);
  }

  const auto leafLastUnderscore = node.find_last_of('_');
  const auto leafNodeNameTillUnderScore = node.substr(0, leafLastUnderscore);
  const auto channelTypeName = node.substr(leafLastUnderscore + 1);

  std::ostringstream varNameStream;
  if (result.back() == "valid" || result.back() == "ready") {
    varNameStream << (result.back() == "valid" ? ("valid" + variableTypeName)
                                               : ("ready" + variableTypeName))
                  << leafNodeNameTillUnderScore;
  } else {
    const auto leafLastPar = node.find_last_of('[');
    const auto leafNodeNameTillPar = node.substr(0, leafLastPar);
    varNameStream << ("data" + variableTypeName) << leafNodeNameTillPar;
  }
  return varNameStream;
}

std::optional<GRBVar> variableExists(GRBModel &model,
                                     const std::string &varName) {
  GRBVar *vars = model.getVars();
  int numVars = model.get(GRB_IntAttr_NumVars);

  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    if (vars[i].get(GRB_StringAttr_VarName).find(varName) !=
        std::string::npos) {
      return vars[i]; // Variable exists
    }
  }
  return {}; // Variable does not exist
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
          experimental::BufferSubjectGraph::createBuffers(op, user);
        }
      }
    }
  });
}

void MAPBUFBuffers::findMinimumFeedbackArcSet() {
  GRBEnv envFeedback = GRBEnv(true);
  envFeedback.set(GRB_IntParam_OutputFlag, 0);
  envFeedback.start();
  GRBModel modelFeedback = GRBModel(envFeedback);

  int numOps = 0;
  DenseMap<Operation *, GRBVar> opToGRB;

  funcInfo.funcOp.walk([&](Operation *op) {
    // create a Gurobi variable for each operation, which will hold the order of
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

  modelFeedback.optimize();

  for (const auto &entry : edgeToOps) {
    auto edgeVar = entry.second;
    auto ops = entry.first;
    auto *inputOp = ops.first;
    auto *outputOp = ops.second;
    if (edgeVar.get(GRB_DoubleAttr_X) > 0) {
      for (Value channel : outputOp->getOperands()) {
        if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
            !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
          // no buffers on MCs because they form self loops
          if (channel.getDefiningOp() == inputOp) {
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
            experimental::BufferSubjectGraph::createBuffers(inputOp, outputOp);
          }
        }
      }
    }
  }
}

void MAPBUFBuffers::addClockPeriodConstraintsNodes() {
  // Lambda function to retrieve and search for Gurobi variables.
  auto retrieveAndSearchGRBVar =
      [&](const std::string &node,
          const std::string &variableType) -> std::optional<GRBVar> {
    std::string channelName = retrieveChannelName(node, variableType).str();
    return variableExists(model, channelName);
  };

  for (auto *node : blifData->getNodesInOrder()) {
    experimental::MILPVarsSubjectGraph *vars = node->gurobiVars;
    GRBVar &nodeVarIn = vars->tIn;
    GRBVar &nodeVarOut = vars->tOut;
    std::optional<GRBVar> &nodeBufVar = vars->bufferVar;

    if (node->isChannelEdgeNode()) {
      // If a Subject Graph Edge is also a DFG edge, then Gurobi variables for
      // it was already created in addChannelVars(). Here, we retrieve those
      // Gurobi variables by doing a search on the Gurobi variables. If found,
      // we assign these Gurobi Variables to the nodeVarIn, nodeVarOut and
      // nodeBufVar, which are member variables of Node Class.

      std::optional<GRBVar> pathInVar =
          retrieveAndSearchGRBVar(node->str(), "pathIn");
      std::optional<GRBVar> pathOutVar =
          retrieveAndSearchGRBVar(node->str(), "pathOut");
      std::optional<GRBVar> bufferVar =
          retrieveAndSearchGRBVar(node->str(), "buffer");

      if (pathInVar.has_value() && pathOutVar.has_value() &&
          bufferVar.has_value()) {
        // Just to make sure that the variables are not empty.
        nodeVarIn = pathInVar.value();
        nodeVarOut = pathOutVar.value();
        nodeBufVar = bufferVar;
        continue;
      }
    }

    // Create the timing variable for Subject Graph Node. If the node is a
    // Primary Input, the delay is 0. Also, there is only one timing variable is
    // needed for these Nodes, therefore nodeVarOut and nodeVarIn are the same.
    nodeVarIn = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node->str());
    nodeVarOut = nodeVarIn;
    if (node->isPrimaryInput()) {
      model.addConstr(nodeVarIn == 0, "input_delay");
    } else {
      model.addConstr(nodeVarIn <= targetPeriod, "clock_period_constraint");
    }

    model.update();
  }
}

void MAPBUFBuffers::addClockPeriodConstraintsChannels(Value channel,
                                                      SignalType signal) {
  ChannelVars &channelVars = vars.channelVars[channel];
  std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
  ChannelSignalVars &signalVars = channelVars.signalVars[signal];
  GRBVar &bufVarSignal =
      vars.channelVars[channel].signalVars[signal].bufPresent;
  GRBVar &t1 = signalVars.path.tIn;
  GRBVar &t2 = signalVars.path.tOut;

  model.addConstr(t1 <= targetPeriod, "pathIn_period");
  model.addConstr(t2 <= targetPeriod, "pathOut_period");
  model.addConstr(t2 - t1 + bigConstant * bufVarSignal >= 0, "buf_delay");
}

void MAPBUFBuffers::connectSubjectGraphs() {
  for (auto &module : experimental::BaseSubjectGraph::subjectGraphMap) {
    module.first->connectInputNodes();
  }

  experimental::BlifData *mergedBlif = new experimental::BlifData();

  mergedBlif->setModuleName("merged");

  for (auto &module : experimental::BaseSubjectGraph::subjectGraphMap) {
    experimental::BlifData *blifModule = module.first->getBlifData();
    for (auto &latch : blifModule->getLatches()) {
      mergedBlif->addLatch(latch.first, latch.second);
    }
    for (auto &node : blifModule->getNodesInOrder()) {
      mergedBlif->addNode(node);
    }
  }

  // Sort the nodes of the newly created Merged BlifData in topological order
  mergedBlif->traverseNodes();
  blifData = mergedBlif;
}

void MAPBUFBuffers::addDelayPropagationConstraints(
    experimental::Node *root, std::vector<experimental::Cut> &cutVector) {
  // Using cuts map to loop over subject graph edges, and adds delay
  // propagation constraints to the nodes that have cuts
  GRBVar &nodeVar = root->gurobiVars->tIn;
  std::set<experimental::Node *> fanIns = root->getFanins();

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
      model.addConstr(nodeVar + (1 - cutSelectionVar) * bigConstant >=
                          leafVar + lutDelay,
                      name);
    };

    auto &leaves = cut.getLeaves();
    for (auto *leaf : leaves) {
      // Loop over leaves of the cut
      if ((leaves.size() == 1) && (leaf == root)) {
        // Trivial cut. Delay is propagated from the fanins of the root
        for (auto &fanIn : fanIns) {
          addDelayPropagationConstraint(fanIn, "trivial_cut_delay");
        }
        continue;
      }
      addDelayPropagationConstraint(leaf, "delay_propagation");
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
    addChannelElasticityConstraints(channel, bufGroups);
    addBlackboxConstraints(channel);

    for (SignalType signal : signals) {
      addClockPeriodConstraintsChannels(channel, signal);
    }
  }

  ChannelFilter channelFilter = [&](Value channel) -> bool {
    Operation *defOp = channel.getDefiningOp();
    return !isa_and_present<handshake::MemoryOpInterface>(defOp) &&
           !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
  };

  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitElasticityConstraints(&op, channelFilter);
  }

  // The generator class to initialize the subject graphs of the modules
  experimental::SubjectGraphGenerator generateSubjectGraph(funcInfo.funcOp,
                                                           blifFiles);

  // boolean to choose between different acyclic graph convertion methods
  bool acyclicType = true;
  if (!acyclicType) {
    addCutLoopbackBuffers();
  } else {
    findMinimumFeedbackArcSet();
  }

  connectSubjectGraphs();

  // Generates the cuts and saves them in the static
  // experimental::CutManager::cuts map
  experimental::CutManager generateCuts(blifData, 6);

  addClockPeriodConstraintsNodes();

  for (auto &[rootNode, cutVector] : experimental::CutManager::cuts) {
    addCutSelectionConstraints(cutVector);
    addDelayPropagationConstraints(rootNode, cutVector);
  }

  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    addChannelThroughputConstraints(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  addObjective(allChannels, cfdfcs);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
