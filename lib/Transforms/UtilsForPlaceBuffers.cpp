//===- UtilsForPlaceBuffers.cpp - functions for placing buffer  -*- C++ -*-===//
//
// This file implements function supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

bool buffer::isEntryOp(Operation *op, std::vector<Operation *> &visitedOp) {
  for (auto operand : op->getOperands())
    if (!operand.getDefiningOp())
      return true;
  return false;
}

int buffer::getBBIndex(Operation *op) {
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == BB_ATTR)
      return dyn_cast<IntegerAttr>(attr.getValue()).getValue().getZExtValue();
  }
  return -1;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  if (opDst->isProperAncestor(opSrc))
    return true;

  return false;
}

unit *buffer::getUnitWithOp(Operation *op, std::vector<unit *> &unitList) {
  for (auto u : unitList) {
    if (u->op == op)
      return u;
  }
  return nullptr;
}

void buffer::connectInChannel(unit *unitNode, channel *inChannel) {
  port *inPort = new port(inChannel->valPort);
  inPort->cntChannels.push_back(inChannel);
  unitNode->inPorts.push_back(inPort);
}

void buffer::dfsHandshakeGraph(Operation *opNode, std::vector<unit *> &unitList,
                               std::vector<Operation *> &visited,
                               channel *inChannel) {

  // ensure inChannel is marked as connected to the unit
  if (inChannel != nullptr)
    if (unit *unitNode = getUnitWithOp(opNode, unitList); unitNode != nullptr) {
      connectInChannel(unitNode, inChannel);
    }

  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }

  // marked as visited
  visited.push_back(opNode);

  // initialize the unit node
  unit *unitNode = new unit(opNode);
  unitList.push_back(unitNode);
  if (inChannel != nullptr)
    connectInChannel(unitNode, inChannel);

  // dfs the successor operation
  for (auto resOperand : opNode->getResults()) {
    // initialize the out port
    port *outPort = new port(&resOperand);
    for (auto sucOp : resOperand.getUsers()) {
      // create the channel connected to the outport
      channel *outChannel = new channel(opNode, sucOp, &resOperand);
      outChannel->isBackEdge = isBackEdge(opNode, sucOp);
      outPort->cntChannels.push_back(outChannel);

      // dfs the successor operation
      dfsHandshakeGraph(sucOp, unitList, visited, outChannel);
    }
    unitNode->outPorts.push_back(outPort);
  }
}

/// ================== dataFlowCircuit Function ================== ///
void buffer::dataFlowCircuit::printCircuits() {
  for (auto unit : units) {
    llvm::errs() << "===========================\n";
    llvm::errs() << "operation: " << *(unit->op) << "\n";
    llvm::errs() << "-------------inPorts: \n";
    for (auto port : unit->outPorts)
      for (auto ch : port->cntChannels)
        ch->print();
    llvm::errs() << "-------------outPorts: \n";
    for (auto port : unit->inPorts)
      for (auto ch : port->cntChannels)
        ch->print();
  }
}

std::vector<std::vector<float>>
buffer::dataFlowCircuit::readInfoFromFile(const std::string &filename) {
  std::vector<std::vector<float>> info;

  std::ifstream file(filename);
  assert(file.is_open() && "Error opening delay info file");

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
      float num = std::stof(value);
      row.push_back(num);
    }

    assert(!row.empty() && "Error reading delay info file");
    info.push_back(row);
  }

  file.close();

  return info;
}

void buffer::dataFlowCircuit::createMILPVars(GRBModel &modelMILP, 
                          std::vector<unit *> &units,
                          std::vector<channel *> &channels,
                          std::vector<port *> &ports,
                          std::map<std::string, GRBVar> &timeVars,
                          std::map<std::string, GRBVar> &elasticVars,
                          std::map<std::string, GRBVar> &thrptVars,
                          std::map<std::string, GRBVar> &bufferVars,
                          std::map<std::string, GRBVar> &retimeVars,
                          std::map<std::string, GRBVar> &outputVars) {
  // clear the variables
  timeVars.clear();
  thrptVars.clear();
  bufferVars.clear();
  retimeVars.clear();
  // create the the circuit throuput variables
  thrptVars["thrpt"] = modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);

  for (size_t i = 0; i < units.size(); i++) {
    unit *unit = units[i];
    // create the retiming variables for the throughput
    retimeVars["retimeIn_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    retimeVars["retimeOut_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    retimeVars["retimeBubble_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);

    // time path defined in input port
    for (size_t j = 0; j < unit->inPorts.size(); j++) {
      port *inP = unit->inPorts[j];
      if (std::find(ports.begin(), ports.end(), inP) != ports.end()) {
        timeVars["timeIn_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeInValid_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeInReady_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        elasticVars["timeInElastic_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
      }
    }

    // time path defined in output port 
    for (size_t j = 0; j < unit->outPorts.size(); j++) {
      port *outP = unit->outPorts[j];
      // if the port is in the path, create time path variables
      if (std::find(ports.begin(), ports.end(), outP) != ports.end()) {
        timeVars["timeOut_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeOutValid_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeOutReady_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        elasticVars["timeOutElastic_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
      }
    }

  }

  // create throughput variables for the channels
  for(int i = 0; i < channels.size(); i++) {
    channel *channel = channels[i];
    unit *src = channel->opSrc;
    unit *dst = channel->opDst;
    std::string varName = std::to_string(i) + 
                          "_u" + findUnitIndex(src->op) + 
                          "_u" + findUnitIndex(dst->op);
    thrptVars["thTokens_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    thrptVars["thBubbles_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    thrptVars["bufferFlopValid_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    thrptVars["bufferFlopReady_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    outputVars["bufferFlop_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    outputVars["bufferSlots_" + varName] =
        modelMILP.addVar(0.0, INFINITY, 0.0, GRB_INTEGER);
    outputVars["hasBuffer_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
  }
}

static unit* findUnitOfVarName(std::vector<unit *> units, std::string varName) {
  size_t underscorePos1 = varName.find('_');
  size_t underscorePos2 = varName.find('_', underscorePos1 + 1);
  
  // Extract the substrings containing i and j
  std::string unitInd = varName.substr(underscorePos1 + 1, underscorePos2 - underscorePos1 - 1);

  return units[std::stoi(unitInd)];
}

static port* getSrcPort(channel *ch, std::vector<unit *> unitList ) {
  unit *srcUnit = getUnitWithOp(ch->opSrc, unitList);
  for (auto p : srcUnit->outPorts) 
    for (auto cntCh : p->cntChannels)
      if (cntCh == ch) 
        return p;
    
  return nullptr;
}

// static port* getDstPort(channel *ch, )


void buffer::dataFlowCircuit::createPathConstraints(GRBModel &modelMILP, 
                                  std::map<std::string, GRBVar> &timeVars,
                                  std::map<std::string, GRBVar> &bufferVars,
                                  double period) {
  // create the constraints for the path
  for (auto ch : channels) {
    // get srcPort val
    port *srcPort = getSrcPort(ch, this->units);
  }
  // for (auto const &timeVar : timeVars) {
  //   // Find the positions of the src units
  //   unit *srcUnit = findUnitOfVarName(units, timeVar.first);
  //   varName = timeVar.first;
  //       std::string outPortInd = varName.substr(underscorePos2 + 1);
  // }
}

void buffer::dataFlowCircuit::createMILPModel(BufferPlacementStrategy &strategy,
                          std::map<std::string, GRBVar> &outputVars) {
  // init the model
  GRBEnv env = GRBEnv(true);
  env.start();
  GRBModel modelMILP = GRBModel(env);

  double period = strategy.period;
  // double periodMax = strategy.periodMax;

  // internal variables
  std::map<std::string, GRBVar> timeVars;
  std::map<std::string, GRBVar> elasticVars;
  std::map<std::string, GRBVar> bufferVars;
  std::map<std::string, GRBVar> retimeVars;
  std::map<std::string, GRBVar> thrptVars;

  createMILPVars(modelMILP, this->units, this->channels, this->ports, 
                 timeVars, elasticVars, thrptVars, bufferVars, retimeVars, outputVars);

  createPathConstraints(modelMILP, timeVars, bufferVars, period);

  // create the variables
}