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

void buffer::channel::print() {
  llvm::errs() << "opSrc: " << *(unitSrc->op) << " ---> ";
  llvm::errs() << "opDst: " << *(unitDst->op) << "\n";
}

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

  // initialize the unit, if the unit is initialized by its predecessor,
  // get from the list. Otherwise, create a new unit and add to the list
  unit *unitNode = getUnitWithOp(opNode, unitList);
  if (!unitNode) {
    unitNode = new unit(opNode);
    unitList.push_back(unitNode);
  }

  if (inChannel != nullptr)
    connectInChannel(unitNode, inChannel);

  // dfs the successor operation
  for (auto resOperand : opNode->getResults()) {
    // initialize the out port
    port *outPort = new port(&resOperand);
    for (auto sucOp : resOperand.getUsers()) {
      // create the channel connected to the outport
      unit *sucUnit  = getUnitWithOp(sucOp, unitList);
      // if the successor unit not exists, create a new one
      if (!sucUnit) {
        sucUnit = new unit(sucOp);
        unitList.push_back(sucUnit);
      }
      channel *outChannel = new channel(unitNode, sucUnit, &resOperand);
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