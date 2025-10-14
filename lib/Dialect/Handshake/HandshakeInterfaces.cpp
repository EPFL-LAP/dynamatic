//===- HandshakeInterfaces.cpp - Handshake interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of Handshake dialect's interfaces' methods for specific
// Handshake operations.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {
namespace handshake {

//===----------------------------------------------------------------------===//
// Operand and Result Names
//===----------------------------------------------------------------------===//

std::string getOperandName(Operation *op, size_t oprdIdx) {

  if (auto nameInterface = dyn_cast<handshake::CustomNamedIOInterface>(op)) {
    return nameInterface.getOperandName(oprdIdx);
  } else if (auto nameInterface =
                 dyn_cast<handshake::SimpleNamedIOInterface>(op)) {
    return nameInterface.getOperandName(oprdIdx);
  } else if (auto nameInterface =
                 dyn_cast<handshake::BinaryArithNamedIOInterface>(op)) {
    return nameInterface.getOperandName(oprdIdx);
  }

  op->emitError() << "must specify operand names, op: " << *op;
  assert(0);
}

std::string getResultName(Operation *op, size_t resIdx) {

  if (auto nameInterface =
                 dyn_cast<handshake::SimpleNamedIOInterface>(op)) {
    return nameInterface.getResultName(resIdx);
  } else if (auto nameInterface =
                 dyn_cast<handshake::BinaryArithNamedIOInterface>(op)) {
    return nameInterface.getResultName(resIdx);
  } else if (auto nameInterface = dyn_cast<handshake::CustomNamedIOInterface>(op)) {
    return nameInterface.getResultName(resIdx);
  } 

  op->emitError() << "must specify result names, op: " << *op;
  assert(0);
}

std::string getInputPortName(Operation *op, size_t portIdx) {
  if(auto operandPortsInterface = 
              dyn_cast<handshake::InputRTLPortsAreOperandsInterface>(op)){
    return operandPortsInterface.getInputPortName(portIdx);
  } else if (auto customPortsInterface = 
              dyn_cast<handshake::CustomRTLInputPortsInterface>(op)){
    return customPortsInterface.getInputPortName(portIdx);
  }

  op->emitError("All operations must specify input ports");
  assert(0);
}

std::string getOutputPortName(Operation *op, size_t portIdx) {
  if(auto resultsPortsInterface = 
              dyn_cast<handshake::OutputRTLPortsAreResultsInterface>(op)){
    return resultsPortsInterface.getOutputPortName(portIdx);
  } else if (auto customPortsInterface = 
              dyn_cast<handshake::CustomRTLOutputPortsInterface>(op)){
    return customPortsInterface.getOutputPortName(portIdx);
  }

  op->emitError("All operations must specify output ports");
  assert(0);
}


} // namespace handshake
} // namespace dynamatic


//===----------------------------------------------------------------------===//
// MemoryOpInterface
//===----------------------------------------------------------------------===//

bool MemoryControllerOp::isMasterInterface() { return true; }

bool LSQOp::isMasterInterface() { return !isConnectedToMC(); }

TypedValue<MemRefType> LSQOp::getMemRef() {
  if (handshake::MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemRef();
  return cast<TypedValue<MemRefType>>(getInputs().front());
}

TypedValue<ControlType> LSQOp::getMemStart() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemStart();
  return cast<TypedValue<ControlType>>(getOperand(1));
}

TypedValue<ControlType> LSQOp::getMemEnd() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemStart();
  return cast<TypedValue<ControlType>>(getResults().back());
}

TypedValue<ControlType> LSQOp::getCtrlEnd() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getCtrlEnd();
  return cast<TypedValue<ControlType>>(getOperands().back());
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
