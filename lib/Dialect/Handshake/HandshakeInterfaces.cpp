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

//===----------------------------------------------------------------------===//
// PortNameGenerator (uses NamedIOInterface)
//===----------------------------------------------------------------------===//

PortNamer::PortNamer(Operation *op) {
  assert(op && "cannot generate port names for null operation");
  if (auto funcOp = dyn_cast<handshake::FuncOp>(op)){
    inferFromFuncOp(funcOp);
  } else {
    inferFromInterface(op);
  }
}

void PortNamer::inferFromInterface(Operation *op) {
  IdxToStrF inF, outF;
  if(auto nameInterface = dyn_cast<handshake::CustomNamedIOInterface>(op)){
    inF = [&](unsigned idx) { return nameInterface.getOperandName(idx); };
    outF = [&](unsigned idx) { return nameInterface.getResultName(idx); };
  } else if (auto nameInterface = dyn_cast<handshake::SimpleNamedIOInterface>(op)) {
    inF = [&](unsigned idx) { return nameInterface.getOperandName(idx); };
    outF = [&](unsigned idx) { return nameInterface.getResultName(idx); };
  } else if (auto nameInterface = dyn_cast<handshake::ArithNamedIOInterface>(op)) {
    inF = [&](unsigned idx) { return nameInterface.getOperandName(idx); };
    outF = [&](unsigned idx) { return nameInterface.getResultName(idx); };
  } else {
    op->emitError("all normal operations must specify port names");
    assert(false);
  }

  for (size_t idx = 0, e = op->getNumOperands(); idx < e; ++idx)
    inputs.push_back(inF(idx));
  for (size_t idx = 0, e = op->getNumResults(); idx < e; ++idx)
    outputs.push_back(outF(idx));

  // The Handshake terminator forwards its non-memory inputs to its outputs, so
  // it needs port names for them
  if (handshake::EndOp endOp = dyn_cast<handshake::EndOp>(op)) {
    handshake::FuncOp funcOp = endOp->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "end must be child of handshake function");
    size_t numResults = funcOp.getFunctionType().getNumResults();
    for (size_t idx = 0, e = numResults; idx < e; ++idx)
      outputs.push_back(detail::simpleOutputPortName(idx));
  }
}

void PortNamer::inferFromFuncOp(handshake::FuncOp funcOp) {
  llvm::transform(funcOp.getArgNames(), std::back_inserter(inputs),
                  [](Attribute arg) { return cast<StringAttr>(arg).str(); });
  llvm::transform(funcOp.getResNames(), std::back_inserter(outputs),
                  [](Attribute res) { return cast<StringAttr>(res).str(); });
}

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

std::string simpleInputPortName(unsigned idx) {
  return "ins_" + std::to_string(idx);
}

std::string simpleOutputPortName(unsigned idx) {
  return "outs_" + std::to_string(idx);
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
