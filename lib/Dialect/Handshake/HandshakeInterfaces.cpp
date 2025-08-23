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

/// Returns the name of an operand which is either provided by the
/// handshake::NamedIOInterface interface  or, failing that, is its index.
std::string getOperandName(Operation *op, size_t oprdIdx) {

  if(auto nameInterface = dyn_cast<handshake::CustomNamedIOInterface>(op)){
    return nameInterface.getOperandName(oprdIdx);
  } else if (auto nameInterface = dyn_cast<handshake::SimpleNamedIOInterface>(op)) {
    return nameInterface.getOperandName(oprdIdx);
  } else if (auto nameInterface = dyn_cast<handshake::ArithNamedIOInterface>(op)) {
    return nameInterface.getOperandName(oprdIdx);
  };

  op->emitError("all operations must specify operand names");
  assert(0);
}

/// Returns the name of a result which is either provided by the
/// handshake::NamedIOInterface interface or, failing that, is its index.
std::string getResultName(Operation *op, size_t resIdx) {

  if(auto nameInterface = dyn_cast<handshake::CustomNamedIOInterface>(op)){
    return nameInterface.getResultName(resIdx);
  } else if (auto nameInterface = dyn_cast<handshake::SimpleNamedIOInterface>(op)) {
    return nameInterface.getResultName(resIdx);
  } else if (auto nameInterface = dyn_cast<handshake::ArithNamedIOInterface>(op)) {
    return nameInterface.getResultName(resIdx);
  };

  op->emitError("all operations must specify result names");
  assert(0);
}

}
}

//===----------------------------------------------------------------------===//
// Operand and Result Names to Port Names
//===----------------------------------------------------------------------===//

PortNamer::PortNamer(Operation *op) {
  assert(op && "cannot generate port names for null operation");

  // special case: input and output port names
  // are actually stored in dictionary attributes
  if (auto funcOp = dyn_cast<handshake::FuncOp>(op)){
    llvm::transform(funcOp.getArgNames(), std::back_inserter(inputs),
                    [](Attribute arg) { return cast<StringAttr>(arg).str(); });
    llvm::transform(funcOp.getResNames(), std::back_inserter(outputs),
                    [](Attribute res) { return cast<StringAttr>(res).str(); });
  } else {
    // all other operations must directly provide names for their
    // inputs and outputs

    for (size_t idx = 0, e = op->getNumOperands(); idx < e; ++idx)
      inputs.push_back(getOperandName(op, idx));
    for (size_t idx = 0, e = op->getNumResults(); idx < e; ++idx)
      outputs.push_back(getResultName(op, idx));

    // The Handshake terminator forwards its non-memory inputs to its outputs, so
    // it needs port names for them
    if (handshake::EndOp endOp = dyn_cast<handshake::EndOp>(op)) {
      handshake::FuncOp funcOp = endOp->getParentOfType<handshake::FuncOp>();
      assert(funcOp && "end must be child of handshake function");
      size_t numResults = funcOp.getFunctionType().getNumResults();
      for (size_t idx = 0; idx < numResults; ++idx)
        outputs.push_back(detail::simpleResultName(idx, numResults));
    }
  }
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


#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
