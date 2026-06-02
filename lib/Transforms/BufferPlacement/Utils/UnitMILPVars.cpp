//===- UnitMILPVars.cpp - Per-unit MILP variable storage --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for UnitVars: construction from an op's retiming-path partition
// and accessors that resolve operand/result indices to the owning path's
// retiming variables.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/Utils/UnitMILPVars.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

// builds the MILP unit-based variables for an operation
UnitVars::UnitVars(Operation *unit) {
  // get the possible retiming paths through a unit
  for (RetimingPath &descriptor : getRetimingPaths(unit))
    // and use the path to make a path variable struct
    retPathVarList.emplace_back(descriptor);
}

CPVar &UnitVars::getRetIn(unsigned operandIdx) {
  // for each retiming path
  for (RetPathVars &pathVars : retPathVarList) {
    // check if the operand belongs to it
    if (pathVars.operands.count(operandIdx)) {
      // if so
      // make sure it has a real operand retiming variable
      assert(pathVars.retIn.has_value() && "retIn not allocated for this path");
      // and return
      return *pathVars.retIn;
    }
  }
  // allow release mode optimizations
  // about assuming the operand belongs
  // to the last path if others are checked
  llvm_unreachable("operand index has no declared path");
}

CPVar &UnitVars::getRetOut(unsigned resultIdx) {
  // for each retiming path
  for (RetPathVars &pathVars : retPathVarList) {
    // check if the result belongs to it
    if (pathVars.results.count(resultIdx)) {
      // if so
      // make sure it has a real result retiming variable
      assert(pathVars.retOut.has_value() &&
             "retOut not allocated for this path");
      // and return
      return *pathVars.retOut;
    }
  }
  // allow release mode optimizations
  // about assuming the result belongs
  // to the last path if others are checked
  llvm_unreachable("result index has no declared path");
}

// small encapsulation function
llvm::SmallVector<CPVar> UnitVars::getInputRetimingVars() {
  llvm::SmallVector<CPVar> inputVars;
  for (RetPathVars &pathVars : retPathVarList) {
    // make sure the input retiming variable is a real retiming variable
    assert(pathVars.retIn.has_value() && "retIn not allocated for this path");
    inputVars.push_back(*pathVars.retIn);
  }
  return inputVars;
}

// small encapsulation function
llvm::SmallVector<CPVar> UnitVars::getOutputRetimingVars() {
  llvm::SmallVector<CPVar> outputVars;
  for (RetPathVars &pathVars : retPathVarList) {
    // make sure the output retiming variable is a real retiming variable
    assert(pathVars.retOut.has_value() && "retOut not allocated for this path");
    outputVars.push_back(*pathVars.retOut);
  }
  return outputVars;
}

void UnitVars::validate() const {
  for (const RetPathVars &path : retPathVarList) {
    assert(path.latency.has_value() &&
           "retiming-path latency must be set before adding constraints");
    assert(path.retIn.has_value() &&
           "retIn must be allocated before adding constraints");
    assert(path.retOut.has_value() &&
           "retOut must be allocated before adding constraints");
  }
}
