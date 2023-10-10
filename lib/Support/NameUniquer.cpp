//===- NameUniquer.cpp - Unique Handshake ops and values --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the name uniquing infrastructure.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/NameUniquer.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace circt;
using namespace dynamatic;

/// Name "part separator" used in operands names.
const static std::string SEP = "_";

/// Returns the name of an operation, which is its fully qualified name striped
/// of the dialect name.
inline static std::string getOpName(Operation &op) {
  return op.getName().stripDialect().str();
}

/// Returns the name of an operand which is either provided by the
/// handshake::NamedIOInterface or, failing that, is its index.
static std::string getOperandName(Operation &op, size_t oprdIdx) {
  std::string oprName;
  if (auto namedIO = dyn_cast<handshake::NamedIOInterface>(&op))
    return namedIO.getOperandName(oprdIdx);
  return std::to_string(oprdIdx);
}

NameUniquer::NameUniquer(mlir::ModuleOp modOp) {
  DenseMap<mlir::OperationName, unsigned> opCounters;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
    registerFunc(funcOp, opCounters);
}

NameUniquer::NameUniquer(handshake::FuncOp funcOp) {
  DenseMap<mlir::OperationName, unsigned> opCounters;
  registerFunc(funcOp, opCounters);
}

StringRef NameUniquer::getName(Operation &op) {
  assert(opToName.contains(&op) && "operation does not have a name");
  return opToName[&op];
}

StringRef NameUniquer::getName(OpOperand &oprd) {
  assert(oprdToName.contains(&oprd) && "operand does not have a name");
  return oprdToName[&oprd];
}

Operation &NameUniquer::getOp(StringRef opName) {
  assert(nameToOp.contains(opName) && "name does not map to any operation");
  return *nameToOp[opName];
}

OpOperand &NameUniquer::getOperand(StringRef oprdName) {
  assert(nameToOprd.contains(oprdName) && "name does not map to any operand");
  return *nameToOprd[oprdName];
}

void NameUniquer::registerFunc(
    handshake::FuncOp funcOp,
    DenseMap<mlir::OperationName, unsigned> &opCounters) {
  // Give a unique name to each operation
  for (Operation &op : funcOp.getOps()) {
    opToName[&op] = getOpName(op) + std::to_string(opCounters[op.getName()]++);
    llvm::errs() << "Op: " << opToName[&op] << "\n";
    nameToOp[opToName[&op]] = &op;
  }

  // Name all operands corresponding to a range of uses
  auto nameAllOperands = [&](Value::use_range uses, StringRef prefix) -> void {
    for (auto [oprdIdx, oprd] : llvm::enumerate(uses)) {
      Operation *user = oprd.getOwner();
      oprdToName[&oprd] = prefix.str() + SEP + opToName[user] + SEP +
                          getOperandName(*user, oprdIdx);
      llvm::errs() << "Oprd: " << oprdToName[&oprd] << "\n";
      nameToOprd[oprdToName[&oprd]] = &oprd;
    }
  };

  // Give a unique name to each use of function argument
  std::string funcName = funcOp.getName().str();
  for (auto [funcArg, argAttr] :
       llvm::zip_equal(funcOp.getArguments(), funcOp.getArgNames())) {
    std::string argName = cast<mlir::StringAttr>(argAttr).str();
    nameAllOperands(funcArg.getUses(), funcName + SEP + argName);
  }

  // Give a unique name to each use of an operation result within the function
  for (Operation &op : funcOp.getOps()) {
    std::string srcName = opToName[&op];
    auto namedIO = dyn_cast<handshake::NamedIOInterface>(&op);
    for (auto [resIdx, res] : llvm::enumerate(op.getResults())) {
      std::string resName =
          namedIO ? namedIO.getResultName(resIdx) : std::to_string(resIdx);
      nameAllOperands(res.getUses(), srcName + SEP + resName);
    }
  }
}