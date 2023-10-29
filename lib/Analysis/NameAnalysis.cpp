//===- NameAnalysis.cpp - Uniquely name all IR operations -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the name analysis infrastructure.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

inline static handshake::NameAttr getNameAttr(Operation *op) {
  return op->getAttrOfType<handshake::NameAttr>(
      handshake::NameAttr::getMnemonic());
}

/// Returns the name of a result which is either provided by the
/// handshake::NamedIOInterface interface or, failing that, is its index.
static std::string getResultName(Operation *op, size_t resIdx) {
  std::string oprName;
  if (auto namedIO = dyn_cast<handshake::NamedIOInterface>(op))
    return namedIO.getResultName(resIdx);
  return std::to_string(resIdx);
}

/// Returns the name of an operand which is either provided by the
/// handshake::NamedIOInterface interface  or, failing that, is its index.
static std::string getOperandName(Operation *op, size_t oprdIdx) {
  std::string oprName;
  if (auto namedIO = dyn_cast<handshake::NamedIOInterface>(op))
    return namedIO.getOperandName(oprdIdx);
  return std::to_string(oprdIdx);
}

/// When getting the "producer part" of an operand's name and it is a block
/// argument, we can only derive a name when it is a function's argument of some
/// sort. Succeeds and sets the two strings, respectively, to the function's
/// name and the argument's name when `val` is a function argument; fails
/// otherwise.
static LogicalResult getSourceNameFromFunc(Value val, std::string &defName,
                                           std::string &resName) {
  // If the parent is some kind of function, use the function name as the
  // channel source name
  BlockArgument arg = cast<BlockArgument>(val);
  Operation *parentOp = arg.getParentBlock()->getParentOp();
  if (auto funcOp = dyn_cast<handshake::FuncOp>(parentOp)) {
    defName = funcOp.getNameAttr().str();
    resName = funcOp.getArgName(arg.getArgNumber()).str();
    return success();
  }
  if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
    defName = funcOp.getNameAttr().str();
    resName = std::to_string(arg.getArgNumber());
    return success();
  }
  return failure();
}

LogicalResult NameAnalysis::getName(Operation *op, StringRef &name) {
  auto nameIt = opToName.find(op);
  if (nameIt == opToName.end())
    return failure();
  name = nameIt->getSecond();
  return success();
}

LogicalResult NameAnalysis::getName(OpOperand &oprd, std::string &name) {
  // The defining operation must have a name (if the defining operation is a
  // Handshake function, the function symbol is used instead)
  std::string defName, resName;
  Value val = oprd.get();
  if (Operation *defOp = val.getDefiningOp()) {
    if (auto defNameIt = opToName.find(defOp); defNameIt != opToName.end()) {
      defName = defNameIt->second;
      resName = getResultName(defOp, cast<OpResult>(val).getResultNumber());
    } else {
      return failure();
    }
  } else if (failed(getSourceNameFromFunc(val, defName, resName))) {
    return failure();
  }

  // The user operation must have a name
  std::string userName, oprName;
  Operation *userOp = oprd.getOwner();
  if (auto userNameIt = opToName.find(userOp); userNameIt != opToName.end()) {
    userName = userNameIt->second;
    oprName = getOperandName(userOp, oprd.getOperandNumber());
  } else {
    return failure();
  }

  name = defName + "_" + resName + "_" + oprName + "_" + userName;
  return success();
}

StringRef NameAnalysis::setName(Operation *op) {
  assert(namesValid && "analysis invariant is broken");
  StringRef existingName;
  if (succeeded(getName(op, existingName)))
    // If the operation already has a name, do nothing and return this one
    return existingName;

  // Set the attribute on the operation and update our mapping
  std::string name = genUniqueName(op->getName());
  auto newAttr = handshake::NameAttr::get(op->getContext(), name);
  op->setAttr(handshake::NameAttr::getMnemonic(), newAttr);
  addMapping(op, name);
  return opToName[op];
}

LogicalResult NameAnalysis::setName(Operation *op, StringRef name,
                                    bool uniqueWhenTaken) {
  assert(namesValid && "analysis invariant is broken");
  assert(!name.empty() && "name can't be empty");
  if (handshake::NameAttr attr = getNameAttr(op))
    // If the operation already has a name, this is a failure
    return failure();

  std::string uniqueName = name.str();
  if (nameToOp.contains(name)) {
    if (!uniqueWhenTaken)
      return failure();
    uniqueName = deriveUniqueName(uniqueName);
  }

  // Set the attribute on the operation and update our mapping
  auto newAttr = handshake::NameAttr::get(op->getContext(), uniqueName);
  op->setAttr(handshake::NameAttr::getMnemonic(), newAttr);
  addMapping(op, uniqueName);
  return success();
}

LogicalResult NameAnalysis::setName(Operation *op, Operation *ascendant,
                                    bool uniqueWhenTaken) {
  assert(namesValid && "analysis invariant is broken");
  if (handshake::NameAttr attr = getNameAttr(op))
    // If the operation already has a name, this is a failure
    return failure();

  auto fromNameIt = opToName.find(ascendant);
  if (fromNameIt == opToName.end())
    // If the ascendant operation doesn't have a name, this is a failure
    return failure();

  std::string uniqueName = genUniqueName(op->getName()) + fromNameIt->second;
  if (nameToOp.contains(uniqueName)) {
    if (!uniqueWhenTaken)
      return failure();
    uniqueName = deriveUniqueName(uniqueName);
  }

  // Set the attribute on the operation and update our mapping
  auto newAttr = handshake::NameAttr::get(op->getContext(), uniqueName);
  op->setAttr(handshake::NameAttr::getMnemonic(), newAttr);
  addMapping(op, uniqueName);
  return success();
}

LogicalResult NameAnalysis::walk(UnnamedBehavior onUnnamed) {
  // Reset the flags
  allOpsNamed = true;
  namesValid = true;

  topLevelOp->walk([&](Operation *nestedOp) {
    // Do not name the top-level module
    if (isa<mlir::ModuleOp>(nestedOp))
      return;

    // Functions already have a unique name; store it in our mapping so that we
    // avoid naming conflicts in case a smart cookie decides one day to name
    // their function "merge0"
    if (auto funcOp = dyn_cast<func::FuncOp>(nestedOp)) {
      addMapping(funcOp, funcOp.getNameAttr());
      return;
    }
    if (auto funcOp = dyn_cast<handshake::FuncOp>(nestedOp)) {
      addMapping(funcOp, funcOp.getNameAttr());
      return;
    }

    handshake::NameAttr attr = getNameAttr(nestedOp);
    if (!attr) {
      // Check what we must do when we encounter an unnamed operation
      switch (onUnnamed) {
      case UnnamedBehavior::DO_NOTHING:
        allOpsNamed = false;
        break;
      case UnnamedBehavior::NAME:
        setName(nestedOp);
        break;
      case UnnamedBehavior::FAIL:
        allOpsNamed = false;
        // All operations must be named
        nestedOp->emitError()
            << "Operation has no name. All operations must be uniquely named.";
        break;
      }
      return;
    }

    StringRef name = attr.getName();

    // Names cannot change during operations' lifetimes
    if (auto nameIt = opToName.find(nestedOp); nameIt != opToName.end()) {
      if (nameIt->getSecond() != name) {
        nestedOp->emitError()
            << "Operation's name was '" << nameIt->second << "' but now is '"
            << name
            << "'. The name of an operation cannot change during its lifetime.";
        namesValid = false;
        return;
      }
    }

    // Names must be unique
    if (auto opIt = nameToOp.find(name); opIt != nameToOp.end()) {
      if (opIt->second != nestedOp) {
        nestedOp->emitError() << "Operation has name '" << name
                              << "' but another operation already has this "
                                 "name. Names must be unique.";
        namesValid = false;
        return;
      }
    } else {
      addMapping(nestedOp, name);
    }
  });

  return success(namesValid &&
                 (onUnnamed != UnnamedBehavior::FAIL || allOpsNamed));
}

std::string NameAnalysis::genUniqueName(mlir::OperationName opName) {
  std::string prefix = opName.stripDialect().str();
  std::string candidate;
  do {
    candidate = prefix + std::to_string(counters[opName]++);
  } while (nameToOp.contains(candidate));
  return candidate;
}

std::string NameAnalysis::deriveUniqueName(StringRef base) {
  unsigned counter = 0;
  std::string candidate;
  do {
    candidate = base.str() + std::to_string(counter++);
  } while (nameToOp.contains(candidate));
  return candidate;
}

std::string dynamatic::getUniqueName(Operation *op) {
  if (handshake::NameAttr attr = getNameAttr(op))
    return attr.getName().str();
  return "";
}

std::string dynamatic::getUniqueName(OpOperand &oprd) {
  // The defining operation must have a name (if the defining operation is a
  // Handshake function, the function symbol is used instead)
  std::string defName, resName;
  Value val = oprd.get();
  if (Operation *defOp = val.getDefiningOp()) {
    if (handshake::NameAttr attr = getNameAttr(defOp)) {
      defName = attr.getName().str();
      resName = getResultName(defOp, cast<OpResult>(val).getResultNumber());
    } else {
      return "";
    }
  } else if (failed(getSourceNameFromFunc(val, defName, resName))) {
    return "";
  }

  // The user operation must have a name
  std::string userName, oprName;
  Operation *userOp = oprd.getOwner();
  if (handshake::NameAttr attr = getNameAttr(userOp)) {
    userName = attr.getName().str();
    oprName = getOperandName(userOp, oprd.getOperandNumber());
  } else {
    return "";
  }

  return defName + "_" + resName + "_" + oprName + "_" + userName;
}