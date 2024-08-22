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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace mlir;
using namespace dynamatic;

/// Shortcut to get the name attribute of an operation.
inline static mlir::StringAttr getNameAttr(Operation *op) {
  return op->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
}

/// If the operation has an intrinsic name, returns it. Returns an empty
/// `StringRef` if the operation does not have an intrinsic name.
static StringRef getIntrinsicName(Operation *op) {
  // Functions already have a unique name; store it in our mapping so that we
  // avoid naming conflicts in case a smart cookie decides one day to name
  // their function "merge0"
  if (func::FuncOp funcOp = dyn_cast<func::FuncOp>(op))
    return funcOp.getNameAttr().strref();
  if (handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(op))
    return funcOp.getNameAttr().strref();
  return StringRef();
}

/// Returns the index of the region and block the block argument belongs to in
/// its parent operaion's region and block list, respectively.
static std::pair<unsigned, unsigned>
findRegionAndBlockNumber(BlockArgument arg) {
  Block *parentBlock = arg.getParentBlock();
  Operation *parentOp = parentBlock->getParentOp();
  Region *parentRegion = arg.getParentRegion();
  for (auto [regionIdx, region] : llvm::enumerate(parentOp->getRegions())) {
    if (&region != parentRegion)
      continue;

    for (auto [blockIdx, block] : llvm::enumerate(region)) {
      if (&block == parentBlock)
        return {regionIdx, blockIdx};
    }
  }
  llvm_unreachable("failed to find region and/or block");
}

/// Internal version of `getBlockArgName`, which returns false if the parent
/// operation doesn't have a known name (as specified in `parentOpName`).
/// Returns true otherwise.
static bool tryToGetBlockArgName(BlockArgument arg, StringRef parentOpName,
                                 std::string &prodName, std::string &resName) {
  return llvm::TypeSwitch<Operation *, bool>(
             arg.getParentBlock()->getParentOp())
      .Case<handshake::FuncOp>([&](handshake::FuncOp funcOp) {
        prodName = funcOp.getNameAttr().str();
        resName = funcOp.getArgName(arg.getArgNumber()).str();
        return true;
      })
      .Case<func::FuncOp>([&](func::FuncOp funcOp) {
        prodName = funcOp.getNameAttr().str();
        resName = std::to_string(arg.getArgNumber());
        return true;
      })
      .Default([&](Operation *parentOp) {
        if (parentOpName.empty())
          return false;
        prodName = parentOpName;
        auto [regionNumber, blockNumber] = findRegionAndBlockNumber(arg);
        resName =
            std::to_string(regionNumber) + "_" + std::to_string(blockNumber);
        return true;
      });
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

StringRef NameAnalysis::getName(Operation *op) {
  assert(namesValid && "analysis invariant is broken");
  // If the operation already has a name or is intrinsically named , do nothing
  // and return the name
  if (mlir::StringAttr attr = getNameAttr(op))
    return attr;
  if (StringRef name = getIntrinsicName(op); !name.empty())
    return name;

  // Set the attribute on the operation and update our mapping
  std::string name = genUniqueName(op->getName());
  op->setAttr(ATTR_NAME, StringAttr::get(op->getContext(), name));
  namedOperations[name] = op;
  return getNameAttr(op);
}

bool NameAnalysis::hasName(Operation *op) { return getNameAttr(op) != nullptr; }

std::string NameAnalysis::getName(OpOperand &oprd) {
  // The defining operation must have a name (if the defining operation is a
  // Handshake function, the function symbol is used instead)
  std::string defName, resName;
  Value val = oprd.get();
  if (Operation *defOp = val.getDefiningOp()) {
    defName = getName(defOp);
    resName = getResultName(defOp, cast<OpResult>(val).getResultNumber());
  } else {
    getBlockArgName(cast<BlockArgument>(val), defName, resName);
  }

  // The user operation must have a name
  std::string userName, oprName;
  Operation *userOp = oprd.getOwner();
  userName = getName(userOp);
  oprName = getOperandName(userOp, oprd.getOperandNumber());
  return defName + "_" + resName + "_" + oprName + "_" + userName;
}

bool NameAnalysis::replaceOp(Operation *op, Operation *newOp) {
  assert(namesValid && "analysis invariant is broken");
  mlir::StringAttr attr = getNameAttr(op);

  // Derive an entirely new name for the operation
  if (!attr) {
    setName(newOp);
    return false;
  }

  // Transfer the name to the new operation and update our internal mapping
  op->removeAttr(ATTR_NAME);
  newOp->setAttr(ATTR_NAME, attr);
  namedOperations[attr.strref()] = newOp;
  return true;
}

LogicalResult NameAnalysis::setName(Operation *op, StringRef name,
                                    bool uniqueWhenTaken) {
  assert(namesValid && "analysis invariant is broken");
  assert(!name.empty() && "name can't be empty");
  // The operation cannot already have a name or be intrinsically named
  if (mlir::StringAttr attr = getNameAttr(op))
    return failure();
  if (isIntrinsicallyNamed(op))
    return failure();

  std::string uniqueName = name.str();
  if (namedOperations.contains(name)) {
    if (!uniqueWhenTaken)
      return failure();
    uniqueName = deriveUniqueName(uniqueName);
  }

  // Set the attribute on the operation and update our mapping
  auto newAttr = mlir::StringAttr::get(op->getContext(), uniqueName);
  op->setAttr(ATTR_NAME, newAttr);
  namedOperations[uniqueName] = op;
  return success();
}

LogicalResult NameAnalysis::setName(Operation *op, Operation *ascendant,
                                    bool uniqueWhenTaken) {
  assert(namesValid && "analysis invariant is broken");
  // The operation cannot already have a name or be intrinsically named
  if (mlir::StringAttr attr = getNameAttr(op))
    return failure();
  if (isIntrinsicallyNamed(op))
    return failure();

  StringRef ascendantName = getName(ascendant);
  std::string uniqueName = genUniqueName(op->getName()) + ascendantName.str();
  if (namedOperations.contains(uniqueName)) {
    if (!uniqueWhenTaken)
      return failure();
    uniqueName = deriveUniqueName(uniqueName);
  }

  // Set the attribute on the operation and update our mapping
  auto newAttr = mlir::StringAttr::get(op->getContext(), uniqueName);
  op->setAttr(ATTR_NAME, newAttr);
  namedOperations[uniqueName] = op;
  return success();
}

LogicalResult NameAnalysis::walk(UnnamedBehavior onUnnamed) {
  // Reset the flags
  allOpsNamed = true;
  namesValid = true;

  topLevelOp->walk([&](Operation *nestedOp) {
    // Do not name the top-level module or intrinsically named operations
    if (isa<mlir::ModuleOp>(nestedOp) || isIntrinsicallyNamed(nestedOp))
      return;

    mlir::StringAttr attr = getNameAttr(nestedOp);
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

    // Check that the name is unique with respect to other knwon operations
    if (auto namedOp = namedOperations.find(attr);
        namedOp != namedOperations.end()) {
      if (namedOp->second != nestedOp) {
        nestedOp->emitError() << "Operation has name " << attr
                              << " but another operation already has this "
                                 "name. Names must be unique.";
        namesValid = false;
        return;
      }
    } else {
      namedOperations[attr] = nestedOp;
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
  } while (namedOperations.contains(candidate));
  return candidate;
}

std::string NameAnalysis::deriveUniqueName(StringRef base) {
  unsigned counter = 0;
  std::string candidate;
  do {
    candidate = base.str() + std::to_string(counter++);
  } while (namedOperations.contains(candidate));
  return candidate;
}

bool NameAnalysis::isIntrinsicallyNamed(Operation *op) {
  if (StringRef name = getIntrinsicName(op); !name.empty()) {
    namedOperations[name] = op;
    return true;
  }
  return false;
}

void NameAnalysis::getBlockArgName(BlockArgument arg, std::string &prodName,
                                   std::string &resName) {
  tryToGetBlockArgName(arg, getName(arg.getParentBlock()->getParentOp()),
                       prodName, resName);
}

StringRef dynamatic::getUniqueName(Operation *op) {
  if (mlir::StringAttr attr = getNameAttr(op))
    return attr;
  if (StringRef name = getIntrinsicName(op); !name.empty())
    return name;
  return StringRef();
}

std::string dynamatic::getUniqueName(OpOperand &oprd) {
  // The defining operation must have a name (if the defining operation is a
  // Handshake function, the function symbol is used instead)
  std::string defName, resName;
  Value val = oprd.get();
  if (Operation *defOp = val.getDefiningOp()) {
    if (mlir::StringAttr attr = getNameAttr(defOp)) {
      defName = attr.str();
      resName = getResultName(defOp, cast<OpResult>(val).getResultNumber());
    } else {
      return "";
    }
  } else if (!tryToGetBlockArgName(
                 cast<BlockArgument>(val),
                 getUniqueName(val.getParentBlock()->getParentOp()), defName,
                 resName)) {
    return "";
  }

  // The user operation must have a name
  std::string userName, oprName;
  Operation *userOp = oprd.getOwner();
  if (mlir::StringAttr attr = getNameAttr(userOp)) {
    userName = attr.str();
    oprName = getOperandName(userOp, oprd.getOperandNumber());
  } else {
    return "";
  }

  return defName + "_" + resName + "_" + oprName + "_" + userName;
}
