//===- HandshakeExecutableOps.cpp - Handshake executable Operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of execution semantics for Handshake
// operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/tools/handshake-simulator/ExecModels.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

using namespace circt;

/*
PROBLEMS & TODO
  - A bit of 'boileplate' and similar code (struct definition & declaration..)
  - Some functions don't have an 'execute' functions, but removing the virtual
    status of it makes the struct without 'execute' abstract, so impossible
    to store in our map
  - Boring dyn_cast, which is mandatory for correct overriding I think
    (Maybe C++ polymorphism can be exploited better ?)
  - Might want can get rid of the tryExecute+execute system which isn't very
    clear and modular-friendly (Have to redesign a bit more then!)
  - Might want to centralize models map and structures so that users only
    code in one file
  - If configuration isn't found, maybe we can automaticaly substitude with
    the default configuration ?
*/

#define INDEX_WIDTH 32

using ModelMap =
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>>;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

// Convert ValueRange to vectors
std::vector<mlir::Value> toVector(mlir::ValueRange range) {
  return std::vector<mlir::Value>(range.begin(), range.end());
}

// Returns whether the precondition holds for a general op to execute
bool isReadyToExecute(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                      llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

// Fetch values from the value map and consume them
std::vector<llvm::Any>
fetchValues(ArrayRef<mlir::Value> values,
            llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].has_value());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

// Store values to the value map
void storeValues(std::vector<llvm::Any> &values, ArrayRef<mlir::Value> outs,
                 llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

// Update the time map after the execution
void updateTime(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                llvm::DenseMap<mlir::Value, double> &timeMap, double latency) {
  double time = 0;
  for (auto &in : ins)
    time = std::max(time, timeMap[in]);
  time += latency;
  for (auto &out : outs)
    timeMap[out] = time;
}

} // namespace

bool dynamatic::experimental::tryToExecute(
    circt::Operation *op, llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (isReadyToExecute(ins, outs, valueMap)) {
    auto in = fetchValues(ins, valueMap);
    std::vector<llvm::Any> out(outs.size());

    auto opName = op->getName().getStringRef().str();
    auto &execModel = models[opName];
    if (!execModel)
      op->emitOpError("Undefined execution for the current op");

    models[opName].get()->execute(in, out, *op);
    storeValues(out, outs, valueMap);
    updateTime(ins, outs, timeMap, latency);
    scheduleList = outs;
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                     Execution models definitions                           //
//===----------------------------------------------------------------------===//

namespace dynamatic {
namespace experimental {
  
} // namespace experimental
} // namespace dynamatic