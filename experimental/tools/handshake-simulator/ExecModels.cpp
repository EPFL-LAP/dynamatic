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

using namespace mlir;
using namespace circt;
using namespace dynamatic::experimental;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

SmallVector<Value> toVector(ValueRange range) {
  return SmallVector<Value>(range.begin(), range.end());
}

// Returns whether the precondition holds for a general op to execute
bool isReadyToExecute(ArrayRef<Value> ins, ArrayRef<Value> outs,
                      llvm::DenseMap<Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

// Fetch values from the value map and consume them
std::vector<llvm::Any> fetchValues(ArrayRef<Value> values,
                                   llvm::DenseMap<Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].has_value());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

// Store values to the value map
void storeValues(std::vector<llvm::Any> &values, ArrayRef<Value> outs,
                 llvm::DenseMap<Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

// Update the time map after the execution
void updateTime(ArrayRef<Value> ins, ArrayRef<Value> outs,
                llvm::DenseMap<Value, double> &timeMap, double latency) {
  double time = 0;
  for (auto &in : ins)
    time = std::max(time, timeMap[in]);
  time += latency;
  for (auto &out : outs)
    timeMap[out] = time;
}

/// Wrapper method for constant time simple operations that just update the
/// output and do not modify the timeMap or valueMap in a very specific way
bool tryToExecute(
    circt::Operation *op, llvm::DenseMap<Value, llvm::Any> &valueMap,
    llvm::DenseMap<Value, double> &timeMap, SmallVector<Value> &scheduleList,
    ModelMap &models,
    const std::function<void(std::vector<llvm::Any> &, std::vector<llvm::Any> &,
                             circt::Operation &)> &executeFunc,
    double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (isReadyToExecute(ins, outs, valueMap)) {
    auto in = fetchValues(ins, valueMap);
    std::vector<llvm::Any> out(outs.size());
    executeFunc(in, out, *op);
    storeValues(out, outs, valueMap);
    updateTime(ins, outs, timeMap, latency);
    scheduleList = outs;
    return true;
  }
  return false;
}

} // namespace

/*
  - 'modelStructuresMap' stores all known models and identifies each structure
      a string name -> any exec model structure
  - 'funcMap' stores the entered configuration, parsed json + --change-model cmd
      mlir operation name -> string name correspond to a exec model struct
  - 'models' links both maps, so the configuration and all known models
      mlir operation name -> configurated exec model structure
*/
bool dynamatic::experimental::initialiseMap(
    llvm::StringMap<std::string> &funcMap, ModelMap &models) {
  // This maps the configuration file / command string name to it's
  // corresponding structure
  ModelMap modelStructuresMap;
  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE BELOW MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  // ------------------------------------------------------------------------ //
  modelStructuresMap["defaultFork"] = std::make_unique<DefaultFork>();
  modelStructuresMap["defaultMerge"] = std::make_unique<DefaultMerge>();
  modelStructuresMap["defaultMux"] = std::make_unique<DefaultMux>();
  modelStructuresMap["defaultBranch"] = std::make_unique<DefaultBranch>();
  modelStructuresMap["defaultSink"] = std::make_unique<DefaultSink>();
  modelStructuresMap["defaultConstant"] = std::make_unique<DefaultConstant>();
  modelStructuresMap["defaultBuffer"] = std::make_unique<DefaultBuffer>();
  modelStructuresMap["defaultConditionalBranch"] =
      std::make_unique<DefaultConditionalBranch>();
  modelStructuresMap["defaultControlMerge"] =
      std::make_unique<DefaultControlMerge>();
  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE ABOVE MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  // ------------------------------------------------------------------------ //

  // Fill the map containing the final execution models structures
  for (auto &elem : funcMap) {
    auto &chosenStruct = modelStructuresMap[elem.getValue()];
    // Stop the program if the corresponding struct wasn't found
    if (!chosenStruct)
       return false;
    models[elem.getKey().str()] = std::move(chosenStruct);
  }
  return true;
}

//===----------------------------------------------------------------------===//
//                     Execution models definitions
//===----------------------------------------------------------------------===//

namespace dynamatic {
namespace experimental {

// Default CIRCT fork
bool DefaultFork::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                             llvm::DenseMap<unsigned, unsigned> &memoryMap,
                             llvm::DenseMap<Value, double> &timeMap,
                             std::vector<std::vector<llvm::Any>> &store,
                             SmallVector<Value> &scheduleList, ModelMap &models,
                             circt::Operation &opArg) {

  auto op = dyn_cast<circt::handshake::ForkOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, circt::Operation &op) {
    for (auto &out : outs)
      out = ins[0];
  };
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, executeFunc, 1);
}

// Default CIRCT merge
bool DefaultMerge::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                              llvm::DenseMap<unsigned, unsigned> &memoryMap,
                              llvm::DenseMap<Value, double> &timeMap,
                              std::vector<std::vector<llvm::Any>> &store,
                              SmallVector<Value> &scheduleList,
                              ModelMap &models, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MergeOp>(opArg);
  bool found = false;
  for (Value in : op.getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        op.emitOpError("More than one valid input to Merge!");
      auto t = valueMap[in];

      valueMap[op.getResult()] = t;
      timeMap[op.getResult()] = timeMap[in];
      // Consume the inputs.
      valueMap.erase(in);
      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to Merge!");
  scheduleList.push_back(op.getResult());
  return true;
}

// Default CIRCT control merge
bool DefaultControlMerge::tryExecute(
    llvm::DenseMap<Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    SmallVector<Value> &scheduleList, ModelMap &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ControlMergeOp>(opArg);
  bool found = false;
  for (auto in : llvm::enumerate(op.getOperands())) {
    if (valueMap.count(in.value()) == 1) {
      if (found)
        op.emitOpError("More than one valid input to CMerge!");
      valueMap[op.getResult()] = valueMap[in.value()];
      timeMap[op.getResult()] = timeMap[in.value()];
      valueMap[op.getIndex()] =
          APInt(IndexType::kInternalStorageBitWidth, in.index());
      timeMap[op.getIndex()] = timeMap[in.value()];

      // Consume the inputs.
      valueMap.erase(in.value());

      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to CMerge!");
  scheduleList = toVector(op.getResults());
  return true;
}

// Default CIRCT mux
bool DefaultMux::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                            llvm::DenseMap<unsigned, unsigned> &memoryMap,
                            llvm::DenseMap<Value, double> &timeMap,
                            std::vector<std::vector<llvm::Any>> &store,
                            SmallVector<Value> &scheduleList, ModelMap &models,
                            circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MuxOp>(opArg);
  Value control = op.getSelectOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  auto opIdx = llvm::any_cast<APInt>(controlValue).getZExtValue();
  assert(opIdx < op.getDataOperands().size() &&
         "Trying to select a non-existing mux operand");

  Value in = op.getDataOperands()[opIdx];
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  double time = std::max(controlTime, inTime);
  valueMap[op.getResult()] = inValue;
  timeMap[op.getResult()] = time;

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  scheduleList.push_back(op.getResult());
  return true;
}

// Default CIRCT branch
bool DefaultBranch::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                               llvm::DenseMap<unsigned, unsigned> &memoryMap,
                               llvm::DenseMap<Value, double> &timeMap,
                               std::vector<std::vector<llvm::Any>> &store,
                               SmallVector<Value> &scheduleList,
                               ModelMap &models, circt::Operation &opArg) {
  llvm::errs() << "[EXECMODELS] in the DEFAULT branch"
               << "\n";
  auto op = dyn_cast<circt::handshake::BranchOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        circt::Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, executeFunc, 0);
}

// Default CIRCT conditional branch
bool DefaultConditionalBranch::tryExecute(
    llvm::DenseMap<Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    SmallVector<Value> &scheduleList, ModelMap &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConditionalBranchOp>(opArg);
  Value control = op.getConditionOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  Value in = op.getDataOperand();
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op.getTrueResult()
                                                       : op.getFalseResult();
  double time = std::max(controlTime, inTime);
  valueMap[out] = inValue;
  timeMap[out] = time;
  scheduleList.push_back(out);

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  return true;
}

// Default CIRCT sink
bool DefaultSink::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                             llvm::DenseMap<unsigned, unsigned> &memoryMap,
                             llvm::DenseMap<Value, double> &timeMap,
                             std::vector<std::vector<llvm::Any>> &store,
                             SmallVector<Value> &scheduleList, ModelMap &models,
                             circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::SinkOp>(opArg);
  valueMap.erase(op.getOperand());
  return true;
}

// Default CIRCT constant
bool DefaultConstant::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                                 llvm::DenseMap<unsigned, unsigned> &memoryMap,
                                 llvm::DenseMap<Value, double> &timeMap,
                                 std::vector<std::vector<llvm::Any>> &store,
                                 SmallVector<Value> &scheduleList,
                                 ModelMap &models, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConstantOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, circt::Operation &op) {
    auto attr = op.getAttrOfType<IntegerAttr>("value");
    outs[0] = attr.getValue();
  };
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, executeFunc, 0);
}

// Default CIRCT buffer
bool DefaultBuffer::tryExecute(llvm::DenseMap<Value, llvm::Any> &valueMap,
                               llvm::DenseMap<unsigned, unsigned> &memoryMap,
                               llvm::DenseMap<Value, double> &timeMap,
                               std::vector<std::vector<llvm::Any>> &store,
                               SmallVector<Value> &scheduleList,
                               ModelMap &models, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::BufferOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        circt::Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, executeFunc, op.getNumSlots());
}

} // namespace experimental
} // namespace dynamatic
