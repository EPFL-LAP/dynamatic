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
using namespace dynamatic::experimental;

/*
PROBLEMS & TODO
  - A bit of 'boileplate' and similar code (struct definition & declaration..)
  - (DONE) Some functions don't have an 'execute' functions, but removing the
virtual status of it makes the struct without 'execute' abstract, so impossible
    to store in our map
  - Boring dyn_cast, which is mandatory for correct overriding I think
    (Maybe C++ polymorphism can be exploited better ?)
  - Might want can get rid of the tryExecute+execute system which isn't very
    clear and modular-friendly (Have to redesign a bit more then!)
  - Might want to centralize models map and structures so that users only
    code in one file
  - If configuration isn't found, maybe we can automaticaly substitude with
    the default configuration ?
  - ModelMap type in .cpp but not in .h which is weird
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

bool tryToExecute(
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
} // namespace

/* MAYBE REMOVE THIS COMM, CURRENTLY HERE TO CLARIFY PR
  - 'modelStructuresMap' stores all known models and identifies each structure
      a string name -> any exec model structure
  - 'funcMap' stores the entered configuration, parsed json + --change-model cmd
      mlir operation name -> string name correspond to a exec model struct
  - 'models' links both maps, so the configuration and all known models
      mlir operation name -> configurated exec model structure

  This allows user to change its models by command or configuration file,
  with the only re-compilation being the one when he adds the structure
  and inserts it in the modelStructuresMap. There could be a better way to do it
  but this one allows dynamic changes.

  modelStructuresMap is the only piece of code the user needs to touch here.
*/
bool dynamatic::experimental::initialiseMap(
    llvm::StringMap<std::string> &funcMap, ModelMap &models) {
  // This maps the configuration file / command string name to it's
  // corresponding structure
  ModelMap modelStructuresMap;
  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE BELOW MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  // ------------------------------------------------------------------------ //
  modelStructuresMap["defaultFork"] =
      std::unique_ptr<ExecutableModel>(new DefaultFork);
  modelStructuresMap["defaultMerge"] =
      std::unique_ptr<ExecutableModel>(new DefaultMerge);
  modelStructuresMap["defaultControlMerge"] =
      std::unique_ptr<ExecutableModel>(new DefaultControlMerge);
  modelStructuresMap["defaultMux"] =
      std::unique_ptr<ExecutableModel>(new DefaultMux);
  modelStructuresMap["defaultBranch"] =
      std::unique_ptr<ExecutableModel>(new DefaultBranch);
  modelStructuresMap["defaultConditionalBranch"] =
      std::unique_ptr<ExecutableModel>(new DefaultConditionalBranch);
  modelStructuresMap["defaultSink"] =
      std::unique_ptr<ExecutableModel>(new DefaultSink);
  modelStructuresMap["defaultConstant"] =
      std::unique_ptr<ExecutableModel>(new DefaultConstant);
  modelStructuresMap["defaultBuffer"] =
      std::unique_ptr<ExecutableModel>(new DefaultBuffer);
  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE ABOVE MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  // ------------------------------------------------------------------------ //

  // Fill the map containing the final execution models structures
  for (auto &elem : funcMap) {
    auto &chosenStruct = modelStructuresMap[elem.getValue()];
    // if (!chosenStruct)
    //   return false;
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
void DefaultFork::execute(std::vector<llvm::Any> &ins,
                          std::vector<llvm::Any> &outs, circt::Operation &op) {
  for (auto &out : outs)
    out = ins[0];
}
bool DefaultFork::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ForkOp>(opArg);
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, 1);
}

// Default CIRCT merge
bool DefaultMerge::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MergeOp>(opArg);
  bool found = false;
  for (mlir::Value in : op.getOperands()) {
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
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ControlMergeOp>(opArg);
  bool found = false;
  for (auto in : llvm::enumerate(op.getOperands())) {
    if (valueMap.count(in.value()) == 1) {
      if (found)
        op.emitOpError("More than one valid input to CMerge!");
      valueMap[op.getResult()] = valueMap[in.value()];
      timeMap[op.getResult()] = timeMap[in.value()];
      valueMap[op.getIndex()] = APInt(INDEX_WIDTH, in.index());
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
bool DefaultMux::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MuxOp>(opArg);
  mlir::Value control = op.getSelectOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  auto opIdx = llvm::any_cast<APInt>(controlValue).getZExtValue();
  assert(opIdx < op.getDataOperands().size() &&
         "Trying to select a non-existing mux operand");

  mlir::Value in = op.getDataOperands()[opIdx];
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
void DefaultBranch::execute(std::vector<llvm::Any> &ins,
                            std::vector<llvm::Any> &outs,
                            circt::Operation &op) {
  outs[0] = ins[0];
}
bool DefaultBranch::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) { // FAUT TOUT RENAME AAAAA

  llvm::errs() << "[EXECMODELS] in the DEFAULT branch"
               << "\n";
  auto op = dyn_cast<circt::handshake::BranchOp>(opArg);
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, 0);
}

// Default CIRCT conditional branch
bool DefaultConditionalBranch::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConditionalBranchOp>(opArg);
  mlir::Value control = op.getConditionOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = op.getDataOperand();
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  mlir::Value out = llvm::any_cast<APInt>(controlValue) != 0
                        ? op.getTrueResult()
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
bool DefaultSink::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::SinkOp>(opArg);
  valueMap.erase(op.getOperand());
  return true;
}

// Default CIRCT constant
void DefaultConstant::execute(std::vector<llvm::Any> &ins,
                              std::vector<llvm::Any> &outs,
                              circt::Operation &op) {
  auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
  outs[0] = attr.getValue();
}
bool DefaultConstant::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConstantOp>(opArg);
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, 0);
}

// Default CIRCT buffer
void DefaultBuffer::execute(std::vector<llvm::Any> &ins,
                            std::vector<llvm::Any> &outs,
                            circt::Operation &op) {
  outs[0] = ins[0];
}
bool DefaultBuffer::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList,
    std::map<std::string,
             std::unique_ptr<dynamatic::experimental::ExecutableModel>> &models,
    circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::BufferOp>(opArg);
  return tryToExecute(op.getOperation(), valueMap, timeMap, scheduleList,
                      models, op.getNumSlots());
}

// TODO : Add dynamatic store/load/end

} // namespace experimental
} // namespace dynamatic