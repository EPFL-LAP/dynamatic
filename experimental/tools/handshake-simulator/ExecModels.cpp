//===- HandshakeExecutableOps.cpp - Handshake executable Operations -------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace dynamatic::experimental;

/// Cycles it take for corresponding operations to execute
/// Will be subject to changes in the future.
#define CYCLE_TIME_LOAD_OP 4
#define CYCLE_TIME_STORE_OP 2

//===----------------------------------------------------------------------===//
// State tracking
//===----------------------------------------------------------------------===//

/// Returns true if an internal state exists for the operation
static inline bool internalStateExists(circt::Operation &opArg,
                                       StateMap &stateMap) {
  return stateMap.count(&opArg);
}

/// Set the entry on the internal state for the entered operation
template <typename ContentType>
static inline void setInternalState(circt::Operation &opArg,
                                    ContentType content, StateMap &stateMap) {
  stateMap[&opArg] = content;
}

/// Gets the current state of the operation and puts it on content
/// Returns false if a state does not exist for this operation
template <typename ContentType>
static bool getInternalState(circt::Operation &opArg, ContentType &content,
                             StateMap &stateMap) {
  if (!stateMap.count(&opArg)) {
    return false;
  }
  content = llvm::any_cast<ContentType>(stateMap[&opArg]);
  return true;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static SmallVector<Value> toVector(ValueRange range) {
  return SmallVector<Value>(range.begin(), range.end());
}

/// Returns whether the precondition holds for a general op to execute
static bool isReadyToExecute(ArrayRef<Value> ins, ArrayRef<Value> outs,
                             llvm::DenseMap<Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

/// Fetches values from the value map and consume them
static std::vector<llvm::Any>
fetchValues(ArrayRef<Value> values,
            llvm::DenseMap<Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].has_value());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

/// Stores values to the value map for each output
static void storeValues(std::vector<llvm::Any> &values, ArrayRef<Value> outs,
                        llvm::DenseMap<Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

/// Updates the time map after the execution
static void updateTime(ArrayRef<Value> ins, ArrayRef<Value> outs,
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
static bool tryToExecute(
    circt::Operation *op, llvm::DenseMap<Value, llvm::Any> &valueMap,
    llvm::DenseMap<Value, double> &timeMap, SmallVector<Value> &scheduleList,
    ModelMap &models,
    const std::function<void(std::vector<llvm::Any> &, std::vector<llvm::Any> &,
                             circt::Operation &)> &executeFunc,
    double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (!isReadyToExecute(ins, outs, valueMap))
    return false;
  auto in = fetchValues(ins, valueMap);
  std::vector<llvm::Any> out(outs.size());
  executeFunc(in, out, *op);
  storeValues(out, outs, valueMap);
  updateTime(ins, outs, timeMap, latency);
  scheduleList = outs;
  return true;
}

/// Parses mem_controller operation operands and puts the corresponding index
/// inside vectors to track each operand
static MemoryControllerState
parseOperandIndex(circt::handshake::MemoryControllerOp &op) {
  MemoryControllerState memControllerData;
  unsigned operandIndex = 1; // ignores memref operand (at index 0)

  // Parses the operand list
  auto accessesPerBB = op.getAccesses();
  for (auto [bbIndex, accesses] : llvm::enumerate(accessesPerBB)) {
    auto accessesArray = accesses.dyn_cast<ArrayAttr>();

    if (op.bbHasControl(bbIndex))
      operandIndex++; // Skip the %bbX

    for (auto &access : accessesArray) {
      auto type = cast<circt::handshake::AccessTypeEnumAttr>(access).getValue();
      memControllerData.accesses.push_back(type);
      if (type == AccessTypeEnum::Store) {
        memControllerData.storesAddr.push_back(operandIndex++);
        memControllerData.storesData.push_back(operandIndex);
      } else {
        memControllerData.loadsAddr.push_back(operandIndex);
      }
      operandIndex++;
    }
    bbIndex++;
  }

  return memControllerData;
}

/// Transfers time and data between to stored element
static void memoryTransfer(Value from, Value to, ExecutableData &data) {
  data.valueMap[to] = data.valueMap[from];
  data.timeMap[to] = data.timeMap[from];
  // Tells all operations the data is available
  data.scheduleList.push_back(to);
  data.valueMap.erase(from);
}

/// Adds execution models to the map using the default.OPNAME name format
template <typename Op, typename Model>
static void addDefault(ModelMap &modelStructuresMap) {
  modelStructuresMap[std::string("default.") + Op::getOperationName().str()] =
      std::make_unique<Model>();
}

LogicalResult
dynamatic::experimental::initialiseMap(llvm::StringMap<std::string> &funcMap,
                                       ModelMap &models) {
  /*
  - 'modelStructuresMap' stores all known models and identifies each structure
      a string name -> any exec model structure
  - 'funcMap' stores the entered configuration, parsed json + --change-model cmd
      mlir operation name -> string name correspond to a exec model struct
  - 'models' links both maps, so the configuration and all known models
      mlir operation name -> configurated exec model structure
  */

  // This maps the configuration file / command string name to it's
  // corresponding structure
  ModelMap modelStructuresMap;
  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE ABOVE MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  //                                                                          //
  //   You only have to map a name, the one found in the configuration, to a  //
  //   unique_ptr of your structure. See an example in the addDefault method  //
  // ------------------------------------------------------------------------ //

  // Default operations
  addDefault<handshake::ForkOp, DefaultFork>(modelStructuresMap);
  addDefault<handshake::MergeOp, DefaultMerge>(modelStructuresMap);
  addDefault<handshake::MuxOp, DefaultMux>(modelStructuresMap);
  addDefault<handshake::BranchOp, DefaultBranch>(modelStructuresMap);
  addDefault<handshake::SinkOp, DefaultSink>(modelStructuresMap);
  addDefault<handshake::ConstantOp, DefaultConstant>(modelStructuresMap);
  addDefault<handshake::BufferOp, DefaultBuffer>(modelStructuresMap);
  addDefault<handshake::ConditionalBranchOp, DefaultConditionalBranch>(
      modelStructuresMap);
  addDefault<handshake::ControlMergeOp, DefaultControlMerge>(
      modelStructuresMap);

  // Dynamatic operations
  addDefault<handshake::MemoryControllerOp, DynamaticMemController>(
      modelStructuresMap);
  addDefault<handshake::DynamaticLoadOp, DynamaticLoad>(modelStructuresMap);
  addDefault<handshake::DynamaticStoreOp, DynamaticStore>(modelStructuresMap);
  addDefault<handshake::DynamaticReturnOp, DynamaticReturn>(modelStructuresMap);
  addDefault<handshake::EndOp, DynamaticEnd>(modelStructuresMap);

  // ------------------------------------------------------------------------ //
  //   ADD YOUR STRUCT TO THE ABOVE MAP IF YOU WANT TO ADD EXECUTION MODELS   //
  //                                                                          //
  //   You only have to map a name, the one found in the configuration, to a  //
  //   unique_ptr of your structure. See an example in the addDefault method  //
  // ------------------------------------------------------------------------ //

  // Fill the map containing the final execution models structures
  for (auto &[opName, modelName] : funcMap) {
    auto &chosenStruct = modelStructuresMap[modelName];
    // Stop the program if the corresponding struct wasn't found
    if (!chosenStruct)
      return failure();
    models[opName.str()] = std::move(chosenStruct);
  }
  return success();
}

//===----------------------------------------------------------------------===//
//                     Execution models definitions
//===----------------------------------------------------------------------===//

namespace dynamatic {
namespace experimental {

//--- Default CIRCT models ---------------------------------------------------//

bool DefaultFork::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ForkOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, circt::Operation &op) {
    for (auto &out : outs)
      out = ins[0];
  };
  return tryToExecute(op.getOperation(), data.valueMap, data.timeMap,
                      data.scheduleList, data.models, executeFunc, 1);
}

bool DefaultMerge::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MergeOp>(opArg);
  bool found = false;
  for (Value in : op.getOperands()) {
    if (data.valueMap.count(in) == 1) {
      if (found)
        op.emitOpError("More than one valid input to Merge!");
      auto t = data.valueMap[in];

      data.valueMap[op.getResult()] = t;
      data.timeMap[op.getResult()] = data.timeMap[in];
      // Consume the inputs.
      data.valueMap.erase(in);
      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to Merge!");
  data.scheduleList.push_back(op.getResult());
  return true;
}

bool DefaultControlMerge::tryExecute(ExecutableData &data,
                                     circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ControlMergeOp>(opArg);
  bool found = false;
  for (auto in : llvm::enumerate(op.getOperands())) {
    if (data.valueMap.count(in.value()) == 1) {
      if (found)
        op.emitOpError("More than one valid input to CMerge!");
      data.valueMap[op.getResult()] = data.valueMap[in.value()];
      data.timeMap[op.getResult()] = data.timeMap[in.value()];
      data.valueMap[op.getIndex()] =
          APInt(IndexType::kInternalStorageBitWidth, in.index());
      data.timeMap[op.getIndex()] = data.timeMap[in.value()];

      // Consume the inputs.
      data.valueMap.erase(in.value());

      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to CMerge!");
  data.scheduleList = toVector(op.getResults());
  return true;
}

bool DefaultMux::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MuxOp>(opArg);
  Value control = op.getSelectOperand();
  if (data.valueMap.count(control) == 0)
    return false;
  auto controlValue = data.valueMap[control];
  auto controlTime = data.timeMap[control];
  auto opIdx = llvm::any_cast<APInt>(controlValue).getZExtValue();
  assert(opIdx < op.getDataOperands().size() &&
         "Trying to select a non-existing mux operand");

  Value in = op.getDataOperands()[opIdx];
  if (data.valueMap.count(in) == 0)
    return false;
  auto inValue = data.valueMap[in];
  auto inTime = data.timeMap[in];
  double time = std::max(controlTime, inTime);
  data.valueMap[op.getResult()] = inValue;
  data.timeMap[op.getResult()] = time;

  // Consume the inputs.
  data.valueMap.erase(control);
  data.valueMap.erase(in);
  data.scheduleList.push_back(op.getResult());
  return true;
}

bool DefaultBranch::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::BranchOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        circt::Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op.getOperation(), data.valueMap, data.timeMap,
                      data.scheduleList, data.models, executeFunc, 0);
}

bool DefaultConditionalBranch::tryExecute(ExecutableData &data,
                                          circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConditionalBranchOp>(opArg);
  Value control = op.getConditionOperand();
  if (data.valueMap.count(control) == 0)
    return false;
  auto controlValue = data.valueMap[control];
  auto controlTime = data.timeMap[control];
  Value in = op.getDataOperand();
  if (data.valueMap.count(in) == 0)
    return false;
  auto inValue = data.valueMap[in];
  auto inTime = data.timeMap[in];
  Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op.getTrueResult()
                                                       : op.getFalseResult();
  double time = std::max(controlTime, inTime);
  data.valueMap[out] = inValue;
  data.timeMap[out] = time;
  data.scheduleList.push_back(out);

  // Consume the inputs.
  data.valueMap.erase(control);
  data.valueMap.erase(in);
  return true;
}

bool DefaultSink::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::SinkOp>(opArg);
  data.valueMap.erase(op.getOperand());
  return true;
}

bool DefaultConstant::tryExecute(ExecutableData &data,
                                 circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::ConstantOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, circt::Operation &op) {
    auto attr = op.getAttrOfType<IntegerAttr>("value");
    outs[0] = attr.getValue();
  };
  return tryToExecute(op.getOperation(), data.valueMap, data.timeMap,
                      data.scheduleList, data.models, executeFunc, 0);
}

bool DefaultBuffer::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::BufferOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        circt::Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op.getOperation(), data.valueMap, data.timeMap,
                      data.scheduleList, data.models, executeFunc,
                      op.getNumSlots());
}

//--- Dynamatic models -------------------------------------------------------//

bool DynamaticMemController::tryExecute(ExecutableData &data,
                                        circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::MemoryControllerOp>(opArg);
  bool completed = true;
  unsigned bufferStart =
      llvm::any_cast<unsigned>(data.valueMap[op.getMemref()]);

  // Add an internal state to keep track of completed load/store requests
  if (!internalStateExists(opArg, data.stateMap))
    setInternalState<MemoryControllerState>(opArg, parseOperandIndex(op),
                                            data.stateMap);

  MemoryControllerState mcData;
  getInternalState<MemoryControllerState>(opArg, mcData, data.stateMap);

  // First do all the stores possible
  for (size_t i = 0; i < mcData.storesAddr.size(); ++i) {
    unsigned addressIdx = mcData.storesAddr[i];
    unsigned dataIdx = mcData.storesData[i];
    Value address = op.getOperand(addressIdx);
    Value dataOperand = op.getOperand(dataIdx);
    // Verify if the operands are ready
    if ((!data.valueMap.count(dataOperand) || !data.valueMap.count(address))) {
      completed = false;
      continue;
    }

    // Store the data accordingly
    auto addressValue = data.valueMap[address];
    auto addressTime = data.timeMap[address];
    auto dataValue = data.valueMap[dataOperand];
    auto dataTime = data.timeMap[dataOperand];

    assert(bufferStart < data.store.size());
    auto &mem = data.store[bufferStart];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < mem.size());
    mem[offset] = dataValue;

    double time = std::max(dataTime, addressTime) + CYCLE_TIME_STORE_OP;
    data.timeMap[address] = time;
    data.timeMap[dataOperand] = time;

    mcData.storesAddr.erase(mcData.storesAddr.begin() + i);
    mcData.storesData.erase(mcData.storesData.begin() + i);
    data.stateMap[&opArg] = mcData;

    // Tell the store instruction it is ready to go
    data.scheduleList.push_back(dataOperand);
  }

  // Now do all the loads possible
  for (size_t i = 0; i < mcData.loadsAddr.size(); ++i) {
    unsigned addressIdx = mcData.loadsAddr[i];
    Value address = op.getOperand(addressIdx);
    Value dataOperand = op.getResult(i);
    // Verify if the operand is ready
    if (!data.valueMap.count(address)) {
      completed = false;
      continue;
    }

    // Load the data accordingly
    auto addressValue = data.valueMap[address];
    auto addressTime = data.timeMap[address];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    auto &mem = data.store[bufferStart];
    assert(offset < mem.size());
    data.valueMap[dataOperand] = mem[offset];
    data.timeMap[dataOperand] = addressTime + CYCLE_TIME_LOAD_OP;

    mcData.loadsAddr.erase(mcData.loadsAddr.begin() + i);
    data.stateMap[&opArg] = mcData;

    // Tell the load instruction it is ready to go
    data.scheduleList.push_back(dataOperand);
  }

  return completed;
}

bool DynamaticLoad::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::DynamaticLoadOp>(opArg);

  // Send address to mem controller if available
  if (!data.valueMap.count(op.getAddressResult()))
    memoryTransfer(op.getAddress(), op.getAddressResult(), data);

  // Send data to successor if available
  if (data.valueMap.count(op.getData())) {
    memoryTransfer(op.getData(), op.getDataResult(), data);
    return true;
  }

  return false;
}

bool DynamaticStore::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::DynamaticStoreOp>(opArg);

  // Add internal state to verify if the other operand has been sent
  // Not necessary but avoids some code duplication
  if (!internalStateExists(opArg, data.stateMap))
    setInternalState<bool>(opArg, false, data.stateMap);

  // Sends some data to memory and memorize that the data was sent
  auto sendIfAvailable = [&](Value from, Value to) -> bool {
    bool otherOperand = false;
    if (data.valueMap.count(from)) {
      memoryTransfer(from, to, data);
      // Verify that the other operand was sent to the mem controller, in which
      // case we have to reschedule the operation or not
      getInternalState(opArg, otherOperand, data.stateMap);
      setInternalState(opArg, true, data.stateMap);
    }
    return otherOperand;
  };

  // Send address to mem controller if available
  if (sendIfAvailable(op.getAddress(), op.getAddressResult()))
    return true;
  // Send data to mem controller if available
  if (sendIfAvailable(op.getData(), op.getDataResult()))
    return true;

  return false;
}

bool DynamaticReturn::tryExecute(ExecutableData &data,
                                 circt::Operation &opArg) {
  auto op = dyn_cast<circt::handshake::DynamaticReturnOp>(opArg);
  auto executeFunc = [&](std::vector<llvm::Any> &ins,
                         std::vector<llvm::Any> &outs, circt::Operation &op) {
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      outs[i] = ins[i];
    data.stateMap[&op] = true;
  };
  return tryToExecute(op.getOperation(), data.valueMap, data.timeMap,
                      data.scheduleList, data.models, executeFunc, 0);
}

bool DynamaticEnd::tryExecute(ExecutableData &data, circt::Operation &opArg) {
  double time = 0.0;
  for (auto &[opKey, state] : data.stateMap) {
    // Verify that all returns have been completed
    if (isa<circt::handshake::DynamaticReturnOp>(opKey)) {
      bool completed = llvm::any_cast<bool>(state);
      if (!completed)
        return false;

      for (unsigned i = 0; i < opKey->getNumResults(); ++i)
        time = std::max(time, data.timeMap[opKey->getResult(i)]);
    }
    // Verify all memory controllers are finished
    if (isa<circt::handshake::MemoryControllerOp>(opKey)) {
      auto mcData = llvm::any_cast<MemoryControllerState>(state);
      if (!mcData.loadsAddr.empty())
        return false;
      if (!mcData.storesAddr.empty())
        return false;
    }
  }

  // Final time
  data.stateMap[&opArg] = time;
  return true;
}

} // namespace experimental
} // namespace dynamatic
