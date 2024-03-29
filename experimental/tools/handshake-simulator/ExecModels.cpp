//===- HandshakeExecutableOps.cpp - Handshake executable Operations -------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of execution semantics for Handshake/MLIR
// operations.
//
//===----------------------------------------------------------------------===//

#include "experimental/tools/handshake-simulator/ExecModels.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

#include <queue>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

/// Cycles it take for corresponding operations to execute
/// Will be subject to changes in the future.
#define CYCLE_TIME_LOAD_OP 3
#define CYCLE_TIME_STORE_OP 1

//===----------------------------------------------------------------------===//
// Dataflow
//===----------------------------------------------------------------------===//

/// Stores a value in a channel, and sets its state to VALID.
void dynamatic::experimental::CircuitState::storeValue(
    mlir::Value channel, std::optional<llvm::Any> data) {
  channelMap[channel].data = std::move(data);
}

/// Performs multiples storeValue's at once.
void dynamatic::experimental::CircuitState::storeValues(
    std::vector<llvm::Any> &values, llvm::ArrayRef<mlir::Value> outs) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    storeValue(outs[i], values[i]);
}

/// Removes a value from a channel, and sets its state to NONE.
void dynamatic::experimental::CircuitState::removeValue(mlir::Value channel) {
  channelMap[channel].data = std::nullopt;
}

inline std::optional<llvm::Any>
dynamatic::experimental::CircuitState::getDataOpt(mlir::Value channel) {
  return channelMap[channel].data;
}

inline llvm::Any
dynamatic::experimental::CircuitState::getData(mlir::Value channel) {
  return *(channelMap[channel].data);
}

inline bool dynamatic::experimental::CircuitState::isNone(mlir::Value channel) {
  return !channelMap[channel].isReady && !channelMap[channel].isValid;
}

bool dynamatic::experimental::CircuitState::onRisingEdge(Operation &op) {
  if (edgeRisenOps.contains(&op))
    return false;
  edgeRisenOps.insert(&op);
  return true;
}

void dynamatic::experimental::CircuitState::storeValueOnRisingEdge(
    mlir::Value channel, std::optional<llvm::Any> data) {
  bufferChannelMap[channel].data = std::move(data);
}

//===----------------------------------------------------------------------===//
// Internal data tracking
//===----------------------------------------------------------------------===//

/// Returns true if an internal data exists for the operation
static inline bool internalDataExists(Operation &opArg,
                                      InternalDataMap &internalDataMap) {
  return internalDataMap.contains(&opArg);
}

/// Set the entry on the internal data for the entered operation
template <typename ContentType>
static inline void setInternalData(Operation &opArg, ContentType content,
                                   InternalDataMap &internalDataMap) {
  internalDataMap[&opArg] = content;
}

/// Gets the current state of the operation and puts it on content
/// Returns false if a state does not exist for this operation
template <typename ContentType>
static bool getInternalData(Operation &opArg, ContentType &content,
                            InternalDataMap &internalDataMap) {
  if (!internalDataMap.contains(&opArg)) {
    return false;
  }
  content = llvm::any_cast<ContentType>(internalDataMap[&opArg]);
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
                             CircuitState &circuitState) {

  for (auto in : ins) {
    if (circuitState.isNone(in))
      return false;
  }
  // We do not care about outputs as this is done for single cycle operations
  return true;
}

/// Fetches values from the value map and consume them
static std::vector<llvm::Any> fetchValues(ArrayRef<Value> values,
                                          CircuitState &circuitState) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    if (circuitState.channelMap[value].isValid)
      ins.push_back(circuitState.getData(value));
    // removeValue(value, channelMap);
  }
  return ins;
}

/// Wrapper method for simple constant time operations
static bool tryToExecute(Operation *op, CircuitState &circuitState,
                         ModelMap &models, ExecuteFunction &executeFunc) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (!isReadyToExecute(ins, outs, circuitState))
    return false;
  auto in = fetchValues(ins, circuitState);
  std::vector<llvm::Any> out(outs.size());
  executeFunc(in, out, *op);
  circuitState.storeValues(out, outs);

  for (auto out : outs)
    circuitState.channelMap[out].isValid = true;

  return true;
}

/// Transfers data between to stored element
static inline void memoryTransfer(Value from, Value to, ExecutableData &data) {
  data.circuitState.channelMap[to].data =
      data.circuitState.channelMap[from].data;
}

/// Parses mem_controller operation operands and puts the corresponding index
/// inside vectors to track each operand
static MemoryControllerState
parseOperandIndex(handshake::MemoryControllerOp &op, unsigned cycle) {
  MemoryControllerState memControllerData;
  // Parses the operand list
  FuncMemoryPorts ports = op.getPorts();
  for (GroupMemoryPorts blockPorts : ports.groups) {
    MemoryRequest request;
    for (MemoryPort &port : blockPorts.accessPorts) {
      request.isReady = false;
      request.lastExecution = 0;
      if (std::optional<StorePort> storePort = dyn_cast<StorePort>(port)) {
        request.isLoad = true;
        request.addressIdx = storePort->getAddrInputIndex();
        request.dataIdx = storePort->getDataInputIndex();
        request.cyclesToComplete = CYCLE_TIME_STORE_OP;
        memControllerData.storeRequests.push_back(request);
      } else {
        std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port);
        assert(loadPort && "port must be load or store");
        request.isLoad = false;
        request.addressIdx = loadPort->getAddrInputIndex();
        request.dataIdx = loadPort->getDataOutputIndex();
        request.cyclesToComplete = CYCLE_TIME_LOAD_OP;
        memControllerData.loadRequests.push_back(request);
      }
    }
  }

  return memControllerData;
}

/// Adds execution models to the map using the default.OPNAME name format
template <typename Op, typename Model>
static inline void addDefault(ModelMap &modelStructuresMap) {
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
  addDefault<handshake::OEHBOp, DefaultOEHB>(modelStructuresMap);
  addDefault<handshake::TEHBOp, DefaultTEHB>(modelStructuresMap);
  addDefault<handshake::ConditionalBranchOp, DefaultConditionalBranch>(
      modelStructuresMap);
  addDefault<handshake::ControlMergeOp, DefaultControlMerge>(
      modelStructuresMap);

  // Dynamatic operations
  addDefault<handshake::MemoryControllerOp, DynamaticMemController>(
      modelStructuresMap);
  addDefault<handshake::MCLoadOp, DynamaticLoad>(modelStructuresMap);
  addDefault<handshake::MCStoreOp, DynamaticStore>(modelStructuresMap);
  addDefault<handshake::ReturnOp, DynamaticReturn>(modelStructuresMap);
  addDefault<handshake::EndOp, DynamaticEnd>(modelStructuresMap);

  // Arith operations
  addDefault<mlir::arith::AddFOp, ArithAddF>(modelStructuresMap);
  addDefault<mlir::arith::AddIOp, ArithAddI>(modelStructuresMap);
  addDefault<mlir::arith::ConstantIndexOp, ConstantIndexOp>(modelStructuresMap);
  addDefault<mlir::arith::ConstantIntOp, ConstantIntOp>(modelStructuresMap);
  addDefault<mlir::arith::XOrIOp, XOrIOp>(modelStructuresMap);
  addDefault<mlir::arith::CmpIOp, CmpIOp>(modelStructuresMap);
  addDefault<mlir::arith::CmpFOp, CmpFOp>(modelStructuresMap);
  addDefault<mlir::arith::SubIOp, SubIOp>(modelStructuresMap);
  addDefault<mlir::arith::SubFOp, SubFOp>(modelStructuresMap);
  addDefault<mlir::arith::MulIOp, MulIOp>(modelStructuresMap);
  addDefault<mlir::arith::MulFOp, MulFOp>(modelStructuresMap);
  addDefault<mlir::arith::DivSIOp, DivSIOp>(modelStructuresMap);
  addDefault<mlir::arith::DivUIOp, DivUIOp>(modelStructuresMap);
  addDefault<mlir::arith::DivFOp, DivFOp>(modelStructuresMap);
  addDefault<mlir::arith::IndexCastOp, IndexCastOp>(modelStructuresMap);
  addDefault<mlir::arith::ExtSIOp, ExtSIOp>(modelStructuresMap);

  // Other operations
  // addDefault<mlir::memref::AllocOp, AllocOp>(modelStructuresMap);
  // addDefault<mlir::cf::BranchOp, BranchOp>(modelStructuresMap);
  // addDefault<mlir::cf::CondBranchOp, CondBranchOp>(modelStructuresMap);

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

bool DefaultFork::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::ForkOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    for (auto &out : outs)
      out = ins[0];
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DefaultMerge::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::MergeOp>(opArg);
  bool found = false;
  for (Value in : op.getOperands()) {
    if (data.circuitState.channelMap[in].isValid) {
      if (found)
        op.emitOpError("More than one valid input to Merge!");
      auto t = data.circuitState.getDataOpt(in);
      data.circuitState.storeValue(op.getResult(), t);
      // Consume the inputs.
      data.circuitState.removeValue(in);
      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to Merge!");
  return true;
}

bool DefaultControlMerge::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::ControlMergeOp>(opArg);
  bool found = false;
  for (auto in : llvm::enumerate(op.getOperands())) {
    if (data.circuitState.channelMap[in.value()].isValid) {
      if (found)
        op.emitOpError("More than one valid input to CMerge!");
      data.circuitState.storeValue(op.getResult(),
                                   data.circuitState.getDataOpt(in.value()));
      data.circuitState.storeValue(
          op.getIndex(),
          APInt(IndexType::kInternalStorageBitWidth, in.index()));
      // Consume the inputs.
      data.circuitState.removeValue(in.value());
      found = true;
    }
  }
  if (!found)
    op.emitOpError("No valid input to CMerge!");
  return true;
}

bool DefaultMux::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::MuxOp>(opArg);
  Value control = op.getSelectOperand();
  if (data.circuitState.isNone(control))
    return false;
  auto controlValue = data.circuitState.getData(control);
  auto opIdx = llvm::any_cast<APInt>(controlValue).getZExtValue();
  assert(opIdx < op.getDataOperands().size() &&
         "Trying to select a non-existing mux operand");

  Value in = op.getDataOperands()[opIdx];
  if (data.circuitState.isNone(in))
    return false;
  auto inValue = data.circuitState.getDataOpt(in);
  data.circuitState.storeValue(op.getResult(), inValue);

  // Consume the inputs.
  data.circuitState.removeValue(in);
  data.circuitState.removeValue(control);
  return true;
}

bool DefaultBranch::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::BranchOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DefaultConditionalBranch::tryExecute(ExecutableData &data,
                                          Operation &opArg) {
  auto op = dyn_cast<handshake::ConditionalBranchOp>(opArg);
  Value control = op.getConditionOperand();
  if (data.circuitState.isNone(control))
    return false;
  auto controlValue = data.circuitState.getData(control);
  Value in = op.getDataOperand();
  if (data.circuitState.isNone(in))
    return false;
  auto inValue = data.circuitState.getDataOpt(in);
  Value out = llvm::any_cast<APInt>(controlValue) != 0 ? op.getTrueResult()
                                                       : op.getFalseResult();
  data.circuitState.storeValue(out, inValue);

  // Consume the inputs.
  data.circuitState.removeValue(in);
  data.circuitState.removeValue(control);

  return true;
}

bool DefaultSink::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::SinkOp>(opArg);
  data.circuitState.removeValue(op.getOperand());
  return true;
}

bool DefaultConstant::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::ConstantOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    auto attr = op.getAttrOfType<IntegerAttr>("value");
    outs[0] = attr.getValue();
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DefaultOEHB::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::OEHBOp>(opArg);

  OEHBdata oehbData;
  if (!internalDataExists(opArg, data.internalDataMap)) {
    oehbData.registerEnable = false;
    oehbData.content = std::nullopt;
    setInternalData<OEHBdata>(opArg, oehbData, data.internalDataMap);
  } else {
    getInternalData<OEHBdata>(opArg, oehbData, data.internalDataMap);
  }

  oehbData.registerEnable =
      data.circuitState.channelMap[op.getOperand()].isReady &&
      data.circuitState.channelMap[op.getOperand()].isValid;

  bool outsValid =
      data.circuitState.channelMap[op.getResult()].isValid; // useless right ?
  bool insReady = !data.circuitState.channelMap[op.getResult()].isValid ||
                  data.circuitState.channelMap[op.getResult()].isReady;
  // On rising edge
  if (data.circuitState.onRisingEdge(opArg)) {
    outsValid = data.circuitState.channelMap[op.getOperand()].isValid &&
                !data.circuitState.channelMap[op.getOperand()].isReady;
    // If input is ready and valid, store the data
    if (oehbData.registerEnable)
      oehbData.content = data.circuitState.getDataOpt(op.getOperand());
  }

  // Update dataflow
  data.circuitState.channelMap[op.getOperand()].isReady = insReady;
  data.circuitState.channelMap[op.getResult()].isValid = outsValid;

  // Whatsoever, the data is pushed
  data.circuitState.channelMap[op.getResult()].data = oehbData.content;

  // Re-store data
  setInternalData(opArg, oehbData, data.internalDataMap);

  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op, data.circuitState, data.models, executeFunc);
}

bool DefaultTEHB::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::TEHBOp>(opArg);

  // Instanciates registers
  TEHBdata tehbData;
  if (!internalDataExists(opArg, data.internalDataMap))
    tehbData.slots = op.getSlots();
  else
    getInternalData<TEHBdata>(opArg, tehbData, data.internalDataMap);

  bool alreadySent = false;

  // On rising edge
  if (data.circuitState.onRisingEdge(opArg)) {
    alreadySent = data.circuitState.channelMap[op.getResult()].isValid &&
                  !data.circuitState.channelMap[op.getResult()].isReady;

    // If something is being done this cycle, stack the data
    if (alreadySent) {
      assert((tehbData.queue.size() + 1) <= tehbData.slots &&
             "FIFO buffer is full!");
      tehbData.queue.push(
          llvm::any_cast<APInt>(data.circuitState.getData(op.getOperand())));
      // Otherwise, let is pass and live
    } else {
      // Either get the data from the input or from the queue
      if (tehbData.queue.empty()) {
        data.circuitState.storeValueOnRisingEdge(
            op.getResult(), data.circuitState.getDataOpt(op.getOperand()));
      } else {
        APInt fifoOut = tehbData.queue.front();
        tehbData.queue.pop();
        data.circuitState.storeValueOnRisingEdge(op.getResult(), fifoOut);
      }
    }
  }

  // Update dataflow states (fifo updates ins ready and outs valid)
  data.circuitState.channelMap[op.getOperand()].isReady = !alreadySent;
  data.circuitState.channelMap[op.getResult()].isValid =
      data.circuitState.channelMap[op.getOperand()].isReady || alreadySent;

  // Re-store data
  setInternalData<TEHBdata>(opArg, tehbData, data.internalDataMap);

  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs,
                        Operation &op) { outs[0] = ins[0]; };
  return tryToExecute(op, data.circuitState, data.models, executeFunc);
}

//--- Dynamatic models -------------------------------------------------------//

bool DynamaticMemController::tryExecute(ExecutableData &data,
                                        Operation &opArg) {
  auto op = dyn_cast<handshake::MemoryControllerOp>(opArg);
  bool hasChanged = false;
  unsigned bufferStart =
      llvm::any_cast<unsigned>(data.circuitState.getData(op.getMemRef()));

  // Add an internal data to keep track of completed load/store requests
  if (!internalDataExists(opArg, data.internalDataMap))
    setInternalData<MemoryControllerState>(
        opArg, parseOperandIndex(op, data.currentCycle), data.internalDataMap);

  MemoryControllerState mcData;
  getInternalData<MemoryControllerState>(opArg, mcData, data.internalDataMap);

  // First do all the stores possible
  for (size_t i = 0; i < mcData.storeRequests.size(); ++i) {
    MemoryRequest &request = mcData.storeRequests[i];
    unsigned addressIdx = request.addressIdx;
    unsigned dataIdx = request.dataIdx;
    Value address = op.getOperand(addressIdx);
    Value dataOperand = op.getOperand(dataIdx);

    // Verify if the operands are ready
    if (data.circuitState.channelMap[dataOperand].isValid &&
        data.circuitState.channelMap[address].isValid) {
      // If this is the cycle the request is made, register it to avoid
      // re-executing the operation
      if (!request.isReady) {
        request.lastExecution = data.currentCycle;
        request.isReady = true;
        --request.cyclesToComplete;
        data.internalDataMap[&opArg] = mcData;
      }

      if (request.lastExecution == data.currentCycle)
        continue;

      // Check if enough cycle passed (simulates the real circuit delay)
      if (request.cyclesToComplete == 0) {
        // Store the data accordingly
        auto addressValue = data.circuitState.getData(address);
        auto dataValue = data.circuitState.getData(dataOperand);

        assert(bufferStart < data.store.size());
        auto &mem = data.store[bufferStart];
        unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
        assert(offset < mem.size());
        mem[offset] = dataValue;

        mcData.storeRequests.erase(mcData.storeRequests.begin() + i);
        hasChanged = true;
        request.lastExecution = data.currentCycle;

      } else {
        --request.cyclesToComplete; // -1 cycle
        request.lastExecution = data.currentCycle;
      }

      data.internalDataMap[&opArg] = mcData;
    }
  }

  // Now do all the loads possible
  for (size_t i = 0; i < mcData.loadRequests.size(); ++i) {
    MemoryRequest &request = mcData.loadRequests[i];
    unsigned addressIdx = request.addressIdx;
    Value address = op.getOperand(addressIdx);
    Value dataOperand = op.getResult(i);
    // Verify if the operand is ready
    if (data.circuitState.channelMap[address].isValid) {
      if (!request.isReady) {
        request.lastExecution = data.currentCycle;
        request.isReady = true;
        --request.cyclesToComplete;
        data.internalDataMap[&opArg] = mcData;
        continue;
      }

      if (request.lastExecution == data.currentCycle)
        continue;

      // Check if enough cycle passed (simulates the real circuit delay)
      if (request.cyclesToComplete == 0) {
        // Load the data accordingly
        auto addressValue = data.circuitState.getData(address);
        unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
        auto &mem = data.store[bufferStart];
        assert(offset < mem.size());
        data.circuitState.storeValue(dataOperand, mem[offset]);

        mcData.loadRequests.erase(mcData.loadRequests.begin() + i);
        hasChanged = true;
      } else {
        --request.cyclesToComplete;
      }
      request.lastExecution = data.currentCycle;
      data.internalDataMap[&opArg] = mcData;
    }
  }

  return hasChanged;
}

bool DynamaticLoad::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::MCLoadOp>(opArg);
  bool hasDoneStuff = false;

  // Send address to mem controller if available
  if (data.circuitState.isNone(op.getAddressResult())) {
    memoryTransfer(op.getAddress(), op.getAddressResult(), data);
    hasDoneStuff = true;
  }
  // Send data to successor if available
  if (data.circuitState.channelMap[op.getData()].isValid &&
      data.circuitState.isNone(op.getDataResult())) {
    memoryTransfer(op.getData(), op.getDataResult(), data);
    hasDoneStuff = true;
  }

  return hasDoneStuff;
}

bool DynamaticStore::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::MCStoreOp>(opArg);
  bool hasDoneStuff = false;

  // Send address to mem controller if available
  if (data.circuitState.channelMap[op.getAddress()].isValid &&
      data.circuitState.isNone(op.getAddressResult())) {
    memoryTransfer(op.getAddress(), op.getAddressResult(), data);
    hasDoneStuff = true;
  }
  // Send data to mem controller if available
  if (data.circuitState.channelMap[op.getData()].isValid &&
      data.circuitState.isNone(op.getDataResult())) {
    memoryTransfer(op.getData(), op.getDataResult(), data);
    hasDoneStuff = true;
  }

  return hasDoneStuff;
}

bool DynamaticReturn::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<handshake::ReturnOp>(opArg);
  auto executeFunc = [&](std::vector<llvm::Any> &ins,
                         std::vector<llvm::Any> &outs, Operation &op) {
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      outs[i] = ins[i];
    data.internalDataMap[&op] = true;
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DynamaticEnd::tryExecute(ExecutableData &data, Operation &opArg) {
  for (auto &[opKey, state] : data.internalDataMap) {
    // Verify that all returns have been completed
    if (isa<handshake::ReturnOp>(opKey)) {
      bool completed = llvm::any_cast<bool>(state);
      if (!completed)
        return false;
    }
    // Verify all memory controllers are finished
    if (isa<handshake::MemoryControllerOp>(opKey)) {
      auto mcData = llvm::any_cast<MemoryControllerState>(state);
      if (!mcData.loadRequests.empty())
        return false;
      if (!mcData.storeRequests.empty())
        return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//                     ARITH IR execution models
//===----------------------------------------------------------------------===//

bool ArithAddF::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::AddFOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APFloat>(ins[0]) + llvm::any_cast<APFloat>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool ArithAddI::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::AddIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APInt>(ins[0]) + llvm::any_cast<APInt>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool ConstantIndexOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::ConstantIndexOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
    outs[0] = attr.getValue().sextOrTrunc(32);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool ConstantIntOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::ConstantIntOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    auto attr = op.getAttrOfType<mlir::IntegerAttr>("value");
    outs[0] = attr.getValue();
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool XOrIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::XOrIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APInt>(ins[0]) ^ llvm::any_cast<APInt>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool CmpIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto castedOp = dyn_cast<mlir::arith::CmpIOp>(opArg);
  auto executeFunc = [&castedOp](std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs, Operation &op) {
    APInt in0 = llvm::any_cast<APInt>(ins[0]);
    APInt in1 = llvm::any_cast<APInt>(ins[1]);
    APInt out0(
        1, mlir::arith::applyCmpPredicate(castedOp.getPredicate(), in0, in1));
    outs[0] = out0;
  };
  return tryToExecute(castedOp.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool CmpFOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto castedOp = dyn_cast<mlir::arith::CmpFOp>(opArg);
  auto executeFunc = [&castedOp](std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs, Operation &op) {
    APFloat in0 = llvm::any_cast<APFloat>(ins[0]);
    APFloat in1 = llvm::any_cast<APFloat>(ins[1]);
    APInt out0(
        1, mlir::arith::applyCmpPredicate(castedOp.getPredicate(), in0, in1));
    outs[0] = out0;
  };
  return tryToExecute(castedOp.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool SubIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::SubIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APInt>(ins[0]) - llvm::any_cast<APInt>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool SubFOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::SubFOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APFloat>(ins[0]) + llvm::any_cast<APFloat>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool MulIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::MulIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APInt>(ins[0]) * llvm::any_cast<APInt>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool MulFOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::MulFOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APFloat>(ins[0]) * llvm::any_cast<APFloat>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DivSIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::DivSIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    if (!llvm::any_cast<APInt>(ins[1]).getZExtValue())
      op.emitOpError() << "Division By Zero!";
    outs[0] = llvm::any_cast<APInt>(ins[0]).sdiv(llvm::any_cast<APInt>(ins[1]));
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DivUIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::DivUIOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    if (!llvm::any_cast<APInt>(ins[1]).getZExtValue())
      op.emitOpError() << "Division By Zero!";
    outs[0] = llvm::any_cast<APInt>(ins[0]).udiv(llvm::any_cast<APInt>(ins[1]));
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool DivFOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto op = dyn_cast<mlir::arith::DivFOp>(opArg);
  auto executeFunc = [](std::vector<llvm::Any> &ins,
                        std::vector<llvm::Any> &outs, Operation &op) {
    outs[0] = llvm::any_cast<APFloat>(ins[0]) / llvm::any_cast<APFloat>(ins[1]);
  };
  return tryToExecute(op.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool IndexCastOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto castedOp = dyn_cast<mlir::arith::IndexCastOp>(opArg);
  auto executeFunc = [&castedOp](std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs, Operation &op) {
    Type outType = castedOp.getOut().getType();
    APInt inValue = llvm::any_cast<APInt>(ins[0]);
    APInt outValue;
    if (outType.isIndex())
      outValue =
          APInt(IndexType::kInternalStorageBitWidth, inValue.getZExtValue());
    else if (outType.isIntOrFloat())
      outValue = APInt(outType.getIntOrFloatBitWidth(), inValue.getZExtValue());
    else {
      op.emitOpError() << "unhandled output type";
    }

    outs[0] = outValue;
  };
  return tryToExecute(castedOp.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool ExtSIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto castedOp = dyn_cast<mlir::arith::ExtSIOp>(opArg);
  auto executeFunc = [&castedOp](std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs, Operation &op) {
    int64_t width = castedOp.getType().getIntOrFloatBitWidth();
    outs[0] = llvm::any_cast<APInt>(ins[0]).sext(width);
  };
  return tryToExecute(castedOp.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

bool ExtUIOp::tryExecute(ExecutableData &data, Operation &opArg) {
  auto castedOp = dyn_cast<mlir::arith::ExtUIOp>(opArg);
  auto executeFunc = [&castedOp](std::vector<llvm::Any> &ins,
                                 std::vector<llvm::Any> &outs, Operation &op) {
    int64_t width = castedOp.getType().getIntOrFloatBitWidth();
    outs[0] = llvm::any_cast<APInt>(ins[0]).zext(width);
  };
  return tryToExecute(castedOp.getOperation(), data.circuitState, data.models,
                      executeFunc);
}

//===----------------------------------------------------------------------===//
//                     CF/MEMREF IR execution models
//===----------------------------------------------------------------------===//

} // namespace experimental
} // namespace dynamatic
