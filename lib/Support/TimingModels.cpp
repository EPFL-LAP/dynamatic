//===- TimingModels.cpp - Parse/Represent comp. timing models ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for timing modeling infrastructure, including the LLVM-style
// fromJSON functions that deserialize a JSON value into a specific object (see
// advanced documentation in ::llvm::json::Value).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/TimingModels.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace dynamatic;

//===----------------------------------------------------------------------===//
// TimingDatabse definitions
//===----------------------------------------------------------------------===//

/// Returns the width of a type which must either be a NoneType, IntegerType, or
/// FloatType.
static unsigned getTypeWidth(Type type) {
  if (isa<mlir::NoneType>(type))
    return 0;
  if (isa<IntegerType, FloatType>(type))
    return type.getIntOrFloatBitWidth();
  llvm_unreachable("unsupported channel type");
}

/// Gets the datawidth of an operation, for use in determining which data point
/// of a bitwidth-dependent metric to pick.
///
/// TODO: Computations for some of the Handshake operations are shady at best
/// due to unclear semantics that we inherit from legacy Dynamatic. Some
/// conservative choices were made, but we should go back to that and clarify
/// everything at some point.
static unsigned getOpDatawidth(Operation *op) {
  // All arithmetic operations are handled the same way
  if (op->getName().getDialectNamespace() == "arith")
    return getTypeWidth(op->getOperand(0).getType());

  // Handshake operations have various semantics and must be handled on a
  // case-by-case basis
  return llvm::TypeSwitch<Operation *, unsigned>(op)
      .Case<handshake::MergeLikeOpInterface>(
          [&](handshake::MergeLikeOpInterface mergeLikeOp) {
            return getTypeWidth(
                mergeLikeOp.getDataOperands().front().getType());
          })
      .Case<handshake::BufferOp, handshake::ForkOp, handshake::LazyForkOp,
            handshake::BranchOp, handshake::SinkOp>(
          [&](auto) { return getTypeWidth(op->getOperand(0).getType()); })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condOp) {
            return getTypeWidth(condOp.getDataOperand().getType());
          })
      .Case<handshake::SourceOp, handshake::ConstantOp>(
          [&](auto) { return getTypeWidth(op->getResult(0).getType()); })
      .Case<handshake::DynamaticReturnOp, handshake::EndOp, handshake::JoinOp>(
          [&](auto) {
            unsigned maxWidth = 0;
            for (Type ty : op->getOperandTypes())
              maxWidth = std::max(maxWidth, getTypeWidth(ty));
            return maxWidth;
          })
      .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>([&](auto) {
        return std::max(getTypeWidth(op->getOperand(0).getType()),
                        getTypeWidth(op->getOperand(1).getType()));
      })
      .Case<handshake::MemoryOpInterface>(
          [&](handshake::MemoryOpInterface memOp) {
            FuncMemoryPorts ports = getMemoryPorts(memOp);
            return std::max(ports.ctrlWidth,
                            std::max(ports.addrWidth, ports.ctrlWidth));
          })
      .Default([&](auto) {
        op->emitError() << "Operation is unsupported in timing model";
        assert(false && "unsupported operation");
        return dynamatic::MAX_DATAWIDTH;
      });
}

template <typename M>
LogicalResult BitwidthDepMetric<M>::getCeilMetric(unsigned bitwidth,
                                                  M &metric) const {
  std::optional<unsigned> widthCeil;
  M metricCeil = 0.0;

  // Iterate over the available bitwidths and determine which is the closest one
  // above the operation's bitwidth
  for (const auto &[width, metric] : data) {
    if (width >= bitwidth) {
      if (!widthCeil.has_value() || *widthCeil > width) {
        widthCeil = width;
        metricCeil = metric;
      }
    }
  }

  if (!widthCeil.has_value())
    // If the maximum bitwidth in the model is strictly lower than the
    // operation's data bitwidth, then we do not know what delay to set and
    // we have to fail
    return failure();

  metric = metricCeil;
  return success();
}

template <typename M>
LogicalResult BitwidthDepMetric<M>::getCeilMetric(Operation *op,
                                                  M &metric) const {
  return getCeilMetric(getOpDatawidth(op), metric);
}

LogicalResult TimingModel::getTotalDataDelay(unsigned bitwidth,
                                             double &delay) const {
  double unitDelay, inPortDelay, outPortDelay;
  if (failed(dataDelay.getCeilMetric(bitwidth, unitDelay)) ||
      failed(inputModel.dataDelay.getCeilMetric(bitwidth, inPortDelay)) ||
      failed(outputModel.dataDelay.getCeilMetric(bitwidth, outPortDelay)))
    return failure();
  delay = unitDelay + inPortDelay + outPortDelay;
  return success();
}

bool TimingDatabase::insertTimingModel(StringRef name, TimingModel &model) {
  return models.insert(std::make_pair(OperationName(name, ctx), model)).second;
}

const TimingModel *TimingDatabase::getModel(OperationName opName) const {
  auto it = models.find(opName);
  if (it == models.end())
    return nullptr;
  return &it->second;
}

const TimingModel *TimingDatabase::getModel(Operation *op) const {
  return getModel(op->getName());
}

LogicalResult TimingDatabase::getLatency(Operation *op, SignalType signalType,
                                         double &latency) const {
  // Our current timing model doesn't have latency information for valid and
  // ready signals, assume it is 0.
  if (signalType != SignalType::DATA) {
    latency = 0.0;
    return success();
  }

  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  return model->latency.getCeilMetric(op, latency);
}

LogicalResult TimingDatabase::getInternalDelay(Operation *op, SignalType type,
                                               double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  switch (type) {
  case SignalType::DATA:
    return model->dataDelay.getCeilMetric(op, delay);
  case SignalType::VALID:
    delay = model->validDelay;
    return success();
  case SignalType::READY:
    delay = model->readyDelay;
    return success();
  }
}

LogicalResult TimingDatabase::getPortDelay(Operation *op, SignalType signalType,
                                           PortType portType,
                                           double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  const TimingModel::PortModel &portModel =
      portType == PortType::IN ? model->inputModel : model->outputModel;

  switch (signalType) {
  case SignalType::DATA:
    return portModel.dataDelay.getCeilMetric(op, delay);
  case SignalType::VALID:
    delay = portModel.validDelay;
    return success();
  case SignalType::READY:
    delay = portModel.readyDelay;
    return success();
  }
}

LogicalResult TimingDatabase::getTotalDelay(Operation *op, SignalType type,
                                            double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();
  switch (type) {
  case SignalType::DATA:
    return model->getTotalDataDelay(getOpDatawidth(op), delay);
  case SignalType::VALID:
    delay = model->getTotalValidDelay();
    return success();
  case SignalType::READY:
    delay = model->getTotalReadyDelay();
    return success();
  }
}

LogicalResult TimingDatabase::readFromJSON(std::string &jsonpath,
                                           TimingDatabase &timingDB) {
  // Open the timing database
  std::ifstream inputFile(jsonpath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open timing database\n";
    return failure();
  }

  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<json::Value> value = json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse timing models in \"" << jsonpath << "\"\n";
    return failure();
  }

  // Deserialize into a timing database
  json::Path::Root jsonRoot(jsonpath);
  return success(fromJSON(*value, timingDB, json::Path(jsonRoot)));
}

//===----------------------------------------------------------------------===//
// JSON parsing
//===----------------------------------------------------------------------===//

/// Returns from the enclosing function if the argument evaluates to false. This
/// is useful in the fromJSON functions.
#define FW_FALSE(ret)                                                          \
  if (!(ret))                                                                  \
    return false;

/// Parses an unsigned number representing a bitwidth from a JSON key. Returns
/// true and sets the second argument to the parsed number if the key represents
/// a valid unsigned number; returns false otherwise.
static bool bitwidthFromJSON(const json::ObjectKey &value, unsigned &bitwidth,
                             json::Path path) {
  StringRef key = value;
  if (std::any_of(key.begin(), key.end(),
                  [](char c) { return !std::isdigit(c); })) {
    path.report("expected unsigned integer for bitwidth");
    return false;
  }
  bitwidth = std::stoi(key.str());
  return true;
}

/// Deserializes an object of type T that is nested under a list of keys inside
/// the passed JSON object. Behaves like the fromJSON functions (See
/// ::llvm::json::Value's documentation).
template <typename T>
static bool deserializeNested(ArrayRef<std::string> keys,
                              const json::Object *object, T &out,
                              json::Path path) {
  assert(!keys.empty() && "list of keys must be non-empty");

  size_t lastElem = keys.size() - 1;
  const json::Object *currentObj = object;
  json::Path currentPath = path;
  for (auto [idx, k] : llvm::enumerate(keys)) {
    currentPath = currentPath.field(k);
    if (idx == lastElem) {
      if (const json::Value *value = currentObj->get(k))
        return fromJSON(*value, out, currentPath);
      path.report("expected last key in path to exist");
      return false;
    }
    if (const json::Object *nextObject = currentObj->getObject(k))
      currentObj = nextObject;
    else {
      path.report("expected last key in path to exist");
      return false;
    }
  }

  return true;
}

bool dynamatic::fromJSON(const json::Value &value,
                         BitwidthDepMetric<double> &metric, json::Path path) {
  const json::Object *object = value.getAsObject();
  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  for (const auto &[bitwidthKey, metricValue] : *object) {
    unsigned bitwidth;
    if (!bitwidthFromJSON(bitwidthKey, bitwidth, path.field(bitwidthKey)))
      return false;
    double dataValue;
    if (!fromJSON(metricValue, dataValue, path.field(bitwidthKey))) {
      return false;
    }
    metric.data[bitwidth] = dataValue;
  }
  return true;
}

static const std::string LATENCY[] = {"latency"};
static const std::string DELAY[] = {"delay", "data"};
static const std::string DELAY_VALID[] = {"delay", "valid", "1"};
static const std::string DELAY_READY[] = {"delay", "ready", "1"};
static const std::string BUF_TRANS[] = {"transparentBuffer"};
static const std::string BUF_OPAQUE[] = {"opaqueBuffer"};
static const std::string DELAY_VR[] = {"delay", "VR"};
static const std::string DELAY_CV[] = {"delay", "CV"};
static const std::string DELAY_CR[] = {"delay", "CR"};
static const std::string DELAY_VC[] = {"delay", "VC"};
static const std::string DELAY_VD[] = {"delay", "VD"};

bool dynamatic::fromJSON(const json::Value &value,
                         TimingModel::PortModel &model, json::Path path) {
  const json::Object *object = value.getAsObject();
  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // Deserialize the data delays
  FW_FALSE(deserializeNested(DELAY, object, model.dataDelay, path));
  // Deserialize the valid/ready delays
  FW_FALSE(deserializeNested(DELAY_VALID, object, model.validDelay, path));
  FW_FALSE(deserializeNested(DELAY_READY, object, model.readyDelay, path));
  // Deserialize the number of buffer slots of each type
  FW_FALSE(deserializeNested(BUF_TRANS, object, model.transparentSlots, path));
  FW_FALSE(deserializeNested(BUF_OPAQUE, object, model.opaqueSlots, path));
  return true;
}

bool dynamatic::fromJSON(const json::Value &value, TimingModel &model,
                         json::Path path) {

  const json::Object *object = value.getAsObject();
  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // Deserialize the latencies
  FW_FALSE(deserializeNested(LATENCY, object, model.latency, path));
  // Deserialize the data delays
  FW_FALSE(deserializeNested(DELAY, object, model.dataDelay, path));
  // Deserialize the valid/ready delay
  FW_FALSE(deserializeNested(DELAY_VALID, object, model.validDelay, path));
  FW_FALSE(deserializeNested(DELAY_READY, object, model.readyDelay, path));

  // Deserialize the wire-to-wire delays
  FW_FALSE(deserializeNested(DELAY_VR, object, model.validToReady, path));
  FW_FALSE(deserializeNested(DELAY_CV, object, model.condToValid, path));
  FW_FALSE(deserializeNested(DELAY_CR, object, model.condToReady, path));
  FW_FALSE(deserializeNested(DELAY_VC, object, model.validToCond, path));
  FW_FALSE(deserializeNested(DELAY_VD, object, model.validToData, path));

  // Deserialize the input ports' model
  if (const json::Value *value = object->get("inport")) {
    FW_FALSE(fromJSON(*value, model.inputModel, path.field("inport")));
  } else {
    path.report("expected to find \"inport\" key");
    return false;
  }

  // Deserialize the output ports' model
  if (const json::Value *value = object->get("outport")) {
    FW_FALSE(fromJSON(*value, model.outputModel, path.field("outport")));
  } else {
    path.report("expected to find \"outport\" key");
    return false;
  }

  return true;
}

bool dynamatic::fromJSON(const json::Value &jsonValue, TimingDatabase &timingDB,
                         json::Path path) {
  const json::Object *components = jsonValue.getAsObject();
  if (!components)
    return false;

  for (const auto &[opName, cmpInfo] : *components) {
    TimingModel model;
    json::Path opPath = path.field(opName);
    fromJSON(cmpInfo, model, opPath);
    if (!timingDB.insertTimingModel(opName, model)) {
      opPath.report("Overriding existing timing model for operation");
      return false;
    }
  }
  return true;
}

bool llvm::json::fromJSON(const json::Value &value, unsigned &number,
                          json::Path path) {
  std::optional<uint64_t> opt = value.getAsUINT64();
  if (!opt.has_value()) {
    path.report("expected unsigned number");
    return false;
  }
  number = opt.value();
  return true;
}
