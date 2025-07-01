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
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <fstream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace ljson = llvm::json;

//===----------------------------------------------------------------------===//
// TimingDatabse definitions
//===----------------------------------------------------------------------===//

unsigned dynamatic::getOpDatawidth(Operation *op) {
  // Handshake operations have various semantics and must be handled on a
  // case-by-case basis
  return llvm::TypeSwitch<Operation *, unsigned>(op)
      .Case<handshake::SelectOp>([&](auto) {
        // The first operand of SelectOp is always an 1-bit wide control signal,
        // so here we look up the width of the second or third input's bitwidth
        return getHandshakeTypeBitWidth(op->getOperand(1).getType());
      })
      .Case<handshake::ArithOpInterface>([&](auto) {
        // This option matches all the handshake equivalent of arith/math
        // operations
        return getHandshakeTypeBitWidth(op->getOperand(0).getType());
      })
      .Case<handshake::MergeLikeOpInterface>(
          [&](handshake::MergeLikeOpInterface mergeLikeOp) {
            return getHandshakeTypeBitWidth(
                mergeLikeOp.getDataOperands().front().getType());
          })
      .Case<handshake::BufferOp, handshake::ForkOp, handshake::LazyForkOp,
            handshake::BranchOp, handshake::SinkOp>([&](auto) {
        return getHandshakeTypeBitWidth(op->getOperand(0).getType());
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condOp) {
            return getHandshakeTypeBitWidth(condOp.getDataOperand().getType());
          })
      .Case<handshake::SourceOp, handshake::ConstantOp>([&](auto) {
        return getHandshakeTypeBitWidth(op->getResult(0).getType());
      })
      .Case<handshake::EndOp, handshake::JoinOp, handshake::BlockerOp>(
          [&](auto) {
            if (op->getNumOperands() == 0)
              return 0u;
            return getHandshakeTypeBitWidth(op->getOperand(0).getType());
          })
      .Case<handshake::LoadOp, handshake::StoreOp>([&](auto) {
        return std::max(getHandshakeTypeBitWidth(op->getOperand(0).getType()),
                        getHandshakeTypeBitWidth(op->getOperand(1).getType()));
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

LogicalResult TimingDatabase::getLatency(
    Operation *op, SignalType signalType, double &latency,
    double targetPeriod) const // Our current timing model doesn't have latency
                               // information for valid and
// ready signals, assume it is 0
{

  if (signalType != SignalType::DATA) {
    latency = 0.0;
    return success();
  }

  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  // First, we extract the DelayDepMetric instance for a specific biwdidth.
  // Then, we use its method (getDelayCeilMetric) to get the latency for the
  // given targetPeriod.
  DelayDepMetric<double> DelayStruct;

  if (failed(model->latency.getCeilMetric(op, DelayStruct)))
    return failure();
  if (failed(DelayStruct.getDelayCeilMetric(targetPeriod, latency)))
    return failure();

  // FIXME: We compensante for the fact that the LSQ has roughly 3 extra cycles
  // of latency on loads compared to an MC here because our timing models are
  // currenty unable to account for this. It's obviosuly very bad to
  // special-case this here so we should find a way to properly express this
  // information in our models.
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    auto memOp = findMemInterface(loadOp.getAddressResult());
    if (isa_and_present<handshake::LSQOp>(memOp))
      latency += 3;
  }
  return success();
}

LogicalResult TimingDatabase::getInternalCombinationalDelay(
    Operation *op, SignalType signalType, double &delay,
    double targetPeriod) const // Our current timing model doesn't have latency
                               // information for valid and
// ready signals, assume it is 0
{
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  // This section now must handle the fact that all latency values are now
  // contained inside an instance of DelayDepMetric. We therefore extract this
  // structure, and use its method to obtain the latency value at the
  // targetPeriod provided.
  DelayDepMetric<double> DelayStruct;

  if (failed(model->latency.getCeilMetric(op, DelayStruct)))
    return failure();
  if (failed(DelayStruct.getDelayCeilValue(targetPeriod, delay)))
    return failure();

  return success();
}

LogicalResult TimingDatabase::getInternalDelay(Operation *op,
                                               SignalType signalType,
                                               double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  switch (signalType) {
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

LogicalResult TimingDatabase::getTotalDelay(Operation *op,
                                            SignalType signalType,
                                            double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();
  switch (signalType) {
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
  llvm::Expected<ljson::Value> value = ljson::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse timing models in \"" << jsonpath << "\"\n";
    return failure();
  }

  // Deserialize into a timing database
  ljson::Path::Root jsonRoot(jsonpath);
  return success(fromJSON(*value, timingDB, ljson::Path(jsonRoot)));
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
static bool bitwidthFromJSON(const ljson::ObjectKey &value, unsigned &bitwidth,
                             ljson::Path path) {
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
                              const ljson::Object *object, T &out,
                              ljson::Path path) {
  assert(!keys.empty() && "list of keys must be non-empty");

  size_t lastElem = keys.size() - 1;
  const ljson::Object *currentObj = object;
  ljson::Path currentPath = path;
  for (auto [idx, k] : llvm::enumerate(keys)) {
    currentPath = currentPath.field(k);
    if (idx == lastElem) {
      if (const ljson::Value *value = currentObj->get(k))
        return fromJSON(*value, out, currentPath);
      path.report("expected last key in path to exist");
      return false;
    }
    if (const ljson::Object *nextObject = currentObj->getObject(k))
      currentObj = nextObject;
    else {
      path.report("expected last key in path to exist");
      return false;
    }
  }

  return true;
}

bool dynamatic::fromJSON(const ljson::Value &value,
                         BitwidthDepMetric<double> &metric, ljson::Path path) {
  const ljson::Object *object = value.getAsObject();
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

bool dynamatic::fromJSON(const ljson::Value &value,
                         BitwidthDepMetric<DelayDepMetric<double>> &metric,
                         ljson::Path path) {

  const ljson::Object *object = value.getAsObject();

  // standard empty object check
  if (!object) {
    path.report("expected JSON object");
    return false;
  }
  // The outer loop is on the bitwidths: each is associated with a
  // DelayDepMetric map in the JSON.
  for (const auto &[bitwidthKey, metricValue] : *object) {
    unsigned bitwidth;
    // we start by obtaining the bitwidth value associated with this key
    if (!bitwidthFromJSON(bitwidthKey, bitwidth, path.field(bitwidthKey)))
      return false;

    // We instantiate inside the loop an internalMap for this specific bitwidth.
    std::map<double, double> internalMap;

    // Validity check to ensure the presence of a map.
    const ljson::Object *nestedMap = metricValue.getAsObject();
    if (!nestedMap) {
      path.field(bitwidthKey).report("expected nested map object");
      return false;
    }

    // nested fromJSON call, which deserializes individual delay & value pairs
    // into the internalMap
    for (const auto &[doubleDelay, doubleValue] : *nestedMap) {
      double key;
      key = std::stod(doubleDelay.str());

      double value;
      if (!fromJSON(doubleValue, value,
                    path.field(bitwidthKey).field(doubleDelay)))
        return false;

      internalMap[key] = value;
    }
    // We save the internal map as the data field of the DelayDepMetric.
    DelayDepMetric<double> DelayDepStruct;
    DelayDepStruct.data = internalMap;

    // Each DelayDepMetric structure is then associated with its bitwidth,
    // completing the 2-level nested map.
    metric.data[bitwidth] = DelayDepStruct;
  }

  return true;
}

static const std::string LATENCY[] = {"latency"};
static const std::string DELAY[] = {"delay", "data"};
static const std::string DELAY_VALID[] = {"delay", "valid", "1"};
static const std::string DELAY_READY[] = {"delay", "ready", "1"};
static const std::string DELAY_VR[] = {"delay", "VR"};
static const std::string DELAY_CV[] = {"delay", "CV"};
static const std::string DELAY_CR[] = {"delay", "CR"};
static const std::string DELAY_VC[] = {"delay", "VC"};
static const std::string DELAY_VD[] = {"delay", "VD"};

bool dynamatic::fromJSON(const ljson::Value &value,
                         TimingModel::PortModel &model, ljson::Path path) {
  const ljson::Object *object = value.getAsObject();
  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // Deserialize the data delays
  FW_FALSE(deserializeNested(DELAY, object, model.dataDelay, path));
  // Deserialize the valid/ready delays
  FW_FALSE(deserializeNested(DELAY_VALID, object, model.validDelay, path));
  FW_FALSE(deserializeNested(DELAY_READY, object, model.readyDelay, path));
  return true;
}

bool dynamatic::fromJSON(const ljson::Value &value, TimingModel &model,
                         ljson::Path path) {

  const ljson::Object *object = value.getAsObject();
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
  if (const ljson::Value *value = object->get("inport")) {
    FW_FALSE(fromJSON(*value, model.inputModel, path.field("inport")));
  } else {
    path.report("expected to find \"inport\" key");
    return false;
  }

  // Deserialize the output ports' model
  if (const ljson::Value *value = object->get("outport")) {
    FW_FALSE(fromJSON(*value, model.outputModel, path.field("outport")));
  } else {
    path.report("expected to find \"outport\" key");
    return false;
  }

  return true;
}

bool dynamatic::fromJSON(const ljson::Value &jsonValue,
                         TimingDatabase &timingDB, ljson::Path path) {
  const ljson::Object *components = jsonValue.getAsObject();
  if (!components)
    return false;

  for (const auto &[opName, cmpInfo] : *components) {
    TimingModel model;
    ljson::Path opPath = path.field(opName);
    fromJSON(cmpInfo, model, opPath);
    if (!timingDB.insertTimingModel(opName, model)) {
      opPath.report("Overriding existing timing model for operation");
      return false;
    }
  }
  return true;
}
