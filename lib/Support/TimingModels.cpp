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
  auto unitDelayOrFail = dataDelay.select(bitwidth);
  if (failed(unitDelayOrFail))
    return failure();
  delay = unitDelayOrFail->get();
  return success();
}

void TimingDatabase::insertTimingModel(StringRef timingModelKey,
                                       TimingModel &model) {
  models.try_emplace(timingModelKey, model);
}

const TimingModel *TimingDatabase::getModel(StringRef timingModelKey) const {
  auto it = models.find(timingModelKey);
  if (it == models.end())
    return nullptr;
  return &it->second;
}

const TimingModel *TimingDatabase::getModel(Operation *op) const {
  StringRef baseName = op->getName().getStringRef();
  // if the operation is a floating point operation with multiple
  // possible implementations
  if (auto fpuImplInterface =
          llvm::dyn_cast<dynamatic::handshake::FPUImplInterface>(op)) {
    // include the implementation in the key
    std::string timingModelKey =
        (baseName + "." + stringifyEnum(fpuImplInterface.getFPUImpl())).str();
    return getModel(timingModelKey);
  }

  return getModel(baseName);
}

FailureOr<double> TimingDatabase::getLatency(Operation *op,
                                             SignalType signalType,
                                             double targetPeriod,
                                             unsigned pathId) const {
  // Our current timing model doesn't have latency information for valid and
  // ready signals; assume it is 0.
  if (signalType != SignalType::DATA)
    return 0.0;

  const TimingModel *model = getModel(op);
  if (!model) {
    op->emitWarning() << "TimingDatabase::getLatency: no timing model for op";
    return failure();
  }

  // Walk inward through the path -> bitwidth -> clock-period nesting.
  auto latAndMaxFreqByBitwidth = model->latAndMaxFreqByPath.select(pathId);
  if (failed(latAndMaxFreqByBitwidth)) {
    op->emitWarning()
        << "TimingDatabase::getLatency: no entry for retiming-path id "
        << pathId;
    return failure();
  }
  auto latAndMaxFreqByClockPeriod = latAndMaxFreqByBitwidth->get().select(op);
  if (failed(latAndMaxFreqByClockPeriod)) {
    op->emitWarning()
        << "TimingDatabase::getLatency: bitwidth not characterised in the "
           "timing model";
    return failure();
  }
  auto latencyOrFail =
      latAndMaxFreqByClockPeriod->get().selectLatency(targetPeriod);
  if (failed(latencyOrFail)) {
    op->emitWarning()
        << "TimingDatabase::getLatency: no latency data for target period "
        << targetPeriod;
    return failure();
  }
  double latency = *latencyOrFail;

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
  return latency;
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

  // Internal combinational delay uses retiming path 0. Walk inward through
  // the path -> bitwidth -> clock-period nesting.
  auto latAndMaxFreqByBitwidth = model->latAndMaxFreqByPath.select(0);
  if (failed(latAndMaxFreqByBitwidth))
    return failure();
  auto latAndMaxFreqByClockPeriod = latAndMaxFreqByBitwidth->get().select(op);
  if (failed(latAndMaxFreqByClockPeriod))
    return failure();
  auto delayOrFail =
      latAndMaxFreqByClockPeriod->get().selectDelay(targetPeriod);
  if (failed(delayOrFail))
    return failure();
  delay = *delayOrFail;

  return success();
}

LogicalResult TimingDatabase::getInternalDelay(Operation *op,
                                               SignalType signalType,
                                               double &delay) const {
  const TimingModel *model = getModel(op);
  if (!model)
    return failure();

  switch (signalType) {
  case SignalType::DATA: {
    auto delayOrFail = model->dataDelay.select(op);
    if (failed(delayOrFail))
      return failure();
    delay = delayOrFail->get();
    return success();
  }
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
  case SignalType::DATA: {
    auto delayOrFail = portModel.dataDelay.select(op);
    if (failed(delayOrFail))
      return failure();
    delay = delayOrFail->get();
    return success();
  }
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
    delay = model->validDelay;
    return success();
  case SignalType::READY:
    delay = model->readyDelay;
    return success();
  }
}

const SpecTimingModel *
TimingDatabase::getSpecModel(StringRef timingModelKey) const {
  auto it = specModels.find(timingModelKey);
  if (it == specModels.end())
    return nullptr;
  return &it->second;
}

const SpecTimingModel *TimingDatabase::getSpecModel(Operation *op) const {
  if (!op)
    return nullptr;
  return getSpecModel(op->getName().getStringRef());
}

static FailureOr<SignalType> signalTypeFromString(StringRef s) {
  if (s == "data")
    return SignalType::DATA;
  if (s == "valid")
    return SignalType::VALID;
  if (s == "ready")
    return SignalType::READY;
  llvm::errs() << "spec timing port has unrecognised signal kind \"" << s
               << "\"\n";
  return failure();
}

// reads an array of delays into a delay list, or fails
// the delay of the path is a function of its bitwidth
static FailureOr<SpecTimingDelayList>
parseDelayList(const ljson::Array *delays) {
  // check array of delays exists
  if (!delays) {
    llvm::errs() << "spec timing object has no array of delays\n";
    return failure();
  }

  // struct to store parsed info in
  SpecTimingDelayList delayList;

  // for each delay
  for (const ljson::Value &delayVal : *delays) {
    // get as json object
    const ljson::Object *delayObj = delayVal.getAsObject();
    if (!delayObj) {
      llvm::errs() << "spec timing delay is not a JSON object\n";
      return failure();
    }

    // declare the delay struct
    SpecTimingDelay delay;

    // read the delay
    std::optional<double> d = delayObj->getNumber("delay");
    if (!d) {
      llvm::errs() << "spec timing delay has no delay\n";
      return failure();
    }
    // store in struct
    delay.delay = *d;

    // read the bitwidth
    std::optional<int64_t> bw = delayObj->getInteger("bitwidth");
    if (!bw) {
      llvm::errs() << "spec timing delay has no bitwidth\n";
      return failure();
    }
    // store bitwidth in struct
    delay.bitwidth = *bw;

    // move into the delay list
    delayList.addDelay(std::move(delay));
  }
  return delayList;
}

// reads a port name and signal kind into a port, or fails
static FailureOr<SpecTimingPort> parsePort(const ljson::Object *portObj) {
  // check port exists
  if (!portObj) {
    llvm::errs() << "spec timing port is not a JSON object\n";
    return failure();
  }

  // struct to save the info in
  SpecTimingPort port;

  // read the port name (whether input or output)
  std::optional<StringRef> portName = portObj->getString("port");
  if (!portName) {
    llvm::errs() << "spec timing port has no name\n";
    return failure();
  }
  // save in struct
  port.name = portName->str();

  // read the signal kind as a string
  std::optional<StringRef> signalName = portObj->getString("signal");
  if (!signalName) {
    llvm::errs() << "spec timing port has no signal\n";
    return failure();
  }

  // convert the signal string to a SignalType enum
  FailureOr<SignalType> signalType = signalTypeFromString(*signalName);
  if (failed(signalType))
    // signalTypeFromString prints the error message if failed
    return failure();

  // save in struct
  port.signal = *signalType;

  return port;
}

// reads a port2port edge [from port, to port, delay list] or fails
static FailureOr<SpecTimingPort2Port>
parsePort2Port(const ljson::Object *edgeObj) {
  // struct to store info in
  SpecTimingPort2Port edge;

  // read the 'from' port
  FailureOr<SpecTimingPort> from = parsePort(edgeObj->getObject("from"));
  if (failed(from))
    // parsePort prints the error message if failed
    return failure();

  // store in struct
  edge.from = *from;

  // read the 'to' port
  FailureOr<SpecTimingPort> to = parsePort(edgeObj->getObject("to"));
  if (failed(to))
    // parsePort prints the error message if failed
    return failure();

  // store in struct
  edge.to = *to;

  // read the list of delays
  FailureOr<SpecTimingDelayList> delayList =
      parseDelayList(edgeObj->getArray("delayList"));
  if (failed(delayList))
    // parseDelayList prints the error message if failed
    return failure();

  // store in struct
  edge.delayList = std::move(*delayList);

  return edge;
}

// reads a port2reg port delay [port, delay list] or fails
static FailureOr<SpecTimingPort2Reg>
parsePort2Reg(const ljson::Object *portDelayObj) {
  // struct to store info
  SpecTimingPort2Reg portDelay;

  // read the port
  FailureOr<SpecTimingPort> port = parsePort(portDelayObj);
  if (failed(port))
    // parsePort prints the error message if failed
    return failure();

  // store in struct
  portDelay.port = *port;

  // read the list of delays
  FailureOr<SpecTimingDelayList> delayList =
      parseDelayList(portDelayObj->getArray("delayList"));
  if (failed(delayList))
    // parseDelayList prints the error message if failed
    return failure();

  // store in struct
  portDelay.delayList = std::move(*delayList);

  return portDelay;
}

// reads a reg2port port delay [port, delay list] or fails
static FailureOr<SpecTimingReg2Port>
parseReg2Port(const ljson::Object *portDelayObj) {
  // struct to save into
  SpecTimingReg2Port portDelay;

  // read the port
  FailureOr<SpecTimingPort> port = parsePort(portDelayObj);
  if (failed(port))
    // parsePort prints the error message if failed
    return failure();

  // store tos truct
  portDelay.port = *port;

  // read the list of delays
  FailureOr<SpecTimingDelayList> delayList =
      parseDelayList(portDelayObj->getArray("delayList"));
  if (failed(delayList))
    // parseDelayList prints the error message if failed
    return failure();

  // store to struct
  portDelay.delayList = std::move(*delayList);

  return portDelay;
}

LogicalResult TimingDatabase::readSpecTimingFromJSON(std::string &jsonpath,
                                                     TimingDatabase &timingDB) {
  // open the file
  std::ifstream inputFile(jsonpath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open spec timing JSON at \"" << jsonpath
                 << "\"\n";
    return failure();
  }

  // read the file into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // parse the string as JSON
  llvm::Expected<ljson::Value> value = ljson::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse spec timing JSON in \"" << jsonpath
                 << "\"\n";
    return failure();
  }

  // get as an object
  const ljson::Object *root = value->getAsObject();
  if (!root) {
    llvm::errs() << "Spec timing JSON root must be an object\n";
    return failure();
  }

  // read one timing model per type of unit
  for (const auto &unitEntry : *root) {
    // key is the operation type this info is for
    StringRef unitKey = unitEntry.first;

    // get characterization info as object
    const ljson::Object *timingModelInfo = unitEntry.second.getAsObject();
    if (!timingModelInfo) {
      llvm::errs() << "spec timing model is not a JSON object\n";
      return failure();
    }

    // declare the model we will fill in
    SpecTimingModel model;

    // read the port2port edges as an arry
    const ljson::Array *port2portArr = timingModelInfo->getArray("port2port");
    if (!port2portArr) {
      llvm::errs() << "spec timing unit has no port2port array\n";
      return failure();
    }

    // for each port2port edge
    for (const ljson::Value &edgeVal : *port2portArr) {
      const ljson::Object *edgeObj = edgeVal.getAsObject();
      if (!edgeObj) {
        llvm::errs() << "port2port delay is not a JSON object\n";
        return failure();
      }
      FailureOr<SpecTimingPort2Port> edge = parsePort2Port(edgeObj);
      if (failed(edge))
        // parsePort2Port prints the error message if failed
        return failure();
      model.port2port.push_back(std::move(*edge));
    }

    // read the port2reg port delays as an array
    const ljson::Array *port2regArr = timingModelInfo->getArray("port2reg");
    if (!port2regArr) {
      llvm::errs() << "spec timing unit has no port2reg array\n";
      return failure();
    }

    // for each port2reg port delay
    for (const ljson::Value &portDelayVal : *port2regArr) {
      const ljson::Object *portDelayObj = portDelayVal.getAsObject();
      if (!portDelayObj) {
        llvm::errs() << "spec timing port delay is not a JSON object\n";
        return failure();
      }
      FailureOr<SpecTimingPort2Reg> portDelay = parsePort2Reg(portDelayObj);
      if (failed(portDelay))
        // parsePort2Reg prints the error message if failed
        return failure();
      model.port2reg.push_back(std::move(*portDelay));
    }

    // read the reg2port port delays as an array
    // TODO: this loop is identical to the port2reg loop except the key, type,
    // and parse function; kept explicit per the one-struct-per-JSON-key
    // decision claude: iterates the reg2port array and parses each element into
    // the model
    const ljson::Array *reg2portArr = timingModelInfo->getArray("reg2port");
    if (!reg2portArr) {
      llvm::errs() << "spec timing unit has no reg2port array\n";
      return failure();
    }

    // for each reg2port port delay
    for (const ljson::Value &portDelayVal : *reg2portArr) {
      const ljson::Object *portDelayObj = portDelayVal.getAsObject();
      if (!portDelayObj) {
        llvm::errs() << "spec timing port delay is not a JSON object\n";
        return failure();
      }
      FailureOr<SpecTimingReg2Port> portDelay = parseReg2Port(portDelayObj);
      if (failed(portDelay))
        // parseReg2Port prints the error message if failed
        return failure();
      model.reg2port.push_back(std::move(*portDelay));
    }

    // store the unit's model under its key
    timingDB.specModels[unitKey] = std::move(model);
  }
  return success();
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

/// Parses an unsigned number representing a retiming-path id from a JSON key.
/// Returns true and sets the second argument to the parsed number if the key
/// represents a valid unsigned number; returns false otherwise.
static bool pathIdFromJSON(const ljson::ObjectKey &value, unsigned &pathId,
                           ljson::Path path) {
  StringRef key = value;
  if (std::any_of(key.begin(), key.end(),
                  [](char c) { return !std::isdigit(c); })) {
    path.report("expected unsigned integer for retiming-path id");
    return false;
  }
  pathId = std::stoi(key.str());
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

bool dynamatic::fromJSON(const ljson::Value &value, DelayDepMetric &metric,
                         ljson::Path path) {
  const ljson::Object *object = value.getAsObject();

  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // Key-value pair of {delay : latency, delay : latency}
  for (const auto &[delayKey, latencyValue] : *object) {
    double delay = std::stod(delayKey.str());
    double latency;
    if (!fromJSON(latencyValue, latency, path.field(delayKey)))
      return false;
    metric.data[delay] = latency;
  }

  return true;
}

bool dynamatic::fromJSON(const ljson::Value &value,
                         BitwidthDepMetric<DelayDepMetric> &metric,
                         ljson::Path path) {

  const ljson::Object *object = value.getAsObject();

  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // The outer keys are bitwidths; each maps to a delay-dependent
  // timing metric.
  for (const auto &[bitwidthKey, metricValue] : *object) {
    unsigned bitwidth;
    if (!bitwidthFromJSON(bitwidthKey, bitwidth, path.field(bitwidthKey)))
      return false;

    DelayDepMetric delayDepStruct;
    if (!fromJSON(metricValue, delayDepStruct, path.field(bitwidthKey)))
      return false;

    metric.data[bitwidth] = delayDepStruct;
  }

  return true;
}

bool dynamatic::fromJSON(const ljson::Value &value,
                         PathDepMetric &latAndMaxFreqByPath, ljson::Path path) {
  const ljson::Object *object = value.getAsObject();
  if (!object) {
    path.report("expected JSON object");
    return false;
  }

  // The outer keys are retiming-path ids; each maps to a bitwidth-dependent
  // timing metric.
  for (const auto &[pathKey, metricValue] : *object) {
    unsigned pathId;
    if (!pathIdFromJSON(pathKey, pathId, path.field(pathKey)))
      return false;

    BitwidthDepMetric<DelayDepMetric> bitwidthMetric;
    if (!fromJSON(metricValue, bitwidthMetric, path.field(pathKey)))
      return false;

    latAndMaxFreqByPath.data[pathId] = bitwidthMetric;
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

  // Deserialize the latency and max frequency
  FW_FALSE(deserializeNested(LATENCY, object, model.latAndMaxFreqByPath, path));
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

  for (const auto &[timingModelKey, cmpInfo] : *components) {
    TimingModel model;
    ljson::Path keyPath = path.field(timingModelKey);
    fromJSON(cmpInfo, model, keyPath);
    timingDB.insertTimingModel(timingModelKey, model);
  }
  return true;
}
