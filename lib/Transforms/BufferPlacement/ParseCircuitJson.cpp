//===- ParseCircuitJson.cpp - Parse circuit json file  ----------*- C++ -*-===//
//
// This file contains functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <iostream>

using namespace dynamatic;
using namespace dynamatic::buffer;

/// Get the full operation name
static std::string getOperationFullName(Operation *op) {
  std::string fullName = op->getName().getStringRef().str();
  return fullName;
}

std::string buffer::getOperationShortStrName(Operation *op) {
  std::string fullName = getOperationFullName(op);
  size_t pos = fullName.find('.');
  return fullName.substr(pos + 1);
}

/// For a channel, indicated by a value, get its port width if exists,
/// otherwise, return CPP_MAX_WIDTH
static unsigned getPortWidth(Value channel) {
  unsigned portBitWidth = bitwidth::CPP_MAX_WIDTH;
  if (isa<NoneType>(channel.getType()))
    return 0;
  if (isa<IntegerType, FloatType>(channel.getType()))
    portBitWidth = channel.getType().getIntOrFloatBitWidth();
  return portBitWidth;
}

/// Get the precise time information w.r.t to the bitwidth from a vector store
/// the {bitwidth, time} info.
static double
getBitWidthMatchedTimeInfo(unsigned bitWidth,
                           std::vector<std::pair<unsigned, double>> &timeInfo) {
  // Sort the vector based on pair.first (unsigned)
  std::sort(
      timeInfo.begin(), timeInfo.end(),
      [](const std::pair<unsigned, double> &a,
         const std::pair<unsigned, double> &b) { return a.first < b.first; });
  for (const auto &[width, opDelay] : timeInfo)
    if (width >= bitWidth)
      return opDelay;

  // return the delay of the largest bitwidth
  return (timeInfo.end() - 1)->second;
}

double buffer::getPortDelay(Value channel,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
                            std::string direction) {
  std::string opName;
  if (direction == "in") {
    opName = getOperationFullName(channel.getDefiningOp());
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].inPortDataDelay);

  } else if (direction == "out") {
    auto dstOp = channel.getUsers().begin();
    opName = getOperationFullName(*dstOp);
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].outPortDataDelay);
  }
  return 0.0;
}

double buffer::getUnitDelay(Operation *op,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
                            std::string type) {
  double delay;
  std::string opName = getOperationFullName(op);
  // check whether delay information exists
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  // get delay w.r.t to bitwidth
  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  if (type == "data")
    delay =
        getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].dataDelay);
  else if (type == "valid")
    delay = unitInfo[opName].validDelay;
  else if (type == "ready")
    delay = unitInfo[opName].readyDelay;
  else
    delay = 0.0;

  return delay;
}

double
buffer::getCombinationalDelay(Operation *op,
                              std::map<std::string, buffer::UnitInfo> &unitInfo,
                              std::string type) {
  std::string opName = getOperationFullName(op);
  if (unitInfo.find(getOperationFullName(op)) == unitInfo.end())
    return 0.0;

  double inPortDelay = 0.0;
  double outPortDelay = 0.0;
  double unitDelay = getUnitDelay(op, unitInfo, type);

  unsigned unitBitWidth = 1;
  for (auto operand : op->getOperands())
    unitBitWidth = std::max(unitBitWidth, getPortWidth(operand));

  if (type == "data") {
    inPortDelay = getBitWidthMatchedTimeInfo(unitBitWidth,
                                             unitInfo[opName].inPortDataDelay);
    outPortDelay = getBitWidthMatchedTimeInfo(
        unitBitWidth, unitInfo[opName].outPortDataDelay);
  } else if (type == "valid") {
    inPortDelay = unitInfo[opName].inPortValidDelay;
    outPortDelay = unitInfo[opName].outPortValidDelay;
  } else if (type == "ready") {
    inPortDelay = unitInfo[opName].inPortReadyDelay;
    outPortDelay = unitInfo[opName].outPortReadyDelay;
  }

  return unitDelay + inPortDelay + outPortDelay;
}

double
buffer::getUnitLatency(Operation *op,
                       std::map<std::string, buffer::UnitInfo> &unitInfo) {
  std::string opName = getOperationFullName(op);
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));

  double latency =
      getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].latency);

  return latency;
}

LogicalResult
buffer::setChannelBufProps(std::vector<Value> &channels,
                           DenseMap<Value, ChannelBufProps> &ChannelBufProps,
                           std::map<std::string, UnitInfo> &unitInfo) {
  for (auto &ch : channels) {
    Operation *srcOp = ch.getDefiningOp();
    Operation *dstOp = *(ch.getUsers().begin());

    // skip the channel that is the block argument
    if (!srcOp || !dstOp)
      continue;

    std::string srcName = srcOp->getName().getStringRef().str();
    std::string dstName = dstOp->getName().getStringRef().str();
    // set merge with multiple input to have at least one transparent buffer
    if (isa<handshake::MergeOp>(srcOp) && srcOp->getNumOperands() > 1)
      ChannelBufProps[ch].minTrans = 1;

    // TODO: set selectOp always select the frequent input
    if (isa<arith::SelectOp>(dstOp))
      if (dstOp->getOperand(2) == ch) {
        ChannelBufProps[ch].maxTrans = 0;
        ChannelBufProps[ch].minOpaque = 0;
      }

    if (isa<handshake::MemoryControllerOp>(srcOp) ||
        isa<handshake::MemoryControllerOp>(dstOp)) {
      ChannelBufProps[ch].maxOpaque = 0;
      ChannelBufProps[ch].maxTrans = 0;
    }

    // set channel buffer properties w.r.t to input file
    if (unitInfo.count(srcName) > 0) {
      ChannelBufProps[ch].minTrans += unitInfo[srcName].outPortTransBuf;
      ChannelBufProps[ch].minOpaque += unitInfo[srcName].outPortOpBuf;
    }

    if (unitInfo.count(dstName) > 0) {
      ChannelBufProps[ch].minTrans += unitInfo[dstName].inPortTransBuf;
      ChannelBufProps[ch].minOpaque += unitInfo[dstName].inPortOpBuf;
    }

    if (ChannelBufProps[ch].minTrans > 0 && ChannelBufProps[ch].minOpaque > 0)
      return failure(); // cannot satisfy the constraint
  }
  return success();
}

static void parseBitWidthPair(llvm::json::Object jsonData,
                              std::vector<std::pair<unsigned, double>> &data) {
  for (const auto &[bitWidth, value] : jsonData) {
    llvm::StringRef bitKey(bitWidth);
    unsigned key = std::stoi(bitKey.str());
    double info = value.getAsNumber().value();
    data.emplace_back(key, info);
  }
}

LogicalResult buffer::parseJson(const std::string &jsonFile,
                                std::map<std::string, UnitInfo> &unitInfo) {

  // Operations that is supported to use its time information.
  std::vector<std::string> opNames = {
      "arith.cmpi",        "arith.addi",
      "arith.subi",        "arith.muli",
      "arith.extsi",       "handshake.d_load",
      "handshake.d_store", "handshake.merge",
      "arith.addf",        "arith.subf",
      "arith.mulf",        "arith.divui",
      "arith.divsi",       "arith.divf",
      "arith.cmpf",        "handshake.control_merge",
      "handshake.fork",    "handshake.d_return",
      "handshake.cond_br", "handshake.end",
      "arith.andi",        "arith.ori",
      "arith.xori",        "arith.shli",
      "arith.shrsi",       "arith.shrui",
      "arith.select",      "handshake.mux"};
  std::string opName;

  std::ifstream inputFile(jsonFile);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open file.\n";
    return failure();
  }

  // Read the JSON content from the file
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line)) {
    jsonString += line;
  }

  // Parse the JSON
  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonString);
  if (!jsonValue)
    return failure();

  auto data = jsonValue->getAsObject();
  for (std::string &op : opNames) {
    auto unitInfoJson = data->getObject(op);
    // parse the bitwidth and its corresponding latency for data
    parseBitWidthPair(*unitInfoJson->getObject("latency"),
                      unitInfo[op].latency);
    parseBitWidthPair(*unitInfoJson->getObject("delay")->getObject("data"),
                      unitInfo[op].dataDelay);
    parseBitWidthPair(
        *unitInfoJson->getObject("inport")->getObject("delay")->getObject(
            "data"),
        unitInfo[op].inPortDataDelay);
    parseBitWidthPair(
        *unitInfoJson->getObject("outport")->getObject("delay")->getObject(
            "data"),
        unitInfo[op].outPortDataDelay);

    // parse the bitwidth and its corresponding latency for valid and ready
    // The valid and ready signal is 1 bit
    unitInfo[op].validDelay = unitInfoJson->getObject("delay")
                                  ->getObject("valid")
                                  ->getNumber("1")
                                  .value();
    unitInfo[op].readyDelay = unitInfoJson->getObject("delay")
                                  ->getObject("ready")
                                  ->getNumber("1")
                                  .value();
    unitInfo[op].inPortValidDelay = unitInfoJson->getObject("inport")
                                        ->getObject("delay")
                                        ->getObject("valid")
                                        ->getNumber("1")
                                        .value();
    unitInfo[op].inPortReadyDelay = unitInfoJson->getObject("inport")
                                        ->getObject("delay")
                                        ->getObject("ready")
                                        ->getNumber("1")
                                        .value();
    unitInfo[op].outPortValidDelay = unitInfoJson->getObject("outport")
                                         ->getObject("delay")
                                         ->getObject("valid")
                                         ->getNumber("1")
                                         .value();
    unitInfo[op].outPortReadyDelay = unitInfoJson->getObject("outport")
                                         ->getObject("delay")
                                         ->getObject("ready")
                                         ->getNumber("1")
                                         .value();

    unitInfo[op].inPortTransBuf = unitInfoJson->getObject("inport")
                                      ->getNumber("transparentBuffer")
                                      .value();
    unitInfo[op].inPortOpBuf =
        unitInfoJson->getObject("inport")->getNumber("opaqueBuffer").value();

    unitInfo[op].outPortTransBuf = unitInfoJson->getObject("outport")
                                       ->getNumber("transparentBuffer")
                                       .value();
    unitInfo[op].outPortOpBuf =
        unitInfoJson->getObject("outport")->getNumber("opaqueBuffer").value();
  }

  return success();
}
