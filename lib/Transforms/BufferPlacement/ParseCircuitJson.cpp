//===- ParseCircuitJson.cpp -Parse circuit json file  -----------*- C++ -*-===//
//
// This file contains functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

using namespace dynamatic;
using namespace dynamatic::buffer;

std::string buffer::getOperationShortStrName(Operation *op) {
  std::string fullName = op->getName().getStringRef().str();
  size_t pos = fullName.find('.');
  return fullName.substr(pos + 1);
}

static inline unsigned getPortWidth(Value channel) {
  unsigned portBitWidth = bitwidth::CPP_MAX_WIDTH;
  if (isa<IntegerType>(channel.getType()))
    portBitWidth = channel.getType().getIntOrFloatBitWidth();
  return portBitWidth;
}

static double
getBitWidthMatchedTimeInfo(unsigned bitWidth,
                           std::vector<std::pair<unsigned, double>> &timeInfo) {
  double delay;
  // Sort the vector based on pair.first (unsigned)
  std::sort(
      timeInfo.begin(), timeInfo.end(),
      [](const std::pair<unsigned, double> &a,
         const std::pair<unsigned, double> &b) { return a.first < b.first; });
  for (const auto &[width, opDelay] : timeInfo)
    if (width >= bitWidth)
      return opDelay;

  // return the delay of the largest bitwidth
  return timeInfo.end()->second;
}

double buffer::getPortDelay(Value channel,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
                            std::string direction) {
  std::string opName;
  if (direction == "in") {
    opName = getOperationShortStrName(channel.getDefiningOp());
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].inPortDataDelay);

  } else if (direction == "out") {
    auto dstOp = channel.getUsers().begin();
    // TODO: handle multiple users
    // assert(dstOp == channel.getUsers().end() && "There are multiple users!");
    opName = getOperationShortStrName(*dstOp);
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
  std::string opName = getOperationShortStrName(op);
  // check whether delay information exists
  if (unitInfo.find(opName) == unitInfo.end()) {
    llvm::errs() << "Cannot find " << opName << " in unitInfo\n";
    return 0.0;
  }

  // get delay w.r.t to bitwidth
  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  if (type == "data")
    delay =
        getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].dataDelay);
  else if (type == "valid")
    delay = unitInfo[opName].validDelay;
  else if (type == "ready")
    delay = unitInfo[opName].readyDelay;
  // llvm::errs() << opName << " : " << delay << "\n";
  return delay;
}

double
buffer::getCombinationalDelay(Operation *op,
                              std::map<std::string, buffer::UnitInfo> &unitInfo,
                              std::string type) {
  std::string opName = getOperationShortStrName(op);
  if (unitInfo.find(getOperationShortStrName(op)) == unitInfo.end())
    return 0.0;

  double inPortDelay, outPortDelay;
  double unitDelay = getUnitDelay(op, unitInfo, type);

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));

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
  std::string opName = getOperationShortStrName(op);
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  double latency =
      getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].latency);
  return latency;
}

// Function to trim leading and trailing whitespace from a string
std::string buffer::trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  if (first == std::string::npos || last == std::string::npos)
    return "";
  return str.substr(first, (last - first));
}

template <typename T1, typename T2>
static bool existKey(const T1 &value,
                     const std::vector<std::pair<T1, T2>> &pairs) {
  for (const auto &pair : pairs)
    if (pair.first == value)
      return true;

  return false;
}

static size_t parseElement(const std::string &vecString, std::string opName,
                           std::vector<std::pair<unsigned, double>> &data) {

  size_t startPos = vecString.find('{', 0);
  size_t endPos = vecString.find('}', 0);
  std::stringstream ss(vecString.substr(startPos + 1, endPos - startPos));

  std::string token;
  while (std::getline(ss, token, ',')) {
    std::size_t colonPos = token.find(':');
    std::string keyString = token.substr(1, colonPos - 2);
    std::string valString = token.substr(colonPos + 1);
    // llvm::errs() << "key " << keyString << " value " << valString << "end\n";
    unsigned key = std::stoi(keyString);
    double value = std::stod(valString);

    if (!existKey(key, data))
      data.emplace_back(key, value);
  };
  return endPos - startPos + 1;
};

static size_t parseDelay(const std::string &delayString, std::string opName,
                         std::map<std::string, buffer::UnitInfo> &unitInfo,
                         bool inPort = false, bool outPort = false) {
  // llvm::errs() << "-----delay\n";
  size_t pos = 0;

  while (delayString.find("\"", pos) < delayString.size()) {
    pos = delayString.find("\"", pos);
    std::string delayKey;
    delayKey =
        delayString.substr(pos + 1, delayString.find("\"", pos + 1) - pos - 1);
    pos = delayString.find(":", pos + 1);

    if (delayKey == "data") {
      if (inPort)
        pos += parseElement(
            delayString.substr(pos, delayString.find('}', pos) - pos + 1),
            opName, unitInfo[opName].inPortDataDelay);
      else if (outPort)
        pos += parseElement(
            delayString.substr(pos, delayString.find('}', pos) - pos + 1),
            opName, unitInfo[opName].outPortDataDelay);
      else
        pos += parseElement(
            delayString.substr(pos, delayString.find('}', pos) - pos + 1),
            opName, unitInfo[opName].dataDelay);
    } else if (delayKey == "valid") {
      std::vector<std::pair<unsigned, double>> valid;
      pos += parseElement(
          delayString.substr(pos, delayString.find('}', pos) - pos + 1), opName,
          valid);
      if (inPort)
        unitInfo[opName].inPortValidDelay = valid[0].second;
      else if (outPort)
        unitInfo[opName].outPortValidDelay = valid[0].second;
      else
        unitInfo[opName].validDelay = valid[0].second;
    } else if (delayKey == "ready") {
      std::vector<std::pair<unsigned, double>> valid;
      pos += parseElement(
          delayString.substr(pos, delayString.find('}', pos) - pos + 1), opName,
          valid);
      if (inPort)
        unitInfo[opName].inPortReadyDelay = valid[0].second;
      else if (outPort)
        unitInfo[opName].outPortReadyDelay = valid[0].second;
      else
        unitInfo[opName].readyDelay = valid[0].second;
    }
  }
  return delayString.size();
};

static size_t parsePort(const std::string &portString, std::string opName,
                        std::map<std::string, UnitInfo> &unitInfo,
                        bool inPort = false, bool outPort = false) {
  size_t pos = 0;
  std::vector<std::pair<unsigned, double>> dataDelay;
  while (portString.find("\"", pos) < portString.size()) {
    pos = portString.find("\"", pos);

    std::string key;
    key = portString.substr(pos + 1, portString.find("\"", pos + 1) - pos - 1);
    // llvm::errs() << key << "\n";
    pos = portString.find(":", pos + 1);

    if (key == "delay")
      // parse contents inside delay
      pos += parseDelay(
          portString.substr(pos, portString.find('}', pos) - pos + 1), opName,
          unitInfo, inPort, outPort);
    else if (key == "opaqueBuffer") {
      std::string buf =
          portString.substr(pos + 1, portString.find(',', pos + 1) - pos - 1);
      if (inPort)
        unitInfo[opName].inPortOpBuf = std::stoi(buf);
      if (outPort)
        unitInfo[opName].outPortOpBuf = std::stoi(buf);
    } else if (key == "transparentBuffer") {
      std::vector<std::pair<unsigned, double>> transBuf;
      std::string buf =
          portString.substr(pos + 1, portString.find(',', pos + 1) - pos - 1);
      if (inPort)
        unitInfo[opName].inPortTransBuf = std::stoi(buf);
      if (outPort)
        unitInfo[opName].outPortTransBuf = std::stoi(buf);
    }
  }
  return portString.size();
};

static void parseBitWidthPair(json jsonData,
                              std::vector<std::pair<unsigned, double>> &data) {
  for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
    auto key = stoi(it.key());
    double value = it.value();
    data.emplace_back(key, value);
  }
}
// Function to parse the JSON string and extract the required data
LogicalResult buffer::parseJson(const std::string &jsonFile,
                                std::map<std::string, UnitInfo> &unitInfo) {

  // reserve for read buffers information to be used in the channel
  // constraints
  size_t pos = 0;
  std::vector<std::string> opNames = {
      "cmpi",          "addi",    "subi",   "muli",    "extsi",
      "d_load",        "d_store", "merge",  "addf",    "subf",
      "mulf",          "divui",   "divsi",  "divf",    "cmpf",
      "control_merge", "fork",    "return", "cond_br", "end",
      "andi",          "ori",     "xori",   "shli",    "shrsi",
      "shrui",         "select",  "mux"};
  std::string opName;

  std::ifstream file(jsonFile);
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file.\n";
  }

  // Read the file contents into a string
  json data;
  file >> data;
  for (auto op : opNames) {
    auto unitInfoJson = data[op];
    auto latencyJson = unitInfoJson["latency"];
    // parse the bitwidth and its corresponding latency
    parseBitWidthPair(unitInfoJson["latency"], unitInfo[op].latency);
    parseBitWidthPair(unitInfoJson["delay"]["data"], unitInfo[op].dataDelay);
    parseBitWidthPair(unitInfoJson["inport"]["delay"]["data"],
                      unitInfo[op].inPortDataDelay);
    parseBitWidthPair(unitInfoJson["outport"]["delay"]["data"],
                      unitInfo[op].outPortDataDelay);
    unitInfo[op].validDelay = unitInfoJson["delay"]["valid"]["1"];
    unitInfo[op].readyDelay = unitInfoJson["delay"]["ready"]["1"];

    unitInfo[op].inPortValidDelay =
        unitInfoJson["inport"]["delay"]["valid"]["1"];
    unitInfo[op].inPortReadyDelay =
        unitInfoJson["inport"]["delay"]["ready"]["1"];

    unitInfo[op].outPortValidDelay =
        unitInfoJson["outport"]["delay"]["valid"]["1"];
    unitInfo[op].outPortReadyDelay =
        unitInfoJson["outport"]["delay"]["ready"]["1"];

    unitInfo[op].inPortTransBuf = unitInfoJson["inport"]["transparentBuffer"];
    unitInfo[op].inPortReadyDelay = unitInfoJson["inport"]["opaqueBuffer"];

    unitInfo[op].inPortTransBuf = unitInfoJson["outport"]["transparentBuffer"];
    unitInfo[op].outPortReadyDelay = unitInfoJson["outport"]["opaqueBuffer"];

    if (unitInfoJson.is_discarded())
      return failure();
  }

  return success();
  // for (auto op : opNames) {
  //   UnitInfo info = UnitInfo();
  //   for (auto val : data[op]["latency"])

  //     llvm::errs() << val.get<std::string>() << "\n";
  //   // unitInfo[op].latency
  // }

  // while (jsonString.find("\"", pos) < jsonString.size()) {
  //   pos = jsonString.find("\"", pos);

  //   std::string key;
  //   key = jsonString.substr(pos + 1, jsonString.find("\"", pos + 1) - pos -
  //   1);

  //   if (std::find(opNames.begin(), opNames.end(), key) != opNames.end()) {
  //     llvm::errs() << "\n=========opName: " << key << "\n";
  //     opName = key;
  //   }
  //   pos = jsonString.find(":", pos + 1);

  //   // parse latency
  //   if (key == "latency") {
  //     pos += parseElement(
  //         jsonString.substr(pos, jsonString.find(']', pos) - pos + 1),
  //         opName, unitInfo[opName].latency);
  //   }

  //   // unit delay
  //   if (key == "delay")
  //     // parse contents inside delay
  //     pos += parseDelay(
  //         jsonString.substr(pos, jsonString.find('}', pos) - pos + 1),
  //         opName, unitInfo);

  //   if (key == "inport")
  //     pos += parsePort(
  //         jsonString.substr(pos, jsonString.find("}}", pos) - pos + 1),
  //         opName, unitInfo, true, false);

  //   if (key == "outport")
  //     pos += parsePort(
  //         jsonString.substr(pos, jsonString.find("}}", pos) - pos + 1),
  //         opName, unitInfo, false, true);
  // }
}
