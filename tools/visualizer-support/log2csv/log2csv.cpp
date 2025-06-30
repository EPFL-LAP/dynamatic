//===- log2csv.cpp - Converts LOG file to CSV for visualizer-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms a Modelsim LOG file into a CSV-formatted sequence of channel state
// changes, utilized by the visualizer.
//
//===----------------------------------------------------------------------===//

#include "../VisualizerSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> handshakeIRFile(cl::Positional, cl::Required,
                                            cl::desc("<input file>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> level(cl::Positional, cl::Required,
                                  cl::desc("<duv level>"),
                                  cl::cat(mainCategory));

static cl::opt<std::string> logFile(cl::Positional, cl::Required,
                                    cl::desc("<path to LOG file>"),
                                    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string> kernelName(cl::Positional, cl::Required,
                                       cl::desc("<kernel name>"), cl::init(""),
                                       cl::cat(mainCategory));

static std::optional<WireReference>
fromSignal(StringRef signalName, const llvm::StringMap<Value> &ports) {
  // Check if this is a data signal
  if (signalName.ends_with(")")) {
    // Try to extract the vector's index from the signal name
    size_t idx = signalName.rfind("(");
    if (idx == std::string::npos)
      return std::nullopt;
    StringRef dataIdxStr = signalName.slice(idx + 1, signalName.size() - 1);

    // Try to convert the index to an unsigned
    unsigned dataIdx;
    if (dataIdxStr.getAsInteger(10, dataIdx))
      return std::nullopt;

    // Try to match the port's name to an SSA value
    StringRef portName = signalName.slice(0, idx);
    auto portIt = ports.find(portName);
    if (portIt == ports.end())
      return std::nullopt;
    return WireReference{portIt->second, SignalType::DATA, dataIdx};
  }

  auto checkSingleWire =
      [&](StringRef suffix,
          SignalType signalType) -> std::optional<WireReference> {
    if (!signalName.ends_with(suffix))
      return std::nullopt;

    // Try to match the port's name to an SSA value
    StringRef portName = signalName.drop_back(suffix.size());
    auto portIt = ports.find(portName);
    if (portIt == ports.end())
      return std::nullopt;
    return WireReference{portIt->second, signalType, std::nullopt};
  };

  if (auto wireOpt =
          checkSingleWire(WireReference::VALID_SUFFIX, SignalType::VALID))
    return wireOpt;
  return checkSingleWire(WireReference::READY_SUFFIX, SignalType::READY);
}

static WireState wireStateFromLog(StringRef token) {
  StringRef state = token.drop_front(2).drop_back(2);
  if (state == "1")
    return WireState::LOGIC_1;
  if (state == "0")
    return WireState::LOGIC_0;
  return WireState::UNDEFINED;
}

static constexpr unsigned long long PERIOD_NS = 4;

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Converts a Modelsim LOG file into a CSV format used by the visualizer.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(handshakeIRFile.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '"
                 << handshakeIRFile << "': " << error.message() << "\n";
    return 1;
  }

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
                      handshake::HandshakeDialect, math::MathDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Open the LOG file
  std::ifstream logFileStream(logFile);
  if (!logFileStream.is_open()) {
    llvm::errs() << "Failed to open LOG file @ \"" << logFile << "\"\n";
    return 1;
  }

  writeCSVHeader(llvm::outs());

  llvm::StringMap<Value> signalNameToValue;
  if (failed(mapSignalsToValues(*modOp, signalNameToValue)))
    return 1;

  mlir::DenseMap<Value, SignalInfo> valueToSignalInfo;
  mlir::DenseMap<Value, ChannelState> state;
  for (auto &[signalName, val] : signalNameToValue) {
    if (isa<MemRefType>(val.getType()))
      continue;
    valueToSignalInfo.try_emplace(val, val, signalName);
    state.try_emplace(val, val.getType());
  }

  std::map<size_t, WireReference> wires;
  mlir::DenseSet<Value> toUpdate;
  size_t cycle = 0;
  unsigned long long period = PERIOD_NS, halfPeriod = period >> 1;

  // Read the LOG file line by line
  std::string event;
  for (size_t lineNum = 1; std::getline(logFileStream, event); ++lineNum) {
    // Split the line into space-separated tokens
    SmallVector<StringRef> tokens;
    StringRef{event}.split(tokens, ' ');
    if (tokens.empty())
      continue;

    auto error = [&](const Twine &msg) -> LogicalResult {
      llvm::errs() << "On line " << lineNum << ": " << msg << "\nIn: " << event
                   << "\n";
      return failure();
    };

    using LogCallback = std::function<LogicalResult()>;
    auto handleLogType = [&](StringRef identifier, size_t minNumTokens,
                             const LogCallback &callback) -> bool {
      if (tokens.front() != identifier)
        return false;
      if (tokens.size() < minNumTokens) {
        (void)error("Expected at least " + std::to_string(minNumTokens) +
                    " tokens, but got " + std::to_string(tokens.size()));
        exit(1);
      }
      if (failed(callback()))
        exit(1);
      return true;
    };

    LogCallback setResolution = [&]() {
      if (tokens[1] != "timestep")
        return success();

      // Resolution is always specified at the top of the log file.

      // Example: tokens[2] = "\"1e-15\"", resolutionStr = "1e-15"
      llvm::StringRef resolutionStr = tokens[2].substr(1, tokens[2].size() - 2);

      // Example: exponent = -15
      int exponent;
      if (resolutionStr.split("e").second.getAsInteger(10, exponent)) {
        return error("expected resolution in scientific notation, but got " +
                     tokens[2]);
      }

      if (exponent > -9) {
        return error(
            "Resolution must be at least nanoseconds (10e-9), but got " +
            tokens[2]);
      }

      period = PERIOD_NS *
               static_cast<unsigned long long>(std::pow(10, -9 - exponent));
      halfPeriod = period >> 1;

      return success();
    };

    LogCallback setSignalAssociation = [&]() {
      unsigned wireID;
      if (tokens[2].getAsInteger(10, wireID)) {
        return error("expected integer identifier for signal, but got " +
                     tokens[2]);
      }
      StringRef signalName = StringRef{tokens[1]}.drop_front(level.size());
      if (auto wire = fromSignal(signalName, signalNameToValue))
        wires.insert({wireID, *wire});
      return success();
    };

    LogCallback newTimestep = [&]() {
      writeChannelStateChanges(llvm::outs(), cycle, toUpdate, state,
                               valueToSignalInfo);
      toUpdate.clear();

      // Parse the current simulation time and derive the current cycle number
      std::string timeStr = tokens[1].str();
      timeStr.erase(std::remove(timeStr.begin(), timeStr.end(), '.'),
                    timeStr.end());
      unsigned long long time;
      if (StringRef{timeStr}.getAsInteger(10, time)) {
        return error("expected integer identifier for time, but got " +
                     timeStr);
      }
      cycle = (time < period) ? 0 : (time - halfPeriod) / period + 1;
      return success();
    };

    LogCallback changeWireState = [&]() {
      unsigned wireID;
      if (tokens[1].getAsInteger(10, wireID)) {
        return error("expected integer identifier for signal, but got " +
                     tokens[1]);
      }

      // Just ignore wires we don't know
      auto wireIt = wires.find(wireID);
      if (wireIt == wires.end())
        return success();

      WireReference &wire = wireIt->second;
      auto channelStateIt = state.find(wire.value);
      assert(channelStateIt != state.end());
      ChannelState &channelState = channelStateIt->second;
      WireState wireState = wireStateFromLog(tokens[2]);

      switch (wire.signalType) {
      case SignalType::VALID:
        channelState.valid = wireState;
        break;
      case SignalType::READY:
        channelState.ready = wireState;
        break;
      case SignalType::DATA:
        if (!wire.idx)
          return error("index undefined for data wire");
        if (*wire.idx >= channelState.data.size()) {
          return error("index out of bounds, expected number between 0 and " +
                       std::to_string(channelState.data.size()) + ", but got " +
                       std::to_string(*wire.idx));
        }
        channelState.data[*wire.idx] = wireState;
        break;
      }

      toUpdate.insert(wire.value);
      return success();
    };

    handleLogType("D", 3, setSignalAssociation) ||
        handleLogType("T", 2, newTimestep) ||
        handleLogType("S", 2, changeWireState) ||
        handleLogType("P", 3, setResolution);
    tokens.clear();
  }

  return 0;
}
