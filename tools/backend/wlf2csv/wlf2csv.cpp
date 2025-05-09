//===- wlf2csv.cpp - Converts WLF file to simpler CSV -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms a WLF file into a CSV-formatted simplified sequence of channel
// state changes.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/System.h"
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> wlfFile(cl::Positional, cl::Required,
                                    cl::desc("<path to WLF file>"),
                                    cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string> kernelName(cl::Positional, cl::Required,
                                       cl::desc("<kernel name>"), cl::init(""),
                                       cl::cat(mainCategory));

namespace {

enum class WireState { UNDEFINED, LOGIC_0, LOGIC_1 };

struct SignalInfo {
  std::string signalName;
  std::string srcComponent;
  unsigned srcPortID;
  std::string dstComponent;
  unsigned dstPortID;

  SignalInfo(Value val, StringRef signalName);
};

struct ChannelState {
  Type type;

  WireState valid = WireState::UNDEFINED;
  WireState ready = WireState::UNDEFINED;
  std::vector<WireState> data;

  ChannelState(Type type)
      : type(type), data(std::vector{handshake::getHandshakeTypeBitWidth(type),
                                     WireState::UNDEFINED}) {}

  std::string decodeData() const {
    if (data.empty())
      return "";

    // Build a string from the vector of bits
    std::string dataString;
    bool partlyUndef = false;
    for (WireState bit : llvm::reverse(data)) {
      switch (bit) {
      case WireState::UNDEFINED:
        dataString.push_back('u');
        partlyUndef = true;
        break;
      case WireState::LOGIC_0:
        dataString.push_back('0');
        break;
      case WireState::LOGIC_1:
        dataString.push_back('1');
        break;
      }
    }
    if (partlyUndef)
      return dataString;

    auto channelType = cast<handshake::ChannelType>(type);
    auto dataType = channelType.getDataType();
    if (auto intType = dyn_cast<IntegerType>(dataType)) {
      APInt intVal(data.size(), dataString, 2);
      if (intType.isSigned())
        return std::to_string(intVal.getSExtValue());
      return std::to_string(intVal.getZExtValue());
    }
    if (auto floatType = dyn_cast<FloatType>(dataType)) {
      APFloat floatVal(floatType.getFloatSemantics(), dataString);
      return std::to_string(floatVal.convertToDouble());
    }
    return "undef";
  }
};

struct WireReference {
  static constexpr StringLiteral VALID_SUFFIX = "_valid",
                                 READY_SUFFIX = "_ready";

  Value value;
  SignalType signalType;
  std::optional<size_t> idx;

  static std::optional<WireReference>
  fromSignal(StringRef signalName, const llvm::StringMap<Value> &ports);
};
} // namespace

SignalInfo::SignalInfo(Value val, StringRef signalName)
    : signalName(signalName) {
  // Derive the source component's name and ID
  if (auto res = dyn_cast<OpResult>(val)) {
    Operation *producerOp = res.getOwner();
    srcPortID = res.getResultNumber();
    srcComponent = getUniqueName(producerOp);
  } else {
    auto arg = cast<BlockArgument>(val);
    auto funcOp = cast<handshake::FuncOp>(arg.getParentBlock()->getParentOp());
    srcPortID = arg.getArgNumber();
    srcComponent = funcOp.getArgName(arg.getArgNumber()).str();
  }

  // Derive the destination component's name and ID
  OpOperand &oprd = *val.getUses().begin();
  Operation *consumerOp = oprd.getOwner();
  if (!isa<MemRefType>(oprd.get().getType()))
    dstPortID = oprd.getOperandNumber();
  if (isa<handshake::EndOp>(consumerOp)) {
    auto funcOp = cast<handshake::FuncOp>(val.getParentBlock()->getParentOp());
    dstComponent = funcOp.getResName(oprd.getOperandNumber()).str();
  } else {
    dstComponent = getUniqueName(consumerOp);
  }
}

std::optional<WireReference>
WireReference::fromSignal(StringRef signalName,
                          const llvm::StringMap<Value> &ports) {
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

  auto checkSingleWire = [&](StringRef suffix,
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

  if (auto wireOpt = checkSingleWire(VALID_SUFFIX, SignalType::VALID))
    return wireOpt;
  return checkSingleWire(READY_SUFFIX, SignalType::READY);
}

static WireState wireStateFromLog(StringRef token) {
  StringRef state = token.drop_front(2).drop_back(2);
  if (state == "1")
    return WireState::LOGIC_1;
  if (state == "0")
    return WireState::LOGIC_0;
  return WireState::UNDEFINED;
}

static LogicalResult mapSignalsToValues(mlir::ModuleOp modOp,
                                        llvm::StringMap<Value> &ports) {
  // Extract the non-external Handshake function from the module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : modOp.getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      return modOp.emitError() << "we currently only support one non-external "
                                  "handshake function per module";
    }
    funcOp = op;
  }
  if (!funcOp)
    return modOp.emitError() << "No Handshake function in input module";

  // Make sure all operations inside the function have names
  NameAnalysis namer(funcOp);
  if (!namer.areAllOpsNamed()) {
    return funcOp.emitError() << "Not all operations in the function have "
                                 "names, this is a requirement";
  }

  // First associate names to all function arguments
  handshake::PortNamer argNameGen(funcOp);
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
    ports.insert({argNameGen.getInputName(idx), arg});

  // Then associate names to each operation's results
  for (Operation &op : funcOp.getOps()) {
    handshake::PortNamer resNameGen(&op);
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      std::string signalName =
          getUniqueName(&op).str() + "_" + resNameGen.getOutputName(idx).str();
      ports.insert({signalName, res});
    }
  }

  return success();
}

static constexpr unsigned long long PERIOD = 4000, HALF_PERIOD = PERIOD >> 1;

static constexpr StringLiteral ACCEPT("accept"), STALL("stall"),
    TRANSFER("transfer"), IDLE("idle"), UNDEFINED("undefined");

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a VHDL design corresponding to an input HW-level IR. "
      "JSON-formatted RTL configuration files encode the procedure to "
      "instantiate/generate external HW modules present in the input IR.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
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

  std::string level = "duv/" + kernelName + "_wrapped/";
  std::string logFile = std::filesystem::temp_directory_path().string() +
                        sys::path::get_separator().str() + "dynamatic_" +
                        kernelName + ".log";
  if (int ret = exec("wlf2log -l", level, "-o", logFile, StringRef{wlfFile})) {
    llvm::errs() << "Failed to convert WLF file @ \"" << wlfFile
                 << "\" into LOG file\n";
    return ret;
  }

  // Open the LOG file
  std::ifstream logFileStream(logFile);
  if (!logFileStream.is_open()) {
    llvm::errs() << "Failed to open LOG file @ \"" << logFile << "\"\n";
    return 1;
  }

  llvm::outs() << "cycle, src_component, src_port, dst_component, dst_port, "
                  "state, data\n";

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

    LogCallback setSignalAssociation = [&]() {
      unsigned wireID;
      if (tokens[2].getAsInteger(10, wireID)) {
        return error("expected integer identifier for signal, but got " +
                     tokens[2]);
      }
      StringRef signalName = StringRef{tokens[1]}.drop_front(level.size());
      if (auto wire = WireReference::fromSignal(signalName, signalNameToValue))
        wires.insert({wireID, *wire});
      return success();
    };

    LogCallback newTimestep = [&]() {
      for (Value val : toUpdate) {
        const ChannelState &channelState = state.at(val);
        WireState valid = channelState.valid;
        WireState ready = channelState.ready;

        StringRef dataflowState;
        if (valid != WireState::LOGIC_1 && ready == WireState::LOGIC_1)
          dataflowState = ACCEPT;
        else if (valid == WireState::LOGIC_1)
          dataflowState = ready != WireState::LOGIC_1 ? STALL : TRANSFER;
        else if (valid == WireState::LOGIC_0 && ready == WireState::LOGIC_0)
          dataflowState = IDLE;
        else
          dataflowState = UNDEFINED;

        const SignalInfo &info = valueToSignalInfo.at(val);
        llvm::outs() << cycle << ", " << info.srcComponent << ", "
                     << info.srcPortID << ", " << info.dstComponent << ", "
                     << info.dstPortID << ", " << dataflowState.str() << ", "
                     << channelState.decodeData() << "\n";
      }
      toUpdate.clear();

      // Parse the current simulation time and derive the current cycle number
      std::string timeStr = tokens[1].str();
      timeStr.erase(std::remove(timeStr.begin(), timeStr.end(), '.'),
                    timeStr.end());
      unsigned time;
      if (StringRef{timeStr}.getAsInteger(10, time)) {
        return error("expected integer identifier for time, but got " +
                     timeStr);
      }
      cycle = (time < PERIOD) ? 0 : (time - HALF_PERIOD) / PERIOD + 1;
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
        handleLogType("S", 2, changeWireState);
    tokens.clear();
  }

  return 0;
}
