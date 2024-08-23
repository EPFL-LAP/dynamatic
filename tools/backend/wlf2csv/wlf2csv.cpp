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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
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
#include "llvm/ADT/TypeSwitch.h"
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

  /// Finds the position (block index and operand index) of a value in the
  /// inputs of a memory interface.
  static std::pair<size_t, size_t> findValueInGroups(FuncMemoryPorts &ports,
                                                     Value val) {
    unsigned numGroups = ports.getNumGroups();
    unsigned accInputIdx = 0;
    for (size_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
      ValueRange groupInputs = ports.getGroupInputs(groupIdx);
      accInputIdx += groupInputs.size();
      for (auto [inputIdx, input] : llvm::enumerate(groupInputs)) {
        if (input == val)
          return std::make_pair(groupIdx, inputIdx);
      }
    }

    // Value must belong to a port with another memory interface, find the one
    for (auto [inputIdx, input] :
         llvm::enumerate(ports.getInterfacesInputs())) {
      if (input == val)
        return std::make_pair(numGroups, inputIdx + accInputIdx);
    }
    llvm_unreachable("value should be an operand to the memory interface");
  }

  /// Corrects for different output port ordering conventions with legacy
  /// Dynamatic.
  static size_t fixOutputPortNumber(Operation *op, size_t idx) {
    return llvm::TypeSwitch<Operation *, size_t>(op)
        .Case<handshake::ConditionalBranchOp>([&](auto) {
          // Legacy Dynamatic has the data operand before the condition operand
          return idx;
        })
        .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>(
            [&](auto) {
              // Legacy Dynamatic has the data operand/result before the address
              // operand/result
              return 1 - idx;
            })
        .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
          // Legacy Dynamatic places the end control signal before the signals
          // going to the MC, if one is connected
          LSQPorts lsqPorts = lsqOp.getPorts();
          if (!lsqPorts.hasAnyPort<MCLoadStorePort>())
            return idx;

          // End control signal succeeded by laad address, store address, store
          // data
          if (idx == lsqOp.getNumResults() - 1)
            return idx - 3;

          // Signals to MC preceeded by end control signal
          unsigned numLoads = lsqPorts.getNumPorts<LSQLoadPort>();
          if (idx >= numLoads)
            return idx + 1;
          return idx;
        })
        .Default([&](auto) { return idx; });
  }

  /// Corrects for different input port ordering conventions with legacy
  /// Dynamatic.
  static size_t fixInputPortNumber(Operation *op, size_t idx) {
    return llvm::TypeSwitch<Operation *, size_t>(op)
        .Case<handshake::ConditionalBranchOp>([&](auto) {
          // Legacy Dynamatic has the data operand before the condition operand
          return 1 - idx;
        })
        .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>(
            [&](auto) {
              // Legacy Dynamatic has the data operand/result before the address
              // operand/result
              return 1 - idx;
            })
        .Case<handshake::MemoryOpInterface>(
            [&](handshake::MemoryOpInterface memOp) {
              Value val = op->getOperand(idx);

              // Legacy Dynamatic puts all control operands before all data
              // operands, whereas for us each control operand appears just
              // before the data inputs of the group it corresponds to
              FuncMemoryPorts ports = getMemoryPorts(memOp);

              // Determine total number of control operands
              unsigned ctrlCount = ports.getNumPorts<ControlPort>();

              // Figure out where the value lies
              auto [groupIDx, opIdx] = findValueInGroups(ports, val);

              if (groupIDx == ports.getNumGroups()) {
                // If the group index is equal to the number of connected
                // groups, then the operand index points directly to the
                // matching port in legacy Dynamatic's conventions
                return opIdx;
              }

              // Figure out at which index the value would be in legacy
              // Dynamatic's interface
              bool valGroupHasControl = ports.groups[groupIDx].hasControl();
              if (opIdx == 0 && valGroupHasControl) {
                // Value is a control input
                size_t fixedIdx = 0;
                for (size_t i = 0; i < groupIDx; i++)
                  if (ports.groups[i].hasControl())
                    fixedIdx++;
                return fixedIdx;
              }

              // Value is a data input
              size_t fixedIdx = ctrlCount;
              for (size_t i = 0; i < groupIDx; i++) {
                // Add number of data inputs corresponding to the group, minus
                // the control input which was already accounted for (if
                // present)
                fixedIdx += ports.groups[i].getNumInputs();
                if (ports.groups[i].hasControl())
                  --fixedIdx;
              }
              // Add index offset in the group the value belongs to
              if (valGroupHasControl)
                fixedIdx += opIdx - 1;
              else
                fixedIdx += opIdx;
              return fixedIdx;
            })
        .Default([&](auto) { return idx; });
  }
};

struct ChannelState {
  Type type;

  WireState valid = WireState::UNDEFINED;
  WireState ready = WireState::UNDEFINED;
  std::vector<WireState> data;

  ChannelState(Type type, size_t datawidth)
      : type(type), data(std::vector{datawidth, WireState::UNDEFINED}) {}

  std::string decodeData() const {
    if (data.empty())
      return "";

    // Build a string from the vector of bits
    std::string dataString;
    for (WireState bit : llvm::reverse(data)) {
      switch (bit) {
      case WireState::UNDEFINED:
        return "undef";
      case WireState::LOGIC_0:
        dataString.push_back('0');
        break;
      case WireState::LOGIC_1:
        dataString.push_back('1');
        break;
      }
    }

    if (auto intType = dyn_cast<IntegerType>(type)) {
      APInt intVal(data.size(), dataString, 2);
      if (intType.isSigned())
        return std::to_string(intVal.getSExtValue());
      return std::to_string(intVal.getZExtValue());
    }
    if (auto floatType = dyn_cast<FloatType>(type)) {
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
  SignalType type;
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
    srcPortID = fixOutputPortNumber(producerOp, res.getResultNumber());
    srcComponent = getUniqueName(producerOp);
  } else {
    auto arg = cast<BlockArgument>(val);
    auto funcOp = cast<handshake::FuncOp>(arg.getParentBlock()->getParentOp());
    srcPortID = 0;
    std::string argName = funcOp.getArgName(arg.getArgNumber()).str();
    if (argName == "start") {
      // Legacy contention that the visualizer expects, follow for now
      argName = "start_0";
    }
    srcComponent = argName;
  }

  // Derive the destination component's name and ID
  OpOperand &oprd = *val.getUses().begin();
  Operation *consumerOp = oprd.getOwner();
  if (!isa<MemRefType>(oprd.get().getType()))
    dstPortID = fixInputPortNumber(consumerOp, oprd.getOperandNumber());
  dstComponent = getUniqueName(consumerOp);
}

std::optional<WireReference>
WireReference::fromSignal(StringRef signalName,
                          const llvm::StringMap<Value> &ports) {
  // Check if this is a data signal
  if (signalName.ends_with(")")) {
    // Try to extract the vector's indexfrom the signal name
    size_t idx = signalName.rfind("(");
    if (idx == std::string::npos)
      return std::nullopt;
    StringRef dataIdxStr = signalName.slice(idx + 1, signalName.size() - 1);

    // Try to convert the index to an unsigned
    unsigned long long dataIdx;
    if (getAsUnsignedInteger(dataIdxStr, 10, dataIdx))
      return std::nullopt;

    // Try to match the port's name to an SSA value
    StringRef portName = signalName.slice(0, idx);
    auto portIt = ports.find(portName);
    if (portIt == ports.end())
      return std::nullopt;
    return WireReference{portIt->second, SignalType::DATA, dataIdx};
  }

  auto checkSingleWire = [&](StringRef suffix,
                             SignalType type) -> std::optional<WireReference> {
    if (!signalName.ends_with(suffix))
      return std::nullopt;

    // Try to match the port's name to an SSA value
    StringRef portName = signalName.drop_back(suffix.size());
    auto portIt = ports.find(portName);
    if (portIt == ports.end())
      return std::nullopt;
    return WireReference{portIt->second, type, std::nullopt};
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
  hw::PortNameGenerator argNameGen(funcOp);
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
    ports.insert({argNameGen.getInputName(idx), arg});

  // First associate names to each operation's results
  for (Operation &op : funcOp.getOps()) {
    hw::PortNameGenerator resNameGen(&op);
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
    valueToSignalInfo.insert({val, SignalInfo{val, signalName}});

    unsigned width = 0;
    if (isa<IntegerType, FloatType>(val.getType()))
      width = val.getType().getIntOrFloatBitWidth();
    state.insert({val, ChannelState{val.getType(), width}});
  }

  std::map<size_t, WireReference> wires;
  mlir::DenseSet<Value> toUpdate;
  size_t cycle = 0;

  // Read the LOG file line by line
  std::string event;
  while (std::getline(logFileStream, event)) {
    // Split the line into space-separated tokens
    SmallVector<StringRef> tokens;
    StringRef{event}.split(tokens, ' ');
    if (tokens.empty())
      continue;

    using LogCallback = std::function<LogicalResult()>;
    auto handleLogType = [&](StringRef identifier, size_t minNumTokens,
                             const LogCallback &callback) -> bool {
      if (tokens.front() != identifier)
        return false;
      if (tokens.size() < minNumTokens) {
        llvm::errs() << "Expected at least " << minNumTokens
                     << " tokens, but got " << tokens.size() << "\n";
        exit(1);
      }
      if (failed(callback())) {
        llvm::errs() << "Failed parsing for log type \"" << identifier
                     << "\"\n";
        exit(1);
      }
      return true;
    };

    LogCallback setSignalAssociation = [&]() {
      unsigned long long wireID;
      if (getAsUnsignedInteger(tokens[2], 10, wireID)) {
        llvm::errs() << "Expected integer identifier for signal, but got "
                     << tokens[2] << "\n";
        return failure();
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
      unsigned long long time;
      if (getAsUnsignedInteger(timeStr, 10, time)) {
        llvm::errs() << "Expected integer identifier for time, but got "
                     << timeStr << "\n";
        return failure();
      }
      cycle = (time < PERIOD) ? 0 : (time - HALF_PERIOD) / PERIOD + 1;
      return success();
    };

    LogCallback changeWireState = [&]() {
      unsigned long long wireID;
      if (getAsUnsignedInteger(tokens[1], 10, wireID)) {
        llvm::errs() << "Expected integer identifier for signal, but got "
                     << tokens[1] << "\n";
        return failure();
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

      switch (wire.type) {
      case SignalType::VALID:
        channelState.valid = wireState;
        break;
      case SignalType::READY:
        channelState.ready = wireState;
        break;
      case SignalType::DATA:
        assert(wire.idx && "index undefined for data wire");
        assert(*wire.idx < channelState.data.size() && "index out of bounds");
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
