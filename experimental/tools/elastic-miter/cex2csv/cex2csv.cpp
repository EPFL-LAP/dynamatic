//===- log2csv.cpp - Converts counterexample to visualizer CSV---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms a NuSMV counterexample into a CSV-formatted sequence of channel
// state changes, utilized by the visualizer.
// experimental/tools/elastic-miter/CMakeLists.txt
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
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
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <optional>
#include <regex>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

/// Specifies the handshake IR file, which is needed to convert channel names
/// into indices due to the CSV format.
static cl::opt<std::string> handshakeIRFile(cl::Positional, cl::Required,
                                            cl::desc("<Handshake IR file>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> resultFile(cl::Positional, cl::Required,
                                       cl::desc("<path to the result file>"),
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
    handshake::PortNamer opNameGen(&op);
    for (auto [idx, oprd] : llvm::enumerate(op.getOperands())) {
      std::string signalName =
          getUniqueName(&op).str() + "_" + opNameGen.getInputName(idx).str();
      ports.insert({signalName, oprd});
    }
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      std::string signalName =
          getUniqueName(&op).str() + "_" + opNameGen.getOutputName(idx).str();
      ports.insert({signalName, res});
    }
  }

  return success();
}

static constexpr StringLiteral ACCEPT("accept"), STALL("stall"),
    TRANSFER("transfer"), IDLE("idle"), UNDEFINED("undefined");

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "Converts a NuSMV counterexample into a CSV "
                              "format used by the visualizer.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(handshakeIRFile.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '"
                 << handshakeIRFile << "': " << error.message() << "\n";
    return 1;
  }

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, handshake::HandshakeDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Open the result file
  std::ifstream resultFileStream(resultFile);
  if (!resultFileStream.is_open()) {
    llvm::errs() << "Failed to open result file @ \"" << resultFile << "\"\n";
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

  std::string line;

  // Match the NuSMV version. The first line is:
  // *** This is NuSMV 2.7.0 (compiled on (unknown))
  std::getline(resultFileStream, line);
  std::regex versionRegex(R"(NuSMV\s+(\d+\.\d+\.\d+))");
  std::smatch match;
  if (std::regex_search(line, match, versionRegex)) {
    llvm::errs() << "Detected NuSMV version: " << match[1]
                 << " (This program is tested with 2.7.0).\n";
  } else {
    llvm::errs()
        << "Warning: Couldn't detect the NuSMV version from the result file.\n";
  }

  bool foundCex = false;
  StringRef lineRef;
  size_t cycle;
  mlir::DenseSet<Value> toUpdate;

  while (std::getline(resultFileStream, line)) {
    lineRef = line;
    if (foundCex) {
      StringRef trimmedLine = lineRef.trim();
      if (trimmedLine.starts_with("-> State:")) {
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
        // Update the cycle
        // trimmedLine is: -> State: 2.1 <-
        // Extract the "1"
        bool failed = trimmedLine.substr(0, trimmedLine.size() - 3)
                          .split('.')
                          .second.getAsInteger(10, cycle);
        if (failed) {
          llvm::errs() << "Failed to parse cycle from line: " << line << "\n";
          return 1;
        }
        llvm::errs() << "Cycle: " << cycle << "\n";
      } else {
        if (!trimmedLine.starts_with(kernelName + ".")) {
          llvm::errs() << "Warning: skipping line: " << line << "\n";
          continue;
        }
        StringRef signal = trimmedLine.split('=').first.trim();
        StringRef logicalValue = trimmedLine.split('=').second.trim();

        SmallVector<StringRef> tokens;
        signal.split(tokens, '.');

        if (tokens.size() < 3) {
          llvm::errs() << "Warning: expected at least 3 tokens in signal '"
                       << signal << "', but got " << tokens.size() << "\n";
          continue;
        }

        StringRef opName = tokens[1];
        StringRef signalName = tokens[2];

        SignalType signalType;
        StringRef channelName;
        if (signalName.ends_with("_valid")) {
          signalType = SignalType::VALID;
          channelName = signalName.drop_back(6);
        } else if (signalName.ends_with("_ready")) {
          signalType = SignalType::READY;
          channelName = signalName.drop_back(6);
        } else {
          signalType = SignalType::DATA;
          channelName = signalName;
        }

        auto valueIt =
            signalNameToValue.find(opName.str() + "_" + channelName.str());
        if (valueIt == signalNameToValue.end()) {
          // llvm::errs() << "Warning: could not find signal '" << opName << "."
          //              << channelName << "' in the Handshake IR file.\n";
          continue;
        }
        auto channelStateIt = state.find(valueIt->second);
        if (channelStateIt == state.end()) {
          llvm::errs() << "Warning: could not find channel state for signal '"
                       << opName << "." << channelName << "'\n";
          continue;
        }
        WireState wireState;
        if (logicalValue == "TRUE") {
          wireState = WireState::LOGIC_1;
        } else if (logicalValue == "FALSE") {
          wireState = WireState::LOGIC_0;
        } else {
          llvm::errs() << "Warning: expected logical value 'TRUE' or 'FALSE', "
                       << "but got '" << logicalValue << "'\n";
          wireState = WireState::UNDEFINED;
        }

        ChannelState &channelState = channelStateIt->second;
        switch (signalType) {
        case SignalType::VALID:
          channelState.valid = wireState;
          break;
        case SignalType::READY:
          channelState.ready = wireState;
          break;
        case SignalType::DATA:
          // Assuming 1-bit data
          channelState.data[0] = wireState;
          break;
        }

        toUpdate.insert(valueIt->second);
      }
    } else {
      if (lineRef.starts_with("Trace Type:")) {
        foundCex = true;
        continue;
      }
    }
  }
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
    llvm::outs() << cycle << ", " << info.srcComponent << ", " << info.srcPortID
                 << ", " << info.dstComponent << ", " << info.dstPortID << ", "
                 << dataflowState.str() << ", " << channelState.decodeData()
                 << "\n";
  }
  return 0;
}