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

#include "../VisualizerSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
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

static std::optional<WireReference>
getWireReference(StringRef signalName, const CSVBuilder &builder) {
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

  auto value = builder.getChannel(channelName);
  if (!value.has_value()) {
    return std::nullopt;
  }

  return WireReference{*value, signalType, std::nullopt};
}

static WireState getWireState(StringRef logicalValue) {
  if (logicalValue == "TRUE") {
    return WireState::Logic1;
  }
  if (logicalValue == "FALSE") {
    return WireState::Logic0;
  }
  llvm::errs() << "Warning: expected logical value 'TRUE' or 'FALSE', "
               << "but got '" << logicalValue << "'\n";
  return WireState::Undefined;
}

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

  CSVBuilder csvBuilder(llvm::outs());
  if (failed(csvBuilder.initialize(*modOp, true))) {
    llvm::errs() << "Failed to initialize CSV builder with the module.\n";
    return 1;
  }

  csvBuilder.writeCSVHeader();

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

  StringRef lineRef;
  while (std::getline(resultFileStream, line)) {
    lineRef = line;
    StringRef trimmedLine = lineRef.trim();
    if (trimmedLine.starts_with("-> State:")) {
      // Update the cycle
      csvBuilder.commitChannelStateChanges();
      // Always increment the cycle by 1, even if another counterexample is
      // found
      // e.g., State 1.1 -> 1.2 -> 2.1 -> 2.2 -> ...
      csvBuilder.updateCycle(csvBuilder.getCycle() + 1);
    } else {
      if (!trimmedLine.starts_with(kernelName + ".")) {
        continue;
      }
      StringRef signal = trimmedLine.split('=').first.trim();
      StringRef logicalValue = trimmedLine.split('=').second.trim();

      SmallVector<StringRef> tokens;
      signal.split(tokens, '.');

      std::string signalName;
      if (tokens.size() == 3) {
        signalName = tokens[1].str() + "_" + tokens[2].str();
      } else if (tokens.size() == 2) {
        // External signals (BlockArgument or Value consumed by EndOp)
        signalName = tokens[1];
      } else {
        continue;
      }

      std::optional<WireReference> wireRef =
          getWireReference(signalName, csvBuilder);
      if (!wireRef) {
        continue;
      }

      auto channelState = csvBuilder.getChannelState(wireRef->value);
      assert(channelState.has_value() &&
             "Expected channel state to be available for the wire reference");

      WireState wireState = getWireState(logicalValue);
      switch (wireRef->signalType) {
      case SignalType::VALID:
        channelState->valid = wireState;
        break;
      case SignalType::READY:
        channelState->ready = wireState;
        break;
      case SignalType::DATA:
        // Assuming 1-bit data
        channelState->data[0] = wireState;
        break;
      }

      csvBuilder.updateChannelState(wireRef->value, *channelState);
    }
  }
  csvBuilder.commitChannelStateChanges();

  return 0;
}