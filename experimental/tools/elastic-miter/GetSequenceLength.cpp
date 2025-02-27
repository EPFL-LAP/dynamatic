//===- GetSequenceLength.cpp ----------------------------------- *- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include <llvm/ADT/StringSet.h>

#include "CreateWrappers.h"
#include "FabricGeneration.h"
#include "GetSequenceLength.h"
#include "SmvUtils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// Remove all leading and trailing whitespace from a string
static std::string stripString(const std::string &string) {
  std::string newString = string;
  size_t start = newString.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
    newString.clear(); // The string contains only whitespace
  } else {
    // Trim leading and trailing spaces
    size_t end = newString.find_last_not_of(" \t\n\r\f\v");
    newString = newString.substr(start, end - start + 1);
  }
  return newString;
}

// Get the set of states given the path to file containing the output to the
// nuXmv command "print_reachable_states -v;". A set element is defined by a
// string which is the concatenation of the state values, as represented in the
// output file.
static FailureOr<llvm::StringSet<>>
getStateSet(const std::filesystem::path &filePath,
            const std::string &modelName) {
  std::ifstream file(filePath);
  llvm::StringSet<> states;
  std::string line, currentState;
  bool recording = false;

  while (std::getline(file, line)) {
    line = stripString(line);

    if (line.find("warning: the states are more than") != std::string::npos) {
      llvm::errs() << "The number of states exceeded the number that can be "
                      "printed by NuSMV/nuXmv.\n";
      return failure();
    }
    if (line.find("-------") != std::string::npos) {
      if (!currentState.empty()) {
        states.insert(currentState);
      }
      currentState.clear();
      recording = true;
      continue;
    }

    if (!recording)
      continue;

    // Skip if it doesn't start with "miter." or starts with "miter.ndw",
    // indicating it is a ND wire
    if (line.find(modelName + ".", 0) != 0 ||
        line.find(modelName + ".ndw", 0) == 0) {
      continue;
    }
    currentState += line + "\n";
  }
  file.close();

  if (!currentState.empty()) {
    states.insert(currentState);
  }

  return states;
}

// Count how many states are in the set reached by infinite tokens but are not
// in the set reached by finite tokens.
static FailureOr<size_t>
compareReachableStates(const std::string &modelName,
                       const std::filesystem::path &infinitePath,
                       const std::filesystem::path &finitePath) {

  auto failOrInfiniteStates = getStateSet(infinitePath, modelName);
  if (failed(failOrInfiniteStates)) {
    llvm::errs() << "Failed to get the state set with infinite tokens.\n";
    return failure();
  }
  auto failOrFiniteStates = getStateSet(finitePath, modelName);
  if (failed(failOrFiniteStates)) {
    llvm::errs() << "Failed to get the state set with finite tokens.\n";
    return failure();
  }

  llvm::StringSet<> infiniteStateSet = failOrInfiniteStates.value();
  llvm::StringSet<> finiteStateSet = failOrFiniteStates.value();

  if (infiniteStateSet.size() > finiteStateSet.size())
    return infiniteStateSet.size() - finiteStateSet.size();

  size_t differenceCount = 0;
  for (const auto &entry : infiniteStateSet) {
    if (finiteStateSet.find(entry.getKey()) == finiteStateSet.end()) {
      differenceCount++;
    }
  }

  return differenceCount;
}

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::filesystem::path &mlirPath) {

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(outputDir);

  // Add ND wires to the circuit
  auto ret =
      dynamatic::experimental::createReachabilityCircuit(context, mlirPath);
  if (failed(ret)) {
    llvm::errs() << "Failed to create reachability module.\n";
    return failure();
  }
  auto [miterModule, config] = ret.value();

  // The filename is "elastic_miter_<funcOpName>.mlir"
  std::string reachabilityMlirFilename =
      "elastic_miter_" + config.funcName + ".mlir";
  std::filesystem::path reachabilityMlirPath =
      outputDir / reachabilityMlirFilename;

  if (failed(createMlirFile(reachabilityMlirPath, miterModule))) {
    llvm::errs() << "Failed to write miter files.\n";
    return failure();
  }

  auto failOrSmvPair = dynamatic::experimental::handshake2smv(
      reachabilityMlirPath, outputDir, true);
  if (failed(failOrSmvPair))
    return failure();
  auto [dstSmv, smvModelName] = failOrSmvPair.value();

  // Create the wrapper with infinite sequence generators
  auto fail = dynamatic::experimental::createWrapper(
      outputDir / "main_inf.smv", config, smvModelName, 0, false,
      SequenceConstraints());
  if (failed(fail)) {
    llvm::errs() << "Failed to create infinite reachability wrapper.\n";
    return failure();
  }

  // create the .cmd file to print all the reachable states with infinite tokens
  if (failed(createCMDfile(outputDir / "reachability_inf.cmd",
                           outputDir / "main_inf.smv",
                           "print_reachable_states -v;")))
    return failure();

  // Run the cmd file with NuSMV / nuXmv for infinite tokens
  int cmdRet = runSmvCmd(outputDir / "reachability_inf.cmd",
                         outputDir / "inf_states.txt");
  if (cmdRet != 0) {
    llvm::errs()
        << "Failed to analyze reachable states with infinite tokens.\n";
    return failure();
  }

  int numberOfTokens = 1;
  while (true) {
    // TODO remove
    llvm::outs() << "Checking " << numberOfTokens << " tokens.\n";

    std::filesystem::path wrapperPath =
        outputDir / ("main_" + std::to_string(numberOfTokens) + ".smv");

    // Create the wrapper with n-token sequence generators
    auto fail = dynamatic::experimental::createWrapper(
        wrapperPath, config, smvModelName, numberOfTokens, false,
        SequenceConstraints(), true);
    if (failed(fail)) {
      llvm::errs() << "Failed to create " << numberOfTokens
                   << " token reachability wrapper.\n";
      return failure();
    }

    // create the .cmd file to print all the reachable states with n tokens
    if (failed(
            createCMDfile(outputDir / ("reachability_" +
                                       std::to_string(numberOfTokens) + ".cmd"),
                          wrapperPath, "print_reachable_states -v;")))
      return failure();

    // Run the cmd file with NuSMV / nuXmv for infinite tokens
    cmdRet = runSmvCmd(
        outputDir / ("reachability_" + std::to_string(numberOfTokens) + ".cmd"),
        outputDir / ("states_" + std::to_string(numberOfTokens) + ".txt"));
    if (cmdRet != 0) {
      llvm::errs() << "Failed to analyze reachable states with"
                   << std::to_string(numberOfTokens) + " tokens.";
      return failure();
    }

    // Count the number of differences of reachable states
    auto failOrNrOfDifferences =
        dynamatic::experimental::compareReachableStates(
            smvModelName, outputDir / "inf_states.txt",
            outputDir / ("states_" + std::to_string(numberOfTokens) + ".txt"));
    if (failed(failOrNrOfDifferences)) {
      llvm::errs() << "Failed to compare the number of reachable states with "
                   << numberOfTokens << " tokens.\n";
      return failure();
    }

    if (failOrNrOfDifferences.value() != 0) {
      numberOfTokens++;
    } else {
      break;
    }
  }
  return numberOfTokens;
}
} // namespace dynamatic::experimental