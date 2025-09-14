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

#include "Constraints.h"
#include "ElasticMiterTestbench.h"
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

static bool isSeqGenerator(StringRef line) {
  return line.startswith("seq_generator_");
}

static bool isSeqGeneratorOuts(StringRef line) {
  if (!isSeqGenerator(line))
    return false;
  return line.split('.').second.starts_with("outs");
}

static bool isSeqGeneratorCounter(StringRef line) {
  if (!isSeqGenerator(line))
    return false;
  return line.split('.').second.starts_with("counter");
}

static bool isSaturatedSeqGeneratorCounter(StringRef line,
                                           unsigned nrOfTokens) {
  if (!isSeqGeneratorCounter(line))
    return false;
  auto strValue = line.split('=').second.trim();
  unsigned counterValue = std::stoi(strValue.str());

  assert(counterValue <= nrOfTokens &&
         "Counter value exceeds the number of tokens");
  return counterValue == nrOfTokens;
}

static StringRef getSeqGeneratorName(StringRef line) {
  return line.split('.').first;
}

static bool isModel(StringRef line, StringRef modelName) {
  return line.starts_with(modelName.str() + ".");
}

static bool isNDWire(StringRef line, StringRef modelName) {
  return line.starts_with(modelName.str() + ".ndw");
}

// Get the set of states given the path to file containing the output to the
// nuXmv command "print_reachable_states -v;". A set element is defined by a
// string which is the concatenation of the state values, as represented in the
// output file.
// Example:
// State 1:
//   seq_generator_C.outs = TRUE
//   model.ndw_in_C.state = running
//   model.fork_control.regBlock0.reg_value = TRUE
// State 2:
//   seq_generator_C.outs = TRUE
//   model.ndw_in_C.state = sleeping
//   model.fork_control.regBlock0.reg_value = TRUE
// State 3:
//   seq_generator_C.outs = TRUE
//   model.ndw_in_C.state = running
//   model.fork_control.regBlock0.reg_value = FALSE
// State 4:
//   seq_generator_C.outs = FALSE
//   model.ndw_in_C.state = running
//   model.fork_control.regBlock0.reg_value = TRUE
// Here State 1 and 2 are equivalent, since they only differ in the NDWire
// state, which is not considered for optimization.
// State 3 is a new distict state, as the model.fork_control.regBlock0.reg_value
// is updated. State 4 is also a new distinct state, as the seq_generator_C.outs
// is updated. So this example has three unique states.
static FailureOr<llvm::StringSet<>>
getStateSet(const std::filesystem::path &filePath, const std::string &modelName,
            unsigned nrOfTokens) {
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

    if (isSeqGeneratorOuts(line)) {
      // Include the value the sequence generator is emitting
      currentState += line + "\n";
    } else if (nrOfTokens > 0 &&
               isSaturatedSeqGeneratorCounter(line, nrOfTokens)) {
      // If the sequence generator is no longer emitting, include the state
      currentState += getSeqGeneratorName(line).str() + ".no_emission = TRUE\n";
    } else if (isModel(line, modelName) && !isNDWire(line, modelName)) {
      // Include the state except for the ND wires
      currentState += line + "\n";
    }
  }
  file.close();

  if (!currentState.empty()) {
    states.insert(currentState);
  }

  return states;
}

// Count how many states are in the set reached by infinite tokens but are not
// in the set reached by finite tokens.
static FailureOr<size_t> compareReachableStates(
    const std::string &modelName, const std::filesystem::path &infinitePath,
    const std::filesystem::path &finitePath, unsigned nrOfTokens) {

  auto failOrInfiniteStates = getStateSet(infinitePath, modelName, nrOfTokens);
  if (failed(failOrInfiniteStates)) {
    llvm::errs() << "Failed to get the state set with infinite tokens.\n";
    return failure();
  }
  auto failOrFiniteStates = getStateSet(finitePath, modelName, nrOfTokens);
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

  llvm::errs() << "nrOfTokens: " << nrOfTokens << "\n";
  llvm::errs() << "infinite states: " << infiniteStateSet.size()
               << ", finite states: " << finiteStateSet.size()
               << ", difference: " << differenceCount << "\n";

  return differenceCount;
}

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::filesystem::path &mlirPath) {

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(outputDir);

  // Add ND wires to the circuit
  auto ret = dynamatic::experimental::createReachabilityCircuit(context,
                                                                mlirPath, true);
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

  // Convert the circuit to SMV and generate a PNG with the circuit's
  // representation.
  auto failOrSmvPair = dynamatic::experimental::handshake2smv(
      reachabilityMlirPath, outputDir, true);
  if (failed(failOrSmvPair))
    return failure();
  std::string smvFilename = config.funcName + ".smv";

  // Create the wrapper with infinite sequence generators
  auto fail = dynamatic::experimental::createSmvSequenceLengthTestbench(
      context, outputDir / "main_inf.smv", config, config.funcName, 0);
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

  // Initially we compare the number of reachable states with just one token, to
  // the number of reachable states with an infinite number of tokens. We then
  // iteratively increase the number of tokens, until all states reachable with
  // inifinite tokens can be reached.
  int numberOfTokens = 1;
  while (true) {
    llvm::errs() << "Analyzing reachable states with "
                 << std::to_string(numberOfTokens) << " tokens.\n";
    std::filesystem::path wrapperPath =
        outputDir / ("main_" + std::to_string(numberOfTokens) + ".smv");

    // Create the wrapper with n-token sequence generators
    auto fail = dynamatic::experimental::createSmvSequenceLengthTestbench(
        context, wrapperPath, config, config.funcName, numberOfTokens);
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
            config.funcName, outputDir / "inf_states.txt",
            outputDir / ("states_" + std::to_string(numberOfTokens) + ".txt"),
            numberOfTokens);
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