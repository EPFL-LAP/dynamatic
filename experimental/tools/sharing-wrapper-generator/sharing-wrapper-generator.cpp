//===- rtl-cmpf-generator.cpp - Generator for arith.cmpf --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL generator for the `handshake.sharing_wrapper` MLIR operation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> outputRTLPath(cl::Positional, cl::Required,
                                          cl::desc("<output file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> entityName(cl::Positional, cl::Required,
                                       cl::desc("<entity name>"),
                                       cl::cat(mainCategory));

static cl::opt<std::string> creditOpt(cl::Positional, cl::Required,
                                      cl::desc("<list of credits>"),
                                      cl::cat(mainCategory));

static cl::opt<std::string> numInputOperandsOpt(
    cl::Positional, cl::Required,
    cl::desc("<number of input operands (space separated)>"),
    cl::cat(mainCategory));

static cl::opt<std::string> dataWidthOpt(cl::Positional, cl::Required,
                                         cl::desc("<dataWidth>"),
                                         cl::cat(mainCategory));

llvm::SmallVector<unsigned, 8> parseCreditOpt(std::string creditString) {
  llvm::SmallVector<unsigned, 8> listOfCredits;

  std::string delimiter = " ";

  size_t pos = 0;
  std::string token;
  while ((pos = creditString.find(delimiter)) != std::string::npos) {
    token = creditOpt.substr(0, pos);
    listOfCredits.emplace_back(std::stoi(token));
    creditString.erase(0, pos + delimiter.length());
  }
  listOfCredits.emplace_back(std::stoi(creditString));

  return listOfCredits;
}

std::string getRangeFromSize(unsigned size) {
  return "(" + std::to_string(size) + " - 1 downto 0)";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "RTL generator for the `arith.cmpf` MLIR operation. Generates the "
      "correct RTL based on the floating comparison predicate.");

  // Open the output file
  std::ofstream outputFile(outputRTLPath);
  if (!outputFile.is_open()) {
    llvm::errs() << "Failed to open output file @ \"" << outputRTLPath
                 << "\"\n";
    return 1;
  }

  // Read the JSON content from the file and into a string
  std::string inputData;

  llvm::SmallVector<unsigned, 8> listOfCredits =
      parseCreditOpt(creditOpt.getValue());

  unsigned dataWidth = std::stoi(dataWidthOpt.getValue());
  unsigned numInputOperands = std::stoi(numInputOperandsOpt.getValue());

  unsigned groupSize = listOfCredits.size();

  outputFile << "-- Sharing Wrapper Circuit for Managing the Shared Unit --\n";
  outputFile << "-- Number of credits of each operation: ";
  for (auto credit : listOfCredits) {
    outputFile << std::to_string(credit) << " ";
  }
  outputFile << "\n";
  outputFile << "-- Total number of shared operations: " << groupSize << "\n";

  outputFile << "-- Number of input operands of the shared op: "
             << numInputOperands << "\n\n";

  unsigned totalNumInputChannels = listOfCredits.size() * numInputOperands + 1;

  unsigned totalNumOutputChannels = listOfCredits.size() + numInputOperands;

  // List of synchronizer units that synchronizes the inputs
  std::vector<std::string> syncs;
  for (unsigned i = 0; i < groupSize; i++) {
    syncs.emplace_back("sync" + std::to_string(i));
  }

  // Output signals of the sync:
  std::vector<std::string> syncsOutputData;
  for (const std::string &sync : syncs) {
    for (unsigned i = 0; i < numInputOperands + 1; i++) {
      syncsOutputData.push_back(sync + "_outs_" + std::to_string(i));
    }
  }

  // Output signals of the sync:
  std::vector<std::string> syncOutputValid;
  for (const std::string &sync : syncs) {
    syncOutputValid.push_back(sync + "_outs_valid");
  }

  // List of mux that synchronizes the inputs
  std::vector<std::string> muxes;
  for (unsigned i = 0; i < numInputOperands; i++) {
    muxes.emplace_back("mux" + std::to_string(i));
  }

  // List of forks that returns the credit
  std::vector<std::string> lazyForks;
  for (unsigned i = 0; i < groupSize; i++) {
    syncs.emplace_back("lazyfork" + std::to_string(i));
  }

  // Header
  outputFile << "entity " << entityName.getValue() << " is\n";
  outputFile << "port(\n";
  outputFile << "clk        : in std_logic;\n";
  outputFile << "rst        : in std_logic;\n";
  outputFile << "ins        : in data_array"
             << getRangeFromSize(totalNumInputChannels)
             << getRangeFromSize(dataWidth) << "\n";
  outputFile << "ins_valid  : in std_logic_vector"
             << getRangeFromSize(totalNumInputChannels) << ";\n";
  outputFile << "ins_ready  : out std_logic_vector"
             << getRangeFromSize(totalNumInputChannels) << ";\n";
  outputFile << "outs       : out data_array"
             << getRangeFromSize(totalNumOutputChannels)
             << getRangeFromSize(dataWidth) << ";\n";
  outputFile << "outs_valid : out std_logic_vector"
             << getRangeFromSize(totalNumOutputChannels) << ";\n";
  outputFile << "outs_ready : in std_logic_vector"
             << getRangeFromSize(totalNumOutputChannels) << "\n";

  outputFile << ");\nend entity"
             << ";\n";

  outputFile << "architecture arch of " << entityName.getValue() << " is\n";

  outputFile << "-- Output data from the synchronizers\n";
  for (const std::string &signal : syncsOutputData)
    outputFile << "signal " << signal << " : std_logic_vector"
               << getRangeFromSize(dataWidth) << ";\n";
  outputFile << "\n";

  outputFile << "-- Output valids from the synchronizers\n";
  for (const std::string &signal : syncOutputValid)
    outputFile << "signal " << signal << " : std_logic;\n";
  outputFile << "\n";

  outputFile << "-- Output data from the muxes\n";
  for (const std::string &mux : muxes)
    outputFile << "signal " << mux << "_outs      : std_logic_vector"
               << getRangeFromSize(dataWidth) << ";\n";
  outputFile << "\n";

  outputFile << "-- Output valids from the muxes\n";
  for (const std::string &mux : muxes)
    outputFile << "signal " << mux << "_out_valid : std_logic;\n";
  outputFile << "\n";

  outputFile << "-- Output valids from the branch\n";
  outputFile << "signal branch_outs_valid : std_logic_vector"
             << getRangeFromSize(groupSize) << ";\n";
  outputFile << "\n";

  outputFile << "-- Output valids from the priority arbiter\n";
  outputFile << "signal arbiter_out : std_logic_vector"
             << getRangeFromSize(groupSize) << ";\n";
  outputFile << "\n";

  outputFile << "-- Flag that says the sharing wrapper is taking a token\n";
  outputFile << "signal fifo_ins_valid : std_logic;\n";
  outputFile << "\n";

  outputFile << "-- FIFO output data\n";
  outputFile << "signal fifo_outs_data : std_logic_vector"
             << getRangeFromSize(groupSize) << ";\n";

  outputFile << "-- FIFO output valid\n";
  outputFile << "signal fifo_outs_valid : std_logic;\n";
  outputFile << "\n";

  outputFile << "begin\n";
  outputFile << "end architecture\n";

  return 0;
}
