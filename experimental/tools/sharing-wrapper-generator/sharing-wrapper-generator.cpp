//===- shairng-wrapper-generator.cpp - Generator--------------- -*- C++ -*-===//
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

static cl::opt<std::string> latencyOpt(cl::Positional, cl::Required,
                                       cl::desc("<latency>"),
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

void printOutPorts(std::ofstream &os, const std::string &unitName,
                   unsigned bitwidth, unsigned numOutputs) {
  for (unsigned i = 0; i < numOutputs; i++) {
    os << unitName << "_out" << i << "_data"
       << " : std_logic_vector" << getRangeFromSize(bitwidth) << ";\n";
    os << unitName << "_out" << i << "_valid"
       << " : std_logic;\n";
    os << unitName << "_out" << i << "_ready"
       << " : std_logic;\n";
    os << "\n";
  }
  os << "\n";
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

  unsigned latency = std::stoi(latencyOpt.getValue());

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
      syncsOutputData.push_back(sync + "_out" + std::to_string(i) + "_data");
    }
  }

  // Output signals of the sync:
  std::vector<std::string> syncOutputValid;
  for (const std::string &sync : syncs) {
    syncOutputValid.push_back(sync + "_out0_valid");
  }

  // List of mux that synchronizes the inputs
  std::vector<std::string> muxes;
  for (unsigned i = 0; i < numInputOperands; i++) {
    muxes.emplace_back("mux" + std::to_string(i));
  }

  // List of forks that returns the credit
  std::vector<std::string> lazyForks;
  for (unsigned i = 0; i < groupSize; i++) {
    lazyForks.emplace_back("lazyfork" + std::to_string(i));
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

  for (const std::string &mux : muxes)
    printOutPorts(outputFile, mux, dataWidth, 1);

  printOutPorts(outputFile, "branch0", dataWidth, groupSize);

  outputFile << "-- Output valids from the priority arbiter\n";
  outputFile << "signal arbiter_out : std_logic_vector"
             << getRangeFromSize(groupSize) << ";\n";
  outputFile << "\n";

  outputFile << "-- Flag that says the sharing wrapper is taking a token\n";
  outputFile << "signal arbiter_out_valid : std_logic;\n";
  outputFile << "\n";

  outputFile << "-- cond FIFO output data\n";
  outputFile << "signal cond_fifo_outs_data : std_logic_vector"
             << getRangeFromSize(groupSize) << ";\n";

  outputFile << "-- cond FIFO output valid\n";
  outputFile << "signal cond_fifo_outs_valid : std_logic;\n";
  outputFile << "\n";

  outputFile << "-- out buffer output data\n";
  for (unsigned i = 0; i < groupSize; i++) {
    std::string unitName = "out_fifo" + std::to_string(i);
    printOutPorts(outputFile, unitName, dataWidth, 1);
  }

  outputFile << "-- out lazy fork output channels\n";
  for (unsigned i = 0; i < groupSize; i++) {
    printOutPorts(outputFile, "out_fork" + std::to_string(i), dataWidth, 2);
  }

  outputFile << "-- credit output channels\n";
  for (unsigned i = 0; i < groupSize; i++) {
    std::string unitName = "credit" + std::to_string(i);
    printOutPorts(outputFile, unitName, dataWidth, 1);
  }

  outputFile << "begin\n";

  for (unsigned i = 0; i < groupSize; i++) {
    std::string sync = "sync" + std::to_string(i);
    std::string credit = "credit" + std::to_string(i);
    outputFile << "-- Wiring for " << sync << ":\n";
    outputFile << sync << " : entity work.crush_sync(arch)\ngeneric map("
               << numInputOperands + 1 << ", " << dataWidth << ")\n";
    outputFile << "port map(\n";
    for (unsigned i = 0; i < numInputOperands; i++) {
      outputFile << "ins(" << i << ") => ins(" << i << "),\n";
    }
    for (unsigned i = 0; i < numInputOperands; i++) {
      outputFile << "ins_valid(" << i << ") => ins_valid(" << i << "),\n";
    }
    for (unsigned i = 0; i < numInputOperands; i++) {
      outputFile << "ins_ready(" << i << ") => ins_ready(" << i << "),\n";
    }

    outputFile << "ins_data(" << numInputOperands << ") => " << credit
               << "_out0"
               << "_data),\n";
    outputFile << "ins_valid(" << numInputOperands << ") => " << credit
               << "_out0"
               << "_valid),\n";
    outputFile << "ins_ready(" << numInputOperands << ") => " << credit
               << "_out0"
               << "_ready),\n";

    for (unsigned i = 0; i < numInputOperands; i++) {
      outputFile << "outs(" << i << ") => " << sync << "_out" << i
                 << "_data),\n";
    }
    outputFile << "outs_valid => " << sync << "_out0"
               << "_valid),\n";
    outputFile << "outs_valid => " << sync << "_out0"
               << "_ready)\n";

    outputFile << ");\n\n";
  }

  for (unsigned i = 0; i < numInputOperands; i++) {
    std::string mux = "mux" + std::to_string(i);
    outputFile << mux << " : entity work.crush_oh_mux(arch)\ngeneric map("
               << groupSize << ", " << dataWidth << ")\n";
    for (unsigned j = 0; j < groupSize; j++) {
      std::string sync = "sync" + std::to_string(j);
      outputFile << "ins(" << j << ") => " << sync << "_out" << i << "_data,\n";
    }
    outputFile << "sel => arbiter_out,\n";
    outputFile << "outs => " << mux << "_out0_data\n";
    outputFile << ");\n\n";
  }

  outputFile << "arbiter0 : work.bitscan(arch)\ngeneric map(" << groupSize
             << ")\n";
  outputFile << "port map(\n";
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "request(" << i << ") => sync" << i << "_out0_valid,\n";
  }
  outputFile << "grant => arbiter_out\n";
  outputFile << ");\n";

  // Generating valid signals to the shared unit
  outputFile << "or_n0 : work.or_n0(arch)\ngeneric map(" << groupSize << ")\n";
  outputFile << "port map(\n";
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "ins(" << i << ") => sync" << i << "_out0_valid,\n";
  }
  outputFile << "outs => arbiter_out_valid\n";
  outputFile << ");\n";

  // Wiring the valid signals to the shared unit
  for (unsigned i = groupSize; i < groupSize + numInputOperands; i++) {
    outputFile << "outs_valid(" << i << ") <= arbiter_out_valid;\n";
    outputFile << "outs(" << i << ") <= mux" << i - groupSize
               << "_out0_data;\n";
  }
  outputFile << "\n";

  outputFile << "cond_fifo : work.ofifo(arch)\ngeneric map(" << latency << ", "
             << dataWidth << ")\n";
  outputFile << "port map(\n";
  outputFile << "clk => clk, rst => rst,\n";
  outputFile << "ins => arbiter_out,\n";
  outputFile << "ins_valid => arbiter_out_valid,\n";
  outputFile << "outs => cond_fifo_outs_data,\n";
  outputFile << "outs_valid => cond_fifo_outs_valid,\n";
  outputFile << "outs_ready => cond_fifo_outs_ready\n";
  outputFile << ");\n\n";

  // -- Branch -- //
  outputFile << "branch : work.crush_oh_branch(arch)\ngeneric map(" << groupSize
             << ", " << dataWidth << ")\nport map(\n";
  outputFile << "ins => ins(" << groupSize << "),\n";
  outputFile << "ins_valid => ins_valid(" << groupSize << "),\n";
  outputFile << "ins_ready => ins_ready(" << groupSize << "),\n";
  outputFile << "sel => cond_fifo_outs_data,\n";
  outputFile << "sel_valid => cond_fifo_outs_valid,\n";
  outputFile << "sel_ready => conf_fifo_outs_ready,\n";
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "outs(" << i << ") => branch0_out" << i << "_data,\n";
    outputFile << "outs_valid(" << i << ") => branch0_out" << i << "_valid,\n";
    outputFile << "outs_ready(" << i << ") => branch0_out" << i << "_ready";
    if (i < groupSize - 1) {
      outputFile << ", ";
    }
    outputFile << "\n";
  }
  outputFile << ");\n\n";

  // -- Output Buffers --//
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "out_buffer" << i << " : work.tfifo(arch)\ngeneric map("
               << listOfCredits[i] << ", " << dataWidth << ")\n";
    outputFile << "port map(\n";
    outputFile << "ins => branch0_out" << i << "_data,\n";
    outputFile << "ins_valid => branch0_out" << i << "_valid,\n";
    outputFile << "ins_ready => branch0_out" << i << "_ready,\n";
    outputFile << "outs => out_fifo" << i << "_out0_data,\n";
    outputFile << "outs_valid => out_fifo" << i << "_out0_valid,\n";
    outputFile << "outs_ready => out_fifo" << i << "_out0_ready\n";
    outputFile << ");\n\n";
  }

  // -- Output Forks -- //
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "fork" << i << " : work.lazy_fork(arch)\ngeneric map(2, "
               << dataWidth << ")\n";
    outputFile << "port map(\n";
    outputFile << "ins => out_fifo" << i << "_out0_data,\n";
    outputFile << "ins_valid => out_fifo" << i << "_out0_valid,\n";
    outputFile << "ins_ready => out_fifo" << i << "_out0_ready,\n";
    outputFile << "outs(0) => out_fork" << i << "_out0_data,\n";
    outputFile << "outs_valid(0) => out_fork" << i << "_out0_valid,\n";
    outputFile << "outs_ready(0) => out_fork" << i << "_out0_ready,\n";
    outputFile << "outs(1) => out_fork" << i << "_out1_data,\n";
    outputFile << "outs_valid(1) => out_fork" << i << "_out1_valid,\n";
    outputFile << "outs_ready(1) => out_fork" << i << "_out1_ready\n";
    outputFile << ");\n\n";
  }

  // -- Credits -- //
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "credit" << i << " : work.credit(arch)\ngeneric map("
               << listOfCredits[i] << ", " << dataWidth << ")\n";
    outputFile << "port map(\n";
    outputFile << "ins_valid => out_fork" << i << "_out1_valid,\n";
    outputFile << "ins_ready => out_fork" << i << "_out1_ready,\n";
    outputFile << "outs_valid => credit" << i << "_out0_valid,\n";
    outputFile << "outs_ready => credit" << i << "_out0_ready\n";
    outputFile << ");\n\n";
  }

  // -- Output data
  for (unsigned i = 0; i < groupSize; i++) {
    outputFile << "outs(" << i << ") <= fork" << i << "_out0_data;\n";
    outputFile << "outs_valid(" << i << ") <= fork" << i << "_out0_valid;\n";
    outputFile << "outs_ready(" << i << ") <= fork" << i << "_out0_ready;\n\n";
  }

  outputFile << "end architecture\n\n";

  return 0;
}
