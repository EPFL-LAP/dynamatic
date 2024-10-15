//===- shairng-wrapper-generator.cpp - Generator--------------- -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL generator for the `handshake.sharing_wrapper` MLIR operation.
// See operation's TableGen entry for port convention and functionality.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

using namespace llvm;
using namespace mlir;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> clOptOutputRTLPath(cl::Positional, cl::Required,
                                               cl::desc("<output file>"),
                                               cl::cat(mainCategory));

static cl::opt<std::string> clOptEntityName(cl::Positional, cl::Required,
                                            cl::desc("<entity name>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string>
    clOptListOfCredits(cl::Positional, cl::Required,
                       cl::desc("<list of credits (space separated)>"),
                       cl::cat(mainCategory));

static cl::opt<std::string>
    clOptNumInputOperands(cl::Positional, cl::Required,
                          cl::desc("<number of input operands>"),
                          cl::cat(mainCategory));

static cl::opt<std::string> clOptDataWidth(cl::Positional, cl::Required,
                                           cl::desc("<dataWidth>"),
                                           cl::cat(mainCategory));

static cl::opt<std::string> clOptLatency(cl::Positional, cl::Required,
                                         cl::desc("<latency>"),
                                         cl::cat(mainCategory));

llvm::SmallVector<unsigned, 8> parseCreditOpt(std::string creditString) {
  llvm::SmallVector<unsigned, 8> listOfCredits;

  std::string delimiter = " ";

  size_t pos = 0;
  std::string token;
  while ((pos = creditString.find(delimiter)) != std::string::npos) {
    token = clOptListOfCredits.substr(0, pos);
    listOfCredits.emplace_back(std::stoi(token));
    creditString.erase(0, pos + delimiter.length());
  }
  listOfCredits.emplace_back(std::stoi(creditString));

  return listOfCredits;
}

std::string getRangeFromSize(unsigned size) {
  return "(" + std::to_string(size) + " - 1 downto 0)";
}

void declareDataSignal(mlir::raw_indented_ostream &os,
                       const std::string &unitName, unsigned bitwidth,
                       unsigned numOutputs) {
  for (unsigned i = 0; i < numOutputs; i++) {
    os << "signal " << unitName << "_out" << i << "_data"
       << " : std_logic_vector" << getRangeFromSize(bitwidth) << ";\n";
  }
}

void declareHandshakeSignals(mlir::raw_indented_ostream &os,
                             const std::string &unitName, unsigned bitwidth,
                             unsigned numOutputs) {
  for (unsigned i = 0; i < numOutputs; i++) {
    os << "signal " << unitName << "_out" << i << "_valid"
       << " : std_logic;\n";
    os << "signal " << unitName << "_out" << i << "_ready"
       << " : std_logic;\n";
  }
}

void printVhdlImpl(mlir::raw_indented_ostream &os, const unsigned &dataWidth,
                   const unsigned &numInputOperands, const unsigned &groupSize,
                   const unsigned &latency,
                   const llvm::SmallVector<unsigned, 8> &listOfCredits) {
  os << "---------------------------------------------------------\n";
  os << "-- Sharing Wrapper Circuit for Managing the Shared Unit\n";
  os << "-- Number of credits of each operation: ";
  for (auto credit : listOfCredits) {
    os << std::to_string(credit) << " ";
  }
  os << "\n";
  os << "-- Total number of shared operations: " << groupSize << "\n";

  os << "-- Number of input operands of the shared op: " << numInputOperands
     << "\n";

  os << "-- Latency of the shared operation: " << latency << "\n";
  os << "---------------------------------------------------------\n\n";
  os << "library ieee;\n";
  os << "use ieee.std_logic_1164.all;\n";
  os << "use ieee.numeric_std.all;\n";
  os << "use work.types.all;\n";

  // Header
  os << "entity " << clOptEntityName.getValue() << " is\n";
  os << "port(\n";
  os << "clk        : in std_logic;\n";
  os << "rst        : in std_logic;\n";
  os << "-- input channels:\n";
  for (unsigned i = 0; i < listOfCredits.size(); i++) {
    for (unsigned j = 0; j < numInputOperands; j++) {
      os << "op" << i << "in" << j << " : in std_logic_vector"
         << getRangeFromSize(dataWidth) << ";\n";
      os << "op" << i << "in" << j << "_valid"
         << " : in std_logic"
         << ";\n";
      os << "op" << i << "in" << j << "_ready"
         << " : out std_logic"
         << ";\n";
    }
  }
  os << "fromSharedUnitOut0"
     << " : in std_logic_vector" << getRangeFromSize(dataWidth) << ";\n";
  os << "fromSharedUnitOut0"
     << "_valid : in std_logic;\n";
  os << "fromSharedUnitOut0"
     << "_ready : out std_logic;\n";
  os << "-- output channels:\n";
  for (unsigned i = 0; i < listOfCredits.size(); i++) {
    os << "op" << i << "out0"
       << " : out std_logic_vector" << getRangeFromSize(dataWidth) << ";\n";
    os << "op" << i << "out0_valid"
       << " : out std_logic;\n";
    os << "op" << i << "out0_ready"
       << " : in std_logic;\n";
  }
  for (unsigned i = 0; i < numInputOperands; i++) {
    os << "toSharedUnitIn" << i << " : out std_logic_vector"
       << getRangeFromSize(dataWidth) << ";\n";
    os << "toSharedUnitIn" << i << "_valid : out std_logic;\n";
    os << "toSharedUnitIn" << i << "_ready : out std_logic";
    if (i < numInputOperands - 1)
      os << ";";
    os << "\n";
  }
  os << ");\nend entity"
     << ";\n";

  os << "architecture arch of " << clOptEntityName.getValue() << " is\n\n";

  for (unsigned i = 0; i < groupSize; i++) {
    for (unsigned j = 0; j < numInputOperands + 1; j++)
      os << "signal sync" << i << "_out" << j << "_data : std_logic_vector"
         << getRangeFromSize(dataWidth) << ";\n";
    os << "signal sync" << i << "_out0_valid : std_logic;\n";
  }

  for (unsigned i = 0; i < numInputOperands; i++) {
    declareDataSignal(os, "mux" + std::to_string(i), dataWidth, 1);
  }

  declareDataSignal(os, "branch0", dataWidth, groupSize);
  declareHandshakeSignals(os, "branch0", dataWidth, groupSize);

  // Output valids from the priority arbiter
  os << "signal arbiter_out : std_logic_vector" << getRangeFromSize(groupSize)
     << ";\n";
  os << "\n";

  // Flag that says the sharing wrapper is taking a token
  os << "signal arbiter_out_valid : std_logic;\n";
  os << "\n";

  // Conditon FIFO output channel
  os << "signal cond_buffer_out0_data : std_logic_vector"
     << getRangeFromSize(groupSize) << ";\n";

  os << "signal cond_buffer_out0_valid : std_logic;\n";
  os << "signal cond_buffer_out0_ready : std_logic;\n\n";

  // Output buffer output data
  for (unsigned i = 0; i < groupSize; i++) {
    std::string unitName = "out_buffer" + std::to_string(i);
    declareDataSignal(os, unitName, dataWidth, 1);
    declareHandshakeSignals(os, unitName, dataWidth, 1);
  }

  // Out lazy fork output channels
  for (unsigned i = 0; i < groupSize; i++) {
    declareDataSignal(os, "out_fork" + std::to_string(i), dataWidth, 2);
    declareHandshakeSignals(os, "out_fork" + std::to_string(i), dataWidth, 2);
  }

  // Credit output channels
  for (unsigned i = 0; i < groupSize; i++) {
    std::string unitName = "credit" + std::to_string(i);
    declareHandshakeSignals(os, unitName, dataWidth, 1);
  }

  os << "\nbegin\n\n";

  // Wiring for sync
  for (unsigned i = 0; i < groupSize; i++) {
    std::string sync = "sync" + std::to_string(i);
    std::string credit = "credit" + std::to_string(i);
    os << sync << " : entity work.crush_sync(arch)\ngeneric map("
       << numInputOperands + 1 << ", " << dataWidth << ")\n";
    os << "port map(\n";
    for (unsigned j = 0; j < numInputOperands; j++) {
      os << "ins(" << j << ") => op" << i << "in" << j << ",\n";
    }
    os << "ins(" << numInputOperands << ") => (others => \'0\'),\n";
    for (unsigned j = 0; j < numInputOperands; j++) {
      os << "ins_valid(" << j << ") => op" << i << "in" << j << "_valid,\n";
    }
    os << "ins_valid(" << numInputOperands << ") => " << credit << "_out0"
       << "_valid,\n";
    for (unsigned j = 0; j < numInputOperands; j++) {
      os << "ins_ready(" << j << ") => op" << i << "in" << j << "_ready,\n";
    }
    os << "ins_ready(" << numInputOperands << ") => " << credit << "_out0"
       << "_ready,\n";

    for (unsigned j = 0; j < numInputOperands + 1; j++) {
      os << "outs(" << j << ") => " << sync << "_out" << j << "_data,\n";
    }
    os << "outs_valid => " << sync << "_out0"
       << "_valid,\n";
    os << "outs_ready => arbiter_out(" << i << ")\n";

    os << ");\n\n";
  }

  for (unsigned i = 0; i < numInputOperands; i++) {
    std::string mux = "mux" + std::to_string(i);
    os << mux << " : entity work.crush_oh_mux(arch)\ngeneric map(" << groupSize
       << ", " << dataWidth << ")\nport map(\n";
    for (unsigned j = 0; j < groupSize; j++) {
      std::string sync = "sync" + std::to_string(j);
      os << "ins(" << j << ") => " << sync << "_out" << i << "_data,\n";
    }
    os << "sel => arbiter_out,\n";
    os << "outs => " << mux << "_out0_data\n";
    os << ");\n\n";
  }

  os << "arbiter : entity work.bitscan(arch)\ngeneric map(" << groupSize
     << ")\n";
  os << "port map(\n";
  for (unsigned i = 0; i < groupSize; i++) {
    os << "request(" << i << ") => sync" << i << "_out0_valid,\n";
  }
  os << "grant => arbiter_out\n";
  os << ");\n\n";

  // Generating valid signals to the shared unit
  os << "or_n : entity work.or_n(arch)\ngeneric map(" << groupSize << ")\n";
  os << "port map(\n";
  for (unsigned i = 0; i < groupSize; i++) {
    os << "ins(" << i << ") => sync" << i << "_out0_valid,\n";
  }
  os << "outs => arbiter_out_valid\n";
  os << ");\n\n";

  // Wiring the valid signals to the shared unit
  for (unsigned i = 0; i < numInputOperands; i++) {
    os << "toSharedUnitIn" << i << "_valid <= arbiter_out_valid;\n";
    os << "toSharedUnitIn" << i << " <= mux" << i << "_out0_data;\n";
  }
  os << "\n";

  // Condition buffer
  os << "cond_buffer : entity work.ofifo(arch)\ngeneric map(" << latency << ", "
     << groupSize << ")\n";
  os << "port map(\n";
  os << "clk => clk, rst => rst,\n";
  os << "ins => arbiter_out,\n";
  os << "ins_valid => arbiter_out_valid,\n";
  os << "outs => cond_buffer_out0_data,\n";
  os << "outs_valid => cond_buffer_out0_valid,\n";
  os << "outs_ready => cond_buffer_out0_ready\n";
  os << ");\n\n";

  // Branch
  os << "branch : entity work.crush_oh_branch(arch)\ngeneric map(" << groupSize
     << ", " << dataWidth << ")\nport map(\n";
  os << "ins => fromSharedUnitOut0,\n";
  os << "ins_valid => fromSharedUnitOut0_valid,\n";
  os << "ins_ready => fromSharedUnitOut0_ready,\n";
  os << "sel => cond_buffer_out0_data,\n";
  os << "sel_valid => cond_buffer_out0_valid,\n";
  os << "sel_ready => cond_buffer_out0_ready,\n";
  for (unsigned i = 0; i < groupSize; i++) {
    os << "outs(" << i << ") => branch0_out" << i << "_data,\n";
  }
  for (unsigned i = 0; i < groupSize; i++) {
    os << "outs_valid(" << i << ") => branch0_out" << i << "_valid,\n";
  }
  for (unsigned i = 0; i < groupSize; i++) {
    os << "outs_ready(" << i << ") => branch0_out" << i << "_ready";
    if (i < groupSize - 1) {
      os << ", ";
    }
    os << "\n";
  }
  os << ");\n\n";

  // Output buffers
  for (unsigned i = 0; i < groupSize; i++) {
    os << "out_buffer" << i << " : entity work.tfifo(arch)\ngeneric map("
       << listOfCredits[i] << ", " << dataWidth << ")\n";
    os << "port map(\n";
    os << "clk => clk, rst => rst,\n";
    os << "ins => branch0_out" << i << "_data,\n";
    os << "ins_valid => branch0_out" << i << "_valid,\n";
    os << "ins_ready => branch0_out" << i << "_ready,\n";
    os << "outs => out_buffer" << i << "_out0_data,\n";
    os << "outs_valid => out_buffer" << i << "_out0_valid,\n";
    os << "outs_ready => out_buffer" << i << "_out0_ready\n";
    os << ");\n\n";
  }

  // Output lazy forks
  for (unsigned i = 0; i < groupSize; i++) {
    os << "out_fork" << i << " : entity work.lazy_fork(arch)\ngeneric map(2, "
       << dataWidth << ")\n";
    os << "port map(\n";
    os << "clk => clk, rst => rst,\n";
    os << "ins => out_buffer" << i << "_out0_data,\n";
    os << "ins_valid => out_buffer" << i << "_out0_valid,\n";
    os << "ins_ready => out_buffer" << i << "_out0_ready,\n";
    os << "outs(0) => out_fork" << i << "_out0_data,\n";
    os << "outs(1) => out_fork" << i << "_out1_data,\n";
    os << "outs_valid(0) => out_fork" << i << "_out0_valid,\n";
    os << "outs_valid(1) => out_fork" << i << "_out1_valid,\n";
    os << "outs_ready(0) => op" << i << "out0_ready,\n";
    os << "outs_ready(1) => out_fork" << i << "_out1_ready\n";
    os << ");\n\n";
  }

  // Credit counters
  for (unsigned i = 0; i < groupSize; i++) {
    os << "credit" << i
       << " : entity work.crush_credit_dataless(arch)\ngeneric map("
       << listOfCredits[i] << ")\n";
    os << "port map(\n";
    os << "clk => clk, rst => rst,\n";
    os << "ins_valid => out_fork" << i << "_out1_valid,\n";
    os << "ins_ready => out_fork" << i << "_out1_ready,\n";
    os << "outs_valid => credit" << i << "_out0_valid,\n";
    os << "outs_ready => credit" << i << "_out0_ready\n";
    os << ");\n\n";
  }

  // Output data
  for (unsigned i = 0; i < groupSize; i++) {
    os << "op" << i << "out0 <= out_fork" << i << "_out0_data;\n";
    os << "op" << i << "out0_valid <= out_fork" << i << "_out0_valid;\n";
  }

  os << "\nend architecture;\n\n";
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "RTL generator for the `arith.cmpf` MLIR operation. Generates the "
      "correct RTL based on the floating comparison predicate.");

  // Open the output file
  std::error_code ec;
  llvm::raw_fd_ostream fileStream(clOptOutputRTLPath, ec);
  if (ec.value() != 0) {
    llvm::errs() << "Failed to open output file @ \"" << clOptOutputRTLPath
                 << "\"\n";
    return 1;
  }
  mlir::raw_indented_ostream os(fileStream);

  llvm::SmallVector<unsigned, 8> listOfCredits =
      parseCreditOpt(clOptListOfCredits.getValue());

  printVhdlImpl(os, std::stoi(clOptDataWidth.getValue()),
                std::stoi(clOptNumInputOperands.getValue()),
                listOfCredits.size(), std::stoi(clOptLatency.getValue()),
                listOfCredits);

  return 0;
}
