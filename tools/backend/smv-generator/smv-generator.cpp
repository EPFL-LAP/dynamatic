//===- rtl-cmpf-generator.cpp - Generator for arith.cmpf --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL generator for the `arith.cmpf` MLIR operation. Generates the correct RTL
// based on the floating comparison predicate.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <map>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace handshake {
enum class OpTypeEnum : uint64_t {
  DEFAULT = 0,
  JOIN = 1,
  FORK = 2,
  LAZY_FORK = 3,
  MERGE = 4,
  CONTROL_MERGE = 5,
  MUX = 6,
  CONSTANT = 7,
  BOP = 8,
  UOP = 9
};

std::unordered_map<std::string, std::unordered_map<std::string, int>>
    operationLatencies = {{"addf", {{"32", 9}, {"64", 12}}},
                          {"subf", {{"32", 9}, {"64", 12}}},
                          {"mulf", {{"32", 4}, {"64", 4}}},
                          {"muli", {{"32", 4}}},
                          {"divf", {{"32", 29}, {"64", 36}}},
                          {"divsi", {{"32", 36}}},
                          {"divui", {{"32", 36}}},
                          {"maximumf", {{"32", 2}, {"64", 2}}},
                          {"minimumf", {{"32", 2}, {"64", 2}}},
                          {"sitofp", {{"32", 5}}}};

OpTypeEnum symbolizeOp(const std::string &name) {
  if (name == "join")
    return OpTypeEnum::JOIN;
  if (name == "fork")
    return OpTypeEnum::FORK;
  if (name == "lazy_fork")
    return OpTypeEnum::LAZY_FORK;
  if (name == "merge")
    return OpTypeEnum::MERGE;
  if (name == "control_merge")
    return OpTypeEnum::CONTROL_MERGE;
  if (name == "mux")
    return OpTypeEnum::MUX;
  if (name == "constant")
    return OpTypeEnum::CONSTANT;
  if (name == "bop")
    return OpTypeEnum::BOP;
  if (name == "uop")
    return OpTypeEnum::UOP;
  return OpTypeEnum::DEFAULT;
}

} // namespace handshake
} // namespace dynamatic

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> outputRTLPath(cl::Positional, cl::Required,
                                          cl::desc("<output file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> entityName(cl::Positional, cl::Required,
                                       cl::desc("<entity name>"),
                                       cl::cat(mainCategory));

static cl::opt<std::string> parameters(cl::Positional, cl::Optional,
                                       cl::desc("<parameters>"),
                                       cl::cat(mainCategory));

std::string generateJoin(unsigned int inputSignals) {
  std::string mod =
      "MODULE join_" + std::to_string(inputSignals) + "_1 (ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += ", ins_valid_" + std::to_string(i);
  mod += ", outs_ready)\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "DEFINE ins_ready" + std::to_string(i) + "  := outs_ready";
    for (unsigned int j = 0; j < inputSignals; j++)
      if (i != j)
        mod += " & ins_valid_" + std::to_string(j);
    mod += ";\n";
  }

  mod += "DEFINE outs_valid  := ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += " & inst_valid_" + std::to_string(i);
  mod += ";\n\n";

  return mod;
}

std::string generateFork(unsigned int outputSignals, bool isDataless) {
  std::string mod =
      "MODULE fork_dataless_1_" + std::to_string(outputSignals) + "(ins_valid";
  unsigned int i;
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + ")\n";

  for (unsigned int i = 0; i < outputSignals; i++) {
    mod += "VAR regblock_" + std::to_string(i) +
           " : eager_fork_register_block(ins_valid, outs_ready_" +
           std::to_string(i) + ", backpressure);\n";
  }

  mod += "DEFINE any_block_stop := ";
  for (i = 0; i < outputSignals - 1; i++) {
    mod += "regblock_" + std::to_string(i) + ".block_stop | ";
  }
  mod += "regblock_" + std::to_string(i) + ".block_stop;\n";

  mod += "DEFINE ins_ready := !any_block_stop;\n";

  for (unsigned int i = 0; i < outputSignals; i++) {
    mod += "DEFINE outs_valid_" + std::to_string(i) + " := regblock_" +
           std::to_string(i) + ".outs_valid;\n";
  }

  if (isDataless)
    return mod;

  mod += "\nMODULE fork_1_" + std::to_string(outputSignals) + "(ins, ins_valid";
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + ")\n";

  mod += "VAR fork_inner : fork_dataless_1_" + std::to_string(outputSignals) +
         "(ins_valid, ";
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + ");\n";

  for (i = 0; i < outputSignals; i++) {
    mod += "DEFINE outs_" + std::to_string(i) + " := ins;\n";
    mod += "DEFINE outs_ready_" + std::to_string(i) +
           " := fork_inner.outs_ready_" + std::to_string(i) + ";\n";
  }
  return mod;
}

std::string generateLazyFork(unsigned int outputSignals, bool isDataless) {
  std::string mod = "MODULE lazy_fork_dataless_1_" +
                    std::to_string(outputSignals) + "(ins_valid";
  unsigned int i;
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + ")\n";

  mod += "DEFINE ins_ready := ";
  for (i = 0; i < outputSignals - 1; i++) {
    mod += "outs_ready_" + std::to_string(i) + " & ";
  }
  mod += "outs_ready_" + std::to_string(i) + ";\n";

  mod += "VAR tmp_ready : array 0.." + std::to_string(outputSignals) +
         " of boolean;\n";
  mod += "ASSIGN\ninit(tmp_ready) := [TRUE];\n";
  for (i = 0; i < outputSignals; i++)
    for (unsigned int j = 0; j < outputSignals; j++)
      if (i != j)
        mod += "next(tmp_ready[" + std::to_string(i) + "]) := tmp_ready[" +
               std::to_string(i) + "] & outs_ready[" + std::to_string(j) +
               "];\n";

  for (unsigned int i = 0; i < outputSignals; i++) {
    mod += "DEFINE outs_valid_" + std::to_string(i) +
           " := ins_valid & tmp_ready[" + std::to_string(i) + "];\n";
  }

  if (isDataless)
    return mod;

  mod += "\nMODULE lazy_fork_1_" + std::to_string(outputSignals) +
         "(ins, ins_valid";
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + "\n";

  mod += "VAR fork_inner : lazy_fork_dataless_1_" +
         std::to_string(outputSignals) + "(ins_valid, ";
  for (i = 0; i < outputSignals - 1; i++)
    mod += ", outs_ready_" + std::to_string(i);
  mod += ", outs_ready_" + std::to_string(i) + ");\n";

  for (i = 0; i < outputSignals; i++) {
    mod += "DEFINE outs_" + std::to_string(i) + " := ins;\n";
    mod += "DEFINE outs_ready_" + std::to_string(i) +
           " := fork_inner.outs_ready_" + std::to_string(i) + ";\n";
  }
  return mod;
}

std::string generateMerge(unsigned int inputSignals, bool isDataless) {
  std::string mod = (isDataless ? "MODULE merge_dataless_" : "MODULE merge_");

  mod += std::to_string(inputSignals) + "_1 (" + (isDataless ? "" : "ins_0") +
         "ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    if (!isDataless)
      mod += ", ins_" + std::to_string(i);
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", outs_ready)\n";

  if (isDataless)
    mod += "VAR tehb_inner : tehb_dataless(tehb_valid, outs_ready);\n";
  else {
    mod += "DEFINE tehb_data_in := case\n";
    for (unsigned int i = 0; i < inputSignals; i++)
      mod += "ins_valid_" + std::to_string(i) + " : ins_" + std::to_string(i) +
             ";\n";
    mod += "TRUE : ins_0;\nesac;\n";
    mod += "VAR tehb_inner : tehb(tehb_data_in, tehb_valid, outs_ready);\n";
  }

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod +=
        "DEFINE ins_ready" + std::to_string(i) + "  := tehb_inner.ins_ready;\n";
  }

  mod += "DEFINE outs_valid := tehb_inner.outs_valid;\n";
  if (!isDataless)
    mod += "DEFINE outs := tehb_inner.outs;\n";

  mod += "DEFINE tehb_valid := ins_valid_0;";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += " | ins_valid_" + std::to_string(i);
  mod += ";\n";

  return mod;
}

std::string generateMergeNoTehb(unsigned int inputSignals, bool isDataless) {
  std::string mod =
      (isDataless ? "MODULE merge_notehb_dataless_" : "MODULE merge_notehb_");
  mod += std::to_string(inputSignals) + "_1 (" + (isDataless ? "" : "ins_0") +
         "ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    if (!isDataless)
      mod += ", ins_" + std::to_string(i);
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", outs_ready)\n";
  if (!isDataless) {
    mod += "VAR outs : DATA_TYPE;\n";
    mod += "ASSIGN\ninit(outs) := 0;\n";
    mod += "next(outs) := case\n";
    for (unsigned int i = 1; i < inputSignals; i++) {
      mod += "ins_valid_" + std::to_string(i) + ": ins_" + std::to_string(i) +
             ";\n";
    }
    mod += "TRUE : outs;\nesac;\n\n";
  }

  mod += "VAR outs_valid : boolean;\n";
  mod += "ASSIGN\ninit(outs_valid) := FALSE;\n";
  mod += "next(outs_valid) := ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += " | ins_valid_" + std::to_string(i);
  }
  mod += ";\n";
  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += "DEFINE ins_ready_" + std::to_string(i) + " := outs_ready;\n";
  }

  return mod;
}

std::string generateControlMerge(unsigned int inputSignals, bool isDataless) {
  std::string mod = generateMergeNoTehb(inputSignals, true);

  mod += "\n\nMODULE control_merge_dataless_";

  mod += std::to_string(inputSignals) + "_1 (" + (isDataless ? "" : "ins_0") +
         "ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", index_ready, outs_ready)\n";

  mod += "VAR index_tehb : integer;\n";
  mod += "ASSIGN\ninit(index_tehb) := 0;\n";
  mod += "next(index_tehb) := case\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "ins_valid_" + std::to_string(i) + " : " + std::to_string(i) + ";\n";
  }
  mod += "TRUE : index_tehb;\nesac;\n";

  mod += std::string("VAR inner_merge : merge_notehb_dataless_") +
         std::to_string(inputSignals) + "(ins_valid_0";

  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", inner_tehb.ins_ready)\n";

  mod +=
      "VAR tehb(index_tehb, inner_merge.outs_valid, inner_fork.ins_ready);\n";

  mod += "VAR inner_fork : fork_dataless_generic(tehb.outs_valid, outs_ready, "
         "index_ready);\n";

  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += "DEFINE ins_ready_" + std::to_string(i) +
           ":= inner_merge.ins_ready_" + std::to_string(i) + ";\n";
  }

  mod += "DEFINE outs_valid := inner_fork.outs_valid_0;\n";
  mod += "DEFINE index := inner_tehb.outs;\n";
  mod += "DEFINE index_valid := inner_fork.outs_valid_1;\n";

  if (isDataless)
    return mod;

  mod += "\n\nMODULE control_merge_";

  mod += std::to_string(inputSignals) + "_1 (" + (isDataless ? "" : "ins_0") +
         "ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    if (!isDataless)
      mod += ", ins_" + std::to_string(i);
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", index_ready, outs_ready)\n";
  mod += "VAR inner_control_merge : control_merge_dataless_";
  mod += std::to_string(inputSignals) + "_1 (ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", index_ready, outs_ready)\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "DEFINE ins_ready_" + std::to_string(i) +
           " : inner_control_merge.ins_ready" + std::to_string(i) + ";\n";
  }

  mod += "DEFINE index := inner_control_merge.index;\n";
  mod += "DEFINE index_valid := inner_control_merge.index_valid;\n";

  mod += "DEFINE outs := case\n";
  for (unsigned int i = 0; i < inputSignals; i++) {
    mod +=
        "index = " + std::to_string(i) + " : ins_" + std::to_string(i) + ";\n";
  }
  mod += "esac;\n\n";
  mod += "DEFINE outs_valid := inner_control_merge.outs_valid;\n";

  return mod;
}

std::string generateMux(unsigned int inputSignals, bool isDataless) {
  std::string mod = (isDataless ? "MODULE mux_dataless_" : "MODULE mux_");

  mod += std::to_string(inputSignals) + "_1 (" + (isDataless ? "" : "ins_0, ") +
         "ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++) {
    if (!isDataless)
      mod += ", ins_" + std::to_string(i);
    mod += ", ins_valid_" + std::to_string(i);
  }
  mod += ", index, index_valid, outs_ready)\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "DEFINE ins_ready" + std::to_string(i) +
           "  := (index == " + std::to_string(i) +
           " & index_valid & tehb_inner.ins_ready & "
           "ins_valid_" +
           std::to_string(i) + ") | !ins_valid_" + std::to_string(i) + ";\n";
  }

  mod += "DEFINE tehb_ins_valid := case\n";
  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "index == " + std::to_string(i) + " : index_valid & ins_valid_" +
           std::to_string(i) + ";\n";
  }

  if (!isDataless) {
    mod += "DEFINE tehb_ins := case\n";
    for (unsigned int i = 0; i < inputSignals; i++) {
      mod += "index == " + std::to_string(i) + " : index_valid & ins_valid_" +
             std::to_string(i) + ";\n";
    }
    mod += "VAR tehb_inner : tehb(tehb_ins, tehb_ins_valid, outs_ready);\n";
  } else {
    mod += "VAR tehb_inner : tehb_dataless(tehb_ins_valid, outs_ready);\n";
  }

  mod += "DEFINE outs_valid := tehb_inner.outs_valid;\n";
  if (!isDataless)
    mod += "DEFINE outs := tehb_inner.outs;\n";

  return mod;
}

std::string generateConstant(int val) {
  std::string mod = "MODULE constant_" + std::to_string(val) +
                    "(ctrl_valid, outs_ready)\n"
                    "DEFINE\n"
                    "outs  := " +
                    std::to_string(val) +
                    ";\n"
                    "outs_valid  := ctrl_valid;\n"
                    "ctrl_ready  := outs_ready;\n";

  return mod;
}

std::string generateBOP(const std::string &name, int latency) {
  std::string mod =
      "MODULE " + name +
      "(lhs, lhs_valid, rhs, rhs_valid, result_ready)\n"
      "VAR inner_join : join_generic(lhs_valid, rhs_valid, "
      "inner_oehb.ins_ready);\n"
      "VAR inner_delay_buffer : delay_buffer(inner_join.outs_valid, "
      "inner_oehb.ins_ready, " +
      std::to_string(latency - 1) +
      ");\n"
      "VAR inner_oehb : oehb_1(inner_delay_buffer.outs_valid, "
      "result_ready);\n"
      "DEFINE result := lhs;\n"
      "DEFINE result_valid := inner_oehb.valid_out;\n"
      "DEFINE lhs_ready := inner_join.ins_ready_0;\n"
      "DEFINE rhs_ready := inner_join.ins_ready_1;\n";

  return mod;
}

std::string generateUOP(const std::string &name, int latency) {
  std::string mod =
      "MODULE " + name +
      "(ins, ins_valid, outs_ready)\n"
      "VAR inner_delay_buffer : delay_buffer(ins_valid, "
      "inner_oehb.ins_ready, " +
      std::to_string(latency - 1) +
      ");\n"
      "VAR inner_oehb : oehb_1(inner_delay_buffer.outs_valid, outs_ready);\n"
      "DEFINE outs := ins;\n"
      "DEFINE outs_valid := inner_oehb.valid_out;\n"
      "DEFINE ins_ready := inner_oehb.ins_ready;\n";
  return mod;
}

std::string generateComponent(handshake::OpTypeEnum name,
                              const std::string &params) {

  switch (name) {
  case handshake::OpTypeEnum::JOIN: {
    return generateJoin(std::stoi(params));
  }
  case handshake::OpTypeEnum::FORK: {
    auto pos = params.find_first_of(',');

    if (pos == std::string::npos) {
      int nOutputs = std::stoi(params);
      return generateFork(nOutputs, false);
    }
    std::string firstParam = params.substr(0, pos);
    std::string secondParam = params.substr(pos + 1);
    int nOutputs = std::stoi(firstParam);
    bool isDataless = secondParam == "dataless";
    return generateFork(nOutputs, isDataless);
  }
  case handshake::OpTypeEnum::LAZY_FORK: {
    auto pos = params.find_first_of(',');
    if (pos == std::string::npos) {
      int nOutputs = std::stoi(params);
      return generateLazyFork(nOutputs, false);
    }
    std::string firstParam = params.substr(0, pos);
    std::string secondParam = params.substr(pos + 1);
    int nOutputs = std::stoi(firstParam);
    bool isDataless = secondParam == "dataless";
    return generateLazyFork(nOutputs, isDataless);
  }
  case handshake::OpTypeEnum::MERGE: {
    auto pos = params.find_first_of(',');
    if (pos == std::string::npos) {
      int nInputs = std::stoi(params);
      return generateMerge(nInputs, false);
    }
    std::string firstParam = params.substr(0, pos);
    std::string secondParam = params.substr(pos + 1);
    int nInputs = std::stoi(firstParam);
    bool isDataless = secondParam == "dataless";
    return generateMerge(nInputs, isDataless);
  }
  case handshake::OpTypeEnum::CONTROL_MERGE: {
    auto pos = params.find_first_of(',');
    if (pos == std::string::npos) {
      int nInputs = std::stoi(params);
      return generateControlMerge(nInputs, false);
    }
    std::string firstParam = params.substr(0, pos);
    std::string secondParam = params.substr(pos + 1);
    int nInputs = std::stoi(firstParam);
    bool isDataless = secondParam == "dataless";
    return generateControlMerge(nInputs, isDataless);
  }
  case handshake::OpTypeEnum::MUX: {
    auto pos = params.find_first_of(',');
    if (pos == std::string::npos) {
      int nInputs = std::stoi(params);
      return generateMux(nInputs, false);
    }
    std::string firstParam = params.substr(0, pos);
    std::string secondParam = params.substr(pos + 1);
    int nInputs = std::stoi(firstParam);
    bool isDataless = secondParam == "dataless";
    return generateMux(nInputs, isDataless);
  }
  case handshake::OpTypeEnum::CONSTANT: {
    int val = std::stoi(params);
    return generateConstant(val);
  }
  case handshake::OpTypeEnum::BOP: {
    int pos = params.find_first_of(',');
    std::string name = params.substr(0, pos);
    std::string dataType = params.substr(pos + 1);
    return generateBOP(
        name, dynamatic::handshake::operationLatencies[name][dataType]);
  }
  case handshake::OpTypeEnum::UOP: {
    int pos = params.find_first_of(',');
    std::string name = params.substr(0, pos);
    std::string dataType = params.substr(pos + 1);
    return generateUOP(
        name, dynamatic::handshake::operationLatencies[name][dataType]);
  }
  default:
    return "";
  }
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "SMV generator. Generates the "
      "correct SMV model based operation type and parameters.");

  std::optional<handshake::OpTypeEnum> opType =
      handshake::symbolizeOp(entityName);
  if (!opType) {
    llvm::errs() << "Unknown operation type \"" << entityName << "\"\n";
    return 1;
  }

  // Open the output file
  std::ofstream outputFile(outputRTLPath);
  if (!outputFile.is_open()) {
    llvm::errs() << "Failed to open output file @ \"" << outputRTLPath
                 << "\"\n";
    return 1;
  }

  outputFile << generateComponent(opType.value(), parameters);

  return 0;
}
