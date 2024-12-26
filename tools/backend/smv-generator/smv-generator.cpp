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
// allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
// 12, 13, 14, 15
enum class OpTypeEnum : uint64_t {
  DEFAULT = 0,
  JOIN = 1,
  MERGE = 2,
  MUX = 3,
  CONSTANT = 4,
};

OpTypeEnum symbolizeOp(const std::string &name) {
  if (name == "join")
    return OpTypeEnum::JOIN;
  if (name == "merge")
    return OpTypeEnum::MERGE;
  if (name == "mux")
    return OpTypeEnum::MUX;
  if (name == "constant")
    return OpTypeEnum::CONSTANT;
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
  assert(inputSignals > 0);
  std::string mod =
      "MODULE join_" + std::to_string(inputSignals) + "_1 (ins_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += ", inst_valid_" + std::to_string(i);
  mod += "outs_ready)\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod += "DEFINE ins_ready" + std::to_string(i) + "  := outs_ready";
    for (unsigned int j = 1; j < inputSignals; j++)
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

std::string generateMerge(unsigned int inputSignals) {
  assert(inputSignals > 0);
  // MODULE merge_*_1_dataless(ins_valid_0, ins_valid_1, ..., outs_ready)
  // VAR tehb_inner : tehb_dataless(tehb_pvalid, outs_ready);
  // DEFINE ins_ready_0 := tehb_inner.ins_ready;
  // DEFINE ins_ready_1 := tehb_inner.ins_ready;
  // ...
  // DEFINE outs_valid := tehb_inner.outs_valid;
  // DEFINE tehb_pvalid := ins_valid_0 | ins_valid_1 | ...

  std::string mod =
      "MODULE merge_" + std::to_string(inputSignals) + "_1 (inst_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += ", inst_valid_" + std::to_string(i);
  mod += ", outs_ready)\n";

  mod += "VAR tehb_inner : tehb_dataless(tehb_pvalid, outs_ready);\n";

  for (unsigned int i = 0; i < inputSignals; i++) {
    mod +=
        "DEFINE ins_ready" + std::to_string(i) + "  := tehb_inner.ins_ready;\n";
  }

  mod += "DEFINE outs_valid := tehb_inner.outs_valid;\n";

  mod += "DEFINE tehb_pvalid := ins_valid_0;";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += " | ins_valid_" + std::to_string(i);
  mod += ";\n";

  return mod;
}

std::string generateMux(unsigned int inputSignals) {
  // assert(inputSignals > 0);
  //  MODULE mux_*_1(ins_valid_0, ins_valid_1, ...,  index, index_valid,
  //  outs_ready)
  //  DEFINE ins_ready_0 := (index == 0 & index_valid &
  //  tehb_inner.ins_ready & ins_valid_0) | !ins_valid_0;
  //  DEFINE ins_ready_1 :=
  //  (index == 1 & index_valid & tehb_inner.ins_ready & ins_valid_1) |
  //  !ins_valid_1;
  //  ...
  //  DEFINE tehb_ins_valid := case
  //                            index == 0 : index_valid & ins_valid_0;
  //                            index == 1 : index_valid & ins_valid_1;
  //                            ...
  //                            esac;
  //  VAR tehb_inner : tehb_dataless(tehb_ins_valid, outs_ready);
  //  DEFINE outs_valid := tehb_inner.outs_valid;

  std::string mod =
      "MODULE mux_" + std::to_string(inputSignals) + "_1 (inst_valid_0";
  for (unsigned int i = 1; i < inputSignals; i++)
    mod += ", inst_valid_" + std::to_string(i);
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

  mod += "VAR tehb_inner : tehb_dataless(tehb_ins_valid, outs_ready);\n";

  mod += "DEFINE outs_valid := tehb_inner.outs_valid;";

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

std::string generateComponent(handshake::OpTypeEnum name,
                              const std::string &params) {

  switch (name) {
  case handshake::OpTypeEnum::JOIN: {
    int nInputs = std::stoi(params);
    return generateJoin(nInputs);
  }
  case handshake::OpTypeEnum::MERGE: {
    int nInputs = std::stoi(params);
    return generateMerge(nInputs);
  }
  case handshake::OpTypeEnum::MUX: {
    int nInputs = std::stoi(params);
    return generateMux(nInputs);
  }
  case handshake::OpTypeEnum::CONSTANT: {
    int val = std::stoi(params);
    return generateConstant(val);
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
