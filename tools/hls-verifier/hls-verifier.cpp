//===- hls-verifier.cpp - C/VHDL co-simulation ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legacy hls_verifier tool, somewhat cleaned up.
//
//===----------------------------------------------------------------------===//

#include "HlsLogging.h"
#include "HlsVhdlTb.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;

using namespace hls_verify;

static const string LOG_TAG = "[HLS_VERIFIER] ";

static const string COVER_CMD = "cover";
static const string VVER_CMD = "vver";

void generateModelsimScripts(const VerificationContext &ctx) {
  vector<string> filelistVhdl =
      getListOfFilesInDirectory(ctx.getVhdlSrcDir(), ".vhd");
  vector<string> filelistVerilog =
      getListOfFilesInDirectory(ctx.getVhdlSrcDir(), ".v");

  ofstream sim(ctx.getModelsimDoFileName());
  // sim << "vdel -all" << endl;
  sim << "vlib work" << endl;
  sim << "vmap work work" << endl;
  sim << "project new . simulation work modelsim.ini 0" << endl;
  sim << "project open simulation" << endl;
  for (auto &it : filelistVhdl)
    sim << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << endl;

  for (auto &it : filelistVerilog)
    sim << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << endl;

  sim << "project calculateorder" << endl;
  sim << "project compileall" << endl;
  sim << "eval vsim " << ctx.getVhdlDuvEntityName() << "_tb" << endl;
  sim << "log -r *" << endl;
  sim << "run -all" << endl;
  sim << "exit" << endl;
  sim.close();
}

void generateVhdlTestbench(const VerificationContext &ctx) {
  HlsVhdlTb vhdlTb(ctx);
  std::error_code ec;
  std::string filepath = ctx.getVhdlTestbenchPath().c_str();
  llvm::raw_fd_ostream fileStream(filepath, ec);
  mlir::raw_indented_ostream os(fileStream);
  vhdlTb.codegen(os);
}

mlir::LogicalResult compareCAndVhdlOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  cout << "\n--- Comparison Results ---\n" << endl;
  for (const auto &outputParam : outputParams) {
    mlir::LogicalResult result = compareFiles(
        ctx.getCOutPath(outputParam), ctx.getVhdlOutPath(outputParam),
        ctx.getTokenComparator(outputParam));
    cout << "Comparison of [" + outputParam.parameterName + "] : "
         << (mlir::succeeded(result) ? "Pass" : "Fail") << endl;
    if (mlir::failed(result)) {
      return failure();
    }
  }
  return mlir::success();
}

void executeVhdlTestbench(const VerificationContext &ctx,
                          const std::string &resourceDir) {
  string command;

  // Generating VHDL testbench

  logInf(LOG_TAG,
         "Generating VHDL testbench for entity " + ctx.getVhdlDuvEntityName());
  generateVhdlTestbench(ctx);

  // Copying supplementary files
  char sep = std::filesystem::path::preferred_separator;
  auto copyToVHDLDir = [&](const std::string &from,
                           const std::string &to) -> void {
    command =
        "cp " + resourceDir + sep + from + " " + ctx.getVhdlSrcDir() + sep + to;
    logInf(LOG_TAG, "Copying supplementary files: [" + command + "]");
    executeCommand(command);
  };
  auto copyToHLSDir = [&](const std::string &from,
                          const std::string &to) -> void {
    command = "cp " + resourceDir + sep + from + " " + ctx.getHlsVerifyDir() +
              sep + to;
    logInf(LOG_TAG, "Copying supplementary files: [" + command + "]");
    executeCommand(command);
  };

  copyToVHDLDir("template_tb_join.vhd", "tb_join.vhd");
  copyToVHDLDir("template_two_port_RAM.vhd", "two_port_RAM.vhd");
  copyToVHDLDir("template_single_argument.vhd", "single_argument.vhd");
  copyToVHDLDir("template_simpackage.vhd", "simpackage.vhd");
  copyToHLSDir("modelsim.ini", "modelsim.ini");

  // Generating modelsim script for the simulation
  generateModelsimScripts(ctx);

  // Cleaning-up exisiting outputs
  command = "rm -rf " + ctx.getVhdlOutDir();
  logInf(LOG_TAG, "Cleaning VHDL output files [" + command + "]");
  executeCommand(command);

  command = "mkdir -p " + ctx.getVhdlOutDir();
  logInf(LOG_TAG, "Creating VHDL output files directory [" + command + "]");
  executeCommand(command);

  // Executing modelsim
  command = "vsim -c -do " + ctx.getModelsimDoFileName();
  logInf(LOG_TAG, "Executing modelsim: [" + command + "]");
  system(("vsim -c -do " + ctx.getModelsimDoFileName()).c_str());
}

int main(int argc, char **argv) {
  cl::opt<std::string> resourcePathName(
      "resource-path",
      cl::desc("Name of the resource path (with two_port_RAM.vhd, "
               "single_argument.vhd, etc.)"),
      cl::value_desc("resource-path"), cl::Required);
  cl::opt<std::string> cTbPathName(
      "ctb-path", cl::desc("Name of the C file with the main function"),
      cl::value_desc("ctb-path"), cl::Required);
  cl::opt<std::string> cDuvPathName(
      "cduv-path",
      cl::desc("Name of the C file with the kernel to be verified"),
      cl::value_desc("cduv-path"), cl::Required);
  cl::opt<std::string> cFuvFunctionName(
      "cfuv-function-name", cl::desc("Name of the C function name"),
      cl::value_desc("cfuv-function-name"), cl::Required);

  cl::opt<std::string> vhdlDuvEntityName(
      "vhdl-duv-entity-name", cl::desc("Name of the VHDL entity name"),
      cl::value_desc("vhdl-duv-entity-name"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv, R"PREFIX(
    This is the hls-verifier tool for comparing C and VHDL/Verilog outputs.

    Note: All C source files should be in the same subdirectory. Assumes
    hls-verifier is run from a subdirectory (called HLS_VERIFY), which is in the
    same level as the subdirectories for C sources (C_SRC) and the vhdl sources
    (VHDL_SRC). Also assumes that the golden references are in a directory
    called C_OUT in the same level.
    )PREFIX");

  VerificationContext ctx(cTbPathName, cDuvPathName, cFuvFunctionName,
                          vhdlDuvEntityName);
  executeVhdlTestbench(ctx, resourcePathName);

  return succeeded(compareCAndVhdlOutputs(ctx));
}
