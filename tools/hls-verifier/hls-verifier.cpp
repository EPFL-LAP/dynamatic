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

#include "Help.h"
#include "HlsLogging.h"
#include "HlsVhdlTb.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
  vhdlTb.generateVhdlTestbench(os);
}

void checkVhdlTestbenchOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  cout << "\n--- Comparison Results ---\n" << endl;
  for (const auto &outputParam : outputParams) {
    bool result = compareFiles(ctx.getRefOutPath(outputParam),
                               ctx.getVhdlOutPath(outputParam),
                               ctx.getTokenComparator(outputParam));
    cout << "Comparison of [" + outputParam.parameterName + "] : "
         << (result ? "Pass" : "Fail") << endl;
  }
  cout << "\n--------------------------\n" << endl;
}

bool compareCAndVhdlOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  cout << "\n--- Comparison Results ---\n" << endl;
  for (const auto &outputParam : outputParams) {
    bool result = compareFiles(ctx.getCOutPath(outputParam),
                               ctx.getVhdlOutPath(outputParam),
                               ctx.getTokenComparator(outputParam));
    cout << "Comparison of [" + outputParam.parameterName + "] : "
         << (result ? "Pass" : "Fail") << endl;
    return result;
  }
  cout << "\n--------------------------\n" << endl;
  return false;
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

bool runVhdlVerification(vector<string> args) {
  if (args.size() < 2) {
    logErr(LOG_TAG, "Not enough arguments.");
    cout << getVHDLVerificationHelpMessage() << endl;
    return true;
  }

  vector<string> temp;

  for (auto &arg : args)
    if (arg.empty() || arg[0] != '-')
      temp.push_back(arg);

  args = temp;

  string resourceDir = args[0];
  string cTbPath = args[1];
  string vhdlDuvEntityName = args[2];
  string cFuvFunctionName = args.size() > 3 ? args[3] : vhdlDuvEntityName;

  vector<string> otherCPaths;
  VerificationContext ctx(cTbPath, "", cFuvFunctionName, vhdlDuvEntityName,
                          otherCPaths);
  executeVhdlTestbench(ctx, resourceDir);
  checkVhdlTestbenchOutputs(ctx);
  return true;
}

bool runCoverification(vector<string> args) {
  if (args.size() < 5) {
    logErr("[COVER]", "Not enough arguments.");
    cout << getCoVerificationHelpMessage() << endl;
    return true;
  }

  vector<string> temp;
  for (auto &arg : args)
    if (arg.empty() || arg[0] != '-')
      temp.push_back(arg);

  args = temp;

  string resourceDir = args[0];
  string cTbPath = args[1];
  string cDuvPath = args[2];
  string cFuvFunctionName = args[3];
  string vhdlDuvEntityName = args[4];

  vector<string> otherCPaths;
  for (size_t i = 6; i < args.size(); i++)
    otherCPaths.push_back(args[i]);

  VerificationContext ctx(cTbPath, cDuvPath, cFuvFunctionName,
                          vhdlDuvEntityName, otherCPaths);
  executeVhdlTestbench(ctx, resourceDir);
  return compareCAndVhdlOutputs(ctx);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    std::string firstArg(argv[1]);
    vector<string> remainingArgs;
    for (int i = 2; i < argc; i++)
      remainingArgs.emplace_back(argv[i]);
    if (firstArg == VVER_CMD)
      return runVhdlVerification(remainingArgs) ? 0 : -1;
    if (firstArg == COVER_CMD)
      return runCoverification(remainingArgs) ? 0 : -1;
    std::cout << std::endl << "Invalid arguments!" << std::endl;
  } else {
    std::cout << std::endl << "No arguments!" << std::endl;
  }
  std::cout << getGeneralHelpMessage() << std::endl;
  return -1;
}
