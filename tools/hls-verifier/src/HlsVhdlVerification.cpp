//===- HlsVhdlVerification.cpp ----------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HlsVhdlVerification.h"
#include "Help.h"
#include "HlsLogging.h"
#include "HlsVhdlTb.h"
#include "Utilities.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace hls_verify {
const string LOG_TAG = "VVER";

bool runVhdlVerification(vector<string> args) {
  if (args.size() < 2) {
    logErr(LOG_TAG, "Not enough arguments.");
    cout << getVHDLVerificationHelpMessage() << endl;
    return true;
  }

  bool useAddrWidth32 = false;

  vector<string> temp;

  for (auto &arg : args) {
    if (!arg.empty() && arg[0] == '-') {
      if (arg == "-aw32") {
        useAddrWidth32 = true;
      }
    } else {
      temp.push_back(arg);
    }
  }

  args = temp;

  string resourceDir = args[0];
  string cTbPath = args[1];
  string vhdlDuvEntityName = args[2];
  string cFuvFunctionName = args.size() > 3 ? args[3] : vhdlDuvEntityName;

  vector<string> otherCPaths;
  VerificationContext ctx(cTbPath, "", cFuvFunctionName, vhdlDuvEntityName,
                          otherCPaths);
  ctx.useAddrWidth32 = useAddrWidth32;
  executeVhdlTestbench(ctx, resourceDir);
  checkVhdlTestbenchOutputs(ctx);
  return true;
}

void generateVhdlTestbench(const VerificationContext &ctx) {
  HlsVhdlTb vhdlTb(ctx);
  ofstream fout(ctx.getVhdlTestbenchPath());
  fout << vhdlTb.generateVhdlTestbench();
  fout.close();
}

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
  for (auto &it : filelistVhdl) {
    sim << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << endl;
  }

  for (auto &it : filelistVerilog) {
    sim << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << endl;
  }
  sim << "project calculateorder" << endl;
  sim << "project compileall" << endl;
  sim << "eval vsim " << ctx.getVhdlDuvEntityName() << "_tb" << endl;
  sim << "log -r *" << endl;
  sim << "run -all" << endl;
  sim << "exit" << endl;
  sim.close();
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
} // namespace hls_verify
