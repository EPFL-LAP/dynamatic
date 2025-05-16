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
#include "Utilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;

using namespace hls_verify;
static const char SEP = std::filesystem::path::preferred_separator;

static const string LOG_TAG = "[HLS_VERIFIER] ";

void generateModelsimScripts(const VerificationContext &ctx) {
  vector<string> filelistVhdl =
      getListOfFilesInDirectory(ctx.getVhdlSrcDir(), ".vhd");
  vector<string> filelistVerilog =
      getListOfFilesInDirectory(ctx.getVhdlSrcDir(), ".v");

  std::error_code ec;
  llvm::raw_fd_ostream os(ctx.getModelsimDoFileName(), ec);
  // os << "vdel -all" << endl;
  os << "vlib work\n";
  os << "vmap work work\n";
  os << "project new . simulation work modelsim.ini 0\n";
  os << "project open simulation\n";
  for (auto &it : filelistVhdl)
    os << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << "\n";

  for (auto &it : filelistVerilog)
    os << "project addfile " << ctx.getVhdlSrcDir() << "/" << it << "\n";

  os << "project calculateorder\n";
  os << "project compileall\n";
  os << "eval vsim " << ctx.getVhdlDuvEntityName() << "_tb\n";
  os << "log -r *\n";
  os << "run -all\n";
  os << "exit\n";
}

void generateVhdlTestbench(const VerificationContext &ctx) {
  logInf(LOG_TAG,
         "Generating VHDL testbench for entity " + ctx.getVhdlDuvEntityName());
  HlsVhdlTb vhdlTb(ctx);
  std::error_code ec;
  std::string filepath = ctx.getVhdlTestbenchPath().c_str();
  llvm::raw_fd_ostream fileStream(filepath, ec);
  mlir::raw_indented_ostream os(fileStream);
  vhdlTb.codegen(os);
}

void copySupplementaryFiles(const VerificationContext &ctx,
                            const std::string &resourcePathName) {
  auto copyToVHDLDir = [&](const std::string &from,
                           const std::string &to) -> void {
    string command;
    command = "cp " + resourcePathName + SEP + from + " " +
              ctx.getVhdlSrcDir() + SEP + to;
    logInf(LOG_TAG, "Copying supplementary files: [" + command + "]");
    executeCommand(command);
  };
  auto copyToHLSDir = [&](const std::string &from,
                          const std::string &to) -> void {
    string command;
    command = "cp " + resourcePathName + SEP + from + " " +
              ctx.getHlsVerifyDir() + SEP + to;
    logInf(LOG_TAG, "Copying supplementary files: [" + command + "]");
    executeCommand(command);
  };

  copyToVHDLDir("template_tb_join.vhd", "tb_join.vhd");
  copyToVHDLDir("template_two_port_RAM.vhd", "two_port_RAM.vhd");
  copyToVHDLDir("template_single_argument.vhd", "single_argument.vhd");
  copyToVHDLDir("template_simpackage.vhd", "simpackage.vhd");
  copyToHLSDir("modelsim.ini", "modelsim.ini");
}

mlir::LogicalResult compareCAndVhdlOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  llvm::errs() << "\n--- Comparison Results ---\n";
  for (const auto &outputParam : outputParams) {
    mlir::LogicalResult result = compareFiles(
        ctx.getCOutPath(outputParam), ctx.getVhdlOutPath(outputParam),
        ctx.getTokenComparator(outputParam));
    llvm::errs() << "Comparison of [" + outputParam.parameterName + "] : "
                 << (mlir::succeeded(result) ? "Pass" : "Fail") << "\n";
    if (failed(result)) {
      return failure();
    }
  }
  return mlir::success();
}

void executeVhdlTestbench(const VerificationContext &ctx) {
  string command;

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
  executeCommand(command);
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
      "hdl-duv-entity-name", cl::desc("Name of the HDL entity name"),
      cl::value_desc("hdl-duv-entity-name"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv, R"PREFIX(
    This is the hls-verifier tool for comparing C and VHDL/Verilog outputs.

    HlsVerifier assumes the following directory structure:
    - All the C source files must be in a directory as cDuvPathName in C_SRC.
    - All HDL sources must be in VHDL_SRC.
    - hls-verifier must run from a subdirectory called HLS_VERIFY
    - The golden references must be in a directory called C_OUT.
    - C_SRC, VHDL_SRC, C_OUT, and HLS_VERIFY must be in the same directory.
    
    )PREFIX");

  VerificationContext ctx(cTbPathName, cDuvPathName, cFuvFunctionName,
                          vhdlDuvEntityName);

  // Generate hls_verify_<cFuvFunctionName>.vhd
  generateVhdlTestbench(ctx);

  // Copy two_port_RAM.vhd, single_argument.vhd, etc. to the VHDL source
  copySupplementaryFiles(ctx, resourcePathName);

  // Need to first copy the supplementary files to the VHDL source before
  // generating the scripts (it looks at the existing files to generate the
  // scripts).
  generateModelsimScripts(ctx);

  // Run modelsim to simulate the testbench and write the outputs to the
  // VHDL_OUT
  executeVhdlTestbench(ctx);

  if (succeeded(compareCAndVhdlOutputs(ctx))) {
    logInf(LOG_TAG, "C and VHDL outputs match");
  } else {
    logErr(LOG_TAG, "C and VHDL outputs do not match");
    return 1;
  }
  return 0;
}
