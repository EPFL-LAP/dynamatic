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
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h" // Include the header for OwningOpRef
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static const char SEP = std::filesystem::path::preferred_separator;

static const string LOG_TAG = "[HLS_VERIFIER] ";

void generateModelsimScripts(const VerificationContext &ctx) {
  vector<string> filelistVhdl =
      getListOfFilesInDirectory(ctx.getHdlSrcDir(), ".vhd");
  vector<string> filelistVerilog =
      getListOfFilesInDirectory(ctx.getHdlSrcDir(), ".v");

  std::error_code ec;
  llvm::raw_fd_ostream os(ctx.getModelsimDoFileName(), ec);
  // os << "vdel -all" << endl;
  os << "vlib work\n";
  os << "vmap work work\n";
  os << "project new . simulation work modelsim.ini 0\n";
  os << "project open simulation\n";
  for (auto &it : filelistVhdl)
    os << "project addfile " << ctx.getHdlSrcDir() << "/" << it << "\n";

  for (auto &it : filelistVerilog)
    os << "project addfile " << ctx.getHdlSrcDir() << "/" << it << "\n";

  os << "project calculateorder\n";
  os << "project compileall\n";
  os << "eval vsim tb\n";
  os << "log -r *\n";
  os << "run -all\n";
  os << "exit\n";
}

void copySupplementaryFiles(const VerificationContext &ctx,
                            const std::string &resourcePathName) {
  auto copyToVHDLDir = [&](const std::string &from,
                           const std::string &to) -> void {
    string command;
    command = "cp " + resourcePathName + SEP + from + " " + ctx.getHdlSrcDir() +
              SEP + to;
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

  // mlir::raw_indented_ostream &os = ctx.testbenchStream;
  handshake::FuncOp *funcOp = ctx.funcOp;
  // os << "-- Write [[[runtime]]], [[[/runtime]]] for output transactor\n";

  llvm::SmallVector<std::pair<std::string, Type>> argAndTypeMap;

  // Connect the memory elements to the DUV
  for (auto [arg, portAttr] : llvm::zip_equal(
           funcOp->getBodyBlock()->getArguments(), funcOp->getArgNames())) {

    std::string argName = portAttr.dyn_cast<StringAttr>().data();

    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(arg.getType())) {
      argAndTypeMap.emplace_back(argName, type.getDataType());

    } else if (mlir::MemRefType type =
                   dyn_cast<mlir::MemRefType>(arg.getType())) {
      argAndTypeMap.emplace_back(argName, type.getElementType());
    }
  }

  // Connect the output channels to the DUV
  for (auto [resType, portAttr] :
       llvm::zip_equal(funcOp->getResultTypes(), funcOp->getResNames())) {
    std::string argName = portAttr.dyn_cast<StringAttr>().str();
    if (handshake::ChannelType type =
            dyn_cast<handshake::ChannelType>(resType)) {
      argAndTypeMap.emplace_back(argName, type.getDataType());
    }
  }

  for (auto [argName, type] : argAndTypeMap) {
    std::string vhdlOutFile =
        ctx.getHdlOutDir() + SEP + "output_" + argName + ".dat";

    std::string cOutFile =
        ctx.getCOutDir() + SEP + "output_" + argName + ".dat";

    LogicalResult result = failure();
    if (isa<Float32Type>(type)) {
      std::unique_ptr<TokenCompare> comparator =
          std::make_unique<FloatCompare>();
      result = compareFiles(cOutFile, vhdlOutFile, std::move(comparator));
      llvm::errs() << "FP32 comparison of [" + argName + "] : "
                   << (mlir::succeeded(result) ? "Pass" : "Fail") << "\n";
    } else if (isa<Float64Type>(type)) {
      std::unique_ptr<TokenCompare> comparator =
          std::make_unique<DoubleCompare>();
      result = compareFiles(cOutFile, vhdlOutFile, std::move(comparator));
      llvm::errs() << "FP64 comparison of [" + argName + "] : "
                   << (mlir::succeeded(result) ? "Pass" : "Fail") << "\n";
    } else if (isa<IntegerType>(type)) {
      std::unique_ptr<TokenCompare> comparator =
          std::make_unique<IntegerCompare>();
      result = compareFiles(cOutFile, vhdlOutFile, std::move(comparator));
      llvm::errs() << "Comparison of [" + argName + "] : "
                   << (mlir::succeeded(result) ? "Pass" : "Fail") << "\n";
    }
    if (failed(result)) {
      return failure();
    }
  }

  return mlir::success();
}

void executeVhdlTestbench(const VerificationContext &ctx) {
  string command;

  // Cleaning-up exisiting outputs
  command = "rm -rf " + ctx.getHdlOutDir();
  logInf(LOG_TAG, "Cleaning VHDL output files [" + command + "]");
  executeCommand(command);

  command = "mkdir -p " + ctx.getHdlOutDir();
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
  cl::opt<std::string> cFuvFunctionName(
      "cfuv-function-name", cl::desc("Name of the C function name"),
      cl::value_desc("cfuv-function-name"), cl::Required);

  cl::opt<std::string> vhdlDuvEntityName(
      "hdl-duv-entity-name", cl::desc("Name of the HDL entity name"),
      cl::value_desc("hdl-duv-entity-name"), cl::Required);
  cl::opt<std::string> mlirPathName(
      "handshake-mlir",
      cl::desc("Name of the handshake MLIR file with the kernel"),
      cl::value_desc("handshake-mlir"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv, R"PREFIX(
    This is the hls-verifier tool for comparing C and VHDL/Verilog outputs.

    HlsVerifier assumes the following directory structure:
    - All the C source files must be in a directory as cDuvPathName in C_SRC.
    - All HDL sources must be in VHDL_SRC.
    - hls-verifier must run from a subdirectory called HLS_VERIFY
    - The golden references must be in a directory called C_OUT.
    - C_SRC, VHDL_SRC, C_OUT, and HLS_VERIFY must be in the same directory.
    
    )PREFIX");

  // We only need the Handshake dialect
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(mlirPathName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << mlirPathName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  handshake::FuncOp funcOp =
      dyn_cast<handshake::FuncOp>(modOp->lookupSymbol(cFuvFunctionName));

  std::string vhdlSrcDir = "../VHDL_SRC";

  std::string vhdlTestbenchPath =
      vhdlSrcDir + SEP + "hls_verify_" + cFuvFunctionName + "_tb.vhd";

  VerificationContext ctx(cFuvFunctionName, vhdlDuvEntityName, &funcOp);

  // Generate hls_verify_<cFuvFunctionName>.vhd
  vhdlTbCodegen(ctx);

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
