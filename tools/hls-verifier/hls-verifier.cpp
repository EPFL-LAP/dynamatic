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
#include "Simulators.h"
#include "Utilities.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
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

static const string LOG_TAG = "HLS_VERIFIER";

mlir::LogicalResult compareCAndVhdlOutputs(const VerificationContext &ctx) {

  // mlir::raw_indented_ostream &os = ctx.testbenchStream;
  handshake::FuncOp *funcOp = ctx.funcOp;
  // os << "-- Write [[[runtime]]], [[[/runtime]]] for output transactor\n";

  llvm::SmallVector<std::pair<std::string, Type>> argAndTypeMap;

  // Collecting the types and names of the arguments to be compared. In
  // Dynamatic, the input data/control channel and the memory block connected to
  // the circuit are presented at the arguments.
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

  // Collecting the types and names of the output channels. Skipping the
  // ControlType as there is no value to be compared. The outputs can only be
  // data/control channels (no arrays).
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

int main(int argc, char **argv) {

  cl::opt<std::string> simPathName(
      "sim-path",
      cl::desc("Path where the simulation files and directories are located"),
      cl::value_desc("Simulation directory"), cl::Required);

  cl::opt<std::string> hlsKernelName(
      "kernel-name", cl::desc("Name of the HLS kernel"),
      cl::value_desc("HLS kernel name"), cl::Required);

  cl::opt<std::string> mlirPathName(
      "handshake-mlir",
      cl::desc("Name of the handshake MLIR file with the kernel"),
      cl::value_desc("handshake-mlir"), cl::Required);

  cl::opt<bool> vivadoFPU(
      "vivado-fpu",
      cl::desc("Use Vivado FPU for floating-point operations (default: false)"),
      cl::value_desc("vivado-fpu"), cl::init(false));

  cl::opt<std::string> simulatorType(
      "simulator", cl::desc("Simulator of choice (options: xsim, ghdl, vsim)"),
      cl::value_desc("Simulator of choice"), cl::init("ghdl"));

  cl::ParseCommandLineOptions(argc, argv, R"PREFIX(
    This is the hls-verifier tool for comparing C and VHDL/Verilog outputs.

    HlsVerifier assumes the following directory structure:
    - All the C source files must be in a directory as cDuvPathName in C_SRC.
    - All HDL sources must be in HDL_SRC.
    - hls-verifier must run from a subdirectory called HLS_VERIFY
    - The golden references must be in a directory called C_OUT.
    - C_SRC, HDL_SRC, C_OUT, and HLS_VERIFY must be in the same directory.
    
    )PREFIX");

  // We only need the Handshake dialect
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(mlirPathName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file '" << mlirPathName
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
      dyn_cast<handshake::FuncOp>(modOp->lookupSymbol(hlsKernelName));

  VerificationContext ctx(simPathName, hlsKernelName, &funcOp, vivadoFPU);

  // Generate hls_verify_<hlsKernelName>.vhd
  vhdlTbCodegen(ctx);

  std::unique_ptr<Simulator> simulator;

  if (simulatorType == "ghdl") {
    simulator = std::make_unique<GHDLSimulator>(&ctx);
  } else if (simulatorType == "vsim") {
    simulator = std::make_unique<VSimSimulator>(&ctx);
  } else if (simulatorType == "xsim") {
    simulator = std::make_unique<XSimSimulator>(&ctx);
  } else {
    logErr(LOG_TAG, "Wrong Simulator (use vsim, xsim, ghdl, verilator)");
    return 1;
  }

  if (failed(simulator->generateScripts())) {
    logInf(LOG_TAG, "Failed to generate Simulation Script");
  }

  // Run the simulator to simulate the testbench and write the outputs to the
  // VHDL_OUT
  simulator->execSimulation();

  if (succeeded(compareCAndVhdlOutputs(ctx))) {
    logInf(LOG_TAG, "C and VHDL outputs match");
  } else {
    logErr(LOG_TAG, "C and VHDL outputs do not match");
    return 1;
  }
  return 0;
}
