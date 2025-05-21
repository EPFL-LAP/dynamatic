//===- VerificationContext.h ------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_VERIFICATION_CONTEXT_H
#define HLS_VERIFIER_VERIFICATION_CONTEXT_H

#include "Utilities.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/IndentedOstream.h"
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace dynamatic;

static const std::string MODELSIM_DIR = "MGC_MSIM";
static const std::string HDL_SRC_DIR = "HDL_SRC";
static const std::string HDL_OUT_DIR = "HDL_OUT";
static const std::string INPUT_VECTORS_DIR = "INPUT_VECTORS";
static const std::string C_SOURCE_DIR = "C_SRC";
static const std::string C_OUT_DIR = "C_OUT";
static const std::string VSIM_SCRIPT_FILE = "simulation.do";
static const std::string HLS_VERIFY_DIR = "HLS_VERIFY";

struct VerificationContext {
  VerificationContext(const std::string &simPath,
                      const std::string &cFuvFunctionName,
                      handshake::FuncOp *funcOp)
      : simPath(simPath), funcOp(funcOp), kernelName(cFuvFunctionName) {}

  static const char SEP = std::filesystem::path::preferred_separator;

  // Path to the simulation directory
  std::string simPath;

  // Pointer to the funcOp of the top-level handshake function
  handshake::FuncOp *funcOp;

  // The name of the top-level handshake function
  std::string kernelName;

  std::string getVhdlTestbenchPath() const {
    return getHdlSrcDir() + SEP + "tb_" + kernelName + ".vhd";
  }

  std::string getModelsimDoFilePath() const { return VSIM_SCRIPT_FILE; }

  std::string getCOutDir() const { return simPath + SEP + C_OUT_DIR; }

  std::string getInputVectorDir() const {
    return simPath + "/" + INPUT_VECTORS_DIR;
  }

  std::string getHdlOutDir() const { return simPath + "/" + HDL_OUT_DIR; }

  std::string getHlsVerifyDir() const { return simPath + "/" + HLS_VERIFY_DIR; }

  std::string getHdlSrcDir() const { return simPath + "/" + HDL_SRC_DIR; }
};

#endif // HLS_VERIFIER_VERIFICATION_CONTEXT_H
