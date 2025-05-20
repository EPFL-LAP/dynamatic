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
  VerificationContext(const std::string &cFuvFunctionName,
                      const std::string &vhdlDuvEntityName,
                      handshake::FuncOp *funcOp)
      : funcOp(funcOp), cFUVFunctionName(cFuvFunctionName),
        vhdlDUVEntityName(vhdlDuvEntityName) {}

  std::string getCFuvFunctionName() const {
    if (!cFUVFunctionName.empty())
      return cFUVFunctionName;
    return vhdlDUVEntityName;
  }

  std::string getVhdlDuvEntityName() const {
    if (!vhdlDUVEntityName.empty())
      return vhdlDUVEntityName;
    return cFUVFunctionName;
  }

  std::string getBaseDir() const { return ".."; }

  std::string getVhdlTestbenchPath() const {
    return getHdlSrcDir() + "/" + "tb_" + getCFuvFunctionName() + ".vhd";
  }

  std::string getModelsimDoFileName() const { return VSIM_SCRIPT_FILE; }

  std::string getCOutDir() const { return getBaseDir() + "/" + C_OUT_DIR; }

  std::string getInputVectorDir() const {
    return getBaseDir() + "/" + INPUT_VECTORS_DIR;
  }

  std::string getHdlOutDir() const { return getBaseDir() + "/" + HDL_OUT_DIR; }

  std::string getHlsVerifyDir() const { return "."; }

  std::string getHdlSrcDir() const { return getBaseDir() + "/" + HDL_SRC_DIR; }

  handshake::FuncOp *funcOp;

  std::string cFUVFunctionName;
  std::string vhdlDUVEntityName;
};

#endif // HLS_VERIFIER_VERIFICATION_CONTEXT_H
