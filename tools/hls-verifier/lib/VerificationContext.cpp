//===- VerificationContext.cpp ----------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VerificationContext.h"
#include "HlsLogging.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/IndentedOstream.h"
#include <cassert>

const std::string Properties::KEY_MODELSIM_DIR = "MSIM_DIRECTORY";
const std::string Properties::KEY_VHDL_SRC_DIR = "VHDL_SRC_DIRECTORY";
const std::string Properties::KEY_VHDL_OUT_DIR = "VHDL_OUT_DIRECTORY";
const std::string Properties::KEY_INPUT_DIR = "INPUT_FILE_DIRECTORY";
const std::string Properties::KEY_C_SRC_DIR = "C_SRC_DIRECTORY";
const std::string Properties::KEY_C_OUT_DIR = "C_OUT_DIRECTORY";
const std::string Properties::KEY_MODELSIM_DO_FILE = "MODELSIM_DO_FILE_NAME";
const std::string Properties::KEY_REF_OUT_DIR = "REF_OUT_DIRECTORY";
const std::string Properties::KEY_HLSVERIFY_DIR = "HLS_VERIFY_DIR";
const std::string Properties::KEY_FLOAT_COMPARE_THRESHOLD =
    "FLOAT_COMPARE_THRESHOLD";
const std::string Properties::KEY_DOUBLE_COMPARE_THRESHOLD =
    "DOUBLE_COMPARE_THRESHOLD";
const std::string Properties::DEFAULT_MODELSIM_DIR = "MGC_MSIM";
const std::string Properties::DEFAULT_VHDL_SRC_DIR = "HDL_SRC";
const std::string Properties::DEFAULT_VHDL_OUT_DIR = "HDL_OUT";
const std::string Properties::DEFAULT_INPUT_DIR = "INPUT_VECTORS";
const std::string Properties::DEFAULT_C_SRC_DIR = "C_SRC";
const std::string Properties::DEFAULT_C_OUT_DIR = "C_OUT";
const std::string Properties::DEFAULT_MODLELSIM_DO_FILE = "simulation.do";
const std::string Properties::DEFAULT_REF_OUT_DIR = "REF_OUT";
const std::string Properties::DEFAULT_HLSVERIFY_DIR = "HLS_VERIFY";
const std::string Properties::DEFAULT_FLOAT_COMPARE_THRESHOLD = "0.00001";
const std::string Properties::DEFAULT_DOUBLE_COMPARE_THRESHOLD =
    "0.000000000001";

Properties::Properties() {
  properties[KEY_MODELSIM_DIR] = DEFAULT_MODELSIM_DIR;
  properties[KEY_VHDL_SRC_DIR] = DEFAULT_VHDL_SRC_DIR;
  properties[KEY_VHDL_OUT_DIR] = DEFAULT_VHDL_OUT_DIR;
  properties[KEY_INPUT_DIR] = DEFAULT_INPUT_DIR;
  properties[KEY_C_SRC_DIR] = DEFAULT_C_SRC_DIR;
  properties[KEY_C_OUT_DIR] = DEFAULT_C_OUT_DIR;
  properties[KEY_MODELSIM_DO_FILE] = DEFAULT_MODLELSIM_DO_FILE;
  properties[KEY_REF_OUT_DIR] = DEFAULT_REF_OUT_DIR;
  properties[KEY_HLSVERIFY_DIR] = DEFAULT_HLSVERIFY_DIR;
  properties[KEY_FLOAT_COMPARE_THRESHOLD] = DEFAULT_FLOAT_COMPARE_THRESHOLD;
  properties[KEY_DOUBLE_COMPARE_THRESHOLD] = DEFAULT_DOUBLE_COMPARE_THRESHOLD;
}

string Properties::get(const string &key) const {
  assert(properties.count(key) != 0 && "key not found");
  return properties.at(key);
}

string VerificationContext::getCFuvFunctionName() const {
  if (!cFUVFunctionName.empty())
    return cFUVFunctionName;
  return vhdlDUVEntityName;
}

string VerificationContext::getVhdlDuvEntityName() const {
  if (!vhdlDUVEntityName.empty())
    return vhdlDUVEntityName;
  return cFUVFunctionName;
}

string VerificationContext::getVhdlTestbenchPath() const {
  return getHdlSrcDir() + "/" + "tb_" + getCFuvFunctionName() + ".vhd";
}

string VerificationContext::getModelsimDoFileName() const {
  return properties.get(Properties::KEY_MODELSIM_DO_FILE);
}

string VerificationContext::getCOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_C_OUT_DIR);
}

string VerificationContext::getInputVectorDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_INPUT_DIR);
}

string VerificationContext::getRefOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_REF_OUT_DIR);
}

string VerificationContext::getHdlOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_VHDL_OUT_DIR);
}

string VerificationContext::getBaseDir() const { return ".."; }

string VerificationContext::getHlsVerifyDir() const { return "."; }

string VerificationContext::getHdlSrcDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_VHDL_SRC_DIR);
}
