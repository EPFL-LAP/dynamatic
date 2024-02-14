//===- VerificationContext.cpp ----------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VerificationContext.h"
#include "HlsLogging.h"
#include <cassert>

namespace hls_verify {

const string Properties::KEY_MODELSIM_DIR = "MSIM_DIRECTORY";
const string Properties::KEY_VHDL_SRC_DIR = "VHDL_SRC_DIRECTORY";
const string Properties::KEY_VHDL_OUT_DIR = "VHDL_OUT_DIRECTORY";
const string Properties::KEY_INPUT_DIR = "INPUT_FILE_DIRECTORY";
const string Properties::KEY_C_SRC_DIR = "C_SRC_DIRECTORY";
const string Properties::KEY_C_OUT_DIR = "C_OUT_DIRECTORY";
const string Properties::KEY_MODELSIM_DO_FILE = "MODELSIM_DO_FILE_NAME";
const string Properties::KEY_REF_OUT_DIR = "REF_OUT_DIRECTORY";
const string Properties::KEY_HLSVERIFY_DIR = "HLS_VERIFY_DIR";
const string Properties::KEY_FLOAT_COMPARE_THRESHOLD =
    "FLOAT_COMPARE_THRESHOLD";
const string Properties::KEY_DOUBLE_COMPARE_THRESHOLD =
    "DOUBLE_COMPARE_THRESHOLD";
const string Properties::DEFAULT_MODELSIM_DIR = "MGC_MSIM";
const string Properties::DEFAULT_VHDL_SRC_DIR = "VHDL_SRC";
const string Properties::DEFAULT_VHDL_OUT_DIR = "VHDL_OUT";
const string Properties::DEFAULT_INPUT_DIR = "INPUT_VECTORS";
const string Properties::DEFAULT_C_SRC_DIR = "C_SRC";
const string Properties::DEFAULT_C_OUT_DIR = "C_OUT";
const string Properties::DEFAULT_MODLELSIM_DO_FILE = "simulation.do";
const string Properties::DEFAULT_REF_OUT_DIR = "REF_OUT";
const string Properties::DEFAULT_HLSVERIFY_DIR = "HLS_VERIFY";
const string Properties::DEFAULT_FLOAT_COMPARE_THRESHOLD = "1";
const string Properties::DEFAULT_DOUBLE_COMPARE_THRESHOLD = "0.000000000001";

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

VerificationContext::VerificationContext(const string &cTbPath,
                                         const string &cFuvPath,
                                         const string &cFuvFunctionName,
                                         const string &vhdlDuvEntityName,
                                         vector<string> &otherCSrcPaths)
    : properties(), cTBPath(cTbPath), cFUVPath(cFuvPath),
      otherCSrcPaths(otherCSrcPaths), cFUVFunctionName(cFuvFunctionName),
      vhdlDUVEntityName(vhdlDuvEntityName) {
  string preprocessedCTb = CAnalyser::getPreprocOutput(
      getCTbPath(), extractParentDirectoryPath(getCTbPath()));
  CAnalyser::parseCFunction(preprocessedCTb, getCFuvFunctionName(), fuv);
  defaultComparator = TokenCompare();
  unsignedIntComparator = IntegerCompare(false);
  signedIntComparator = IntegerCompare(true);
  floatComparator = FloatCompare(
      stof(properties.get(Properties::KEY_FLOAT_COMPARE_THRESHOLD)));
  doubleComparator = DoubleCompare(
      stod(properties.get(Properties::KEY_FLOAT_COMPARE_THRESHOLD)));
}

string VerificationContext::getCTbPath() const { return cTBPath; }

string VerificationContext::getCFuvPath() const { return cFUVPath; }

string VerificationContext::getCFuvFunctionName() const {
  if (!cFUVFunctionName.empty())
    return cFUVFunctionName;
  return vhdlDUVEntityName;
}

string VerificationContext::getVhdlDuvEntityName() const {
  if (!vhdlDUVEntityName.empty()) {
    return vhdlDUVEntityName;
  }
  return cFUVFunctionName;
}

vector<string> VerificationContext::getOtherCSrcPaths() const {
  return otherCSrcPaths;
}

string VerificationContext::getCExecutablePath() const {
  return extractParentDirectoryPath(cFUVPath) + "/" + "hls_verify_" +
         getCFuvFunctionName() + ".out";
}

string VerificationContext::getVhdlTestbenchPath() const {
  return getVhdlSrcDir() + "/" + "hls_verify_" + getVhdlDuvEntityName() +
         "_tb.vhd";
}

string VerificationContext::getModelsimDoFileName() const {
  return properties.get(Properties::KEY_MODELSIM_DO_FILE);
}

string VerificationContext::getInjectedCFuvPath() const {
  return extractParentDirectoryPath(cFUVPath) + "/" + "hls_verify_" +
         getCFuvFunctionName() + ".c";
}

string VerificationContext::getCOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_C_OUT_DIR);
}

string VerificationContext::getCOutPath(const CFunctionParameter &param) const {
  assert(param.isOutput && "Parameter is not an output type.");
  return getCOutDir() + "/output_" + param.parameterName + ".dat";
}

string VerificationContext::getInputVectorDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_INPUT_DIR);
}

string
VerificationContext::getInputVectorPath(const CFunctionParameter &param) const {
  assert(param.isInput && "Parameter is not an input type.");
  return getInputVectorDir() + "/input_" + param.parameterName + ".dat";
}

string VerificationContext::getRefOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_REF_OUT_DIR);
}

string
VerificationContext::getRefOutPath(const CFunctionParameter &param) const {
  assert(param.isOutput && "Parameter is not an output type.");
  return getCOutDir() + "/output_" + param.parameterName + ".dat";
}

string VerificationContext::getVhdlOutDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_VHDL_OUT_DIR);
}

string
VerificationContext::getVhdlOutPath(const CFunctionParameter &param) const {
  assert(param.isOutput && "Parameter is not an output type.");
  return getVhdlOutDir() + "/output_" + param.parameterName + ".dat";
}

string VerificationContext::getBaseDir() const { return ".."; }

string VerificationContext::getHlsVerifyDir() const { return "."; }

string VerificationContext::getVhdlSrcDir() const {
  return getBaseDir() + "/" + properties.get(Properties::KEY_VHDL_SRC_DIR);
}

const TokenCompare *
VerificationContext::getTokenComparator(const CFunctionParameter &param) const {
  if (param.isIntType) {
    if (param.actualType.find("unsigned") != string::npos)
      return &unsignedIntComparator;
    return &signedIntComparator;
  }
  if (param.isFloatType && param.dtWidth == 32) {
    return &floatComparator;
  }
  if (param.isFloatType && param.dtWidth == 64) {
    return &doubleComparator;
  }
  return &defaultComparator;
}

CFunction VerificationContext::getCFuv() const { return fuv; }

vector<CFunctionParameter> VerificationContext::getFuvInputParams() const {
  vector<CFunctionParameter> result;
  for (const auto &param : fuv.params) {
    if (param.isInput)
      result.push_back(param);
  }
  return result;
}

vector<CFunctionParameter> VerificationContext::getFuvOutputParams() const {
  vector<CFunctionParameter> result;
  if (fuv.returnVal.isOutput) {
    result.push_back(fuv.returnVal);
  }
  for (const auto &param : fuv.params) {
    if (param.isOutput)
      result.push_back(param);
  }
  return result;
}

vector<CFunctionParameter> VerificationContext::getFuvParams() const {
  vector<CFunctionParameter> result;
  result.push_back(fuv.returnVal);
  for (const auto &param : fuv.params)
    result.push_back(param);
  return result;
}

} // namespace hls_verify
