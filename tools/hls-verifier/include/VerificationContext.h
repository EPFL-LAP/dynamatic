//===- VerificationContext.h ------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_VERIFICATION_CONTEXT_H
#define HLS_VERIFIER_VERIFICATION_CONTEXT_H

#include "CAnalyser.h"
#include "Utilities.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

namespace hls_verify {

class Properties {
public:
  static const string KEY_MODELSIM_DIR;
  static const string KEY_VHDL_SRC_DIR;
  static const string KEY_VHDL_OUT_DIR;
  static const string KEY_INPUT_DIR;
  static const string KEY_C_SRC_DIR;
  static const string KEY_C_OUT_DIR;
  static const string KEY_MODELSIM_DO_FILE;
  static const string KEY_REF_OUT_DIR;
  static const string KEY_HLSVERIFY_DIR;
  static const string KEY_FLOAT_COMPARE_THRESHOLD;
  static const string KEY_DOUBLE_COMPARE_THRESHOLD;
  static const string DEFAULT_MODELSIM_DIR;
  static const string DEFAULT_VHDL_SRC_DIR;
  static const string DEFAULT_VHDL_OUT_DIR;
  static const string DEFAULT_INPUT_DIR;
  static const string DEFAULT_C_SRC_DIR;
  static const string DEFAULT_C_OUT_DIR;
  static const string DEFAULT_MODLELSIM_DO_FILE;
  static const string DEFAULT_REF_OUT_DIR;
  static const string DEFAULT_HLSVERIFY_DIR;
  static const string DEFAULT_FLOAT_COMPARE_THRESHOLD;
  static const string DEFAULT_DOUBLE_COMPARE_THRESHOLD;

  Properties();
  Properties(const string &propertiesFileName);

  string get(const string &key) const;

private:
  map<string, string> properties;
};

class VerificationContext {
public:
  VerificationContext(const string &cTbPath, const string &cFuvPath,
                      const string &cFuvFunctionName,
                      const string &vhdlDuvEntityName,
                      vector<string> &otherCSrcPaths);

  string getCTbPath() const;
  string getCFuvPath() const;
  vector<string> getOtherCSrcPaths() const;
  string getCFuvFunctionName() const;
  string getVhdlDuvEntityName() const;

  string getInjectedCFuvPath() const;
  string getCExecutablePath() const;
  string getVhdlTestbenchPath() const;

  string getBaseDir() const;
  string getHlsVerifyDir() const;
  string getVhdlSrcDir() const;

  string getCOutDir() const;
  string getRefOutDir() const;
  string getVhdlOutDir() const;
  string getInputVectorDir() const;

  string getCOutPath(const CFunctionParameter &param) const;
  string getRefOutPath(const CFunctionParameter &param) const;
  string getVhdlOutPath(const CFunctionParameter &param) const;
  string getInputVectorPath(const CFunctionParameter &param) const;

  string getModelsimDoFileName() const;

  const TokenCompare *getTokenComparator(const CFunctionParameter &param) const;

  CFunction getCFuv() const;
  vector<CFunctionParameter> getFuvOutputParams() const;
  vector<CFunctionParameter> getFuvInputParams() const;
  vector<CFunctionParameter> getFuvParams() const;

private:
  Properties properties;
  CFunction fuv;
  string cTBPath;
  string cFUVPath;
  std::vector<string> otherCSrcPaths;
  string cFUVFunctionName;
  string vhdlDUVEntityName;
  TokenCompare defaultComparator;
  IntegerCompare signedIntComparator;
  IntegerCompare unsignedIntComparator;
  FloatCompare floatComparator;
  DoubleCompare doubleComparator;
};

} // namespace hls_verify

#endif // HLS_VERIFIER_VERIFICATION_CONTEXT_H
