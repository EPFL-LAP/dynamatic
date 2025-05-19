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

using namespace std;
using namespace dynamatic;

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

struct VerificationContext {
  VerificationContext(const string &cFuvFunctionName,
                      const string &vhdlDuvEntityName,
                      handshake::FuncOp *funcOp)
      : funcOp(funcOp), properties(), cFUVFunctionName(cFuvFunctionName),
        vhdlDUVEntityName(vhdlDuvEntityName) {}

  string getCFuvFunctionName() const;
  string getVhdlDuvEntityName() const;

  string getInjectedCFuvPath() const;
  string getCExecutablePath() const;
  string getVhdlTestbenchPath() const;

  string getBaseDir() const;
  string getHlsVerifyDir() const;
  string getHdlSrcDir() const;

  string getCOutDir() const;
  string getRefOutDir() const;
  string getHdlOutDir() const;
  string getInputVectorDir() const;

  string getModelsimDoFileName() const;

  handshake::FuncOp *funcOp;

  Properties properties;
  string cFUVFunctionName;
  string vhdlDUVEntityName;
};

#endif // HLS_VERIFIER_VERIFICATION_CONTEXT_H
