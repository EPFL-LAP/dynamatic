//===- CAnalyser.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_C_ANALYSER_H
#define HLS_VERIFIER_C_ANALYSER_H

#include <string>
#include <vector>

using namespace std;

namespace hls_verify {

struct CFunctionParameter {
  string immediateType;
  string actualType;
  string parameterName;
  bool isPointer;
  bool isInput;
  bool isOutput;
  bool isReturn;
  bool isFloatType;
  bool isIntType;
  int arrayLength;
  vector<int> dims;
  int dtWidth;
};

struct CFunction {
  CFunctionParameter returnVal;
  vector<CFunctionParameter> params;
  string functionName;
};

class CAnalyser {
public:
  static bool parseCFunction(const string &cSrc, const string &fuvName,
                             CFunction &func);
  static bool parseAsArrayType(const string &str, CFunctionParameter &param);
  static bool parseAsPointerType(const string &str, CFunctionParameter &param);
  static bool parseAsSimpleType(const string &str, CFunctionParameter &param);
  static bool paramFromString(const string &str, CFunctionParameter &param);
  static string getActualType(const string &cSrc, string type);
  static int getBitWidth(const string &type);
  static bool isFloatType(const string &type);
  static string getPreprocOutput(const string &cFilePath,
                                 const string &cIncludeDir = "");

private:
};
} // namespace hls_verify

#endif // HLS_VERIFIER_C_ANALYSER_H
