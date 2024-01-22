//===- CInjector.h ----------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_C_INJECTOR_H
#define HLS_VERIFIER_C_INJECTOR_H

#include <string>

#include "CAnalyser.h"
#include "VerificationContext.h"

using namespace std;

namespace hls_verify {

class CInjector {
public:
  CInjector(const VerificationContext &ctx);
  string getInjectedCFuv();

private:
  VerificationContext ctx;
  string injectedFuvSrc;
  string getFileIoCodeForInputParam(const CFunctionParameter &param);
  string getFileIoCodeForOutputParam(const CFunctionParameter &param);
  string getFileIoCodeForReturnValue(const CFunctionParameter &param,
                                     const string &actualReturnValue);
  string getFileIoCodeForInput(const CFunction &func);
  string getFileIoCodeForOutput(const CFunction &func,
                                const string &actualReturnValue);
  string getVariableDeclarations(const CFunction &func);
};
} // namespace hls_verify

#endif // HLS_VERIFIER_C_INJECTOR_H
