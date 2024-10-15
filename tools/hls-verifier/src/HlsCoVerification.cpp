//===- HlsCoVerifcation.cpp -------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HlsCoVerification.h"
#include "Help.h"
#include "HlsLogging.h"
#include "HlsVhdlVerification.h"
#include "Utilities.h"
#include <iostream>

namespace hls_verify {
const string LOG_TAG = "COVER";

bool runCoverification(vector<string> args) {
  if (args.size() < 5) {
    logErr(LOG_TAG, "Not enough arguments.");
    cout << getCoVerificationHelpMessage() << endl;
    return true;
  }

  bool useAddrWidth32 = false;

  vector<string> temp;
  for (auto &arg : args) {
    if (!arg.empty() && arg[0] == '-') {
      if (arg == "-aw32")
        useAddrWidth32 = true;
    } else {
      temp.push_back(arg);
    }
  }
  args = temp;

  string resourceDir = args[0];
  string cTbPath = args[1];
  string cDuvPath = args[2];
  string cFuvFunctionName = args[3];
  string vhdlDuvEntityName = args[4];

  vector<string> otherCPaths;
  for (size_t i = 6; i < args.size(); i++)
    otherCPaths.push_back(args[i]);

  VerificationContext ctx(cTbPath, cDuvPath, cFuvFunctionName,
                          vhdlDuvEntityName, otherCPaths);
  ctx.useAddrWidth32 = useAddrWidth32;
  executeVhdlTestbench(ctx, resourceDir);
  return compareCAndVhdlOutputs(ctx);
}

bool compareCAndVhdlOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  cout << "\n--- Comparison Results ---\n" << endl;
  for (const auto &outputParam : outputParams) {
    bool result = compareFiles(ctx.getCOutPath(outputParam),
                               ctx.getVhdlOutPath(outputParam),
                               ctx.getTokenComparator(outputParam));
    cout << "Comparison of [" + outputParam.parameterName + "] : "
         << (result ? "Pass" : "Fail") << endl;
    return result;
  }
  cout << "\n--------------------------\n" << endl;
  return false;
}
} // namespace hls_verify
