//===- HlsCVerification.cpp -------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HlsCVerification.h"
#include "CAnalyser.h"
#include "CInjector.h"
#include "Help.h"
#include "HlsLogging.h"
#include "Utilities.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

namespace hls_verify {
const string LOG_TAG = "CVER";

bool runCVerification(vector<string> args) {
  if (args.size() < 3) {
    logErr(LOG_TAG, "Not enough arguments.");
    cout << getCVerificationHelpMessage() << endl;
    return true;
  }

  string cTbPath = args[0];
  string cFuvPath = args[1];
  string cFuvName = args[2];
  vector<string> otherCPaths;
  for (size_t i = 3; i < args.size(); i++) {
    otherCPaths.push_back(args[i]);
  }

  VerificationContext ctx(cTbPath, cFuvPath, cFuvName, "", otherCPaths);
  executeCTestbench(ctx);
  checkCTestbenchOutputs(ctx);
  return true;
}

void compileCTestbench(const VerificationContext &ctx) {
  stringstream compCmdBuilder;
  compCmdBuilder << "gcc ";
  compCmdBuilder << ctx.getInjectedCFuvPath() + " ";
  vector<string> otherCPaths = ctx.getOtherCSrcPaths();
  for (auto &otherCPath : otherCPaths)
    compCmdBuilder << otherCPath << " ";
  compCmdBuilder << "-I" << extractParentDirectoryPath(ctx.getCTbPath());
  compCmdBuilder << " -o " << ctx.getCExecutablePath();

  string compCmd = compCmdBuilder.str();
  logInf(LOG_TAG, "Compiling C files [" + compCmd + "]");
  assert(executeCommand(compCmd) && "Compilation failed!");
}

void doPostProcessing(const VerificationContext &ctx) {
  const CFunction &fuv = ctx.getCFuv();
  if (fuv.returnVal.isOutput) {
    addHeaderAndFooter(ctx.getCOutPath(fuv.returnVal));
  }
  for (const auto &param : fuv.params) {
    if (param.isOutput) {
      addHeaderAndFooter(ctx.getCOutPath(param));
    }
    if (param.isInput) {
      addHeaderAndFooter(ctx.getInputVectorPath(param));
    }
  }
}

void checkCTestbenchOutputs(const VerificationContext &ctx) {
  const vector<CFunctionParameter> &outputParams = ctx.getFuvOutputParams();
  cout << "\n--- Comparison Results ---\n" << endl;
  for (const auto &outputParam : outputParams) {
    bool result = compareFiles(ctx.getRefOutPath(outputParam),
                               ctx.getCOutPath(outputParam),
                               ctx.getTokenComparator(outputParam));
    cout << "Comparison of [" + outputParam.parameterName + "] : "
         << (result ? "Pass" : "Fail") << endl;
  }
  cout << "\n--------------------------\n" << endl;
}

void executeCTestbench(const VerificationContext &ctx) {
  string command;

  // Generate IO injected C fuv

  CInjector injector(ctx);

  ofstream injectedCFuv(ctx.getInjectedCFuvPath());
  injectedCFuv << injector.getInjectedCFuv();
  injectedCFuv.close();

  // Compiling

  compileCTestbench(ctx);

  // Cleaning-up exisiting outputs

  command = "rm -rf " + ctx.getCOutDir();
  logInf(LOG_TAG, "Cleaning C output files [" + command + "]");
  executeCommand(command);

  command = "rm -rf " + ctx.getInputVectorDir();
  logInf(LOG_TAG, "Cleaning existing input files [" + command + "]");
  executeCommand(command);

  command = "mkdir -p " + ctx.getCOutDir();
  logInf(LOG_TAG, "Creating C output files directory [" + command + "]");
  executeCommand(command);

  command = "mkdir -p " + ctx.getInputVectorDir();
  logInf(LOG_TAG, "Creating input files directory [" + command + "]");
  executeCommand(command);

  // Running the compiled testbench

  command = ctx.getCExecutablePath();
  logInf(LOG_TAG, "Executing C test-bench [" + command + "]");
  assert(executeCommand(command) && "C testbench execution failed!");
  doPostProcessing(ctx);
  logInf(LOG_TAG, "C simulation finished. Outputs saved in directory " +
                      ctx.getCOutDir() + ".");
}
} // namespace hls_verify
