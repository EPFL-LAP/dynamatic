//===- hls-verifier.cpp - C/VHDL co-simulation ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legacy hls_verifier tool, somewhat cleaned up.
//
//===----------------------------------------------------------------------===//

#include "src/Help.h"
#include "src/HlsCoVerification.h"
#include "src/HlsVhdlVerification.h"
#include <iostream>
#include <string>
#include <vector>

using namespace hls_verify;

int main(int argc, char **argv) {
  if (argc > 1) {
    std::string firstArg(argv[1]);
    vector<string> remainingArgs;
    for (int i = 2; i < argc; i++)
      remainingArgs.emplace_back(argv[i]);
    if (firstArg == VVER_CMD)
      return runVhdlVerification(remainingArgs) ? 0 : -1;
    if (firstArg == COVER_CMD)
      return runCoverification(remainingArgs) ? 0 : -1;
    std::cout << std::endl << "Invalid arguments!" << std::endl;
  } else {
    std::cout << std::endl << "No arguments!" << std::endl;
  }
  std::cout << getGeneralHelpMessage() << std::endl;
  return -1;
}
