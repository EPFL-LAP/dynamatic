//===- Help.h ---------------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HELP_H
#define HLS_VERIFIER_HELP_H

#include <string>

using namespace std;

namespace hls_verify {
string getGeneralHelpMessage();
string getCVerificationHelpMessage();
string getVHDLVerificationHelpMessage();
string getCoVerificationHelpMessage();
} // namespace hls_verify

#endif // HLS_VERIFIER_HELP_H
