//===- HlsCoVerification.h --------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_COVERIFICATION_H
#define HLS_VERIFIER_HLS_COVERIFICATION_H

#include "VerificationContext.h"
#include <string>
#include <vector>

using namespace std;

namespace hls_verify {
const string COVER_CMD = "cover";

bool runCoverification(vector<string> args);

bool compareCAndVhdlOutputs(const VerificationContext &ctx);
} // namespace hls_verify

#endif // HLS_VERIFIER_HLS_COVERIFICATION_H
