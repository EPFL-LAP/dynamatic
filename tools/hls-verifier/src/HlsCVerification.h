//===- HlsCVerification.h ---------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_C_VERIFICATION_H
#define HLS_VERIFIER_C_VERIFICATION_H

#include "VerificationContext.h"
#include <string>
#include <vector>

using namespace std;

namespace hls_verify {
const string CVER_CMD = "cver";

/**
 * Run C verification.
 * @param args remaining arguments to cver
 * @return true if the function executes normally, false otherwise.
 */
bool runCVerification(vector<string> args);

/**
 * Execute the C testbench of the given verification context.
 * @param ctx verification context
 */
void executeCTestbench(const VerificationContext &ctx);

/**
 * Compile the C testbench of the given verification context
 * @param ctx verification context
 */
void compileCTestbench(const VerificationContext &ctx);

/**
 * For every data file (input/output), add opening and closing tags
 * @param ctx verification context
 */
void doPostProcessing(const VerificationContext &ctx);

/**
 * Compares all generated C testbench outputs against references.
 * @param ctx verification context
 */
void checkCTestbenchOutputs(const VerificationContext &ctx);
} // namespace hls_verify

#endif // HLS_VERIFIER_C_VERIFICATION_H
