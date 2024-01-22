//===- HlsVhdlVerification.h ----------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_VHDL_VERIFICATION_H
#define HLS_VERIFIER_HLS_VHDL_VERIFICATION_H

#include <string>
#include <vector>

#include "VerificationContext.h"

using namespace std;

namespace hls_verify {
const string VVER_CMD = "vver";

/**
 * Run VHDL verification.
 * @param args remaining arguments to vver
 * @return true if the function executes normally, false otherwise.
 */
bool runVhdlVerification(vector<string> args);

/**
 * Generate the VHDL testbench of the given verification context.
 * @param ctx
 */
void generateVhdlTestbench(const VerificationContext &ctx);

/**
 * Generate ModelSim scripts to run the generated VHDL testbench.
 * @param ctx verification context
 */
void generateModelsimScripts(const VerificationContext &ctx);

/**
 * Compares all generated VHDL testbench outputs against references.
 * @param ctx verification context
 */
void checkVhdlTestbenchOutputs(const VerificationContext &ctx);

/**
 * Execute the VHDL testbench of the given verification context.
 * @param ctx verification context
 */
void executeVhdlTestbench(const VerificationContext &ctx,
                          const std::string &resourceDir);
} // namespace hls_verify

#endif // HLS_VERIFIER_HLS_VHDL_VERIFICATION_H
