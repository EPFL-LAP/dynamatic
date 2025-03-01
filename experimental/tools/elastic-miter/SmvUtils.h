//===- SmvUtils.h - Utility functions to interact with nuXmv ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides some function which simplify the interaction with nuXmv
// and NuSMV.
// 1. It provides an easy way to generate cmd files that can be sourced
// by nuXmv/NuSMV.
// 2. It provides an interface to call nuXmv or NUSMV and sourcing a cmd file.
// 3. It provides an interface to convert MLIR files in the handshake dialect to
// the SMV format.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SMV_UTILS_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SMV_UTILS_H

#include "mlir/Support/LogicalResult.h"
#include <filesystem>
#include <string>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// Create a cmd file, the file loads the SMV model and then executes the
// additional commands. If showCounterExamples is set, additionally it will
// print an xml trace to trace.xml
LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands,
                            bool showCounterExamples = false);

// Run a cmd file using nuXmv. nuXmv needs to be in the PATH.
// Returns the exit status of the binary.
int runNuXmv(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile);

// Run a cmd file using NuSMV. NuSMV needs to be in the PATH.
// Returns the exit status of the binary.
int runNuSMV(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile);

// Run a cmd using NuSMV or nuXmv. When the define USE_NUXMV is set, nuXmv is
// used, otherwise NuSMV. The used binary needs to be in the PATH.
// Returns the exit status of the binary.
int runSmvCmd(const std::filesystem::path &cmdPath,
              const std::filesystem::path &stdoutFile);

// Implements the handshake to SMV conversion flow. All needed files are placed
// in outputDir. Either failed() or a pair with the path to the top level SMV
// file and the name of the generated SMV module, is returned.
FailureOr<std::pair<std::filesystem::path, std::string>>
handshake2smv(const std::filesystem::path &mlirPath,
              const std::filesystem::path &outputDir, bool png);

} // namespace dynamatic::experimental
#endif