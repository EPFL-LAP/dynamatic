//===- GetSequenceLength.h -.-------------------------------- ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An infinite number of input tokens can be emulated by a lower number of
// tokens, if all internal states of the circuit can be reached. This file
// provides a way to determine how many input tokens are needed.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H

#include "dynamatic/Support/LLVM.h"
#include <filesystem>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// Compute the length of the input sequence which is required to emulate an
// infinite sequence. This is done by enumerating the reachable states and then
// iteratively increasing the number of input tokens until the set of reachble
// states is equivalent.
FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::filesystem::path &mlirPath);
} // namespace dynamatic::experimental

#endif