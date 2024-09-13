//===- Utils.cpp - Common dependency-less entities --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements common entites and utilities.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Utils/Utils.h"

using namespace dynamatic;

std::array<SignalType, 3> dynamatic::getSignalTypes() {
  return {SignalType::DATA, SignalType::VALID, SignalType::READY};
}
