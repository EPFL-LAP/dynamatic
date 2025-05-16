//===- HlsLogging.cpp -------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HlsLogging.h"
#include "llvm/Support/raw_ostream.h"

void logInf(const string &tag, const string &msg) {
  llvm::errs() << "[INFO  " << tag << "] " << msg << "\n";
}

void logErr(const string &tag, const string &msg) {
  llvm::errs() << "[ERROR " << tag << "] " << msg << "\n";
}

void logWrn(const string &tag, const string &msg) {
  llvm::errs() << "[WARN  " << tag << "] " << msg << "\n";
}