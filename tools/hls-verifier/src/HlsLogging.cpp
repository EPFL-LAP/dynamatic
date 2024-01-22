//===- HlsLogging.cpp -------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HlsLogging.h"

void logInf(const string &tag, const string &msg) {
  cout << "[INFO  " << tag << "] " << msg << endl;
}

void logErr(const string &tag, const string &msg) {
  cerr << "[ERROR " << tag << "] " << msg << endl;
}

void logWrn(const string &tag, const string &msg) {
  cout << "[WARN  " << tag << "] " << msg << endl;
}