//===- HlsLogging.h ---------------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HLS_VERIFIER_HLS_LOGGING_H
#define HLS_VERIFIER_HLS_LOGGING_H

#include <iostream>
#include <string>

using namespace std;

#define LOG_DEBUG

#ifdef LOG_DEBUG
#define log_dbg(x)                                                             \
  std::cout << "[DEBUG " << __FILE__ << ": " << __LINE__ << "] " << (x) << endl
#else
#define log_debug(x) ;
#endif

void logInf(const string &tag, const string &msg);
void logErr(const string &tag, const string &msg);
void logWrn(const string &tag, const string &msg);

#endif // HLS_VERIFIER_HLS_LOGGING_H
