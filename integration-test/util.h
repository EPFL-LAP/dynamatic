//===- util.h - Integration testing helper functions -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utility functions to be used for 
// integration testing.
//
//===----------------------------------------------------------------------===//
#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

int runIntegrationTest(const fs::path &name, int &outSimTime);
int getSimulationTime(const fs::path &logFile);
std::vector<fs::path> findTests(const fs::path &start);

#endif // UTIL_H