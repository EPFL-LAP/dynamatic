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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int runIntegrationTest(const std::string &name, int &outSimTime,
                       const std::optional<fs::path> &customPath = std::nullopt,
                       bool useVerilog = false);
bool runSpecIntegrationTest(const std::string &name, int &outSimTime);
int getSimulationTime(const fs::path &logFile);

#endif // UTIL_H
