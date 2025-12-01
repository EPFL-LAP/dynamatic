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

struct IntegrationTestData {
  // Configurations
  std::string name;
  fs::path benchmarkPath;
  bool useVerilog = false;
  // Use resource sharing to reduce the functional unit usage.
  bool useSharing = false;
  // Use model checking to remove redundant logic.
  bool useRigidification = false;
  std::string milpSolver = "gurobi";
  std::string bufferAlgorithm = "fpga20";

  // Results
  int simTime;
};

int runIntegrationTest(IntegrationTestData &config);
bool runSpecIntegrationTest(const std::string &name, int &outSimTime);
int getSimulationTime(const fs::path &logFile);

#endif // UTIL_H
