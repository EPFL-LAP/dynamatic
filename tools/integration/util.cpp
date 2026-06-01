//===- util.h - Integration testing helper functions -----------*- C++-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of integration test helper functions.
//
//===----------------------------------------------------------------------===//
#include "util.h"

#include <regex>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

int runIntegrationTest(IntegrationTestData &config) {
  fs::path cSourcePath =
      config.benchmarkPath / config.name / (config.name + ".c");

  std::string tmpFilename = "tmp_" + config.name + ".dyn";
  std::ofstream scriptFile(tmpFilename);
  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  }

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
             << "set-src " << cSourcePath.string() << std::endl
             << "set-clock-period " << config.clockPeriod << std::endl;

  // clang-format off
  scriptFile << "compile"
             << " --buffer-algorithm " << config.bufferAlgorithm
             << (config.useSharing ? " --sharing" : "")
             << (config.useRigidification ? " --rigidification" : "")
             << (config.useSpeculation ? " --speculation" : "")
             << " --milp-solver " << config.milpSolver << std::endl;
  // clang-format on

  // Assert testVHDL or testVerilog is true
  if (!config.testVHDL && !config.testVerilog) {
    std::cout << "[ERROR] Either testVHDL or testVerilog must be true"
              << std::endl;
    return -1;
  }

  if (config.verifyInvariants) {
    scriptFile << "verify-invariants" << std::endl;
  }

  // Verify Verilog works correctly
  if (config.testVerilog) {
    scriptFile << "write-hdl --hdl verilog" << std::endl
               << "simulate" << std::endl;
  }
  // Verify VHDL works correctly
  if (config.testVHDL) {
    // By default, the report containing the simulation time is re-written
    // during the second simulation (i.e., the VHDL simulation).
    scriptFile << "write-hdl --hdl vhdl" << std::endl
               << "simulate" << std::endl;
  }
  scriptFile << "exit" << std::endl;

  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticOutPath =
      cSourcePath.parent_path() / "out" / "dynamatic_out.txt";
  fs::path dynamaticErrPath =
      cSourcePath.parent_path() / "out" / "dynamatic_err.txt";
  if (!fs::exists(dynamaticOutPath.parent_path())) {
    fs::create_directories(dynamaticOutPath.parent_path());
  }

  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  cmd += " 1> ";
  cmd += dynamaticOutPath;
  cmd += " 2> ";
  cmd += dynamaticErrPath;

  int status = system(cmd.c_str());
  if (status == 0) {
    fs::path logFilePath =
        cSourcePath.parent_path() / "out" / "sim" / "report.txt";
    config.simTime = getSimulationTime(logFilePath);
  }

  return status;
}

int getSimulationTime(const fs::path &logFile) {
  std::ifstream file(logFile);
  if (!file.is_open()) {
    std::cout << "[WARNING] Failed to open " << logFile << std::endl;
    return -1;
  }

  std::vector<std::string> lines;
  std::string line;

  // Read all lines into a vector
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  std::regex pattern("Simulation done! Latency = (\\d+) cycles");
  std::smatch match;

  // Search lines in reverse order
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    if (std::regex_search(*it, match, pattern)) {
      return std::stoi(match[1]);
    }
  }

  std::cout << "[WARNING] Log file does not contain simulation time!"
            << std::endl;
  return -1;
}
