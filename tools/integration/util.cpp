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

int runIntegrationTest(const std::string &name, int &outSimTime) {
  fs::path path =
      fs::path(DYNAMATIC_ROOT) / "integration-test" / name / (name + ".c");

  std::cout << "[INFO] Running " << name << std::endl;
  std::string tmpFilename = "tmp_" + name + ".dyn";
  std::ofstream scriptFile(tmpFilename);
  if (!scriptFile.is_open()) {
    std::cout << "[ERROR] Failed to create .dyn script file" << std::endl;
    return -1;
  }

  scriptFile << "set-dynamatic-path " << DYNAMATIC_ROOT << std::endl
             << "set-src " << path.string() << std::endl
             << "compile" << std::endl
             << "write-hdl" << std::endl
             << "simulate" << std::endl
             << "exit" << std::endl;

  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticLogPath = path.parent_path() / "out" / "dynamatic_out.txt";
  if (!fs::exists(dynamaticLogPath.parent_path())) {
    fs::create_directories(dynamaticLogPath.parent_path());
  }

  std::string cmd = dynamaticPath.string() + " --exit-on-failure --run ";
  cmd += tmpFilename;
  cmd += " &> ";
  cmd += dynamaticLogPath;

  int status = system(cmd.c_str());
  if (status == 0) {
    fs::path logFilePath = path.parent_path() / "out" / "sim" / "report.txt";
    outSimTime = getSimulationTime(logFilePath);
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

  std::regex pattern("Time: (\\d+) ns");
  std::smatch match;

  // Search lines in reverse order
  for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
    if (std::regex_search(*it, match, pattern)) {
      return std::stoi(match[1]) / 4;
    }
  }

  std::cout << "[WARNING] Log file does not contain simulation time!"
            << std::endl;
  return -1;
}