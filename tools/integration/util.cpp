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

#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool runSubprocess(const std::vector<std::string> &args,
                   const fs::path &outputPath) {
  std::ostringstream command;
  command << args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    command << " " << args[i];
  }
  command << " 1>" << outputPath.string();
  std::cout << "[INFO] Running command: " << command.str().c_str() << std::endl;
  return std::system(command.str().c_str()) == 0;
};

int runIntegrationTest(const std::string &name, int &outSimTime,
                       const std::optional<fs::path> &customPath,
                       bool useVerilog) {
  fs::path path =
      customPath.value_or(fs::path(DYNAMATIC_ROOT) / "integration-test") /
      name / (name + ".c");

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
             << "write-hdl --hdl " << (useVerilog ? "verilog" : "vhdl")
             << std::endl
             << "simulate" << std::endl
             << "exit" << std::endl;

  scriptFile.close();

  fs::path dynamaticPath = fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic";
  fs::path dynamaticOutPath = path.parent_path() / "out" / "dynamatic_out.txt";
  fs::path dynamaticErrPath = path.parent_path() / "out" / "dynamatic_err.txt";
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
    fs::path logFilePath = path.parent_path() / "out" / "sim" / "report.txt";
    outSimTime = getSimulationTime(logFilePath);
  }

  return status;
}

bool runSpecIntegrationTest(const std::string &name, int &outSimTime) {
  bool spec = true;

  const std::string DYNAMATIC_OPT_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "dynamatic-opt";

  const std::string EXPORT_DOT_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "export-dot";

  const std::string EXPORT_RTL_BIN =
      fs::path(DYNAMATIC_ROOT) / "bin" / "export-rtl";

  const std::string SIMULATE_SH = fs::path(DYNAMATIC_ROOT) / "tools" /
                                  "dynamatic" / "scripts" / "simulate.sh";

  const std::string RTL_CONFIG =
      fs::path(DYNAMATIC_ROOT) / "data" / "rtl-config-vhdl-beta.json";

  fs::path cFilePath =
      fs::path(DYNAMATIC_ROOT) / "integration-test" / name / (name + ".c");

  std::cout << "[INFO] Running " << name << " with speculation flow"
            << std::endl;

  fs::path cFileDir = cFilePath.parent_path();
  fs::path outDir = cFileDir / "out_spec";
  if (fs::exists(outDir)) {
    fs::remove_all(outDir);
  }
  fs::create_directories(outDir);

  fs::path compOutDir = outDir / "comp";
  fs::create_directories(compOutDir);

  // Copy cf.mlir
  fs::path cfFileBase = cFileDir / "cf.mlir";
  fs::path cfFile = compOutDir / "cf.mlir";
  fs::copy_file(cfFileBase, cfFile, fs::copy_options::overwrite_existing);

  fs::path cfTransformed = compOutDir / "cfTransformed.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, cfFile.string(), "--canonicalize",
                      "--cse", "--sccp", "--symbol-dce", "--control-flow-sink",
                      "--loop-invariant-code-motion", "--canonicalize"},
                     cfTransformed)) {
    std::cerr << "Failed to apply standard transformations to cf\n";
    return false;
  }

  fs::path cfDynTransformed = compOutDir / "cfDynTransformed.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, cfTransformed.string(),
                      "--arith-reduce-strength=max-adder-depth-mul=1",
                      "--push-constants", "--mark-memory-interfaces"},
                     cfDynTransformed)) {
    std::cerr << "Failed to apply Dynamatic transformations to cf\n";
    return false;
  }

  fs::path handshake = compOutDir / "handshake.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, cfDynTransformed.string(),
                      "--lower-cf-to-handshake"},
                     handshake)) {
    std::cerr << "Failed to compile cf to handshake\n";
    return false;
  }

  fs::path handshakeTransformed = compOutDir / "handshakeTransformed.mlir";
  if (!runSubprocess(
          {DYNAMATIC_OPT_BIN, handshake.string(),
           "--handshake-analyze-lsq-usage",
           "--handshake-replace-memory-interfaces",
           "--handshake-minimize-cst-width", "--handshake-optimize-bitwidths",
           "--handshake-materialize", "--handshake-infer-basic-blocks"},
          handshakeTransformed)) {
    std::cerr << "Failed to apply transformations to handshake\n";
    return false;
  }

  fs::path handshakeBuffered = compOutDir / "handshakeBuffered.mlir";
  std::string timingModel =
      (fs::path(DYNAMATIC_ROOT) / "data" / "components-flopoco.json").string();
  if (!runSubprocess(
          {DYNAMATIC_OPT_BIN, handshakeTransformed.string(),
           "--handshake-set-buffering-properties=version=fpga20",
           "\"--handshake-place-buffers=algorithm=on-merges timing-models=" +
               timingModel + "\""},
          handshakeBuffered)) {
    std::cerr << "Failed to place simple buffers\n";
    return false;
  }

  fs::path handshakeCanonicalized = compOutDir / "handshakeCanonicalized.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, handshakeBuffered.string(),
                      "--handshake-canonicalize",
                      "--handshake-hoist-ext-instances"},
                     handshakeCanonicalized)) {
    std::cerr << "Failed to canonicalize Handshake\n";
    return false;
  }

  fs::path handshakeExport;
  if (spec) {
    fs::path handshakeSpeculation = compOutDir / "handshakeSpeculation.mlir";
    fs::path specJson = cFileDir / "spec.json";
    if (!runSubprocess(
            {DYNAMATIC_OPT_BIN, handshakeCanonicalized.string(),
             "--handshake-speculation=json-path=" + specJson.string(),
             "--handshake-materialize", "--handshake-canonicalize"},
            handshakeSpeculation)) {
      std::cerr << "Failed to add speculative units\n";
      return false;
    }

    fs::path bufferJsonPath = cFileDir / "buffer.json";
    std::ifstream bufferFile(bufferJsonPath);
    json buffers;
    bufferFile >> buffers;

    std::vector<std::string> bufferArgs = {DYNAMATIC_OPT_BIN,
                                           handshakeSpeculation.string()};
    for (const auto &buffer : buffers) {
      bufferArgs.push_back(
          "\"--handshake-placebuffers-custom=pred=" +
          buffer["pred"].get<std::string>() +
          " outid=" + std::to_string(buffer["outid"].get<int>()) +
          " slots=" + std::to_string(buffer["slots"].get<int>()) +
          " type=" + buffer["type"].get<std::string>() + "\"");
    }

    handshakeExport = compOutDir / "handshake_export.mlir";
    if (!runSubprocess(bufferArgs, handshakeExport)) {
      std::cerr << "Failed to export Handshake\n";
      return false;
    }
  } else {
    handshakeExport = compOutDir / "handshake_export.mlir";
    fs::copy_file(handshakeCanonicalized, handshakeExport,
                  fs::copy_options::overwrite_existing);
  }

  fs::path dotFile = compOutDir / (name + ".dot");
  if (!runSubprocess({EXPORT_DOT_BIN, handshakeExport.string(),
                      "--edge-style=spline", "--label-type=uname"},
                     dotFile)) {
    std::cerr << "Failed to export dot file\n";
    return false;
  }

  fs::path pngFile = compOutDir / (name + ".png");
  if (!runSubprocess({"dot", "-Tpng", dotFile.string()}, pngFile)) {
    std::cerr << "Failed to create PNG file\n";
    return false;
  }

  fs::path hw = compOutDir / "hw.mlir";
  if (!runSubprocess({DYNAMATIC_OPT_BIN, handshakeExport.string(),
                      "--lower-handshake-to-hw"},
                     hw)) {
    std::cerr << "Failed to lower handshake to hw\n";
    return false;
  }

  fs::path hdlDir = outDir / "hdl";
  if (std::system((EXPORT_RTL_BIN + " " + hw.string() + " " + hdlDir.string() +
                   " " + RTL_CONFIG + " --dynamatic-path " + DYNAMATIC_ROOT +
                   " --hdl vhdl")
                      .c_str()) != 0) {
    std::cerr << "Failed to export hdl\n";
    return false;
  }

  std::cout << "Simulator launching\n";
  if (std::system((SIMULATE_SH + " " + DYNAMATIC_ROOT + " " +
                   cFileDir.string() + " " + outDir.string() + " " + name +
                   " \"\" " + "false")
                      .c_str()) != 0) {
    std::cerr << "Failed to simulate\n";
    return false;
  }

  std::cout << "Simulation succeeded\n";

  fs::path logFilePath = cFileDir / "out_spec" / "sim" / "report.txt";
  outSimTime = getSimulationTime(logFilePath);

  return true;
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