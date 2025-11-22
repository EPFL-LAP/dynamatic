//===- SmvUtils.cpp -------------------------------------------- *- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "SmvUtils.h"

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands,
                            bool showCounterExamples) {

  std::string command = "set verbose_level 0;\n"
                        "set pp_list cpp;\n"
                        "set counter_examples " +
                        std::to_string(showCounterExamples) +
                        ";\n"
                        "set dynamic_reorder 1;\n"
                        "set on_failure_script_quits;\n"
                        "set reorder_method sift;\n"
                        "set enable_sexp2bdd_caching 0;\n"
                        "set bdd_static_order_heuristics basic;\n"
                        "set cone_of_influence;\n"
                        "set use_coi_size_sorting 1;\n"
                        "read_model -i " +
                        smvPath.string() + ";\n" +
                        "flatten_hierarchy;\n"
                        "encode_variables;\n"
                        "build_flat_model;\n"
                        "build_model -f;\n" +
                        additionalCommands + ";\n";
  if (showCounterExamples) {
    command += "show_traces -a -p 4 -o " +
               (cmdPath.parent_path() / "trace.xml").string() + ";\n";
  }
  command += "time;\n"
             "quit;\n";
  std::ofstream mainFile(cmdPath);
  mainFile << command;
  mainFile.close();
  return success();
}

// Runs a shell command and redirects the stdout to the provided file.
static int executeWithRedirect(const std::string &command,
                               const std::filesystem::path &stdoutFile) {

  std::istringstream cmdStream(command);
  std::vector<StringRef> stringRefVec;

  // Parse the command token by token
  std::string word;
  std::vector<std::string> argsVector;
  while (cmdStream >> word) {
    argsVector.push_back(word);
  }

  // Convert the argsVector from std::vector<std::string> to ArrayRef<StringRef>
  stringRefVec.reserve(argsVector.size()); // Preallocate the vector
  for (const auto &arg : argsVector) {
    stringRefVec.emplace_back(arg);
  }
  ArrayRef<StringRef> argsArrayRef(stringRefVec);

  // Redirect stdout, keep default of stdin and stderr
  std::string stdoutFileString = stdoutFile.string();
  ArrayRef<std::optional<StringRef>> redirects = {
      std::nullopt, stdoutFileString, std::nullopt};

  std::string errMsg;
  bool executionFailed;
  // Find the program in the PATH
  auto programName = llvm::sys::findProgramByName(argsArrayRef[0]);
  std::error_code ec = programName.getError();
  if (ec) {
    llvm::errs() << "Could not find program with name: " << argsArrayRef[0]
                 << "\n";
    return -1;
  }

  int exitCode =
      sys::ExecuteAndWait(*programName, argsArrayRef, std::nullopt, redirects,
                          0, 0, &errMsg, &executionFailed);

  return exitCode;
}

int runNuXmv(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile) {
  std::string command = "nuXmv -source " + cmdPath.string();
  return executeWithRedirect(command, stdoutFile);
}

// For the equivalence checking to work a modified NuSMV is required.
// Run build.sh with the --enable-leq-binaries flag
int runNuSMV(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile) {
  std::string command = "ext/NuSMV -source " + cmdPath.string();
  int exitCode = executeWithRedirect(command, stdoutFile);

  // Check if bits 15-8 are set to 0x7F. In this case the command was not found.
  if (exitCode == -1) {
    llvm::errs() << "NuSMV not found. Run build.sh with the "
                    "--enable-leq-binaries flag\n";
  }
  return exitCode;
}

int runSmvCmd(const std::filesystem::path &cmdPath,
              const std::filesystem::path &stdoutFile) {
#ifdef USE_NUXMV
  return runNuXmv(cmdPath, stdoutFile);
#else
  return runNuSMV(cmdPath, stdoutFile);
#endif
}

FailureOr<std::pair<std::filesystem::path, std::string>>
handshake2smv(const std::filesystem::path &mlirPath,
              const std::filesystem::path &outputDir, bool generateCircuitPng) {

  std::filesystem::path dotFile = outputDir / "model.dot";

  // Convert the handshake to dot
  std::string cmd =
      "bin/export-dot " + mlirPath.string() + " --edge-style=spline";
  int ret = executeWithRedirect(cmd, dotFile);
  if (ret != 0) {
    llvm::errs() << "Failed to convert to dot\n";
    return failure();
  }

  // Optionally, generate a visual representation of the circuit from the
  // generated dotfile
  if (generateCircuitPng) {
    std::filesystem::path pngFile = outputDir / "model.png";
    cmd = "dot -Tpng " + dotFile.string() + " -o " + pngFile.string();
    ret = executeWithRedirect(cmd, "/dev/null");
    if (ret != 0) {
      llvm::errs() << "Failed to convert to PNG\n";
      return failure();
    }
  }

  // Convert the dotfile to SMV
  // The current implementation of dot2smv uses the hardcoded name "model.smv"
  // in the dotfile's directory.
  // For the conversion to work dot2smv is required.
  // Run build.sh with the --enable-leq-binaries flag
  std::filesystem::path smvFile = dotFile.parent_path() / "model.smv";
  cmd = "python3 ext/dot2smv/dot2smv " + dotFile.string();
  ret = executeWithRedirect(cmd, "/dev/null");
  // Check if bits 15-8 are set to 0x7F. In this case the command was not
  // found.
  if (ret == -1) {
    llvm::errs() << "dot2smv not found. Run build.sh with the "
                    "--enable-leq-binaries flag\n";
  }
  if (ret != 0) {
    llvm::errs() << "Failed to convert to SMV\n";
    return failure();
  }
  // Currently dot2smv only supports "model" as the model's name
  std::string moduleName = "model";

  return std::make_pair(smvFile, moduleName);
}
} // namespace dynamatic::experimental