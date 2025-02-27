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

int runNuXmv(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile) {
  std::string command =
      "nuXmv -source " + cmdPath.string() + " > " + stdoutFile.string();
  return system(command.c_str());
}

int runNuSMV(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile) {
  std::string command =
      "NuSMV -source " + cmdPath.string() + " > " + stdoutFile.string();
  return system(command.c_str());
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
              const std::filesystem::path &outputDir, bool png) {

  std::filesystem::path dotFile = outputDir / "model.dot";

  // Convert the handshake to dot
  std::string cmd = "bin/export-dot " + mlirPath.string() +
                    " --edge-style=spline > " + dotFile.string();
  int ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to dot\n";
    return failure();
  }

  // Optionally, generate a visual representation of the circuit from the
  // generated dotfile
  if (png) {
    std::filesystem::path pngFile = outputDir / "model.png";
    cmd = "dot -Tpng " + dotFile.string() + " -o " + pngFile.string() +
          " > /dev/null";
    ret = system(cmd.c_str());
    if (ret != 0) {
      llvm::errs() << "Failed to convert to PNG\n";
      return failure();
    }
  }

  // Convert the dotfile to SMV
  // The current implementation of dot2smv uses the hardcoded name "model.smv"
  // in the dotfile's directory.
  std::filesystem::path smvFile = dotFile.parent_path() / "model.smv";
  cmd = "python3 ../dot2smv/dot2smv " + dotFile.string() + " > /dev/null";
  ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to SMV\n";
    return failure();
  }
  // Currently dot2smv only supports "model" as the model's name
  std::string moduleName = "model";

  return std::make_pair(smvFile, moduleName);
}
} // namespace dynamatic::experimental