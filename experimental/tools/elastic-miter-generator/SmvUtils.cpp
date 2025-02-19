#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands) {

  std::string command = "set verbose_level 0;\n"
                        "set pp_list cpp;\n"
                        "set counter_examples 0;\n"
                        "set dynamic_reorder 1;\n"
                        "set on_failure_script_quits;\n"
                        "set reorder_method sift;\n"
                        "set enable_sexp2bdd_caching 0;\n"
                        "set bdd_static_order_heuristics basic;\n"
                        "set cone_of_influence;\n"
                        "set use_coi_size_sorting 1;\n"
                        "read_model -i" +
                        smvPath.string() + ";\n" +
                        "flatten_hierarchy;\n"
                        "encode_variables;\n"
                        "build_flat_model;\n"
                        "build_model -f;\n" +
                        additionalCommands + ";\n" +
                        "time;\n"
                        "quit;\n";
  std::ofstream mainFile(cmdPath);
  mainFile << command;
  mainFile.close();
  return success();
}

// TODO proper output handling...
int runNuXmv(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile) {
  std::string command =
      "nuXmv -source " + cmdPath.string() + " > " + stdoutFile.string();
  return system(command.c_str());
}

FailureOr<std::filesystem::path>
handshake2smv(const std::filesystem::path &mlirPath,
              const std::filesystem::path &outputDir, bool png = false) {

  std::filesystem::path dotFile = outputDir / "model.dot";

  std::string cmd = "bin/export-dot " + mlirPath.string() +
                    " --edge-style=spline > " + dotFile.string();
  int ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to dot\n";
    return failure();
  }

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

  // The current implementation of dot2smv uses the hardcoded name "model.smv"
  // in the dotfile's directory.
  std::filesystem::path smvFile = dotFile.parent_path() / "model.smv";
  cmd = "python3 ../dot2smv/dot2smv " + dotFile.string() + " > /dev/null";
  ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to SMV\n";
    return failure();
  }

  return smvFile;
}
} // namespace dynamatic::experimental