#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "../experimental/tools/elastic-miter-generator/GetStates.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands) {

  std::string command = "set verbose_level 0;"
                        "set pp_list cpp;"
                        "set counter_examples 0;"
                        "set dynamic_reorder 1;"
                        "set on_failure_script_quits;"
                        "set reorder_method sift;"
                        "set enable_sexp2bdd_caching 0;"
                        "set bdd_static_order_heuristics basic;"
                        "set cone_of_influence;"
                        "set use_coi_size_sorting 1;"
                        "read_model -i" +
                        smvPath.string() + ";\n" +
                        "flatten_hierarchy;"
                        "encode_variables;"
                        "build_flat_model;"
                        "build_model - f;" +
                        additionalCommands + "\n" +
                        "time;"
                        "quit;";
  std::ofstream mainFile(cmdPath);
  mainFile << command;
  mainFile.close();
  return success();
}

// TODO proper output handling...
int runNuXmv(const std::string &cmd, const std::string &stdoutFile) {
  std::string command = "nuXmv -source " + cmd + " > " + stdoutFile;
  return system(command.c_str());
}

FailureOr<std::filesystem::path>
handshake2smv(const std::filesystem::path &mlirPath, bool png = false) {

  std::filesystem::path dotFile = mlirPath.parent_path() / "miter.dot";
  std::string cmd = "bin/export-dot " + mlirPath.string() +
                    " --edge-style=spline > " + dotFile.string();
  int ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to dot\n";
    return failure();
  }

  if (png) {
    std::filesystem::path pngFile = mlirPath.parent_path() / "miter.png";
    cmd = "dot -Tpng " + dotFile.string() + " -o " + pngFile.string();
    ret = system(cmd.c_str());
    if (ret != 0) {
      llvm::errs() << "Failed to convert to PNG\n";
      return failure();
    }
  }

  // The current implementation of dot2smv uses the hardcoded name "model.smv"
  // in the dotfile's directory.
  std::filesystem::path smvFile = dotFile.parent_path() / "model.smv";
  cmd = "python3 ../dot2smv/dot2smv " + dotFile.string();
  ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to SMV\n";
    return failure();
  }

  return smvFile;
}
} // namespace dynamatic::experimental