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

// TODO move somewhere else
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
                        smvPath.string() +
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

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::string &mlirFile) {

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(outputDir);

  // Add ND wires to the circuit
  auto ret =
      dynamatic::experimental::createReachabilityCircuit(context, mlirFile);
  if (failed(ret)) {
    llvm::errs() << "Failed to create reachability module.\n";
    return failure();
  }
  auto [miterModule, json] = ret.value();

  std::string mlirFilename =
      "elastic_miter_" + json["funcName"].getAsString()->str() + ".mlir";
  std::filesystem::path mlirPath = outputDir / mlirFilename;

  // TODO ...
  if (failed(createFiles(outputDir, mlirFilename, miterModule, json))) {
    llvm::errs() << "Failed to write miter files.\n";
    return failure();
  }

  auto failOrDstSmv = handshake2smv(mlirFile, true);
  if (failed(failOrDstSmv))
    return failure();
  auto dstSmv = failOrDstSmv.value();

  // TODO json
  auto fail = dynamatic::experimental::createMiterWrapper(
      outputDir / "main_inf.smv", outputDir / "elastic-miter-config.json",
      dstSmv.filename(), 0);
  if (failed(fail)) {
    llvm::errs() << "Failed to create infinite reachability wrapper.\n";
    return failure();
  }

  LogicalResult cmdFail =
      createCMDfile(outputDir / "reachability_inf.cmd",
                    outputDir / "main_inf.smv", "print_reachable_states -v;");
  if (failed(cmdFail))
    return failure();

  // Run nuXmv for infinite tokens
  int nuxmvRet = runNuXmv(outputDir.string() + "/reachability_inf.cmd",
                          outputDir.string() + "/inf_states.txt");
  if (nuxmvRet != 0) {
    llvm::errs()
        << "Failed to analyze reachable states with infinite tokens.\n";
    return failure();
  }

  int n = 1;
  while (true) {
    // TODO remove
    llvm::outs() << "Checking " << n << " tokens.\n";

    std::filesystem::path wrapperPath =
        outputDir / ("main_" + std::to_string(n) + ".smv");

    auto fail = dynamatic::experimental::createMiterWrapper(
        wrapperPath, outputDir / "elastic-miter-config.json", dstSmv.filename(),
        0);
    if (failed(fail)) {
      llvm::errs() << "Failed to create " << n
                   << " token reachability wrapper.\n";
      return failure();
    }

    cmdFail = createCMDfile(outputDir /
                                ("reachability_" + std::to_string(n) + ".cmd"),
                            wrapperPath, "print_reachable_states -v;");
    if (failed(cmdFail))
      return failure();

    nuxmvRet = runNuXmv(
        outputDir.string() + "/reachability_" + std::to_string(n) + ".cmd",
        outputDir.string() + "/" + std::to_string(n) + "_states.txt");
    if (nuxmvRet != 0) {
      llvm::errs() << "Failed to analyze reachable states with"
                   << std::to_string(n) + " tokens.";
      return failure();
    }

    // Check state differences
    int nrOfDifferences = dynamatic::experimental::getStates(
        outputDir.string() + "/inf_states.txt",
        outputDir.string() + "/" + std::to_string(n) + "_states.txt");

    if (nrOfDifferences != 0) {
      n++;
    } else {
      std::cout << n << std::endl;
      break;
    }
  }
  return n;
}
} // namespace dynamatic::experimental