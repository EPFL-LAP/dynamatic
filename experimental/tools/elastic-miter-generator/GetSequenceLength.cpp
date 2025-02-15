#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/GetStates.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// TODO ...
const std::string OUT_DIR = "experimental/tools/elastic-miter-generator/out";
const std::string COMP_DIR = OUT_DIR + "/comp";
const std::string DOT = COMP_DIR + "/miter.dot";
const std::string REWRITES =
    "experimental/test/tools/elastic-miter-generator/rewrites";

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
    cmd = "dot -Tpng " + DOT + " -o " + pngFile.string();
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
                                    const std::string &mlirFile) {

  // TODO type
  auto failOrDstSmv = handshake2smv(mlirFile, true);
  if (failed(failOrDstSmv))
    return failure();

  OwningOpRef<ModuleOp> modRef = parseSourceFile<ModuleOp>(mlirFile, &context);

  ModuleOp mod = modRef.release();

  // TODO use correct name
  // # TODO we should probably just pass the filename
  // Create state wrapper for infinite tokens
  auto failOrInfWrapper =
      dynamatic::experimental::createReachableStateWrapper(mod, 0, true);
  if (failed(failOrInfWrapper))
    return failure();

  std::string infWrapper = failOrInfWrapper.value();

  std::ofstream infFile(OUT_DIR + "/comp/main_inf.smv");
  infFile << infWrapper;
  infFile.close();

  // Run nuXmv for infinite tokens
  int ret = runNuXmv(COMP_DIR + "/prove_inf.cmd", OUT_DIR + "/inf_states.txt");
  if (ret != 0) {
    llvm::errs()
        << "Failed to analyze reachable states with infinite tokens.\n";
    return failure();
  }

  int n = 1;
  while (true) {
    llvm::outs() << "Checking " << n << " tokens.\n";

    auto failOrFinWrapper =
        dynamatic::experimental::createReachableStateWrapper(mod, n, false);
    if (failed(failOrFinWrapper))
      return 1;

    std::string finWrapper = failOrFinWrapper.value();

    std::ofstream finFile(OUT_DIR + "/comp/main_" + std::to_string(n) + ".smv");
    finFile << finWrapper;
    finFile.close();

    // TODO automatically create cmd file
    ret = runNuXmv(COMP_DIR + "/prove_" + std::to_string(n) + ".cmd",
                   OUT_DIR + "/" + std::to_string(n) + "_states.txt");
    if (ret != 0) {
      llvm::errs() << "Failed to analyze reachable states with"
                   << std::to_string(n) + " tokens.";
      return failure();
    }

    // Check state differences
    int nrOfDifferences = dynamatic::experimental::getStates(
        OUT_DIR + "/inf_states.txt",
        OUT_DIR + "/" + std::to_string(n) + "_states.txt");

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