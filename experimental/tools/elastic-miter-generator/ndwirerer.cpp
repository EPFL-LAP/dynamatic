#include "../experimental/tools/elastic-miter-generator/CreateStateWrapper.h"
#include "../experimental/tools/elastic-miter-generator/GetStates.h"
#include "dynamatic/InitAllDialects.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/InitLLVM.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;
using namespace llvm;

// TODO ...
const std::string OUT_DIR = "experimental/tools/elastic-miter-generator/out";
const std::string COMP_DIR = OUT_DIR + "/comp";
const std::string DOT = COMP_DIR + "/miter.dot";
const std::string REWRITES =
    "experimental/test/tools/elastic-miter-generator/rewrites";
const std::string MOD = "z";

void exitOnFail(int ret, const std::string &message) {
  if (ret != 0) {
    std::cerr << message << std::endl;
    exit(1);
  }
}

int runNuXmv(const std::string &cmd, const std::string &stdoutFile) {
  std::string command = "nuXmv -source " + cmd + " > " + stdoutFile;
  return system(command.c_str());
}

FailureOr<std::string> handshake2smv(const std::string &mlir,
                                     bool png = false) {
  std::string cmd = "bin/export-dot " + mlir + " --edge-style=spline > " + DOT;
  int ret = system(cmd.c_str());
  exitOnFail(ret, "Failed to convert to dot");

  if (png) {
    cmd = "dot -Tpng " + DOT + " -o " + COMP_DIR + "/visual.png";
    ret = system(cmd.c_str());
    exitOnFail(ret, "Failed to convert to PNG");
  }

  cmd = "python3 ../dot2smv/dot2smv " + DOT;
  ret = system(cmd.c_str());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to SMV\n";
    return failure();
  }

  std::filesystem::rename(
      std::filesystem::path(COMP_DIR + "/model.smv"),
      std::filesystem::path(COMP_DIR + "/" + MOD + "_lhs.smv"));
  return COMP_DIR + "/" + MOD + "_lhs.smv";
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  auto failOrDstSmv = handshake2smv(REWRITES + "/" + MOD + "_lhs.mlir", true);
  if (failed(failOrDstSmv))
    return 1;

  std::string dstSmv = failOrDstSmv.value();

  OwningOpRef<ModuleOp> modRef =
      parseSourceFile<ModuleOp>(REWRITES + "/" + MOD + "_lhs.mlir", &context);

  ModuleOp mod = modRef.release();

  // TODO use correct name
  // # TODO we should probably just pass the filename
  // Create state wrapper for infinite tokens
  auto failOrInfWrapper =
      dynamatic::experimental::createStateWrapper("z_lhs.smv", mod, 0, true);
  if (failed(failOrInfWrapper))
    return 1;

  std::string infWrapper = failOrInfWrapper.value();

  std::ofstream infFile(OUT_DIR + "/comp/main_inf.smv");
  infFile << infWrapper;
  infFile.close();

  // Run nuXmv for infinite tokens
  int ret = runNuXmv(COMP_DIR + "/prove_inf.cmd", OUT_DIR + "/inf_states.txt");
  exitOnFail(ret, "Failed to analyze reachable states with infinite tokens.");

  int n = 1;
  while (true) {
    llvm::outs() << "Checking " << n << " tokens.\n";

    // TODO use correct name
    auto failOrFinWrapper =
        dynamatic::experimental::createStateWrapper("z_lhs.smv", mod, n, false);
    if (failed(failOrFinWrapper))
      return 1;

    std::string finWrapper = failOrFinWrapper.value();

    std::ofstream finFile(OUT_DIR + "/comp/main_" + std::to_string(n) + ".smv");
    finFile << finWrapper;
    finFile.close();

    // TODO automatically create cmd file
    ret = runNuXmv(COMP_DIR + "/prove_" + std::to_string(n) + ".cmd",
                   OUT_DIR + "/" + std::to_string(n) + "_states.txt");
    exitOnFail(ret, "Failed to analyze reachable states with " +
                        std::to_string(n) + " tokens.");

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
}
