#include <any>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "CreateWrappers.h"
#include "ElasticMiterFabricGeneration.h"
#include "SmvUtils.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include <llvm/ADT/StringSet.h>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

static std::string stripString(const std::string &string) {
  std::string newString = string;
  size_t start = newString.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
    newString.clear(); // The string contains only whitespace
  } else {
    // Trim leading and trailing spaces
    size_t end = newString.find_last_not_of(" \t\n\r\f\v");
    newString = newString.substr(start, end - start + 1);
  }
  return newString;
}

// TODO handle too many states to print
static std::vector<std::string> getStateSet(const std::string &filename,
                                            const std::string &modelName) {
  std::ifstream file(filename);
  std::vector<std::string> states;
  std::string line, currentState;
  bool recording = false;

  while (std::getline(file, line)) {
    line = stripString(line);

    if (line.find("warning: the states are more than") != std::string::npos) {
      // TODO
    }
    if (line.find("-------") != std::string::npos) {
      if (!currentState.empty()) {
        states.push_back(currentState);
      }
      currentState.clear();
      recording = true;
      continue;
    }

    if (!recording)
      continue;
    // Skip if it doesn't start with "miter."
    if (line.rfind(modelName + ".", 0) != 0) {
      continue;
    }
    currentState += line + "\n";
  }

  if (!currentState.empty()) {
    states.push_back(currentState);
  }

  return states;
}

int compareReachableStates(const std::string &infFile,
                           const std::string &finFile,
                           const std::string &modelName) {
  std::vector<std::string> finVector = getStateSet(finFile, modelName);
  std::vector<std::string> infVector = getStateSet(infFile, modelName);

  // TODO use StringSet directly
  llvm::StringSet<> setFin;
  llvm::StringSet<> setInf;
  for (const auto &entry : finVector) {
    setFin.insert(entry);
  }
  for (const auto &entry : infVector) {
    setInf.insert(entry);
  }

  // llvm::outs() << setInf.size() << "\n";
  // llvm::outs() << setFin.size() << "\n";

  // for (auto &a : setInf) {
  //   llvm::outs() << a.getKey() << "\n";
  // }
  // llvm::outs() << "-----------\n";
  // for (auto &a : setFin) {
  //   llvm::outs() << a.getKey() << "\n";
  // }

  int diffCount = 0;
  for (const auto &entry : infVector) {
    if (std::find(finVector.begin(), finVector.end(), entry) ==
        finVector.end()) {
      diffCount++;
    }
  }
  return diffCount;
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
  auto [miterModule, config] = ret.value();

  std::string ndWireMlirFilename =
      "elastic_miter_" + std::any_cast<std::string>(config.funcName) + ".mlir";
  std::filesystem::path ndWireMlirPath = outputDir / ndWireMlirFilename;

  if (failed(createMlirFile(outputDir, ndWireMlirFilename, miterModule))) {
    llvm::errs() << "Failed to write miter files.\n";
    return failure();
  }

  auto failOrDstSmv =
      dynamatic::experimental::handshake2smv(ndWireMlirPath, outputDir, true);
  if (failed(failOrDstSmv))
    return failure();
  auto dstSmv = failOrDstSmv.value();

  // Currently handshake2smv only supports "model" as the model's name
  auto fail = dynamatic::experimental::createWrapper(outputDir / "main_inf.smv",
                                                     config, "model", 0);
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
  int nuxmvRet = runNuXmv(outputDir / "reachability_inf.cmd",
                          outputDir / "inf_states.txt");
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

    // Currently handshake2smv only supports "model" as the model's name
    auto fail =
        dynamatic::experimental::createWrapper(wrapperPath, config, "model", n);
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

    nuxmvRet =
        runNuXmv(outputDir / ("reachability_" + std::to_string(n) + ".cmd"),
                 outputDir.string() + "/" + std::to_string(n) + "_states.txt");
    if (nuxmvRet != 0) {
      llvm::errs() << "Failed to analyze reachable states with"
                   << std::to_string(n) + " tokens.";
      return failure();
    }

    // Check state differences
    // Currently handshake2smv only supports "model" as the model's name
    int nrOfDifferences = dynamatic::experimental::compareReachableStates(
        outputDir.string() + "/inf_states.txt",
        outputDir.string() + "/" + std::to_string(n) + "_states.txt", "model");

    if (nrOfDifferences != 0) {
      n++;
    } else {
      break;
    }
  }
  return n;
}
} // namespace dynamatic::experimental