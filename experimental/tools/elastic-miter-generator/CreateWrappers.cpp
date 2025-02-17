#include "mlir/IR/Attributes.h"

#include <any>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <string>
#include <utility>

#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace dynamatic::experimental {

// TODO pass names of in and output and name
std::string createModuleCall(const std::string &moduleName,
                             const SmallVector<std::string> &argNames,
                             const SmallVector<std::string> &resNames) {
  std::ostringstream call;
  // TODO does this work?
  call << "  VAR " << moduleName << " : " << moduleName << "(";

  for (size_t i = 0; i < argNames.size(); ++i) {
    if (i > 0)
      call << ", ";
    call << "seq_generator" << i << ".dataOut0, seq_generator" << i
         << ".valid0";
  }

  call << ", ";

  for (size_t i = 0; i < resNames.size(); ++i) {
    if (i > 0)
      call << ", ";
    call << "sink" << i << ".ready0";
  }

  call << ");\n";
  return call.str();
}

std::string createSequenceGenerators(const std::string &moduleName,
                                     const SmallVector<std::string> &argNames,
                                     size_t nrOfTokens) {
  std::ostringstream sequenceGenerators;
  for (size_t i = 0; i < argNames.size(); ++i) {
    if (nrOfTokens == 0) {
      sequenceGenerators << "  VAR seq_generator" << i << " : bool_input_inf("
                         << moduleName << "." << argNames[i] << "_ready);\n";
    } else {
      sequenceGenerators << "  VAR seq_generator" << i << " : bool_input("
                         << moduleName << "." << argNames[i] << "_ready, "
                         << nrOfTokens << ");\n";
    }
  }
  return sequenceGenerators.str();
}

std::string createSinks(const std::string &moduleName,
                        const SmallVector<std::string> &resNames,
                        size_t nrOfTokens) {
  std::ostringstream sinks;
  sinks << "  -- TODO make sure we have sink_1_0\n";
  for (size_t i = 0; i < resNames.size(); ++i) {
    sinks << "  VAR sink" << i << " : sink_1_0(" << moduleName << "."
          << resNames[i] << "_out, " << moduleName << "." << resNames[i]
          << "_valid);\n";
  }
  return sinks.str();
}

std::string createMiterProperties(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, std::string>> &inputBuffers,
    const SmallVector<std::pair<std::string, std::string>> &outputBuffers,
    const SmallVector<std::string> &results) {
  std::ostringstream properties;

  for (const auto &result : results) {
    properties << "CTLSPEC AG (" << moduleName << "." + result + "_valid -> "
               << moduleName << "." + result + "_out)\n";
  }

  std::string outputProp;
  for (const auto &buffer : outputBuffers) {
    outputProp += moduleName + "." + buffer.first + ".num = 0 & ";
    outputProp += moduleName + "." + buffer.second + ".num = 0 & ";
  }

  if (!outputProp.empty())
    outputProp = outputProp.substr(0, outputProp.size() - 3);

  std::string inputProp;
  for (const auto &bufferPair : inputBuffers) {
    inputProp += moduleName + "." + bufferPair.first + ".num = " + moduleName +
                 "." + bufferPair.second + ".num & ";
  }

  if (!inputProp.empty())
    inputProp = inputProp.substr(0, inputProp.size() - 3);

  std::string finalBufferProp =
      "AF (AG (" + inputProp + " & " + outputProp + "))";
  properties << "CTLSPEC " + finalBufferProp + "\n";

  return properties.str();
}

LogicalResult createWrapper(const std::filesystem::path &wrapperPath,
                            llvm::StringMap<std::any> config,
                            const std::string &modelSmvName, size_t nrOfTokens,
                            bool includeProperties) {
  // TODO remove
  // std::ifstream file(jsonPath);
  // if (!file) {
  //   llvm::errs() << "Config JSON file not found\n";
  //   return failure();
  // }

  // std::stringstream buffer;
  // buffer << file.rdbuf();
  // file.close();

  // llvm::Expected<llvm::json::Value> jsonValue =
  // llvm::json::parse(buffer.str()); if (!jsonValue) {
  //   llvm::errs() << "Failed parsing JSON\n";
  //   return failure();
  // }

  // llvm::json::Object *config = jsonValue->getAsObject();
  // if (!config) {
  //   llvm::errs() << "Failed parsing JSON\n";
  //   return failure();
  // }

  // SmallVector<std::string> argNames;
  // if (auto *args = config->getArray("arguments")) {
  //   for (const auto &arg : *args) {
  //     argNames.push_back(arg.getAsString()->str());
  //   }
  // }

  // SmallVector<std::string> resNames;
  // if (auto *args = config->getArray("results")) {
  //   for (const auto &arg : *args) {
  //     resNames.push_back(arg.getAsString()->str());
  //   }
  // }

  // SmallVector<std::string> outputBufferNames;
  // if (auto *args = config->getArray("output_buffers")) {
  //   for (const auto &arg : *args) {
  //     outputBufferNames.push_back(arg.getAsString()->str());
  //   }
  // }

  // SmallVector<std::pair<std::string, std::string>> inputBufferNames;
  // std::string inputProp;
  // if (auto *inBufs = config->getArray("input_buffers")) {
  //   for (const auto &bufPair : *inBufs) {
  //     if (auto *pair = bufPair.getAsArray()) {
  //       inputBufferNames.push_back(std::make_pair(
  //           (*pair)[0].getAsString()->str(),
  //           (*pair)[1].getAsString()->str()));
  //     }
  //   }
  // }

  // for (auto key : config.keys()) {
  //   llvm::outs() << key.str() << "\n";
  // }

  SmallVector<std::string> argNames;
  // Test if "arguments" exists in config and is of the correct type
  if (config.contains("arguments") &&
      std::any_cast<SmallVector<std::string>>(&config["arguments"])) {
    argNames = std::any_cast<SmallVector<std::string>>(config["arguments"]);
  } else {
    llvm::errs() << "\"arguments\" not in config.\n";
    return failure();
  }

  // Test if "results" exists in config and is of the correct type
  SmallVector<std::string> resNames;
  if (config.contains("results") &&
      std::any_cast<SmallVector<std::string>>(&config["results"])) {
    resNames = std::any_cast<SmallVector<std::string>>(config["results"]);

  } else {
    llvm::errs() << "\"results\" not in config.\n";
    return failure();
  }

  // Test if "output_buffers" exists in config and is of the correct type.
  // If includeProperties is not set this is not required
  SmallVector<std::pair<std::string, std::string>> outputBufferNamePairs;
  if (includeProperties && config.contains("output_buffers") &&
      std::any_cast<SmallVector<std::pair<std::string, std::string>>>(
          &config["output_buffers"])) {
    outputBufferNamePairs =
        std::any_cast<SmallVector<std::pair<std::string, std::string>>>(
            config["output_buffers"]);
  } else if (includeProperties) {
    llvm::errs() << "\"output_buffers\" not in config.\n";
    return failure();
  }

  // Test if "input_buffers" exists in config and is of the correct type.
  // If includeProperties is not set this is not required
  SmallVector<std::pair<std::string, std::string>> inputBufferNamePairs;
  if (includeProperties && config.contains("input_buffers") &&
      std::any_cast<SmallVector<std::pair<std::string, std::string>>>(
          &config["input_buffers"])) {
    inputBufferNamePairs =
        std::any_cast<SmallVector<std::pair<std::string, std::string>>>(
            config["input_buffers"]);

  } else if (includeProperties) {
    llvm::errs() << "\"input_buffers\" not in config.\n";
    return failure();
  }

  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + ".smv\"\n";
  wrapper << BOOL_INPUT;
  wrapper << BOOL_INPUT_INF;

  wrapper << "MODULE main\n";

  wrapper << "\n";

  wrapper << createSequenceGenerators(modelSmvName, argNames, nrOfTokens);

  wrapper << createModuleCall(modelSmvName, argNames, resNames) << "\n";

  wrapper << createSinks(modelSmvName, resNames, nrOfTokens);

  wrapper << "\n";

  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, inputBufferNamePairs,
                                     outputBufferNamePairs, resNames);
  }

  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper.str();
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental