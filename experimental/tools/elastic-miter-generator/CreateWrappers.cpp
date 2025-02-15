#include "mlir/IR/Attributes.h"

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

#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace dynamatic::experimental {

// TODO pass names of in and output
std::string createMiterCall(const SmallVector<std::string> &args,
                            const SmallVector<std::string> &res) {
  std::ostringstream miter;
  miter << "VAR miter : elastic_miter(";

  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      miter << ", ";
    miter << "seq_generator" << i << ".dataOut0, seq_generator" << i
          << ".valid0";
  }

  miter << ", ";

  for (size_t i = 0; i < res.size(); ++i) {
    if (i > 0)
      miter << ", ";
    miter << "sink" << i << ".ready0";
  }

  miter << ");\n";
  return miter.str();
}

std::string createMiterProperties(
    const SmallVector<std::pair<std::string, std::string>> &inputBuffers,
    const SmallVector<std::string> &outputBuffers,
    const SmallVector<std::string> &results) {
  std::ostringstream properties;

  for (const auto &result : results) {
    properties << "CTLSPEC AG (miter." + result + "_valid -> miter." + result +
                      "_out)\n";
  }

  std::string outputProp;
  for (const auto &buffer : outputBuffers) {
    outputProp += "miter." + buffer + ".num = 0 & ";
  }

  if (!outputProp.empty())
    outputProp = outputProp.substr(0, outputProp.size() - 3);

  std::string inputProp;
  for (const auto &bufferPair : inputBuffers) {
    inputProp += "miter." + bufferPair.first + ".num = miter." +
                 bufferPair.second + ".num & ";
  }

  if (!inputProp.empty())
    inputProp = inputProp.substr(0, inputProp.size() - 3);

  std::string finalBufferProp =
      "AF (AG (" + inputProp + " & " + outputProp + "))";
  properties << "CTLSPEC " + finalBufferProp + "\n";

  return properties.str();
}

// TODO make this cleaner
LogicalResult createMiterWrapper(const std::filesystem::path &wrapperPath,
                                 const std::filesystem::path &jsonPath,
                                 const std::string &modelSmvName,
                                 size_t nrOfTokens) {
  std::ifstream file(jsonPath);
  if (!file) {
    llvm::errs() << "Config JSON file not found\n";
    return failure();
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(buffer.str());
  if (!jsonValue) {
    llvm::errs() << "Failed parsing JSON\n";
    return failure();
  }

  llvm::json::Object *config = jsonValue->getAsObject();
  if (!config) {
    llvm::errs() << "Failed parsing JSON\n";
    return failure();
  }

  SmallVector<std::string> argNames;
  if (auto *args = config->getArray("arguments")) {
    for (const auto &arg : *args) {
      argNames.push_back(arg.getAsString()->str());
    }
  }

  SmallVector<std::string> resNames;
  if (auto *args = config->getArray("results")) {
    for (const auto &arg : *args) {
      resNames.push_back(arg.getAsString()->str());
    }
  }

  SmallVector<std::string> outputBufferNames;
  if (auto *args = config->getArray("output_buffers")) {
    for (const auto &arg : *args) {
      outputBufferNames.push_back(arg.getAsString()->str());
    }
  }

  SmallVector<std::pair<std::string, std::string>> inputBufferNames;
  std::string inputProp;
  if (auto *inBufs = config->getArray("input_buffers")) {
    for (const auto &bufPair : *inBufs) {
      if (auto *pair = bufPair.getAsArray()) {
        inputBufferNames.push_back(std::make_pair(
            (*pair)[0].getAsString()->str(), (*pair)[1].getAsString()->str()));
      }
    }
  }

  std::string output;
  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + "\"\n";
  wrapper << "#ifndef BOOL_INPUT\n"
             "#define BOOL_INPUT\n"
             "MODULE bool_input(nReady0, max_tokens)\n"
             "  VAR dataOut0 : boolean;\n"
             "  VAR counter : 0..31;\n"
             "  ASSIGN\n"
             "    init(counter) := 0;\n"
             "    next(counter) := case\n"
             "      nReady0 & counter < max_tokens : counter + 1;\n"
             "      TRUE : counter;\n"
             "    esac;\n"
             "\n"
             "  -- bool_input persistent\n"
             "  ASSIGN\n"
             "    next(dataOut0) := case\n"
             "      valid0 & !nReady0 : dataOut0;\n"
             "      TRUE : {TRUE, FALSE};\n"
             "    esac;\n"
             "\n"
             "  DEFINE valid0 := counter < max_tokens;\n"
             "#endif // BOOL_INPUT\n"
             "\n"
             "MODULE main\n";

  for (size_t i = 0; i < argNames.size(); ++i) {
    if (nrOfTokens == 0) {
      wrapper << "  VAR seq_generator" << i << " : bool_input_inf(miter." << i
              << "_ready);\n";
    } else {
      wrapper << "  VAR seq_generator" << i << " : bool_input(miter." << i
              << "_ready, " << nrOfTokens << ");\n";
    }
  }
  wrapper << "\n";

  wrapper << "\n  " << createMiterCall(argNames, resNames) << "\n";
  wrapper << "  -- TODO make sure we have sink_1_0\n";

  for (size_t i = 0; i < resNames.size(); ++i) {
    wrapper << "  VAR sink" << i << " : sink_1_0(miter." << resNames[i]
            << "_out, miter." << resNames[i] << "_valid);\n";
  }

  wrapper << "\n";

  wrapper << createMiterProperties(inputBufferNames, outputBufferNames,
                                   resNames);

  std::ofstream mainFile(wrapperPath);
  mainFile << output;
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental