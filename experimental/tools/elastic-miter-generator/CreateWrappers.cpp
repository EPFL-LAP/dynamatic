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
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

using namespace mlir;

namespace dynamatic::experimental {

// TODO pass names of in and output and name
std::string
createModuleCall(const std::string &moduleName,
                 const SmallVector<std::pair<std::string, Type>> &arguments,
                 const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream call;
  // TODO does this work?
  call << "  VAR " << moduleName << " : " << moduleName << "(";

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (i > 0)
      call << ", ";
    // The current handshake2smv conversion also puts dataOut when it is of type
    // control
    if (false && arguments[i].second.isa<handshake::ControlType>()) {
      call << "seq_generator_" << arguments[i].first << ".valid0";
    } else {
      call << "seq_generator_" << arguments[i].first
           << ".dataOut0, seq_generator_" << arguments[i].first << ".valid0";
    }
  }

  call << ", ";

  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0)
      call << ", ";
    call << "sink_" << results[i].first << ".ready0";
  }

  call << ");\n";
  return call.str();
}

std::string createSequenceGenerators(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    size_t nrOfTokens) {
  std::ostringstream sequenceGenerators;
  for (const auto &argument : arguments) {
    if (nrOfTokens == 0) {
      sequenceGenerators << "  VAR seq_generator_" << argument.first
                         << " : bool_input_inf(" << moduleName << "."
                         << argument.first << "_ready);\n";
    } else {
      sequenceGenerators << "  VAR seq_generator_" << argument.first
                         << " : bool_input(" << moduleName << "."
                         << argument.first << "_ready, " << nrOfTokens
                         << ");\n";
    }
  }
  return sequenceGenerators.str();
}

std::string
createSinks(const std::string &moduleName,
            const SmallVector<std::pair<std::string, Type>> &results,
            size_t nrOfTokens) {
  std::ostringstream sinks;
  sinks << "  -- TODO make sure we have sink_1_0\n";
  for (size_t i = 0; i < results.size(); ++i) {
    sinks << "  VAR sink_" << results[i].first << " : sink_1_0(" << moduleName
          << "." << results[i].first << "_out, " << moduleName << "."
          << results[i].first << "_valid);\n";
  }
  return sinks.str();
}

std::string createMiterProperties(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, std::string>> &inputBuffers,
    const SmallVector<std::pair<std::string, std::string>> &outputBuffers,
    const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream properties;

  for (const auto &result : results) {
    if (result.second.isa<handshake::ChannelType>())
      properties << "INVARSPEC (" << moduleName
                 << "." + result.first + "_valid -> " << moduleName
                 << "." + result.first + "_out)\n";
  }

  std::string inputProp;
  for (const auto &bufferPair : inputBuffers) {
    inputProp += "(" + moduleName + "." + bufferPair.first +
                 ".num = " + moduleName + "." + bufferPair.second + ".num) & ";
  }

  // Remove the final " & "
  if (!inputProp.empty())
    inputProp = inputProp.substr(0, inputProp.size() - 3);

  std::string outputProp;
  for (const auto &buffer : outputBuffers) {
    outputProp += "(" + moduleName + "." + buffer.first + ".num = 0) & ";
    outputProp += "(" + moduleName + "." + buffer.second + ".num = 0) & ";
  }

  // Remove the final " & "
  if (!outputProp.empty())
    outputProp = outputProp.substr(0, outputProp.size() - 3);

  std::string finalBufferProp =
      "AF (AG (" + inputProp + " & " + outputProp + "))";
  properties << "CTLSPEC " + finalBufferProp + "\n";

  return properties.str();
}

LogicalResult createWrapper(const std::filesystem::path &wrapperPath,
                            const ElasticMiterConfig &config,
                            const std::string &modelSmvName, size_t nrOfTokens,
                            bool includeProperties) {

  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + ".smv\"\n";
  wrapper << BOOL_INPUT;
  wrapper << BOOL_INPUT_INF;

  wrapper << "MODULE main\n";

  wrapper << "\n";

  wrapper << createSequenceGenerators(modelSmvName, config.arguments,
                                      nrOfTokens);

  wrapper << createModuleCall(modelSmvName, config.arguments, config.results)
          << "\n";

  wrapper << createSinks(modelSmvName, config.results, nrOfTokens);

  wrapper << "\n";

  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, config.inputBuffers,
                                     config.outputBuffers, config.results);
  }

  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper.str();
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental