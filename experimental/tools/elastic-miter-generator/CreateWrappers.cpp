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
  for (const auto &result : results) {
    sinks << "  VAR sink_" << result.first << " : sink_1_0(" << moduleName
          << "." << result.first << "_out, " << moduleName << "."
          << result.first << "_valid);\n";
  }
  return sinks.str();
}

std::string createSeqContraintAplusBequalC(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments, size_t seqA,
    size_t seqB, size_t seqC) {

  std::ostringstream seqConstraint;
  std::string aSeqName = "seq_generator_" + arguments[seqA].first;
  std::string bSeqName = "seq_generator_" + arguments[seqB].first;
  std::string cSeqName = "seq_generator_" + arguments[seqC].first;
  // Seq-contraint 4 A_EQUALS_FALSE_IN_B
  seqConstraint << "INVAR (" << aSeqName << ".exact_tokens + " << bSeqName
                << ".exact_tokens = " << cSeqName << ".exact_tokens);\n";

  return seqConstraint.str();
}

std::string createSeqContraintAequalB(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments, size_t seqA,
    size_t seqB) {

  std::ostringstream seqConstraint;
  std::string aSeqName = "seq_generator_" + arguments[seqA].first;
  std::string bSeqName = "seq_generator_" + arguments[seqB].first;
  // Seq-contraint 4 A_EQUALS_FALSE_IN_B
  seqConstraint << "INVAR (" << aSeqName << ".exact_tokens = " << bSeqName
                << ".exact_tokens);\n";

  return seqConstraint.str();
}

std::string createSeqContraintLoop(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments, size_t seqA,
    size_t seqB, bool lastFalse) {

  std::ostringstream seqConstraint;
  std::string falseTokenCounter = arguments[seqB].first + "_false_token_cnt";
  std::string falseTokenSeqName = "seq_generator_" + arguments[seqB].first;
  std::string otherSeqName = "seq_generator_" + arguments[seqA].first;
  // Seq-contraint 4 A_EQUALS_FALSE_IN_B
  seqConstraint << "VAR " << falseTokenCounter << " : 0..31;\n"
                << "ASSIGN init(" << falseTokenCounter << ") := 0;\n"
                << "ASSIGN next(" << falseTokenCounter << ") := case\n"
                << "  " << falseTokenSeqName << ".valid0 & "
                << falseTokenSeqName << ".nReady0 & (" << falseTokenSeqName
                << ".dataOut0 = FALSE) & (" << falseTokenCounter << " < "
                << otherSeqName << ".exact_tokens) : (" << falseTokenCounter
                << " + 1);\n"
                << "  TRUE : " << falseTokenCounter << ";\n"
                << "esac;\n"
                << "INVAR (((" << otherSeqName << ".exact_tokens - "
                << falseTokenCounter << ") = (" << falseTokenSeqName
                << ".exact_tokens - " << falseTokenSeqName << ".counter)) & (("
                << otherSeqName << ".exact_tokens - " << falseTokenCounter
                << ") >= 1) ) -> " << falseTokenSeqName
                << ".dataOut0 = FALSE;\n"
                << "INVAR (((" << otherSeqName << ".exact_tokens - "
                << falseTokenCounter << ") = " << lastFalse << " ) & (("
                << falseTokenSeqName << ".exact_tokens - " << falseTokenSeqName
                << ".counter) >= " << 1 + lastFalse << ")) -> ("
                << falseTokenSeqName << ".dataOut0 = TRUE);\n"
                << "INVAR (" << falseTokenSeqName
                << ".exact_tokens >= " << otherSeqName << ".exact_tokens);\n"
                << "INVAR (" << otherSeqName << ".exact_tokens = 0) -> ("
                << falseTokenSeqName << ".exact_tokens = 0);\n";

  return seqConstraint.str();
}

std::string createMiterProperties(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, std::string>> &inputBuffers,
    const SmallVector<std::pair<std::string, std::string>> &outputBuffers,
    const SmallVector<std::pair<std::string, Type>> &arguments,
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

  properties << "\n";

  // properties << createSeqContraintLoop(moduleName, arguments, 0, 1, true);

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
                                     config.outputBuffers, config.arguments,
                                     config.results);
  }

  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper.str();
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental