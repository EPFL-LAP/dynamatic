#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "CreateWrappers.h"
#include "FabricGeneration.h"

using namespace mlir;

namespace dynamatic::experimental {

// Create the call to the module
static std::string
createModuleCall(const std::string &moduleName,
                 const SmallVector<std::pair<std::string, Type>> &arguments,
                 const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream call;

  call << "  VAR " << moduleName << " : " << moduleName << "(";

  for (size_t i = 0; i < arguments.size(); ++i) {
    if (i > 0)
      call << ", ";
    // The current handshake2smv conversion also creates a dataOut port when it
    // is of type control
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

// Create the sequence generator at the inputs of the module
static std::string createSequenceGenerators(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    size_t nrOfTokens, bool exact = false) {
  std::ostringstream sequenceGenerators;
  for (const auto &argument : arguments) {
    if (nrOfTokens == 0) {
      sequenceGenerators << "  VAR seq_generator_" << argument.first
                         << " : bool_input_inf(" << moduleName << "."
                         << argument.first << "_ready);\n";
    } else if (exact) {
      sequenceGenerators << "  VAR seq_generator_" << argument.first
                         << " : bool_input_exact(" << moduleName << "."
                         << argument.first << "_ready, " << nrOfTokens
                         << ");\n";
    } else {
      sequenceGenerators << "  VAR seq_generator_" << argument.first
                         << " : bool_input(" << moduleName << "."
                         << argument.first << "_ready, " << nrOfTokens
                         << ");\n";
    }
  }
  return sequenceGenerators.str();
}

// Create the sinks at the outputs of the module
static std::string
createSinks(const std::string &moduleName,
            const SmallVector<std::pair<std::string, Type>> &results,
            size_t nrOfTokens) {
  std::ostringstream sinks;

  for (const auto &result : results) {
    sinks << "  VAR sink_" << result.first << " : sink_1_0(" << moduleName
          << "." << result.first << "_out, " << moduleName << "."
          << result.first << "_valid);\n";
  }
  return sinks.str();
}

// Create the constraints to restrict the relations of the length of input
// sequences.
static std::string createSeqRelationConstraint(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const std::string &seqLengthRelationConstraint) {

  std::string output = "INVAR";
  output += seqLengthRelationConstraint + "\n";

  for (size_t i = 0; i < arguments.size(); i++) {
    std::regex numberRegex(std::to_string(i));

    output = std::regex_replace(output, numberRegex,
                                " seq_generator_" + arguments[i].first +
                                    ".exact_tokens ");
  }

  return output;
}

// Create the constraints for the loop condition.
static std::string createSeqConstraintLoop(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    LoopSeqConstraint constraint) {

  std::ostringstream seqConstraint;
  std::string falseTokenCounter =
      arguments[constraint.controlSequence].first + "_false_token_cnt";
  std::string falseTokenSeqName =
      "seq_generator_" + arguments[constraint.controlSequence].first;
  std::string otherSeqName =
      "seq_generator_" + arguments[constraint.dataSequence].first;

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
                << falseTokenCounter << ") = " << constraint.lastFalse
                << " ) & ((" << falseTokenSeqName << ".exact_tokens - "
                << falseTokenSeqName
                << ".counter) >= " << 1 + constraint.lastFalse << ")) -> ("
                << falseTokenSeqName << ".dataOut0 = TRUE);\n"
                << "INVAR (" << falseTokenSeqName
                << ".exact_tokens >= " << otherSeqName << ".exact_tokens);\n"
                << "INVAR (" << otherSeqName << ".exact_tokens = 0) -> ("
                << falseTokenSeqName << ".exact_tokens = 0);\n\n";

  return seqConstraint.str();
}

// Create the constraints to limit the number of tokens in the circuit.
static std::string createTokenLimiter(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const SmallVector<std::pair<std::string, std::string>> &inputNDWires,
    const SmallVector<std::pair<std::string, std::string>> &outputNDWires,
    TokenLimitConstraint constraint) {

  std::ostringstream tokenLimitConstraint;

  std::string tokenLimitDef =
      arguments[constraint.inputSequence].first + "_active_token_limit";

  tokenLimitConstraint << "DEFINE " << tokenLimitDef
                       << " := " << constraint.limit << ";\n";

  for (std::string side : {"lhs", "rhs"}) {

    std::string limiterVarName = side + "_" +
                                 arguments[constraint.inputSequence].first +
                                 "_active_tokens";

    std::string inputNDWireName;
    std::string outputNDWireName;
    if (side == "lhs") {
      inputNDWireName =
          moduleName + "." + inputNDWires[constraint.inputSequence].first;
      outputNDWireName =
          moduleName + "." + outputNDWires[constraint.outputSequence].first;
    } else {
      inputNDWireName =
          moduleName + "." + inputNDWires[constraint.inputSequence].second;
      outputNDWireName =
          moduleName + "." + outputNDWires[constraint.outputSequence].second;
    }

    std::string inputTokenCondition =
        inputNDWireName + ".valid0 & " + inputNDWireName + ".nReady0";
    std::string outputTokenCondition =
        outputNDWireName + ".valid0 & " + outputNDWireName + ".nReady0";

    tokenLimitConstraint << "VAR " << limiterVarName << " : 0.."
                         << constraint.limit << ";\n"
                         << "ASSIGN\n"
                         << "init(" << limiterVarName << ") := 0;\n"
                         << "next(" << limiterVarName << ") := case\n"
                         << "  " << inputTokenCondition << " & "
                         << outputTokenCondition << " : " << limiterVarName
                         << ";\n"
                         << "  " << outputTokenCondition << " & "
                         << limiterVarName << " > 0 : " << limiterVarName
                         << " - 1;\n"
                         << "  " << inputTokenCondition << " & "
                         << limiterVarName << " < " << tokenLimitDef << " : "
                         << limiterVarName << " + 1;\n"
                         << "  TRUE : " << limiterVarName << ";\n"
                         << "  esac;\n"
                         << "INVAR " << limiterVarName << " = " << tokenLimitDef
                         << " -> " << inputNDWireName
                         << ".state = sleeping;\n\n";
  }

  return tokenLimitConstraint.str();
}

// Creates all the properties required for the elastic-miter based equivalence
// checking.
// 1. All output tokens with data are true.
// 2. At a certain point, and from then on, all the output buffers remain empty.
// 3. At a certain point, and from then on, all input buffer pair store the same
// number of tokens.
static std::string createMiterProperties(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, std::string>> &inputBuffers,
    const SmallVector<std::pair<std::string, std::string>> &inputNDWires,
    const SmallVector<std::pair<std::string, std::string>> &outputBuffers,
    const SmallVector<std::pair<std::string, std::string>> &outputNDWires,
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

  return properties.str();
}

LogicalResult createWrapper(const std::filesystem::path &wrapperPath,
                            const ElasticMiterConfig &config,
                            const std::string &modelSmvName, size_t nrOfTokens,
                            bool includeProperties,
                            const SequenceConstraints &sequenceConstraints,
                            bool exact) {

  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + ".smv\"\n";
  wrapper << BOOL_INPUT;
  wrapper << BOOL_INPUT_EXACT;
  wrapper << BOOL_INPUT_INF;

  wrapper << "MODULE main\n";

  wrapper << "\n";

  wrapper << createSequenceGenerators(modelSmvName, config.arguments,
                                      nrOfTokens, exact);

  wrapper << createModuleCall(modelSmvName, config.arguments, config.results)
          << "\n";

  wrapper << createSinks(modelSmvName, config.results, nrOfTokens);

  wrapper << "\n";

  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, config.inputBuffers,
                                     config.inputNDWires, config.outputBuffers,
                                     config.outputNDWires, config.arguments,
                                     config.results);

    for (const auto &constraint :
         sequenceConstraints.seqLengthRelationConstraints) {
      wrapper << createSeqRelationConstraint(modelSmvName, config.arguments,
                                             constraint);
    }

    for (const auto &constraint : sequenceConstraints.loopSeqConstraints) {
      wrapper << createSeqConstraintLoop(modelSmvName, config.arguments,
                                         constraint);
    }
    for (const auto &constraint : sequenceConstraints.tokenLimitConstraints) {
      wrapper << createTokenLimiter(modelSmvName, config.arguments,
                                    config.inputNDWires, config.outputNDWires,
                                    constraint);
    }
  }

  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper.str();
  mainFile.close();

  return success();
}
} // namespace dynamatic::experimental