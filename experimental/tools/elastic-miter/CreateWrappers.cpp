#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "CreateWrappers.h"
#include "FabricGeneration.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace dynamatic::experimental {

template <typename T>
static std::string join(const T &v, const std::string &delim) {
  std::ostringstream s;
  for (const auto &i : v) {
    if (&i != &v[0]) {
      s << delim;
    }
    s << i;
  }
  return s.str();
}

// Create the call to the module
// The resulting string will look like:
// VAR <moduleName> : <moduleName> (seq_generator_A.dataOut0,
// seq_generator_A.valid0, ..., sink_F.ready0, ...)
static std::string instantiateModuleUnderTest(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const SmallVector<std::pair<std::string, Type>> &results) {
  SmallVector<std::string> inputVariables;
  for (const auto &[argumentName, argumentType] : arguments) {
    // The current handshake2smv conversion also creates a dataOut port when it
    // is of type control
    if (!LEGACY_DOT2SMV_COMPATIBLE &&
        argumentType.isa<handshake::ControlType>()) {
      inputVariables.push_back("seq_generator_" + argumentName + "." +
                               SEQUENCE_GENERATOR_VALID_NAME.str());
    } else {
      inputVariables.push_back("seq_generator_" + argumentName + "." +
                               SEQUENCE_GENERATOR_DATA_NAME.str());
      inputVariables.push_back("seq_generator_" + argumentName + "." +
                               SEQUENCE_GENERATOR_VALID_NAME.str());
    }
  }

  for (const auto &[resultName, _] : results) {
    inputVariables.push_back("sink_" + resultName + "." +
                             SINK_READY_NAME.str());
  }

  std::ostringstream call;
  call << "  VAR " << moduleName << " : " << moduleName << "(";
  call << join(inputVariables, ", ");
  call << ");\n";

  return call.str();
}

// Create the sequence generator at the inputs of the module
static std::string createSequenceGenerators(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    size_t nrOfTokens, bool exact = false) {
  std::ostringstream sequenceGenerators;
  for (const auto &[argumentName, _] : arguments) {
    if (nrOfTokens == 0) {
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : bool_input_inf(" << moduleName << "."
                         << argumentName << "_ready);\n";
    } else if (exact) {
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : bool_input_exact(" << moduleName << "."
                         << argumentName << "_ready, " << nrOfTokens << ");\n";
    } else {
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : bool_input(" << moduleName << "."
                         << argumentName << "_ready, " << nrOfTokens << ");\n";
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

  for (const auto &[resultName, _] : results) {
    sinks << "  VAR sink_" << resultName << " : sink_1_0(" << moduleName << "."
          << resultName << "_out, " << moduleName << "." << resultName
          << "_valid);\n";
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
    auto [argumentName, _] = arguments[i];
    std::regex numberRegex(std::to_string(i));

    output =
        std::regex_replace(output, numberRegex,
                           " seq_generator_" + argumentName + ".exact_tokens ");
  }

  return output;
}

// Create the constraints for the loop condition.
static std::string createSeqConstraintLoop(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    LoopSeqConstraint constraint) {

  auto [controlSequenceName, _] = arguments[constraint.controlSequence];
  auto [dataSequenceName, _] = arguments[constraint.dataSequence];

  std::ostringstream seqConstraint;
  std::string falseTokenCounter = controlSequenceName + "_false_token_cnt";
  std::string falseTokenSeqName = "seq_generator_" + controlSequenceName;
  std::string otherSeqName = "seq_generator_" + dataSequenceName;

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

  auto [inputSequenceName, _] = arguments[constraint.inputSequence];

  auto [lhsInputNDWire, rhsInputNDWire] =
      inputNDWires[constraint.inputSequence];
  auto [lhsOutputNDWire, rhsOutputNDWire] =
      outputNDWires[constraint.outputSequence];

  std::string tokenLimitDef = inputSequenceName + "_active_token_limit";

  tokenLimitConstraint << "DEFINE " << tokenLimitDef
                       << " := " << constraint.limit << ";\n";

  for (std::string side : {"lhs", "rhs"}) {

    std::string limiterVarName =
        side + "_" + inputSequenceName + "_active_tokens";

    std::string inputNDWireName;
    std::string outputNDWireName;
    if (side == "lhs") {
      inputNDWireName = moduleName + "." + lhsInputNDWire;
      outputNDWireName = moduleName + "." + lhsOutputNDWire;
    } else {
      inputNDWireName = moduleName + "." + rhsInputNDWire;
      outputNDWireName = moduleName + "." + rhsOutputNDWire;
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
static std::string createMiterProperties(const std::string &moduleName,
                                         const ElasticMiterConfig &config) {
  std::ostringstream properties;

  // Create the property that every output data token will be TRUE. The outputs
  // are created by comparing the output from the LHS to the RHS. If the output
  // is of type control it is the output of a join operation. In this case we do
  // not need additional properties. Making sure the buffers will hold no tokens
  // is sufficient.
  // Example (Assuming data outputs A and B control output C):
  // INVARSPEC (model.EQ_A_valid -> model.EQ_A_out)
  // INVARSPEC (model.EQ_B_valid -> model.EQ_B_out)
  for (const auto &[resultName, resultType] : config.results) {
    if (resultType.isa<handshake::ChannelType>())
      properties << "INVARSPEC (" << moduleName
                 << "." + resultName + "_valid -> " << moduleName
                 << "." + resultName + "_out)\n";
  }

  // Make sure the input buffers will have pairwise the same number of tokens.
  // This means both circuits consume the same number of tokens.
  SmallVector<std::string> bufferProperties;
  for (const auto &[lhsBuffer, rhsBuffer] : config.inputBuffers) {
    bufferProperties.push_back("(" + moduleName + "." + lhsBuffer + ".num = " +
                               moduleName + "." + rhsBuffer + ".num)");
  }

  // Make sure the output buffers will be empty.
  // This means both circuits produce the same number of output tokens.
  for (const auto &[lhsBuffer, rhsBuffer] : config.outputBuffers) {
    bufferProperties.push_back("(" + moduleName + "." + lhsBuffer +
                               ".num = 0)");
    bufferProperties.push_back("(" + moduleName + "." + rhsBuffer +
                               ".num = 0)");
  }

  // Make sure the buffer will start to hold at one point and from there on.
  // The final property will be (Assuming inputs D and C, and outputs A and B):
  // CTLSPEC AF (AG ((model.lhs_in_buf_D.num = model.rhs_in_buf_D.num) &
  // (model.lhs_in_buf_C.num = model.rhs_in_buf_C.num) &
  // (model.lhs_out_buf_A.num = 0) & (model.rhs_out_buf_A.num = 0) &
  // (model.lhs_out_buf_B.num = 0) & (model.rhs_out_buf_B.num = 0)))
  std::string finalBufferProp =
      "AF (AG (" + join(bufferProperties, " & ") + "))";
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
  wrapper << SMV_BOOL_INPUT;
  wrapper << SMV_BOOL_INPUT_EXACT;
  wrapper << SMV_BOOL_INPUT_INF;

  wrapper << "MODULE main\n";

  wrapper << "\n";

  wrapper << createSequenceGenerators(modelSmvName, config.arguments,
                                      nrOfTokens, exact);

  wrapper << instantiateModuleUnderTest(modelSmvName, config.arguments,
                                        config.results)
          << "\n";

  wrapper << createSinks(modelSmvName, config.results, nrOfTokens) << "\n";

  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, config);

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