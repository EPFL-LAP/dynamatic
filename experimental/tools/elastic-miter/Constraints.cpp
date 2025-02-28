

#include "Constraints.h"
#include "FabricGeneration.h"

#include <cstddef>

#include <regex>
#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"
namespace dynamatic::experimental {
void TokenLimitConstraint::parseString(const std::string &option) {
  // A Token Limit constraint has the form
  // "<inputSequence>,<outputSequence>,<limit>".
  // At any point in time, the number of tokens which are created at the input
  // with index inputSequence can only be up to "limit" higher than the number
  // of tokens reaching the output with the index outputSequence.
  // Example:
  // `--token_limit="1,1,2"` ensures that there are only two tokens in the
  // circuit which enter at the input with index 1 and leave at the ouput with
  // index 1.

  std::regex pattern(
      R"(^(\d+),(\d+),(\d+)$)"); // Three uint separated by commas
  std::smatch match;

  if (!std::regex_match(option, match, pattern)) {
    llvm::errs() << "Token limit constraints are three positive numbers "
                    "separated by commas\n";
    // TODO return failure();
  }
  inputSequence = std::stoul(match[1]);
  outputSequence = std::stoul(match[2]);
  limit = std::stoul(match[3]);
}

void LoopConstraint::parseString(const std::string &option) {
  std::regex pattern(R"(^(\d+),(\d+)$)"); // Two uint separated by a comma
  std::smatch match;

  // A Loop Condition sequence contraint has the form
  // "<dataSequence>,<controlSequence>". The number of tokens in the input with
  // the index dataSequence is equivalent to the number of false tokens at the
  // output with the index controlSequence.
  // Example:
  // --loop="0,1"
  if (!std::regex_match(option, match, pattern)) {
    llvm::errs() << "Loop sequence constraints are two positive numbers "
                    "separated by a comma\n";
    // TODO return failure();
  }
  dataSequence = std::stoul(match[1]);
  controlSequence = std::stoul(match[2]);
}

// Create the constraints to restrict the relations of the length of input
// sequences.
std::string SequenceLengthRelationConstraint::createConstraintString(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config) const {

  std::string output = "INVAR";
  output += constraint + "\n";

  for (size_t i = 0; i < config.arguments.size(); i++) {
    auto [argumentName, _] = config.arguments[i];
    std::regex numberRegex(std::to_string(i));

    output =
        std::regex_replace(output, numberRegex,
                           " seq_generator_" + argumentName + ".exact_tokens ");
  }

  return output;
}
// Create the constraints for the loop condition.
std::string LoopConstraint::createConstraintString(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config,
    bool lastFalse) const {

  auto [controlSequenceName, _] = config.arguments[controlSequence];
  auto [dataSequenceName, _] = config.arguments[dataSequence];

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
                << falseTokenCounter << ") = " << lastFalse << " ) & (("
                << falseTokenSeqName << ".exact_tokens - " << falseTokenSeqName
                << ".counter) >= " << 1 + lastFalse << ")) -> ("
                << falseTokenSeqName << ".dataOut0 = TRUE);\n"
                << "INVAR (" << falseTokenSeqName
                << ".exact_tokens >= " << otherSeqName << ".exact_tokens);\n"
                << "INVAR (" << otherSeqName << ".exact_tokens = 0) -> ("
                << falseTokenSeqName << ".exact_tokens = 0);\n\n";

  return seqConstraint.str();
}

// Create the constraints to limit the number of tokens in the circuit.
std::string TokenLimitConstraint::createConstraintString(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config) const {

  std::ostringstream tokenLimitConstraint;

  auto [inputSequenceName, _] = config.arguments[inputSequence];

  auto [lhsInputNDWire, rhsInputNDWire] = config.inputNDWires[inputSequence];
  auto [lhsOutputNDWire, rhsOutputNDWire] =
      config.outputNDWires[outputSequence];

  std::string tokenLimitDef = inputSequenceName + "_active_token_limit";

  tokenLimitConstraint << "DEFINE " << tokenLimitDef << " := " << limit
                       << ";\n";

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

    tokenLimitConstraint << "VAR " << limiterVarName << " : 0.." << limit
                         << ";\n"
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
} // namespace dynamatic::experimental
