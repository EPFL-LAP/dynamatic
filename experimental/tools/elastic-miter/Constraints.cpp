#include <cstddef>
#include <regex>
#include <string>

#include "llvm/Support/raw_ostream.h"

#include "Constraints.h"
#include "FabricGeneration.h"

namespace dynamatic::experimental {

// Parse a Token Limit constraint in the form
// "<inputSequence>,<outputSequence>,<limit>".
void TokenLimitConstraint::parseString(const std::string &option) {
  std::regex pattern(
      R"(^(\d+),(\d+),(\d+)$)"); // Three uint separated by commas
  std::smatch match;

  if (!std::regex_match(option, match, pattern)) {
    llvm::errs() << "Token limit constraints are three positive numbers "
                    "separated by commas\n";
  }
  inputSequence = std::stoul(match[1]);
  outputSequence = std::stoul(match[2]);
  limit = std::stoul(match[3]);
}

// Parse a Loop Condition sequence contraint in the form
// "<dataSequence>,<controlSequence>".
void LoopConstraint::parseString(const std::string &option) {
  std::regex pattern(R"(^(\d+),(\d+)$)"); // Two uint separated by a comma
  std::smatch match;

  if (!std::regex_match(option, match, pattern)) {
    llvm::errs() << "Loop sequence constraints are two positive numbers "
                    "separated by a comma\n";
  }
  dataSequence = std::stoul(match[1]);
  controlSequence = std::stoul(match[2]);
}

std::string SequenceLengthRelationConstraint::createSmvConstraint(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config) const {

  std::string output = "INVAR";
  output += constraint + "\n";

  // Replace all the occurences of the sequence with index i with its
  // corresponding sequence generator.
  // Example:
  // Input names: [A, B, C, D].
  // Constraint: 1+2=3+4
  // Resulting SMV constraint:
  // seq_generator_A.exact_tokens + seq_generator_B.exact_tokens =
  // seq_generator_A.exact_tokens + seq_generator_B.exact_tokens
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

  auto [controlSequenceName, ignored1] = config.arguments[controlSequence];
  auto [dataSequenceName, ignored2] = config.arguments[dataSequence];

  // The name of the SMV variable to keep track of false tokens.
  std::string falseCounterTokenCounterVariable =
      controlSequenceName + "_false_token_cnt";
  std::string controlSeqGeneratorName = "seq_generator_" + controlSequenceName;
  std::string dataSeqGeneratorName = "seq_generator_" + dataSequenceName;

  std::ostringstream falseTokenCounterString;
  std::ostringstream forceFalseTokenInvar;
  std::ostringstream limitFalseTokensInvar;
  std::ostringstream seqLengthRelationInvar;
  std::ostringstream emptyDataSequenceInvar;

  // Create a counter, which counts the total number of false token at the
  // control sequence input.
  // Example:
  // VAR Cd_false_token_cnt : 0..31;
  // ASSIGN init(Cd_false_token_cnt) := 0;
  // ASSIGN next(Cd_false_token_cnt) := case
  //   seq_generator_Cd.valid0 & seq_generator_Cd.nReady0 &
  //   (seq_generator_Cd.dataOut0 = FALSE) & (Cd_false_token_cnt <
  //   seq_generator_Dd.exact_tokens) : (Cd_false_token_cnt + 1); TRUE :
  //   Cd_false_token_cnt;
  // esac;
  falseTokenCounterString << "VAR " << falseCounterTokenCounterVariable
                          << " : 0..31;\n"
                          << "ASSIGN init(" << falseCounterTokenCounterVariable
                          << ") := 0;\n"
                          << "ASSIGN next(" << falseCounterTokenCounterVariable
                          << ") := case\n"
                          << "  " << controlSeqGeneratorName << ".valid0 & "
                          << controlSeqGeneratorName << ".nReady0 & ("
                          << controlSeqGeneratorName << ".dataOut0 = FALSE) & ("
                          << falseCounterTokenCounterVariable << " < "
                          << dataSeqGeneratorName << ".exact_tokens) : ("
                          << falseCounterTokenCounterVariable << " + 1);\n"
                          << "  TRUE : " << falseCounterTokenCounterVariable
                          << ";\n"
                          << "esac;\n";

  // Make sure enough false tokens are generated. When there are as many false
  // tokens that still need to be generated, as there are total control tokens
  // that can be generated, every token needs to be false.
  // Example:
  // INVAR (((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) =
  // (seq_generator_Cd.exact_tokens - seq_generator_Cd.counter)) &
  // ((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) >= 1) ) ->
  // seq_generator_Cd.dataOut0 = FALSE;
  forceFalseTokenInvar << "INVAR (((" << dataSeqGeneratorName
                       << ".exact_tokens - " << falseCounterTokenCounterVariable
                       << ") = (" << controlSeqGeneratorName
                       << ".exact_tokens - " << controlSeqGeneratorName
                       << ".counter)) & ((" << dataSeqGeneratorName
                       << ".exact_tokens - " << falseCounterTokenCounterVariable
                       << ") >= 1) ) -> " << controlSeqGeneratorName
                       << ".dataOut0 = FALSE;\n";

  // Make sure that not to many false tokens are generated. If the last tokens
  // needs to be false, this means we only generate false tokens as long as we
  // still have one false token to spare. If the last token can be freely
  // chosen, we only create false token as long as we haven't reached the total
  // number of tokens.
  // Example (last True or False): INVAR
  // (((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) = 0 ) &
  // ((seq_generator_Cd.exact_tokens - seq_generator_Cd.counter) >= 1)) ->
  // (seq_generator_Cd.dataOut0 = TRUE);
  // Example (last False): INVAR
  // (((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) = 1 ) &
  // ((seq_generator_Cd.exact_tokens - seq_generator_Cd.counter) >= 2)) ->
  // (seq_generator_Cd.dataOut0 = TRUE);
  limitFalseTokensInvar << "INVAR (((" << dataSeqGeneratorName
                        << ".exact_tokens - "
                        << falseCounterTokenCounterVariable
                        << ") = " << lastFalse << " ) & (("
                        << controlSeqGeneratorName << ".exact_tokens - "
                        << controlSeqGeneratorName
                        << ".counter) >= " << 1 + lastFalse << ")) -> ("
                        << controlSeqGeneratorName << ".dataOut0 = TRUE);\n";

  // Make sure the control sequence generates at least as many tokens as the
  // data sequence. Otherwise it will be impossible to have the same number of
  // control false tokens as total data tokens.
  seqLengthRelationInvar << "INVAR (" << controlSeqGeneratorName
                         << ".exact_tokens >= " << dataSeqGeneratorName
                         << ".exact_tokens);\n";

  // When there are no data tokens, there should also be no control tokens.
  // Otherwise the control token cannot be consumed.
  emptyDataSequenceInvar << "INVAR (" << dataSeqGeneratorName
                         << ".exact_tokens = 0) -> (" << controlSeqGeneratorName
                         << ".exact_tokens = 0);\n\n";

  return falseTokenCounterString.str() + forceFalseTokenInvar.str() +
         limitFalseTokensInvar.str() + seqLengthRelationInvar.str() +
         emptyDataSequenceInvar.str();
}

// Create the constraints to limit the number of tokens in the circuit.
std::string TokenLimitConstraint::createSmvConstraint(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config) const {

  std::ostringstream tokenLimitConstraint;

  auto [inputSequenceName, _] = config.arguments[inputSequence];

  auto [lhsInputNDWire, rhsInputNDWire] = config.inputNDWires[inputSequence];
  auto [lhsOutputNDWire, rhsOutputNDWire] =
      config.outputNDWires[outputSequence];

  std::string tokenLimitDef = inputSequenceName + "_active_token_limit";

  // The define for the actual active token limit
  tokenLimitConstraint << "DEFINE " << tokenLimitDef << " := " << limit
                       << ";\n";

  // We need to limit the tokens in the LHS and RHS seperately
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

    // The condition which signifies a valid token exchange at the output of the
    // input ND wire.
    std::string inputTokenCondition =
        inputNDWireName + ".valid0 & " + inputNDWireName + ".nReady0";

    // The condition which signifies a valid token exchange at the output of the
    // output ND wire.
    std::string outputTokenCondition =
        outputNDWireName + ".valid0 & " + outputNDWireName + ".nReady0";

    // Active token counter:
    // When there is a token at the input and output we
    // keep the count the same.
    // When there is a token only at the input we increase the active counter.
    // When there is a token only at the output we decrease the active counter.
    // Example:
    // VAR rhs_D_active_tokens : 0..1;
    // ASSIGN
    // init(rhs_D_active_tokens) := 0;
    // next(rhs_D_active_tokens) := case
    //   model.rhs_in_ndw_D.valid0 & model.rhs_in_ndw_D.nReady0 &
    //   model.rhs_out_ndw_F.valid0 & model.rhs_out_ndw_F.nReady0 :
    //   rhs_D_active_tokens; model.rhs_out_ndw_F.valid0 &
    //   model.rhs_out_ndw_F.nReady0 & rhs_D_active_tokens > 0 :
    //   rhs_D_active_tokens - 1; model.rhs_in_ndw_D.valid0 &
    //   model.rhs_in_ndw_D.nReady0 & rhs_D_active_tokens < D_active_token_limit
    //   : rhs_D_active_tokens + 1; TRUE : rhs_D_active_tokens;
    // esac;
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
                         << "esac;\n";

    // Token limiter invar:
    // When we reached the active token limit, the input ND wire is forced to
    // sleep (not pass any tokens). Once a token reaches the output it can run
    // again.
    // Example:
    // INVAR rhs_D_active_tokens = D_active_token_limit ->
    // model.rhs_in_ndw_D.state = sleeping;
    tokenLimitConstraint << "INVAR " << limiterVarName << " = " << tokenLimitDef
                         << " -> " << inputNDWireName
                         << ".state = sleeping;\n\n";
  }

  return tokenLimitConstraint.str();
}
} // namespace dynamatic::experimental
