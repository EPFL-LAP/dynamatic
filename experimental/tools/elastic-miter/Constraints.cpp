#include <cstddef>
#include <regex>
#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
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

  std::string output = "INVAR ";
  output += constraint + ";\n";

  // Replace all the occurences of the sequence with index i with its
  // corresponding sequence generator.
  // Example:
  // Input names: [A, B, C, D].
  // Constraint: 0+1=2+3
  // Resulting SMV constraint:
  // seq_generator_A.exact_tokens + seq_generator_B.exact_tokens =
  // seq_generator_C.exact_tokens + seq_generator_D.exact_tokens

  // Replace 0 with {0}, 1 with {1}, ...
  // This is because the number might be present in the argument names.
  // Argument names cannot use { and }, preventing a conflict.
  output = std::regex_replace(output, std::regex(R"(\d+)"), "{$&}");

  for (size_t i = 0; i < config.arguments.size(); i++) {
    auto [argumentName, _] = config.arguments[i];
    std::regex numberRegex("\\{" + std::to_string(i) + "\\}");

    output =
        std::regex_replace(output, numberRegex,
                           " seq_generator_" + argumentName + ".exact_tokens ");
  }

  return output;
}

std::string SequenceLengthEnhancedRelationConstraint::createSmvConstraint(
    const std::string &moduleName,
    const dynamatic::experimental::ElasticMiterConfig &config) const {

  std::string output = "INVAR ";
  output += constraint + ";\n";

  for (size_t i = 0; i < config.arguments.size(); i++) {
    auto [argumentName, _] = config.arguments[i];
    std::regex numberRegex("\\{in:" + std::to_string(i) + "\\}");

    output =
        std::regex_replace(output, numberRegex,
                           " seq_generator_" + argumentName + ".exact_tokens ");
  }
  for (size_t i = 0; i < config.results.size(); i++) {
    auto [eqResultName, _] = config.results[i];
    StringRef resultName(eqResultName);
    resultName = resultName.substr(3); // Remove EQ_
    std::regex numberRegex("\\{out:" + std::to_string(i) + "\\}");

    output = std::regex_replace(output, numberRegex,
                                " " + config.funcName + ".out_nds_" +
                                    resultName.str() + ".exact_tokens ");
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

  std::string falseTokenCounterString;
  std::string forceFalseTokenInvar;
  std::string limitFalseTokensInvar;
  std::string seqLengthRelationInvar;
  std::string emptyDataSequenceInvar;

  // Create a counter, which counts the total number of false token at the
  // control sequence input.
  falseTokenCounterString = llvm::formatv(
      R"DELIM(VAR {0} : 0..31;
ASSIGN init({0}) := 0;
ASSIGN next({0}) := case
  {1}.outs_valid & {1}.nReady0 & ({1}.outs = FALSE) & ({0} < {2}.exact_tokens) : ({0} + 1);
  TRUE : {0};
esac;
)DELIM",
      falseCounterTokenCounterVariable, controlSeqGeneratorName,
      dataSeqGeneratorName);

  // Make sure enough false tokens are generated. When there are as many false
  // tokens that still need to be generated, as there are total control tokens
  // that can be generated, every token needs to be false.
  forceFalseTokenInvar = llvm::formatv(
      R"DELIM(INVAR ((({0}.exact_tokens - {1}) = ({2}.exact_tokens - {2}.counter)) &
  (({0}.exact_tokens - {1}) >= 1)) -> {2}.outs = FALSE;
)DELIM",
      dataSeqGeneratorName, falseCounterTokenCounterVariable,
      controlSeqGeneratorName);

  // Make sure that not to many false tokens are generated. If the last tokens
  // needs to be false, this means we only generate false tokens as long as we
  // still have one false token to spare. If the last token can be freely
  // chosen, we only create false token as long as we haven't reached the total
  // number of tokens.
  // Example (last True or False): INVAR
  // (((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) = 0 ) &
  // ((seq_generator_Cd.exact_tokens - seq_generator_Cd.counter) >= 1)) ->
  // (seq_generator_Cd.outs = TRUE);
  // Example (last False): INVAR
  // (((seq_generator_Dd.exact_tokens - Cd_false_token_cnt) = 1 ) &
  // ((seq_generator_Cd.exact_tokens - seq_generator_Cd.counter) >= 2)) ->
  // (seq_generator_Cd.outs = TRUE);
  limitFalseTokensInvar = llvm::formatv(
      R"DELIM(INVAR ((({0}.exact_tokens - {1}) = {2}) &
(({3}.exact_tokens - {3}.counter) >= {4})) ->
({3}.outs = TRUE);
)DELIM",
      dataSeqGeneratorName, falseCounterTokenCounterVariable,
      (unsigned)lastFalse, controlSeqGeneratorName, 1 + lastFalse);

  // Make sure the control sequence generates at least as many tokens as the
  // data sequence. Otherwise it will be impossible to have the same number of
  // control false tokens as total data tokens.
  seqLengthRelationInvar = llvm::formatv(
      R"DELIM(INVAR ({0}.exact_tokens >= {1}.exact_tokens);
)DELIM",
      controlSeqGeneratorName, dataSeqGeneratorName);

  // When there are no data tokens, there should also be no control tokens.
  // Otherwise the control token cannot be consumed.
  emptyDataSequenceInvar = llvm::formatv(
      R"DELIM(INVAR ({0}.exact_tokens = 0) -> ({1}.exact_tokens = 0);
)DELIM",
      dataSeqGeneratorName, controlSeqGeneratorName);

  return falseTokenCounterString + forceFalseTokenInvar +
         limitFalseTokensInvar + seqLengthRelationInvar +
         emptyDataSequenceInvar;
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
    std::string activeTokenCounterString =
        llvm::formatv(R"DELIM(VAR {0} : 0..{1}
ASSIGN
init({0}) := 0;
next({0}) := case
  {2} & {3} : {0};
  {3} & {0} > 0 : {0} - 1;
  {2} & {0} < {4} : {0} + 1;
  TRUE : {0};
esac;
)DELIM",
                      limiterVarName, limit, inputTokenCondition,
                      outputTokenCondition, tokenLimitDef);
    tokenLimitConstraint << activeTokenCounterString;

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
