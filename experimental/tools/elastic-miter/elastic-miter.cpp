//===- elastic-miter.cpp - The elastic-miter driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the elastic-miter tool, it takes two MLIR circuits in
// the Handshake dialect and formally verifies their equivalence.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include <filesystem>
#include <fstream>
#include <string>

#include "dynamatic/InitAllDialects.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "Constraints.h"
#include "ElasticMiterTestbench.h"
#include "FabricGeneration.h"
#include "GetSequenceLength.h"
#include "SmvUtils.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic::handshake;

// CLI Settings

static cl::OptionCategory generalCategory("1. General Elastic-Miter Options");
static cl::OptionCategory constraintsCategory("2. Constraints Options");

static cl::opt<std::string> lhsFilenameArg(
    "lhs", cl::Prefix, cl::Required,
    cl::desc("The left-hand side (LHS) input handshake MLIR file"),
    cl::cat(generalCategory));

static cl::opt<std::string> rhsFilenameArg(
    "rhs", cl::Prefix, cl::Required,
    cl::desc("The right-hand side (RHS) input handshake MLIR file"),
    cl::cat(generalCategory));

static cl::opt<std::string> outputDirArg("o", cl::Prefix, cl::Required,
                                         cl::desc("Specify output directory"),
                                         cl::cat(generalCategory));

// Specify a Sequence Length Relation constraint.
// Can be used multiple times. E.g.: --seq_length="0+1=2" --seq_length="1<2"
// It controls the relative length of the input sequences.
// The constraint has the form of an arithmetic equation. The number in the
// equation will be replaced the respective input with the index of the number.
// Example:
// --seq_length="0+1=2" will ensure that the inputs with index 0 and index 1
// together produce as many tokens as the input with index 2.
static cl::list<std::string> seqLengthRelationConstraints(
    "seq_length", cl::Prefix,
    cl::desc("Specify constraints for the relation of sequence lengths."),
    cl::cat(constraintsCategory));

// Specify a Loop Condition sequence contraint.
// Can be used multiple times. E.g.: --loop="0,1" --loop="2,3"
// It has the form "<dataSequence>,<controlSequence>".
// The number of tokens in the input with the index dataSequence is equivalent
// to the number of false tokens at the output with the index controlSequence.
// Example:
// --loop="0,1"
static cl::list<std::string>
    loopSeqConstraints("loop", cl::Prefix,
                       cl::desc("Specify loop constraints."),
                       cl::cat(constraintsCategory));

// Specify a Strict Loop Condition sequence contraint.
// Can be used multiple times. E.g.: --loop_strict="0,1" --loop_strict="2,3"
// Works identically to the loop condition sequence contraint, with the
// addition that the last token also needs to be false.
// Example:
// --loop_strict="0,1"
static cl::list<std::string> strictLoopSeqConstraints(
    "loop_strict", cl::Prefix,
    cl::desc("Specify loop constraints, where the last token is false."),
    cl::cat(constraintsCategory));

// Specify a Token Limit constraint.
// Can be used multiple times. E.g. --token_limit="0,0,1" --token_limit="1,2,2"
// It has the form "<inputSequence>,<outputSequence>,<limit>".
// At any point in time, the number of tokens which are created at the input
// with index inputSequence can only be up to "limit" higher than the number of
// tokens reaching the output with the index outputSequence.
// Example:
// --token_limit="0,0,1"
static cl::list<std::string>
    tokenLimitConstraints("token_limit", cl::Prefix,
                          cl::desc("Specify token limit constraint."),
                          cl::cat(constraintsCategory));

//  Enable counterexamples. In case a property does not pass, this will generate
//  a counterexample and put it in an XML file.
static cl::opt<bool>
    enableCounterExamples("cex",
                          cl::desc("Enable counter examples and create XML "
                                   "files in the output directory."),
                          cl::init(false), cl::cat(generalCategory));

static FailureOr<SmallVector<dynamatic::experimental::ElasticMiterConstraint *>>
parseSequenceConstraints() {

  SmallVector<dynamatic::experimental::ElasticMiterConstraint *> constraints;

  // Parse the sequence length relation constraints. They are string in the
  // style "0+1+..=4+5+..", where the numbers represent the index of the
  // sequence
  for (const auto &constraint : seqLengthRelationConstraints)
    constraints.push_back(
        new dynamatic::experimental::SequenceLengthRelationConstraint(
            constraint));

  // A Loop Condition sequence contraint has the form
  // "<dataSequence>,<controlSequence>". The number of tokens in the input with
  // the index dataSequence is equivalent to the number of false tokens at the
  // output with the index controlSequence.
  // Example:
  // --loop="0,1"
  for (const auto &csv : loopSeqConstraints)
    constraints.push_back(new dynamatic::experimental::LoopConstraint(csv));

  // Identical to the loop condition sequence contraint, with the addition that
  // the last token also needs to be false.
  for (const auto &csv : strictLoopSeqConstraints)
    constraints.push_back(
        new dynamatic::experimental::StrictLoopConstraint(csv));

  // A Token Limit constraint has the form
  // "<inputSequence>,<outputSequence>,<limit>".
  // At any point in time, the number of tokens which are created at the input
  // with index inputSequence can only be up to "limit" higher than the number
  // of tokens reaching the output with the index outputSequence.
  // Example:
  // `--token_limit="1,1,2"` ensures that there are only two tokens in the
  // circuit which enter at the input with index 1 and leave at the ouput with
  // index 1.
  for (const auto &csv : tokenLimitConstraints)
    constraints.push_back(
        new dynamatic::experimental::TokenLimitConstraint(csv));

  return constraints;
}

static FailureOr<bool> checkEquivalence(
    MLIRContext &context, const std::filesystem::path &lhsPath,
    const std::filesystem::path &rhsPath,
    const std::filesystem::path &outputDir,
    const SmallVector<dynamatic::experimental::ElasticMiterConstraint *>
        &constraints) {

  // Find out needed number of tokens for the LHS
  auto failOrLHSseqLen = dynamatic::experimental::getSequenceLength(
      context, outputDir / "lhs_reachability", lhsPath);
  if (failed(failOrLHSseqLen))
    return failure();

  // Find out needed number of tokens for the RHS
  auto failOrRHSseqLen = dynamatic::experimental::getSequenceLength(
      context, outputDir / "rhs_reachability", rhsPath);
  if (failed(failOrRHSseqLen))
    return failure();

  size_t nrOfTokens =
      std::max(failOrLHSseqLen.value(), failOrRHSseqLen.value());

  std::filesystem::path miterDir = outputDir / "miter";
  // Create the miterDir if it doesn't exist
  std::filesystem::create_directories(miterDir);

  // Create an elastic-miter circuit with a needed nrOfTokens to emulate an
  // infinite number of tokens
  auto failOrPair = dynamatic::experimental::createMiterFabric(
      context, lhsPath, rhsPath, miterDir.string(), nrOfTokens);
  if (failed(failOrPair)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return failure();
  }
  auto [mlirPath, config] = failOrPair.value();

  // Convert the MLIR circuit to SMV
  auto failOrSmvPair =
      dynamatic::experimental::handshake2smv(mlirPath, miterDir, true);
  if (failed(failOrSmvPair)) {
    llvm::errs() << "Failed to convert miter module to SMV.\n";
    return failure();
  }
  auto [smvPath, smvModelName] = failOrSmvPair.value();

  std::filesystem::path wrapperPath = miterDir / "main.smv";

  // Create wrapper (main) for the elastic-miter

  std::string testbench = dynamatic::experimental::createElasticMiterTestBench(
      context, config, smvModelName, nrOfTokens, true, constraints);
  std::ofstream mainFile(wrapperPath);
  mainFile << testbench;
  mainFile.close();

  // Put the output of the CTLSPEC check into results.txt. Later we
  // read from that file to check whether all the properties pass.
  std::filesystem::path resultTxtPath = miterDir / "result.txt";
  std::string miterCommand = "check_invar -s forward;\n"
                             "check_ctlspec;\n";
  LogicalResult cmdFail = dynamatic::experimental::createCMDfile(
      miterDir / "prove.cmd", miterDir / "main.smv", miterCommand,
      enableCounterExamples);
  if (failed(cmdFail))
    return failure();

  // Run equivalence checking by calling NuSMV or nuXmv
  dynamatic::experimental::runSmvCmd(miterDir / "prove.cmd", resultTxtPath);

  bool isEquivalent = true;
  std::string line;
  std::ifstream result(resultTxtPath);
  // Iterate through the lines to check if a property has failed.
  while (getline(result, line)) {
    if (line.find("is false") != std::string::npos) {
      isEquivalent = false;
    }
  }
  result.close();
  return isEquivalent;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide the general options that are irrelevent for elastic-miter
  cl::HideUnrelatedOptions({&generalCategory, &constraintsCategory});

  cl::ParseCommandLineOptions(
      argc, argv,
      "Checks the equivalence of two dynamic circuits in the handshake "
      "dialect. At the end it will output whether the circuits are "
      "latency-insensitive equivalent. "
      "Takes two MLIR files as input. The files need to contain exactely "
      "one module each. Each module needs to contain exactely one "
      "handshake.func. "
      "The resulting miter MLIR file and JSON config file are placed in "
      "the specified output directory.\n"
      "Usage Example:\n"
      "elastic-miter --lhs=b_lhs.mlir --rhs=b_rhs.mlir -o out "
      "--seq_length=\"0+1=3\" --seq_length=\"0=2\" "
      "--loop_strict=0,1\n");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  // SmallVector first
  // Create sequence constraint struct. cl::list needs to be converted to
  auto failOrSequenceConstraints = parseSequenceConstraints();
  if (failed(failOrSequenceConstraints)) {
    llvm::errs() << "Failed to parse constraints\n";
    return 1;
  }

  std::filesystem::path lhsPath = lhsFilenameArg.getValue();
  std::filesystem::path rhsPath = rhsFilenameArg.getValue();
  std::filesystem::path outputDir = outputDirArg.getValue();

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(outputDir);

  auto failOrEquivalent = checkEquivalence(context, lhsPath, rhsPath, outputDir,
                                           failOrSequenceConstraints.value());
  if (failed(failOrEquivalent)) {
    llvm::errs() << "Equivalence checking failed.\n";
    return 1;
  }
  bool equivalent = failOrEquivalent.value();
  if (equivalent)
    llvm::outs() << lhsPath.filename() << " <> " << rhsPath.filename()
                 << ": EQUIVALENT.\n";
  else
    llvm::outs() << lhsPath.filename() << " <> " << rhsPath.filename()
                 << ": NOT EQUIVALENT.\n";

  exit(!equivalent);
}