#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <unordered_set>

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "Constraints.h"
#include "ElasticMiterTestbench.h"
#include "FabricGeneration.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace dynamatic::experimental {

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
      properties << llvm::formatv("INVARSPEC ({0}.{1}_valid -> {0}.{1}_out)",
                                  moduleName, resultName)
                        .str();
  }

  // Make sure the input buffers will have pairwise the same number of tokens.
  // This means both circuits consume the same number of tokens.
  SmallVector<std::string> bufferProperties;
  for (const auto &[lhsBuffer, rhsBuffer] : config.inputBuffers) {
    bufferProperties.push_back(llvm::formatv("({0}.{1}.num = {0}.{2}.num)",
                                             moduleName, lhsBuffer, rhsBuffer)
                                   .str());
  }

  // Make sure the output buffers will be empty.
  // This means both circuits produce the same number of output tokens.
  for (const auto &[lhsBuffer, rhsBuffer] : config.outputBuffers) {
    bufferProperties.push_back(
        llvm::formatv("({0}.{1}.num = 0)", moduleName, lhsBuffer).str());
    bufferProperties.push_back(
        llvm::formatv("({0}.{1}.num = 0)", moduleName, rhsBuffer).str());
  }

  // Make sure the buffer property will start to hold at one point and from
  // there on. The final property will be (Assuming inputs D and C, and outputs
  // A and B): CTLSPEC AF (AG ((model.lhs_in_buf_D.num = model.rhs_in_buf_D.num)
  // & (model.lhs_in_buf_C.num = model.rhs_in_buf_C.num) &
  // (model.lhs_out_buf_A.num = 0) & (model.rhs_out_buf_A.num = 0) &
  // (model.lhs_out_buf_B.num = 0) & (model.rhs_out_buf_B.num = 0)))
  properties << llvm::formatv("CTLSPEC AF (AG ({0}))\n\n",
                              join(bufferProperties, " & "))
                    .str();

  return properties.str();
}

std::string createElasticMiterTestBench(
    MLIRContext &context, const ElasticMiterConfig &config,
    const std::string &modelSmvName, size_t nrOfTokens, bool includeProperties,
    const std::optional<
        SmallVector<dynamatic::experimental::ElasticMiterConstraint *>>
        &sequenceConstraints,
    bool generateExactNrOfTokens) {
  std::ostringstream wrapper;

  // replace all arguments' type with boolean
  SmallVector<std::pair<std::string, Type>> arguments;
  Type i1Type = IntegerType::get(&context, 1);
  for (auto [argName, _] : config.arguments) {
    auto newArg = std::make_pair(argName, ChannelType::get(i1Type, {}));
    arguments.push_back(newArg);
  }

  const SmvTestbenchConfig smvConfig = {.arguments = arguments,
                                        .results = config.results,
                                        .modelSmvName = modelSmvName,
                                        .nrOfTokens = nrOfTokens,
                                        .generateExactNrOfTokens =
                                            generateExactNrOfTokens,
                                        .syncOutput = false};

  wrapper << createSmvFormalTestbench(smvConfig);
  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, config);

    if (sequenceConstraints) {
      for (const auto &constraint : *sequenceConstraints) {
        wrapper << constraint->createSmvConstraint(modelSmvName, config);
      }
    }
  }
  return wrapper.str();
}

LogicalResult createSmvSequenceLengthTestbench(
    MLIRContext &context, const std::filesystem::path &wrapperPath,
    const ElasticMiterConfig &config, const std::string &modelSmvName,
    size_t nrOfTokens) {

  // Call the function to generate a general testbench. We do not need to
  // include properties nor sequence constraints.
  std::string wrapper = createElasticMiterTestBench(
      context, config, modelSmvName, nrOfTokens, false, std::nullopt, true);
  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper;
  mainFile.close();
  return success();
}
} // namespace dynamatic::experimental