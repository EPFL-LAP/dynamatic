#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "Constraints.h"
#include "CreateSmvFormalTestbench.h"
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
    size_t nrOfTokens, bool generateExactNrOfTokens = false) {
  std::ostringstream sequenceGenerators;
  for (const auto &[argumentName, _] : arguments) {
    // We support three different kinds of sequence generators:
    // 1. Infinite sequence generator: Will create an infinite number of tokens.
    // 2. Standard finite generator: Will create 0 to the maximal number of
    //    tokens. The exact number of tokens is non-deterministic.
    // 3. Exact finite generator: Will create the exact number of tokens (if it
    //    receives enough ready inputs).
    // When nrOfTokens is set to 0, the infinite sequence generator is created
    // and the value of generateExactNrOfTokens is ignored.
    if (nrOfTokens == 0) {
      // Example: VAR seq_generator_D : bool_input_inf(model.D_ready);
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : bool_input_inf(" << moduleName << "."
                         << argumentName << "_ready);\n";
    } else if (generateExactNrOfTokens) {
      // Example: VAR seq_generator_D : bool_input_exact(model.D_ready, 1);
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : bool_input_exact(" << moduleName << "."
                         << argumentName << "_ready, " << nrOfTokens << ");\n";
    } else {
      // Example: VAR seq_generator_D : bool_input(model.D_ready, 1);
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

  // Make sure the buffer property will start to hold at one point and from
  // there on. The final property will be (Assuming inputs D and C, and outputs
  // A and B): CTLSPEC AF (AG ((model.lhs_in_buf_D.num = model.rhs_in_buf_D.num)
  // & (model.lhs_in_buf_C.num = model.rhs_in_buf_C.num) &
  // (model.lhs_out_buf_A.num = 0) & (model.rhs_out_buf_A.num = 0) &
  // (model.lhs_out_buf_B.num = 0) & (model.rhs_out_buf_B.num = 0)))
  std::string finalBufferProp =
      "AF (AG (" + join(bufferProperties, " & ") + "))";
  properties << "CTLSPEC " + finalBufferProp + "\n";

  properties << "\n";

  return properties.str();
}

LogicalResult createSmvFormalTestbench(
    const std::filesystem::path &wrapperPath, const ElasticMiterConfig &config,
    const std::string &modelSmvName, size_t nrOfTokens, bool includeProperties,
    const std::optional<
        SmallVector<dynamatic::experimental::ElasticMiterConstraint *>>
        &sequenceConstraints,
    bool generateExactNrOfTokens) {

  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + ".smv\"\n";
  wrapper << SMV_BOOL_INPUT;
  wrapper << SMV_BOOL_INPUT_EXACT;
  wrapper << SMV_BOOL_INPUT_INF;

  wrapper << "MODULE main\n";

  wrapper << "\n";

  wrapper << createSequenceGenerators(modelSmvName, config.arguments,
                                      nrOfTokens, generateExactNrOfTokens);

  wrapper << instantiateModuleUnderTest(modelSmvName, config.arguments,
                                        config.results)
          << "\n";

  wrapper << createSinks(modelSmvName, config.results, nrOfTokens) << "\n";

  if (includeProperties) {
    wrapper << createMiterProperties(modelSmvName, config);

    if (sequenceConstraints) {
      for (const auto &constraint : *sequenceConstraints) {
        wrapper << constraint->createSmvConstraint(modelSmvName, config);
      }
    }
  }

  std::ofstream mainFile(wrapperPath);
  mainFile << wrapper.str();
  mainFile.close();

  return success();
}

LogicalResult createSmvSequenceLengthTestbench(
    const std::filesystem::path &wrapperPath, const ElasticMiterConfig &config,
    const std::string &modelSmvName, size_t nrOfTokens) {

  // Call the function to generate a general testbench. We do not need to
  // include properties nor sequence constraints.
  return createSmvFormalTestbench(wrapperPath, config, modelSmvName, nrOfTokens,
                                  false, std::nullopt, true);
}
} // namespace dynamatic::experimental