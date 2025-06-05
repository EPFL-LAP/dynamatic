#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "experimental/Support/CreateSmvFormalTestbench.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace dynamatic::experimental {

/// Create the call to the module
/// The resulting string will look like:
/// VAR <moduleName> : <moduleName> (seq_generator_A.outs,
/// seq_generator_A.outs_valid, ..., sink_F.ins_ready, ...)
static std::string instantiateModuleUnderTest(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, mlir::Type>> &arguments,
    const SmallVector<std::pair<std::string, mlir::Type>> &results,
    bool syncOutput = false) {
  SmallVector<std::string> inputVariables;
  for (const auto &argument : arguments) {
    // The current handshake2smv conversion also creates a dataOut port when it
    // is of type control
    auto argumentName = argument.first;
    auto argumentType = argument.second;

    llvm::TypeSwitch<Type>(argumentType)
        .Case<handshake::ChannelType>([&](auto) {
          inputVariables.push_back(
              llvm::formatv("seq_generator_{0}.{1}", argumentName,
                            SEQUENCE_GENERATOR_DATA_NAME.str()));
          inputVariables.push_back(
              llvm::formatv("seq_generator_{0}.{1}", argumentName,
                            SEQUENCE_GENERATOR_VALID_NAME.str()));
        })
        .Case<handshake::ControlType>([&](auto) {
          // TODO: remove this if statement when updating the elastic-miter to
          // the new SMV backend.
          // This is a hack: we use syncOutput as a proxy to
          // know if we are using the old dot2smv (in elastic-miter syncOutput
          // is always false) or the new backend (in rigidification syncOutput
          // is always true). This hack can be fixed as soon as elastic-miter is
          // updated.
          if (syncOutput) {
            inputVariables.push_back(
                llvm::formatv("seq_generator_{0}.{1}", argumentName,
                              SEQUENCE_GENERATOR_VALID_NAME.str()));
          } else {
            inputVariables.push_back(
                llvm::formatv("seq_generator_{0}.{1}", argumentName,
                              SEQUENCE_GENERATOR_DATA_NAME.str()));
            inputVariables.push_back(
                llvm::formatv("seq_generator_{0}.{1}", argumentName,
                              SEQUENCE_GENERATOR_VALID_NAME.str()));
          }
        })
        .Case<IntegerType>([&](IntegerType intType) {
          // This is the case for data coming from memory (it has no handshake
          // signals)
          if (argumentName != "clk" && argumentName != "rst") {
            inputVariables.push_back(
                llvm::formatv("0ud{0}_0", intType.getWidth()));
          }
        })
        .Default([](Type) {});
  }

  if (syncOutput)
    for (const auto &[i, result] : llvm::enumerate(results)) {
      const auto &[resultName, type] = result;
      if (type.isa<handshake::ControlType, handshake::ChannelType>())
        inputVariables.push_back(llvm::formatv("join_global.ins_{0}_ready", i));
    }
  else
    for (const auto &[resultName, type] : results) {
      if (type.isa<handshake::ControlType, handshake::ChannelType>())
        inputVariables.push_back(
            llvm::formatv("sink_{0}.{1}", resultName, SINK_READY_NAME.str()));
    }

  return llvm::formatv("VAR {0} : {1} ({2});\n", moduleName, moduleName,
                       llvm::join(inputVariables, ", "))
      .str();
}

std::optional<std::string> getPrefixTypeName(const Type &type) {
  return llvm::TypeSwitch<Type, std::optional<std::string>>(type)
      .Case<handshake::ControlType>(
          [&](handshake::ControlType cType) { return std::nullopt; })
      .Case<handshake::ChannelType>([&](handshake::ChannelType cType) {
        if (cType.getDataBitWidth() == 1)
          return std::string("bool");
        return "s" + std::to_string(cType.getDataBitWidth());
      });
}

static std::optional<std::string> convertMLIRTypeToSMV(const Type &type) {
  return llvm::TypeSwitch<Type, std::optional<std::string>>(type)
      .Case<handshake::ControlType>(
          [&](handshake::ControlType cType) { return std::nullopt; })
      .Case<handshake::ChannelType>([&](handshake::ChannelType cType) {
        if (cType.getDataBitWidth() == 1)
          return std::string("boolean");
        return llvm::formatv("signed word [{0}]", cType.getDataBitWidth())
            .str();
      });
}

/// SMV module for a sequence generator with a finite number of tokens. The
/// actual number of generated tokens is non-determinstically set between 0
/// and (inclusive) max_tokens.
std::string smvInput(const Type &type) {
  return llvm::formatv(R"DELIM(
MODULE {0}_input(nReady0, max_tokens)"
  VAR outs : {1};
  VAR counter : 0..31;
  FROZENVAR exact_tokens : 0..max_tokens;
  ASSIGN
    init(counter) := 0;
    next(counter) := case
      nReady0 & counter < exact_tokens : counter + 1;
      TRUE : counter;
    esac;

  DEFINE outs_valid := counter < exact_tokens;

)DELIM",
                       *getPrefixTypeName(type), *convertMLIRTypeToSMV(type));
}

/// SMV module for a sequence generator with a finite number of tokens. The
/// number of generated tokens is exact_tokens.
std::string smvInputExact(const Type &type) {
  return llvm::formatv(R"DELIM(
MODULE {0}_input_exact(nReady0, exact_tokens)"
  VAR outs : {1};
  VAR counter : 0..31;
  ASSIGN
    init(counter) := 0;
    next(counter) := case
      nReady0 & counter < exact_tokens : counter + 1;
      TRUE : counter;
    esac;

  DEFINE outs_valid := counter < exact_tokens;

)DELIM",
                       *getPrefixTypeName(type), *convertMLIRTypeToSMV(type));
}

/// SMV module for a sequence generator with an infinite number of tokens
std::string smvInputInf(const Type &type) {
  return llvm::formatv(R"DELIM(
MODULE {0}_input_inf(nReady0)"
  VAR outs : {1};

  -- make sure outs is persistent
  DEFINE outs_valid := TRUE;

)DELIM",
                       *getPrefixTypeName(type), *convertMLIRTypeToSMV(type));
}

static std::string
createSequenceGenerator(const Type &type, size_t nrOfTokens,
                        bool generateExactNrOfTokens = false) {
  return llvm::TypeSwitch<Type, std::string>(type)
      .Case<handshake::ControlType>([&](auto) {
        if (nrOfTokens == 0)
          return SMV_CTRL_INPUT_INF;
        if (generateExactNrOfTokens)
          return SMV_CTRL_INPUT_EXACT;
        return SMV_CTRL_INPUT;
      })
      .Case<handshake::ChannelType>([&](handshake::ChannelType type) {
        if (type.getDataBitWidth() == 1) {
          if (nrOfTokens == 0)
            return SMV_BOOL_INPUT_INF;
          if (generateExactNrOfTokens)
            return SMV_BOOL_INPUT_EXACT;
          return SMV_BOOL_INPUT;
        }
        {
          if (nrOfTokens == 0)
            return smvInputInf(type);
          if (generateExactNrOfTokens)
            return smvInputExact(type);
          return smvInput(type);
        }
      });
}

static std::string createTBJoin(size_t nrOfOutputs) {
  std::ostringstream tbJoin;
  std::vector<std::string> insValids;
  for (size_t i = 0; i < nrOfOutputs; i++)
    insValids.push_back(llvm::formatv("ins_{0}_valid", i));

  tbJoin << llvm::formatv("MODULE tb_join ({0}, outs_ready)\n",
                          llvm::join(insValids, ", "))
                .str();

  tbJoin << llvm::formatv("  DEFINE\n  outs_valid := {0};\n",
                          llvm::join(insValids, " & "))
                .str();

  for (size_t i = 0; i < nrOfOutputs; i++) {
    std::vector<std::string> tmp = insValids;
    tmp.erase(tmp.begin() + i);
    tbJoin << llvm::formatv("  ins_{0}_ready := {1} & outs_ready;\n", i,
                            llvm::join(tmp, " & "))
                  .str();
  }
  tbJoin << "\n\n";
  return tbJoin.str();
}

static std::string createSupportEntities(
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const SmallVector<std::pair<std::string, Type>> &results, size_t nrOfTokens,
    bool generateExactNrOfTokens = false, bool syncOutput = false) {

  llvm::DenseSet<Type> types;
  for (const auto &[_, type] : arguments)
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      types.insert(type);
  std::ostringstream supportEntities;

  for (const auto &type : types) {
    supportEntities << createSequenceGenerator(type, nrOfTokens,
                                               generateExactNrOfTokens)
                    << "\n\n";
  }
  if (syncOutput) {
    int nrOutChannels = 0;
    for (const auto &[s, t] : results) {
      if (isa<handshake::ChannelType, handshake::ControlType>(t)) {
        nrOutChannels++;
      }
    }

    supportEntities << createTBJoin(nrOutChannels);
  } else
    supportEntities << "MODULE sink_main (ins_valid)\n"
                       "  DEFINE ins_ready   := TRUE;\n\n";

  return supportEntities.str();
}

/// Create the sequence generator at the inputs of the module
/// We support three different kinds of sequence generators:
/// 1. Infinite sequence generator: Will create an infinite number of
/// tokens.
/// 2. Standard finite generator: Will create 0 to the maximal number of
/// tokens. The exact number of tokens is non-deterministic.
/// 3. Exact finite generator: Will create the exact number of tokens (if
/// it receives enough ready inputs).
/// When nrOfTokens is set to 0, the infinite sequence generator is created
/// and the value of generateExactNrOfTokens is ignored.
static std::string instantiateSequenceGenerators(
    const std::string &moduleName,
    const SmallVector<std::pair<std::string, Type>> &arguments,
    size_t nrOfTokens, bool generateExactNrOfTokens = false) {
  std::ostringstream sequenceGenerators;
  for (const auto &[argumentName, type] : arguments) {
    if (!type.isa<handshake::ControlType, handshake::ChannelType>())
      continue;

    std::string typePrefixName =
        llvm::TypeSwitch<Type, std::string>(type)
            .Case<handshake::ControlType>([&](handshake::ControlType cType) {
              return std::string("ctrl");
            })
            .Case<handshake::ChannelType>([&](handshake::ChannelType cType) {
              if (cType.getDataBitWidth() == 1)
                return std::string("bool");
              return "s" + std::to_string(cType.getDataBitWidth());
            });

    if (nrOfTokens == 0) {
      // Example: VAR seq_generator_D : bool_input_inf(model.D_ready);
      sequenceGenerators
          << llvm::formatv(
                 "VAR seq_generator_{0} : {1}_input_inf({2}.{0}_ready);\n",
                 argumentName, typePrefixName, moduleName)
                 .str();

    } else if (generateExactNrOfTokens) {
      // Example: VAR seq_generator_D : bool_input_exact(model.D_ready, 1);
      sequenceGenerators << llvm::formatv(
                                "VAR seq_generator_{0} : "
                                "{1}_input_exact({2}.{0}_ready, {3});\n",
                                argumentName, typePrefixName, moduleName,
                                nrOfTokens)
                                .str();
    } else {
      // Example: VAR seq_generator_D : bool_input(model.D_ready, 1);
      sequenceGenerators << llvm::formatv("VAR seq_generator_{0} : "
                                          "{1}_input({2}.{0}_ready, {3});\n",
                                          argumentName, typePrefixName,
                                          moduleName, nrOfTokens)
                                .str();
    }
  }
  return sequenceGenerators.str();
}

/// Create the sinks at the outputs of the module
static std::string
instantiateSinks(const std::string &moduleName,
                 const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream sinks;

  for (const auto &[resultName, type] : results) {
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      sinks << llvm::formatv("  VAR sink_{0} : sink_main({1}.{0}_valid);\n",
                             resultName, moduleName)
                   .str();
  }
  return sinks.str();
}

/// Create the join at the outputs of the module
static std::string
instantiateJoin(const std::string &moduleName,
                const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream str;
  std::vector<std::string> outputValids;

  str << "  VAR join_global : tb_join(";
  for (const auto &[resultName, type] : results) {
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      outputValids.push_back(
          llvm::formatv("{0}.{1}_valid", moduleName, resultName));
  }
  str << llvm::join(outputValids, ", ") << ", global_ready);\n";

  return str.str();
}

std::string createSmvFormalTestbench(const SmvTestbenchConfig &config) {
  std::ostringstream wrapper;
  wrapper << "#include \"" + config.modelSmvName + ".smv\"\n\n";

  wrapper << createSupportEntities(
      config.arguments, config.results, config.nrOfTokens,
      config.generateExactNrOfTokens, config.syncOutput);

  wrapper << "MODULE main\n\n";

  wrapper << instantiateSequenceGenerators(config.modelSmvName,
                                           config.arguments, config.nrOfTokens,
                                           config.generateExactNrOfTokens);

  wrapper << instantiateModuleUnderTest(config.modelSmvName, config.arguments,
                                        config.results, config.syncOutput)
          << "\n";

  if (config.syncOutput) {

    wrapper << "  DEFINE global_ready := TRUE;\n";

    wrapper << instantiateJoin(config.modelSmvName, config.results) << "\n";
  } else {
    wrapper << instantiateSinks(config.modelSmvName, config.results) << "\n";
  }

  return wrapper.str();
}

} // namespace dynamatic::experimental