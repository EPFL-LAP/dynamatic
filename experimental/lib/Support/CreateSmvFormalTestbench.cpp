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
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

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
          inputVariables.push_back("seq_generator_" + argumentName + "." +
                                   SEQUENCE_GENERATOR_DATA_NAME.str());
          inputVariables.push_back("seq_generator_" + argumentName + "." +
                                   SEQUENCE_GENERATOR_VALID_NAME.str());
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
            inputVariables.push_back("seq_generator_" + argumentName + "." +
                                     SEQUENCE_GENERATOR_VALID_NAME.str());
          } else {
            inputVariables.push_back("seq_generator_" + argumentName + "." +
                                     SEQUENCE_GENERATOR_DATA_NAME.str());
            inputVariables.push_back("seq_generator_" + argumentName + "." +
                                     SEQUENCE_GENERATOR_VALID_NAME.str());
          }
        })
        .Case<IntegerType>([&](IntegerType intType) {
          // This is the case for data coming from memory (it has no handshake
          // signals)
          if (argumentName != "clk" && argumentName != "rst") {
            inputVariables.push_back("0usd" +
                                     std::to_string(intType.getWidth()) + "_0");
          }
        })
        .Default([](Type) {});
  }

  if (syncOutput)
    for (const auto &[i, result] : llvm::enumerate(results)) {
      const auto &[resultName, type] = result;
      if (type.isa<handshake::ControlType, handshake::ChannelType>())
        inputVariables.push_back("join_global.ins_" + std::to_string(i) +
                                 "_ready");
    }
  else
    for (const auto &[resultName, type] : results) {
      if (type.isa<handshake::ControlType, handshake::ChannelType>())
        inputVariables.push_back("sink_" + resultName + "." +
                                 SINK_READY_NAME.str());
    }

  std::ostringstream call;
  call << "  VAR " << moduleName << " : " << moduleName << "(";
  call << join(inputVariables, ", ");
  call << ");\n";

  return call.str();
}
std::string getPrefixTypeName(const std::string &smvType) {
  size_t start = smvType.find('[');
  size_t end = smvType.find(']');

  if (start != std::string::npos && end != std::string::npos &&
      end > start + 1) {
    std::string numberStr = smvType.substr(start + 1, end - start - 1);
    return "s" + numberStr;
  }
  return "";
}

// SMV module for a sequence generator with a finite number of tokens. The
// actual number of generated tokens is non-determinstically set between 0
// and (inclusive) max_tokens.
std::string SMV_INPUT(const std::string &type) {
  return "MODULE " + getPrefixTypeName(type) +
         "_input(nReady0, max_tokens)\n"
         "  VAR dataOut0 : " +
         type +
         ";\n"
         "  VAR counter : 0..31;\n"
         "  FROZENVAR exact_tokens : 0..max_tokens;\n"
         "  ASSIGN\n"
         "    init(counter) := 0;\n"
         "    next(counter) := case\n"
         "      nReady0 & counter < exact_tokens : counter + 1;\n"
         "      TRUE : counter;\n"
         "    esac;\n"
         "\n"
         "  DEFINE valid0 := counter < exact_tokens;\n\n";
}

// SMV module for a sequence generator with a finite number of tokens. The
// number of generated tokens is exact_tokens.
std::string SMV_INPUT_EXACT(const std::string &type) {
  return "MODULE " + getPrefixTypeName(type) +
         "_input_exact(nReady0, exact_tokens)\n"
         "  VAR dataOut0 : " +
         type +
         ";\n"
         "  VAR counter : 0..31;\n"
         "  ASSIGN\n"
         "    init(counter) := 0;\n"
         "    next(counter) := case\n"
         "      nReady0 & counter < exact_tokens : counter + 1;\n"
         "      TRUE : counter;\n"
         "    esac;\n"
         "\n"
         "  DEFINE valid0 := counter < exact_tokens;\n\n";
}

// SMV module for a sequence generator with an infinite number of tokens
std::string SMV_INPUT_INF(const std::string &type) {
  return "MODULE " + getPrefixTypeName(type) +
         "_input_inf(nReady0)\n"
         "  VAR dataOut0 : " +
         type +
         ";\n"
         "    -- make sure dataOut0 is persistent\n"
         "    DEFINE valid0 := TRUE;\n\n";
}

static std::string
createSequenceGenerator(const std::optional<std::string> &type,
                        size_t nrOfTokens,
                        bool generateExactNrOfTokens = false) {
  if (type == std::nullopt) {
    if (nrOfTokens == 0)
      return SMV_CTRL_INPUT_INF;
    if (generateExactNrOfTokens)
      return SMV_CTRL_INPUT_EXACT;
    return SMV_CTRL_INPUT;
  }
  if (*type == "boolean") {
    if (nrOfTokens == 0)
      return SMV_BOOL_INPUT_INF;
    if (generateExactNrOfTokens)
      return SMV_BOOL_INPUT_EXACT;
    return SMV_BOOL_INPUT;
  }
  {
    if (nrOfTokens == 0)
      return SMV_INPUT_INF(*type);
    if (generateExactNrOfTokens)
      return SMV_INPUT_EXACT(*type);
    return SMV_INPUT(*type);
  }
}

static std::string createTBJoin(size_t nrOfOutputs) {
  std::ostringstream tbJoin;
  std::vector<std::string> insValids;
  for (size_t i = 0; i < nrOfOutputs; i++)
    insValids.push_back("ins_" + std::to_string(i) + "_valid");

  tbJoin << "MODULE tb_join (";
  tbJoin << join(insValids, ", ");
  tbJoin << ", outs_ready)\n";

  tbJoin << "  DEFINE\n  outs_valid := ";
  tbJoin << join(insValids, " & ");
  tbJoin << ";\n";

  for (size_t i = 0; i < nrOfOutputs; i++) {
    tbJoin << "  ins_" << i << "_ready := ";
    std::vector<std::string> tmp = insValids;
    tmp.erase(tmp.begin() + i);
    tbJoin << join(tmp, " & ");
    tbJoin << " & outs_ready;\n";
  }
  tbJoin << "\n\n";
  return tbJoin.str();
}

static std::optional<std::string> convertMLIRTypeToSMV(Type type) {
  return llvm::TypeSwitch<Type, std::optional<std::string>>(type)
      .Case<handshake::ControlType>(
          [&](handshake::ControlType cType) { return std::nullopt; })
      .Case<handshake::ChannelType>([&](handshake::ChannelType cType) {
        if (cType.getDataBitWidth() == 1)
          return std::string("boolean");
        return "signed word [" + std::to_string(cType.getDataBitWidth()) + "]";
      });
}

static std::string createSupportEntities(
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const SmallVector<std::pair<std::string, Type>> &results, size_t nrOfTokens,
    bool generateExactNrOfTokens = false, bool syncOutput = false) {

  std::unordered_set<std::optional<std::string>> types;
  for (const auto &[_, type] : arguments)
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      types.insert(convertMLIRTypeToSMV(type));
  std::ostringstream supportEntities;

  for (const auto &smvType : types) {
    supportEntities << createSequenceGenerator(smvType, nrOfTokens,
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
                       "  DEFINE ready0 := TRUE;\n\n";

  return supportEntities.str();
}

// Create the sequence generator at the inputs of the module
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

    // We support three different kinds of sequence generators:
    // 1. Infinite sequence generator: Will create an infinite number of
    // tokens.
    // 2. Standard finite generator: Will create 0 to the maximal number of
    // tokens. The exact number of tokens is non-deterministic.
    // 3. Exact finite generator: Will create the exact number of tokens (if
    // it receives enough ready inputs).
    // When nrOfTokens is set to 0, the infinite sequence generator is created
    // and the value of generateExactNrOfTokens is ignored.
    if (nrOfTokens == 0) {
      // Example: VAR seq_generator_D : bool_input_inf(model.D_ready);
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : " + typePrefixName + "_input_inf(" << moduleName
                         << "." << argumentName << "_ready);\n";
    } else if (generateExactNrOfTokens) {
      // Example: VAR seq_generator_D : bool_input_exact(model.D_ready, 1);
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : " + typePrefixName + "_input_exact("
                         << moduleName << "." << argumentName << "_ready, "
                         << nrOfTokens << ");\n";
    } else {
      // Example: VAR seq_generator_D : bool_input(model.D_ready, 1);
      sequenceGenerators << "  VAR seq_generator_" << argumentName
                         << " : " + typePrefixName + "_input(" << moduleName
                         << "." << argumentName << "_ready, " << nrOfTokens
                         << ");\n";
    }
  }
  return sequenceGenerators.str();
}

// Create the sinks at the outputs of the module
static std::string
instantiateSinks(const std::string &moduleName,
                 const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream sinks;

  for (const auto &[resultName, type] : results) {
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      sinks << "  VAR sink_" << resultName << " : sink_main(" << moduleName
            << "." << resultName << "_valid);\n";
  }
  return sinks.str();
}

// Create the join at the outputs of the module
static std::string
instantiateJoin(const std::string &moduleName,
                const SmallVector<std::pair<std::string, Type>> &results) {
  std::ostringstream str;
  std::vector<std::string> outputValids;

  str << "  VAR join_global : tb_join(";
  for (const auto &[resultName, type] : results) {
    if (type.isa<handshake::ControlType, handshake::ChannelType>())
      outputValids.push_back(moduleName + "." + resultName + "_valid");
  }
  str << join(outputValids, ", ") << ", global_ready);\n";

  return str.str();
}

std::string createSmvFormalTestbench(
    const SmallVector<std::pair<std::string, Type>> &arguments,
    const SmallVector<std::pair<std::string, Type>> &results,
    const std::string &modelSmvName, size_t nrOfTokens,
    bool generateExactNrOfTokens, bool syncOutput) {

  std::ostringstream wrapper;
  wrapper << "#include \"" + modelSmvName + ".smv\"\n\n";

  wrapper << createSupportEntities(arguments, results, nrOfTokens,
                                   generateExactNrOfTokens, syncOutput);

  wrapper << "MODULE main\n\n";

  wrapper << instantiateSequenceGenerators(modelSmvName, arguments, nrOfTokens,
                                           generateExactNrOfTokens);

  wrapper << instantiateModuleUnderTest(modelSmvName, arguments, results,
                                        syncOutput)
          << "\n";

  if (syncOutput) {

    wrapper << "  VAR global_ready : boolean;\n"
               "  ASSIGN\n"
               "  init(global_ready) := TRUE;\n"
               "  next(global_ready) := join_global.outs_valid ? FALSE : "
               "global_ready;\n\n";

    wrapper << instantiateJoin(modelSmvName, results) << "\n";
  } else {
    wrapper << instantiateSinks(modelSmvName, results) << "\n";
  }

  return wrapper.str();
}

} // namespace dynamatic::experimental