//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Experimental tool that exports VHDL from a netlist-level IR expressed in a
// combination of the HW and ESI dialects. The result is produced on standart
// llvm output.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <string>
#include <system_error>
#include <utility>

#include <array>
#include <cstdio>
#include <iostream>
#include <memory>

#define assertm(exp, msg) assert(((void)(msg), exp))

const static std::string JSON_LIBRARY =
    "experimental/tools/export-vhdl/library.json";
static const std::string CONCRET_METHOD_STR = "concretization_method",
                         COMPS_STR = "components",
                         GENERATORS_STR = "generators",
                         GENERICS_STR = "generics", PORTS_STR = "ports",
                         VALUE_STR = "value", CHANNEL_STR = "channel",
                         ARRAY_STR = "array", STD_LOGIC_STR = "std_logic",
                         STD_LOGIC_VECTOR_STR = "std_logic_vector",
                         DATA_ARRAY_STR = "data_array", CLOCK_STR = "clock",
                         RESET_STR = "reset", DATAFLOW_STR = "dataflow",
                         DATA_STR = "data", CONTROL_STR = "control",
                         NONE_VALUE = "-1", NAME_STR = "name",
                         TYPE_STR = "type", SIZE_STR = "size",
                         BITWIDTH_STR = "bitwidth", PATH_STR = "path";

using namespace llvm;
using namespace mlir;
using namespace circt;

// ============================================================================
// Helper functions
// ============================================================================

/// Execute binary file
std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    llvm::errs() << "popen() failed!";
    exit(1);
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

/// Assert condition: if object doesn't contain the field s, return false
bool checkObj(json::Object *&obj, const std::string &s) {
  return obj->find(s) != obj->end();
}

/// Split the string with discriminating parameters into string vector for
/// convenience
static llvm::SmallVector<std::string>
parseDiscriminatingParameters(std::string &modParameters) {
  llvm::SmallVector<std::string> s;
  std::stringstream str(modParameters);
  std::string temp;
  while (str.good()) {
    std::getline(str, temp, '_');
    s.push_back(temp);
  }
  return s;
}

/// Check if the given string is numerical
static bool isNumber(const std::string &str) {
  auto it = str.begin();
  while (it != str.end() && std::isdigit(*it))
    ++it;
  return !str.empty() && it == str.end();
}

/// Extract an integer from mlir::type
static std::pair<std::string, size_t> extractType(mlir::Type &t) {
  std::pair<std::string, size_t> p;
  if (t.isIntOrFloat()) {
    size_t type = t.getIntOrFloatBitWidth();
    p = std::pair(VALUE_STR, std::max(type, 1UL));
  } else if (auto ch = t.dyn_cast<esi::ChannelType>()) {
    // In case of !esi.channel<i...>
    p = std::pair(CHANNEL_STR, ch.getInner().getIntOrFloatBitWidth());
  } else {
    llvm_unreachable("Unsupported type");
  }
  return p;
}

/// extName consists of modName and modParameters (e.g. handshake_fork.3_32)
/// and this function splits these parameters into 2 strings (in our example
/// handshake_fork and 3_32 respectively)
static std::pair<std::string, std::string> splitExtName(StringRef extName) {
  size_t firstUnd = extName.find('.');
  std::string modName = extName.substr(0, firstUnd).str();
  std::string modParameters = extName.substr(firstUnd + 1).str();
  return std::pair(modName, modParameters);
}

// ============================================================================
// Necessary structures
// ============================================================================
namespace {

struct VHDLDescParameter;
struct VHDLModParameter;
struct VHDLInstParameter;
struct VHDLModuleDescription;
struct VHDLModule;
struct VHDLInstance;

/// The library that proves to be a cpp representation of library.json - a file
/// that contains all the components we can meet, with their inputs, outputs,
/// pathes to declaration files and so on.
using VHDLComponentLibrary = llvm::StringMap<VHDLModuleDescription>;
/// After we constructed the "template" VHDLComponentLibrary we need to
/// concretize it, that is to choose modules, only the components existing in
/// the input IR - a representation from which we want to export VHDL. This
/// step, after processing extern module operations in the input IR, results in
/// VHDLModuleLibrary.
using VHDLModuleLibrary = llvm::StringMap<VHDLModule>;
/// We obtained components with exact parameters, i.e. forks with 2 or 3 outputs
/// or buffers with different bitwidthes. Now we should process the situation
/// when more then one of these "concretized templates" exist: get instances,
/// different members of the same module structures. VHDLInstanceLibrary
/// contains all these instances with mlir::operation * as keys
using VHDLInstanceLibrary = llvm::MapVector<Operation *, VHDLInstance>;

/// A helper structure to describe parameters inside VHDLComponentLibrary
/// generics
struct VHDLGenericParam {
  VHDLGenericParam(std::string tempName = "", std::string tempType = "")
      : name{std::move(tempName)}, type{std::move(tempType)} {}
  /// Name of the parameter, e.g. bitwidthStr
  std::string name;
  /// Type of the parameter, e.g. "integer"
  std::string type;
};

/// A helper structure to describe parameters among VHDLComponentLibrary
/// input and output ports
struct VHDLDescParameter {
  enum class Type { DATAFLOW, DATA, CONTROL };
  VHDLDescParameter(std::string tempName = "", Type tempType = {},
                    std::string tempSize = "", std::string tempBitwidth = "")
      : name{std::move(tempName)}, type{tempType}, size{std::move(tempSize)},
        bitwidth{std::move(tempBitwidth)} {};

  /// Name of the parameter, e.g. "ins"
  std::string name;
  /// Type of the parameter. There're three possible options:
  /// 1. dataflowStr, when we have a data signal as well as valid and ready
  /// signals
  /// 2. dataStr, when we have onle a data signal without valid and ready
  /// signals
  /// 3. controlStr, when we don't have a data signal, only valid and ready
  /// signals
  Type type;
  /// Size of the parameter, that is the length of the corresponding array. It
  /// can be either a string ("LOAD_COUNT"), which will be updated with exact
  /// data later, or a number ("32") in case it is always the same
  std::string getSize() const {
    std::string temp;
    if (size.empty())
      temp = NONE_VALUE;
    else
      temp = size;
    return temp;
  }
  /// Bitwidth of the parameter. It can be either a string (bitwidthStr), which
  /// will be updated with exact data later, or a number ("32") in case it is
  /// always the same
  std::string getBitwidth() const {
    std::string temp;
    if (bitwidth.empty())
      temp = NONE_VALUE;
    else
      temp = bitwidth;
    return temp;
  }

private:
  std::string size;
  std::string bitwidth;
};

// A helper structure to describe components' inputs and outputs inside
// VHDLModuleLibrary. Each VHDLModParameter corresponds to either a signal or
// an "empty" signal
struct VHDLModParameter {
  enum class Type { STD_LOGIC, STD_LOGIC_VECTOR };
  enum class Amount { VALUE, ARRAY };
  VHDLModParameter(bool tempFlag = false, std::string tempName = "",
                   Type tempType = {}, Amount tempAmount = {},
                   size_t tempSize = 0, size_t tempBitwidth = 0)
      : flag{tempFlag}, name{std::move(tempName)}, type{tempType},
        amount{tempAmount}, size{tempSize}, bitwidth{tempBitwidth} {};
  /// Value that shows if the parameter exists inside signals' inputs / outputs.
  /// True, if exists, false, if not. It's useful in components' instantiation
  /// and so called empty signals, signals needed only as VHDL components'
  /// outputs but not participated in wiring
  bool flag;
  /// Name of the parameter, e.g. "ins"
  std::string name;
  /// Type of the parameter. There're two possible options:
  /// 1. stdLogicStr, when bitwidth = 1
  /// 2. stdLogicVectorStr, when bitwidth > 1
  Type type;
  /// Amount of the parameter. There're two possible options:
  /// 1. valueStr, when the length of the corresponding array is 1
  /// 2. arrayStr, when the length of the corresponding array is more than 1
  Amount amount;
  /// Size of the parameter, that is the length of the corresponding array.
  size_t size;
  /// Bitwidth of the parameter
  size_t bitwidth;
};

/// A helper structure to describe components' inputs and outputs inside
/// VHDLInstanceLibrary. Each VHDLInstParameter is a signal
struct VHDLInstParameter {
  enum class Type { VALUE, CHANNEL };
  VHDLInstParameter(mlir::Value tempValue = {}, std::string tempName = "",
                    Type tempType = {}, size_t tempBitwidth = (size_t)0)
      : value{tempValue}, name{std::move(tempName)}, type{tempType},
        bitwidth{tempBitwidth} {};
  /// mlir::value of the input / output (for identification)
  mlir::Value value;
  /// Name of the parameter, e.g. "extsi1_in0"
  std::string name;
  /// Type of the parameter. There're two possible options:
  /// 1. valueStr, when there's only data signal
  /// 2. channelStr, when there're also valid and control signals additionally
  /// to data signal
  Type type;
  /// Bitwidth of the parameter
  size_t bitwidth;
};

/// Description of the component in the library.json. Obtained after parsing
/// library.json.
struct VHDLModuleDescription {
  enum class Method { GENERATOR, GENERIC };
  VHDLModuleDescription(
      std::string tempPath = {}, Method tempConcrMethod = {},
      llvm::SmallVector<std::string> tempGenerators = {},
      llvm::SmallVector<VHDLGenericParam> tempGenerics = {},
      llvm::SmallVector<VHDLDescParameter> tempInputPorts = {},
      llvm::SmallVector<VHDLDescParameter> tempOutputPorts = {})
      : path(std::move(tempPath)), concretizationMethod(tempConcrMethod),
        generators(std::move(tempGenerators)),
        generics(std::move(tempGenerics)),
        inputPorts(std::move(tempInputPorts)),
        outputPorts(std::move(tempOutputPorts)) {}
  /// Path to a file with VHDL component's achitecture.
  /// Either .vhd or binary, depends on concretization method
  std::string path;
  /// Method of concretization, that is the way we get the component's
  /// architecture. There're two possible options:
  /// 1. "GENERIC". Means that we simply obtain an architecture from a file
  /// 2. "GENERATOR". Means that the architecture needs further concretization
  /// and it's necessary to get some parameters for it from the input IR
  Method concretizationMethod;
  /// Parameters' names that will be used in generation
  llvm::SmallVector<std::string> generators;
  /// Parameters' names that are used in VHDL components' declarations (that is
  /// VHDL generics)
  llvm::SmallVector<VHDLGenericParam> generics;
  /// Input ports as they exist in library.json
  llvm::SmallVector<VHDLDescParameter> inputPorts;
  /// Output ports as they exist in library.json
  llvm::SmallVector<VHDLDescParameter> outputPorts;
  /// Function that concretizes a library component, that is gets a module with
  /// exact values (BITWIDTH, SIZE and so on). It uses modParameters string,
  /// obtained from processing the input IR, to get all required information.
  VHDLModule concretize(std::string modName, std::string modParameters) const;
};

/// VHDL module, that is VHDLModuleDescription + extern module operations from
/// the input IR
struct VHDLModule {
  VHDLModule(std::string tempModName = "", std::string tempMtext = "",
             llvm::SmallVector<std::string> tempModParameters = {},
             llvm::SmallVector<VHDLModParameter> tempInputs = {},
             llvm::SmallVector<VHDLModParameter> tempOutputs = {},
             const VHDLModuleDescription &tempModDesc = {})
      : modName(std::move(tempModName)), modText(std::move(tempMtext)),
        modParameters(std::move(tempModParameters)),
        concrInputs(std::move(tempInputs)),
        concrOutputs(std::move(tempOutputs)), modDesc(tempModDesc) {}
  /// Concretized module's name, i.g. "fork"
  std::string modName;
  /// Component's definition and architecture, obtained from "concretize"
  /// function
  std::string modText;
  /// Discriminating parameters, i.e. parameters string parsed into separate
  /// parts
  llvm::SmallVector<std::string> modParameters;
  /// Concretized module's inputs
  llvm::SmallVector<VHDLModParameter> concrInputs;
  /// Concretized module's outputs
  llvm::SmallVector<VHDLModParameter> concrOutputs;
  /// Reference to the corresponding template in VHDLComponentLibrary
  const VHDLModuleDescription &modDesc;
  /// Function that instantiates a module component, that is gets an instance
  /// with an exact number. It uses innerOp operation, obtained from processing
  /// the input IR, to get all required information.
  VHDLInstance instantiate(std::string instName,
                           circt::hw::InstanceOp &innerOp) const;
};

llvm::SmallVector<VHDLModParameter>
getConcretizedPorts(const llvm::SmallVector<VHDLDescParameter> &descPorts,
                    llvm::StringMap<std::string> &genericsMap) {
  // Future VHDLModule inputs / outputs
  llvm::SmallVector<VHDLModParameter> ports;
  for (const VHDLDescParameter &i : descPorts) {
    size_t modSize;
    VHDLModParameter::Amount modAmount;
    if (i.getSize() == NONE_VALUE) {
      // If parameter sizeStr for the input / output doesn't exist in
      // library.json it means that input / output is one-dimentional, valueStr
      modSize = 1;
      modAmount = VHDLModParameter::Amount::VALUE;
    } else {
      // Otherwise it's an arrayStr, and we have the length of it in params
      auto s = genericsMap.find(i.getSize());
      modSize = std::atoi(s->second.c_str());
      modAmount = VHDLModParameter::Amount::ARRAY;
    }
    VHDLModParameter::Type modType;
    size_t modBitwidth;
    if (i.getBitwidth() == NONE_VALUE) {
      // If parameter bitwidthStr for the input doesn't exist in library.json it
      // means that input is 1-bit, stdLogicStr
      modType = VHDLModParameter::Type::STD_LOGIC;
      modBitwidth = 1;
    } else {
      // Otherwise the concrete bitwidth is either given or should also be found
      // in params
      modType = VHDLModParameter::Type::STD_LOGIC_VECTOR;
      if (isNumber(i.getBitwidth()))
        modBitwidth = std::atoi(i.getBitwidth().c_str());
      else {
        auto b = genericsMap.find(i.getBitwidth());
        modBitwidth = std::atoi(b->second.c_str());
      }
    }
    // If input doesn't exist among the signals (0 - size), it's an empty signal
    if (modSize != 0)
      ports.push_back(VHDLModParameter(true, i.name, modType, modAmount,
                                       modSize, modBitwidth));
    else
      ports.push_back(VHDLModParameter(false, i.name, modType, modAmount,
                                       modSize, modBitwidth));
  }
  return ports;
};

VHDLModule VHDLModuleDescription::concretize(std::string modName,
                                             std::string modParameters) const {
  // Split the given modParameters string into separate parts for convenience
  llvm::SmallVector<std::string> modParametersVec =
      parseDiscriminatingParameters(modParameters);
  // Fill the map that links between themselves template and exact names.
  // Template names - parameters that are contained in generators and generics
  // arrays
  llvm::StringMap<std::string> generatorsMap, genericsMap;
  auto *paramValIter = modParametersVec.begin();
  for (const std::string &paramName : generators)
    generatorsMap[paramName] = *(paramValIter++);
  for (const VHDLGenericParam &param : generics)
    genericsMap[param.name] = *(paramValIter++);

  // Future VHDLModule inputs
  llvm::SmallVector<VHDLModParameter> inputs =
      getConcretizedPorts(inputPorts, genericsMap);

  // Future VHDLModule outputs
  llvm::SmallVector<VHDLModParameter> outputs =
      getConcretizedPorts(outputPorts, genericsMap);

  std::string modText;
  // Open a file with component concretization data
  std::ifstream file;
  // Read as file
  std::stringstream buffer;
  if (concretizationMethod == VHDLModuleDescription::Method::GENERATOR) {
    // In case of the generator we're looking for binary
    std::string commandLineArguments;
    // Collecting discriminating params for command line. We might need both
    // generators and generics for a script (e.g. constant), and also generators
    // for modName because of the distinction in architectures (constant_1 and
    // constant_100 should differ)
    for (auto &clArg : generatorsMap) {
      commandLineArguments += " " + clArg.getValue();
      modName += "_" + clArg.getValue();
    }
    for (auto &clArg : genericsMap) {
      commandLineArguments += " " + clArg.getValue();
    }
    // Get module architecture
    auto fullPath = path + commandLineArguments;
    modText = exec(fullPath.c_str());
  } else if (concretizationMethod == VHDLModuleDescription::Method::GENERIC) {
    // In case of generic we're looking for ordinary file
    file.open(path);
    if (!file.is_open()) {
      llvm::errs() << "Generic filepath is incorrect\n";
      exit(1);
    }
    buffer << file.rdbuf();
    modText = buffer.str();
  } else {
    // Error
    llvm::errs() << "Wrong concredization method";
    exit(1);
  }
  // Obtain VHDLModule::modParameters vector
  llvm::SmallVector<std::string> resultParamArr;
  for (size_t i = generators.size(); i < generators.size() + generics.size();
       ++i)
    resultParamArr.push_back(modParametersVec[i]);

  // Get rid of "handshake" and "arith" prefixes in module name
  modName = modName.substr(modName.find('_') + 1);
  return VHDLModule(modName, modText, resultParamArr, inputs, outputs, *this);
}

/// VHDL instance, that is VHDLModule + inner module operations from
/// the input IR
struct VHDLInstance {
  VHDLInstance(std::string tempInstanceName, std::string tempItext,
               llvm::SmallVector<VHDLInstParameter> tempInputs,
               llvm::SmallVector<VHDLInstParameter> tempOutputs,
               const VHDLModule &tempMod)
      : instanceName(std::move(tempInstanceName)),
        instanceText(std::move(tempItext)), inputs(std::move(tempInputs)),
        outputs(std::move(tempOutputs)), mod(tempMod) {}
  /// Instance name, for instance "fork0"
  std::string instanceName;
  /// Instantiation of the component, i.e. "port mapping", assighnment between
  /// template component's inputs / outputs and other signals
  std::string instanceText;
  /// Instance inputs, i.e. signals
  llvm::SmallVector<VHDLInstParameter> inputs;
  /// Instance outputs, i.e. signals
  llvm::SmallVector<VHDLInstParameter> outputs;
  /// Reference to the corresponding module in VHDLModuleLibrary
  const VHDLModule &mod;
};

VHDLInstance VHDLModule::instantiate(std::string instName,
                                     circt::hw::InstanceOp &innerOp) const {
  // Shorten the name
  instName = instName.substr(instName.find('_') + 1);
  // Counter for innerOp argumentss or results array
  size_t num = 0;
  // Process the input signal
  llvm::SmallVector<VHDLInstParameter> inputs;
  for (Value opr : innerOp.getOperands()) {
    std::string name = instName + "_" + innerOp.getArgumentName(num).str();
    mlir::Type t = opr.getType();
    std::string type = extractType(t).first;
    VHDLInstParameter::Type eType;
    if (type == VALUE_STR)
      eType = VHDLInstParameter::Type::VALUE;
    else if (type == CHANNEL_STR)
      eType = VHDLInstParameter::Type::CHANNEL;
    else {
      llvm::errs() << "Wrong type: it must be either value, or channel.\n";
      exit(1);
    }
    size_t bitwidth = extractType(t).second;
    // Both clock and reset always exist. There's no need in them among inputs.
    if (innerOp.getArgumentName(num).str() != CLOCK_STR &&
        innerOp.getArgumentName(num).str() != RESET_STR)
      inputs.push_back(VHDLInstParameter(opr, name, eType, bitwidth));
    ++num;
  }
  num = 0;
  // Process the output signal
  llvm::SmallVector<VHDLInstParameter> outputs;
  for (Value opr : innerOp.getResults()) {
    std::string name = instName + "_" + innerOp.getResultName(num).str();
    mlir::Type t = opr.getType();
    std::string type = extractType(t).first;
    VHDLInstParameter::Type eType;
    if (type == VALUE_STR)
      eType = VHDLInstParameter::Type::VALUE;
    else if (type == CHANNEL_STR)
      eType = VHDLInstParameter::Type::CHANNEL;
    else {
      llvm::errs() << "Wrong type: it must be either value, or channel.\n";
      exit(1);
    }
    size_t bitwidth = extractType(t).second;
    outputs.push_back(VHDLInstParameter(opr, name, eType, bitwidth));
    ++num;
  }

  std::string compName = modName.substr(0, modName.find("."));
  std::string instText =
      instName + " : entity work." + compName + "(arch) generic map(";
  // "0", i.e. none, types are not allowed, so always replace them with 1
  for (const std::string &i : modParameters)
    if (i == "0")
      instText += "1, ";
    else
      instText += i + ", ";

  instText[instText.size() - 2] = ')';
  instText[instText.size() - 1] = '\n';
  instText += "port map(\n";
  // Inputs

  // clock-reset instantiation
  instText += "clk  => " + instName + "_clk,\n";
  instText += "rst  => " + instName + "_rst,\n";
  const VHDLModParameter *concrInpIt = concrInputs.begin();
  VHDLInstParameter *inpIt = inputs.begin();
  const VHDLDescParameter *descInpIt = modDesc.inputPorts.begin();
  // We process the list of template inputPorts (inputs from library.json) and
  // assign them to signals
  while (descInpIt != modDesc.inputPorts.end()) {
    std::string descName = descInpIt->name;
    auto channelType = descInpIt->type;
    if (concrInpIt->flag) {
      // If not an empty signal
      auto count = concrInpIt->amount;
      auto bitLen = concrInpIt->type;
      if (count == VHDLModParameter::Amount::VALUE &&
          bitLen == VHDLModParameter::Type::STD_LOGIC) {
        std::string insName = inpIt->name;
        // std_logic
        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::DATA)
          instText += descName + " => " + insName + ",\n";

        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::CONTROL) {
          instText += descName + "_valid => " + insName + "_valid,\n";
          instText += descName + "_ready => " + insName + "_ready,\n";
        }
        ++inpIt;
      } else if (count == VHDLModParameter::Amount::ARRAY &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC) {
        // std_logic_vector(COUNT)
        VHDLInstParameter *prevIt = inpIt;
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::DATA)
            instText +=
                descName + "(" + std::to_string(k) + ") => " + insName + ",\n";

          ++inpIt;
        }
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_valid(" + std::to_string(k) + ") => " +
                        insName + "_valid,\n";

          ++inpIt;
        }
        // According to VHDL compiler's rules ports of the same logic should go
        // in a row, so cycles for ready and valid signals are splitted up
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_ready(" + std::to_string(k) + ") => " +
                        insName + "_ready,\n";

          ++inpIt;
        }
      } else if (count == VHDLModParameter::Amount::VALUE &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC_VECTOR) {
        // std_logic_vector(BITWIDTH)
        std::string insName = inpIt->name;
        std::string z;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        if (concrInpIt->bitwidth <= 1)
          z = "(0)";

        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::DATA)
          instText += descName + z + " => " + insName + ",\n";

        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::CONTROL) {
          instText += descName + "_valid" + " => " + insName + "_valid,\n";
          instText += descName + "_ready" + " => " + insName + "_ready,\n";
        }
        ++inpIt;
      } else if (count == VHDLModParameter::Amount::ARRAY &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC_VECTOR) {
        // data_array(BITWIDTH)(COUNT)
        VHDLInstParameter *prevIt = inpIt;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          std::string z;
          if (concrInpIt->bitwidth <= 1)
            z = "(0)";
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::DATA)
            instText += descName + "(" + std::to_string(k) + ")" + z + " => " +
                        insName + ",\n";
          ++inpIt;
        }
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_valid(" + std::to_string(k) + ")" +
                        " => " + insName + "_valid,\n";
          ++inpIt;
        }
        // The same about order of valid and ready signals as above
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->size; ++k) {
          std::string insName = inpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_ready(" + std::to_string(k) + ")" +
                        " => " + insName + "_ready,\n";
          ++inpIt;
        }
      }
    } else {
      // If the input doesn't exist among signals in the input IR it's empty
      std::string extraSignalName = instName + "_x" + descName;
      if (channelType == VHDLDescParameter::Type::DATAFLOW ||
          channelType == VHDLDescParameter::Type::DATA)
        instText += descName + "(0) => " + extraSignalName + ",\n";
      if (channelType == VHDLDescParameter::Type::DATAFLOW ||
          channelType == VHDLDescParameter::Type::CONTROL) {
        instText += descName + "_valid(0) => " + extraSignalName + "_valid,\n";
        instText += descName + "_ready(0) => " + extraSignalName + "_ready,\n";
      }
    }
    ++concrInpIt;
    ++descInpIt;
  }

  // Outputs
  const VHDLModParameter *concrOutpIt = concrOutputs.begin();
  VHDLInstParameter *outpIt = outputs.begin();
  const VHDLDescParameter *descOutpIt = modDesc.outputPorts.begin();
  // We process the list of template inputPorts (inputs from library.json) and
  // assign them to signals
  while (descOutpIt != modDesc.outputPorts.end()) {
    std::string descName = descOutpIt->name;
    auto channelType = descOutpIt->type;
    if (concrOutpIt->flag) {
      // If not an empty signal
      auto count = concrOutpIt->amount;
      auto bitLen = concrOutpIt->type;
      if (count == VHDLModParameter::Amount::VALUE &&
          bitLen == VHDLModParameter::Type::STD_LOGIC) {
        std::string insName = outpIt->name;
        // std_logic
        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::DATA)
          instText += descName + " => " + insName + ",\n";
        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::CONTROL) {
          instText += descName + "_valid => " + insName + "_valid,\n";
          instText += descName + "_ready => " + insName + "_ready,\n";
        }
        ++outpIt;
      } else if (count == VHDLModParameter::Amount::ARRAY &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC) {
        // std_logic_vector(COUNT)
        VHDLInstParameter *prevIt = outpIt;
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::DATA)
            instText +=
                descName + "(" + std::to_string(k) + ") => " + insName + ",\n";
          ++outpIt;
        }
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_valid(" + std::to_string(k) + ") => " +
                        insName + "_valid,\n";
          ++outpIt;
        }
        // According to VHDL compiler's rules ports of the same logic should go
        // in a row, so cycles for ready and valid signals are splitted up
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_ready(" + std::to_string(k) + ") => " +
                        insName + "_ready,\n";
          ++outpIt;
        }
      } else if (count == VHDLModParameter::Amount::VALUE &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC_VECTOR) {
        // std_logic_vector(BITWIDTH)
        std::string insName = outpIt->name;
        std::string z;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        if (concrOutpIt->bitwidth <= 1)
          z = "(0)";
        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::DATA)
          instText += descName + z + " => " + insName + ",\n";
        if (channelType == VHDLDescParameter::Type::DATAFLOW ||
            channelType == VHDLDescParameter::Type::CONTROL) {
          instText += descName + "_valid" + " => " + insName + "_valid,\n";
          instText += descName + "_ready" + " => " + insName + "_ready,\n";
        }
        ++outpIt;
      } else if (count == VHDLModParameter::Amount::ARRAY &&
                 bitLen == VHDLModParameter::Type::STD_LOGIC_VECTOR) {
        // data_array(BITWIDTH)(COUNT)
        VHDLInstParameter *prevIt = outpIt;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          std::string z;
          if (concrOutpIt->bitwidth <= 1)
            z = "(0)";
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::DATA)
            instText += descName + "(" + std::to_string(k) + ")" + z + " => " +
                        insName + ",\n";
          ++outpIt;
        }
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_valid(" + std::to_string(k) + ")" +
                        " => " + insName + "_valid,\n";
          ++outpIt;
        }
        outpIt = prevIt;
        // The same about order of valid and ready signals as above
        for (size_t k = 0; k < concrOutpIt->size; ++k) {
          std::string insName = outpIt->name;
          if (channelType == VHDLDescParameter::Type::DATAFLOW ||
              channelType == VHDLDescParameter::Type::CONTROL)
            instText += descName + "_ready(" + std::to_string(k) + ")" +
                        " => " + insName + "_ready,\n";
          ++outpIt;
        }
      }
    } else {
      // If the output doesn't exist among signals in the input IR it's empty
      std::string extraSignalName = instName + "_y" + descName;
      if (channelType == VHDLDescParameter::Type::DATAFLOW ||
          channelType == VHDLDescParameter::Type::DATA)
        instText += descName + "(0) => " + extraSignalName + ",\n";
      if (channelType == VHDLDescParameter::Type::DATAFLOW ||
          channelType == VHDLDescParameter::Type::CONTROL) {
        instText += descName + "_valid(0) => " + extraSignalName + "_valid,\n";
        instText += descName + "_ready(0) => " + extraSignalName + "_ready,\n";
      }
    }
    ++concrOutpIt;
    ++descOutpIt;
  }
  instText[instText.size() - 2] = ')';
  instText[instText.size() - 1] = ';';
  instText += "\n";

  return VHDLInstance(instName, instText, inputs, outputs, *this);
}

} // namespace

// ============================================================================
// Wrappers
// ============================================================================

/// Get a module
static VHDLModule getMod(StringRef extName, VHDLComponentLibrary &jsonLib) {
  // extern Module Name = name + parameters with '.' as delimiter
  std::pair<std::string, std::string> p = splitExtName(extName);
  std::string modName = p.first;
  std::string modParameters = p.second;
  // find external module in VHDLComponentLibrary
  llvm::StringMapIterator<VHDLModuleDescription> comp = jsonLib.find(modName);

  if (comp == jsonLib.end()) {
    llvm::errs() << "Unable to find the element in the components' library\n";
    exit(1);
  }
  const VHDLModuleDescription &desc = (*comp).second;
  VHDLModule mod = desc.concretize(modName, modParameters);
  return mod;
};

/// Get an instance
static VHDLInstance getInstance(StringRef extName, StringRef name,
                                VHDLModuleLibrary &modLib,
                                circt::hw::InstanceOp innerOp) {
  // find external module in VHDLModuleLibrary
  StringMapIterator<VHDLModule> comp = modLib.find(extName);
  if (comp == modLib.end()) {
    llvm::errs() << "Unable to find the element in the instances' library\n";
    exit(1);
  }
  const VHDLModule &desc = (*comp).second;
  VHDLInstance inst = desc.instantiate(name.str(), innerOp);
  return inst;
}

// ============================================================================
// Functions that parse extern files
// ============================================================================

/// Get a cpp representation for given library.json file, i.e.
/// VHDLComponentLibrary
static VHDLComponentLibrary parseJSON() {
  // Load JSON library
  std::ifstream lib(JSON_LIBRARY);

  VHDLComponentLibrary m;
  if (!lib.is_open()) {
    llvm::errs() << "Filepath is incorrect\n";
    exit(1);
  }
  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(lib, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<json::Value> jsonLib = json::parse(jsonString);
  if (!jsonLib) {
    llvm::errs() << "Failed to parse models in \"" << JSON_LIBRARY << "\"\n";
    exit(1);
  }

  if (!jsonLib->getAsObject()) {
    llvm::errs() << "Library JSON is not a valid JSON"
                 << "\n";
    exit(1);
  }
  // Parse elements in json
  for (auto item : *jsonLib->getAsObject()) {
    // ModuleName is either "arith" or "handshake"
    std::string moduleName = item.getFirst().str();
    json::Array *moduleArray = item.getSecond().getAsArray();
    for (json::Value &c : *moduleArray) {
      // c is iterator, which points on a specific component's scheme inside
      // arith / handshake class
      json::Object *obj = c.getAsObject();
      assertm(checkObj(obj, COMPS_STR),
              "Error in library's \"components\" field.");
      json::Array *jsonComponents = obj->get(COMPS_STR)->getAsArray();
      assertm(checkObj(obj, CONCRET_METHOD_STR),
              "Error in library's \"concretization_method\" field.");
      auto jsonConcretizationMethod =
          obj->get(CONCRET_METHOD_STR)->getAsString();
      if (jsonConcretizationMethod)
        assertm(checkObj(obj, GENERATORS_STR),
                "Error in library's \"generators\" field.");
      json::Array *jsonGenerators = obj->get(GENERATORS_STR)->getAsArray();
      assertm(checkObj(obj, GENERICS_STR),
              "Error in library's \"generics\" field.");
      json::Array *jsonGenerics = obj->get(GENERICS_STR)->getAsArray();
      assertm(checkObj(obj, PORTS_STR), "Error in library's \"ports\" field.");
      json::Object *jsonPorts = obj->get(PORTS_STR)->getAsObject();
      // Creating corresponding VHDLModuleDescription variables
      std::string concretizationMethod = jsonConcretizationMethod.value().str();
      VHDLModuleDescription::Method eMethod;
      if (concretizationMethod == "GENERATOR")
        eMethod = VHDLModuleDescription::Method::GENERATOR;
      else if (concretizationMethod == "GENERIC")
        eMethod = VHDLModuleDescription::Method::GENERIC;
      else {
        llvm::errs() << "Wrong concretization method: only GENERATOR and "
                        "GENERIC are allowed.\n";
        exit(1);
      }

      llvm::SmallVector<std::string> generators;
      for (json::Value &jsonGenerator : *jsonGenerators)
        generators.push_back(jsonGenerator.getAsString().value().str());

      llvm::SmallVector<VHDLGenericParam> generics;
      for (json::Value &jsonGeneric : *jsonGenerics) {
        json::Object *ob = jsonGeneric.getAsObject();
        assertm(checkObj(ob, NAME_STR),
                "Error in library's generic \"name\" field.");
        std::string name = ob->get(NAME_STR)->getAsString().value().str();
        assertm(checkObj(ob, TYPE_STR),
                "Error in library's generic \"type\" field.");
        std::string type = ob->get(TYPE_STR)->getAsString().value().str();
        generics.push_back(VHDLGenericParam(name, type));
      }
      // Get input ports
      llvm::SmallVector<VHDLDescParameter> inputPorts;
      json::Array *jsonInputPorts = jsonPorts->get("in")->getAsArray();
      for (json::Value &jsonInputPort : *jsonInputPorts) {
        json::Object *ob = jsonInputPort.getAsObject();
        assertm(checkObj(ob, NAME_STR),
                "Error in library's input \"name\" field.");
        std::string name = ob->get(NAME_STR)->getAsString().value().str();
        assertm(checkObj(ob, TYPE_STR),
                "Error in library's input \"type\" field.");
        std::string type = ob->get(TYPE_STR)->getAsString().value().str();
        VHDLDescParameter::Type eType;
        if (type == DATAFLOW_STR)
          eType = VHDLDescParameter::Type::DATAFLOW;
        else if (type == DATA_STR)
          eType = VHDLDescParameter::Type::DATA;
        else if (type == CONTROL_STR)
          eType = VHDLDescParameter::Type::CONTROL;
        else {
          llvm::errs()
              << "Wrong type value: it must be dataflow, data or control.\n";
          exit(1);
        }
        std::string size;
        if (ob->find(SIZE_STR) != ob->end())
          size = ob->get(SIZE_STR)->getAsString().value().str();
        std::string bitwidth;
        if (ob->find(BITWIDTH_STR) != ob->end())
          bitwidth = ob->get(BITWIDTH_STR)->getAsString().value().str();
        inputPorts.push_back(VHDLDescParameter(name, eType, size, bitwidth));
      }
      // Get output ports
      llvm::SmallVector<VHDLDescParameter> outputPorts;
      json::Array *jsonOutputPorts = jsonPorts->get("out")->getAsArray();
      for (json::Value &jsonOutputPort : *jsonOutputPorts) {
        json::Object *ob = jsonOutputPort.getAsObject();
        assertm(checkObj(ob, NAME_STR),
                "Error in library's output \"name\" field.");
        std::string name = ob->get(NAME_STR)->getAsString().value().str();
        assertm(checkObj(ob, TYPE_STR),
                "Error in library's output \"type\" field.");
        std::string type = ob->get(TYPE_STR)->getAsString().value().str();
        VHDLDescParameter::Type eType;
        if (type == DATAFLOW_STR)
          eType = VHDLDescParameter::Type::DATAFLOW;
        else if (type == DATA_STR)
          eType = VHDLDescParameter::Type::DATA;
        else if (type == CONTROL_STR)
          eType = VHDLDescParameter::Type::CONTROL;
        else {
          llvm::errs()
              << "Wrong type value: it must be dataflow, data or control.\n";
          exit(1);
        }
        std::string size;
        if (ob->find(SIZE_STR) != ob->end())
          size = ob->get(SIZE_STR)->getAsString().value().str();
        std::string bitwidth;
        if (ob->find(BITWIDTH_STR) != ob->end())
          bitwidth = ob->get(BITWIDTH_STR)->getAsString().value().str();
        outputPorts.push_back(VHDLDescParameter(name, eType, size, bitwidth));
      }

      for (json::Value &jsonComponent : *jsonComponents) {
        json::Object *ob = jsonComponent.getAsObject();
        assertm(checkObj(ob, NAME_STR),
                "Error in library's components \"name\" field.");
        std::string name = ob->get(NAME_STR)->getAsString().value().str();
        assertm(checkObj(ob, PATH_STR),
                "Error in library's components\"path\" field.");
        std::string path = ob->get(PATH_STR)->getAsString().value().str();
        std::string keyName = moduleName + "_" + name;
        // Inserting our component into library
        m.insert(std::pair(
            keyName, VHDLModuleDescription(path, eMethod, generators, generics,
                                           inputPorts, outputPorts)));
      }
    }
  }
  lib.close();
  return m;
}

/// Get modules from extern operations in the input IR
static VHDLModuleLibrary parseExternOps(mlir::ModuleOp modOp,
                                        VHDLComponentLibrary &m) {
  VHDLModuleLibrary modLib;
  for (hw::HWModuleExternOp modOp : modOp.getOps<hw::HWModuleExternOp>()) {
    StringRef extName = modOp.getName();
    VHDLModule i = getMod(extName, m);
    modLib.insert(std::pair(extName, i));
  }
  return modLib;
}

/// Get instances from extern operations in the input IR
static VHDLInstanceLibrary
parseInstanceOps(mlir::OwningOpRef<mlir::ModuleOp> &module,
                 VHDLModuleLibrary &modLib) {
  VHDLInstanceLibrary instLib;
  for (hw::HWModuleOp modOp : module->getOps<hw::HWModuleOp>()) {
    for (hw::InstanceOp innerOp : modOp.getOps<hw::InstanceOp>()) {
      StringRef extName = innerOp.getReferencedModuleName();
      StringRef name = innerOp.getInstanceName();
      VHDLInstance i = getInstance(extName, name, modLib, innerOp);
      instLib.insert(std::pair(innerOp.getOperation(), i));
    }
  }
  return instLib;
}

/// Get the description of the head instance, "hw.module"
static std::pair<Operation *, VHDLInstance>
parseModule(hw::HWModuleOp hwModOp) {
  std::string iName;
  llvm::SmallVector<VHDLInstParameter> ins, outs;
  auto inputs = hwModOp.getPorts().inputs;
  auto outputs = hwModOp.getPorts().outputs;
  for (auto i : inputs) {
    mlir::Type t = i.type;
    std::string name = i.getName().str();
    std::string type = extractType(t).first;
    VHDLInstParameter::Type eType;
    if (type == VALUE_STR)
      eType = VHDLInstParameter::Type::VALUE;
    else if (type == CHANNEL_STR)
      eType = VHDLInstParameter::Type::CHANNEL;
    else {
      llvm::errs() << "Wrong type: it must be either value, or channel.\n";
      exit(1);
    }
    size_t bitwidth = extractType(t).second;
    if (name != CLOCK_STR && name != RESET_STR)
      ins.push_back(VHDLInstParameter({}, name, eType, bitwidth));
  }
  for (auto &i : outputs) {
    mlir::Type t = i.type;
    std::string name = i.getName().str();
    std::string type = extractType(t).first;
    VHDLInstParameter::Type eType;
    if (type == VALUE_STR)
      eType = VHDLInstParameter::Type::VALUE;
    else if (type == CHANNEL_STR)
      eType = VHDLInstParameter::Type::CHANNEL;
    else {
      llvm::errs() << "Wrong type: it must be either value, or channel.\n";
      exit(1);
    }
    size_t bitwidth = extractType(t).second;
    outs.push_back(VHDLInstParameter({}, name, eType, bitwidth));
  }
  iName = hwModOp.getName().str();
  return std::pair(hwModOp, VHDLInstance(iName, {}, ins, outs, {}));
}

/// Get the description of the output instance, "hw.output".
/// Names are obtained from hw.module
static std::pair<Operation *, VHDLInstance> parseOut(hw::HWModuleOp hwModOp) {
  Operation *outOp;
  std::string iName;
  llvm::SmallVector<VHDLInstParameter> tInputs, tOutputs;
  auto outputs = hwModOp.getPorts().outputs;
  SmallVector<VHDLInstParameter> outs;
  for (circt::hw::PortInfo &i : outputs) {
    mlir::Type t = i.type;
    std::string name = i.getName().str();
    std::string type = extractType(t).first;
    VHDLInstParameter::Type eType;
    if (type == VALUE_STR)
      eType = VHDLInstParameter::Type::VALUE;
    else if (type == CHANNEL_STR)
      eType = VHDLInstParameter::Type::CHANNEL;
    else {
      llvm::errs() << "Wrong type: it must be either value, or channel.\n";
      exit(1);
    }
    size_t bitwidth = extractType(t).second;
    outs.push_back(VHDLInstParameter({}, name, eType, bitwidth));
  }
  outOp = cast<hw::OutputOp>(hwModOp.getBodyBlock()->getTerminator());
  iName = hwModOp.getName().str();
  tOutputs = outs;
  return std::pair(outOp, VHDLInstance(iName, {}, tOutputs, {}, {}));
}

// ============================================================================
// Export functions
// ============================================================================

/// Get supporting .vhd components that are used in ours, i.g. join
static void getSupportFiles() {
  std::ifstream file;
  file.open("experimental/data/vhdl/supportfiles.vhd");
  if (!file.is_open()) {
    llvm::errs() << "Support filepath is incorrect\n";
    exit(1);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  llvm::outs() << buffer.str() << "\n";
}

/// Get existing components' architectures
static void getModulesArchitectures(VHDLModuleLibrary &modLib) {
  // module arches
  llvm::StringMap<std::string> modulesArchitectures;
  for (auto &i : modLib) {
    if (!modulesArchitectures.contains(i.getValue().modName)) {
      modulesArchitectures.insert(std::pair(i.getValue().modName, ""));
      llvm::outs() << i.getValue().modText << "\n";
    }
  }
}

/// Get the declaration of head instance
static void getEntityDeclaration(std::pair<Operation *, VHDLInstance> &hwmod) {
  VHDLInstance inst = hwmod.second;
  std::string res;
  res += "clock : in std_logic;\n";
  res += "reset : in std_logic;\n";
  for (const VHDLInstParameter &i : inst.inputs) {
    res += i.name;
    if (i.bitwidth > 1)
      res += " : in std_logic_vector (" + std::to_string(i.bitwidth - 1) +
             " downto 0);\n";
    else
      res += " : in std_logic;\n";
    if (i.type == VHDLInstParameter::Type::CHANNEL) {
      res += i.name + "_valid : in std_logic;\n";
      res += i.name + "_ready : out std_logic;\n";
    }
  }

  for (const VHDLInstParameter &i : inst.outputs) {
    res += i.name;
    if (i.bitwidth > 1)
      res += " : out std_logic_vector (" + std::to_string(i.bitwidth - 1) +
             " downto 0);\n";
    else
      res += " : out std_logic;\n";
    if (i.type == VHDLInstParameter::Type::CHANNEL) {
      res += i.name + "_valid : out std_logic;\n";
      res += i.name + "_ready : in std_logic;\n";
    }
  }

  res[res.length() - 2] = ')';
  res[res.length() - 1] = ';';
  res += "\n";
  llvm::outs() << res;
}

/// Support function for signals declaration that prints input / output signals
void getSignalsPorts(llvm::SmallVector<VHDLModParameter> &concrPorts,
                     const VHDLInstParameter *portsIt, VHDLInstance &mod) {
  // inputs / outputs
  for (const VHDLModParameter &i : concrPorts) {
    if (i.flag) {
      // If it's not an empty signal
      for (size_t j = 0; j < i.size; ++j) {
        llvm::outs() << "signal " << portsIt->name << " : ";
        if (i.type == VHDLModParameter::Type::STD_LOGIC)
          llvm::outs() << "std_logic"
                       << ";\n";
        else if (i.type == VHDLModParameter::Type::STD_LOGIC_VECTOR)
          if (portsIt->bitwidth <= 1)
            llvm::outs() << "std_logic;\n";
          else {
            size_t p = portsIt->bitwidth - 1;
            llvm::outs() << "std_logic_vector"
                         << "(" << p << " downto 0);\n";
          }
        else {
          llvm::errs() << "Wrong module type!\n";
          exit(1);
        }
        if (portsIt->type == VHDLInstParameter::Type::CHANNEL) {
          llvm::outs() << "signal " << portsIt->name << "_valid : "
                       << "std_logic;\n";
          llvm::outs() << "signal " << portsIt->name << "_ready : "
                       << "std_logic;\n";
        }
        ++portsIt;
      }
    } else {
      // Empty signals also should be declared
      llvm::outs() << "signal " << mod.instanceName + "_x" + i.name << " : ";
      if (i.type == VHDLModParameter::Type::STD_LOGIC)
        llvm::outs() << "std_logic"
                     << ";\n";
      else if (i.type == VHDLModParameter::Type::STD_LOGIC_VECTOR)
        if (i.bitwidth <= 1)
          llvm::outs() << "std_logic;\n";
        else {
          size_t p = i.bitwidth - 1;
          llvm::outs() << "std_logic_vector"
                       << "(" << p << " downto 0);\n";
        }
      else {
        llvm::errs() << "Wrong module type!\n";
        exit(1);
      }
      llvm::outs() << "signal " << mod.instanceName + "_x" + i.name
                   << "_valid : std_logic;\n";
      llvm::outs() << "signal " << mod.instanceName + "_x" + i.name
                   << "_ready : std_logic;\n";
    }
  }
}

/// Get the signals' declaration
static void getSignalsDeclaration(VHDLInstanceLibrary &instanceLib) {
  for (auto &[op, mod] : instanceLib) {
    VHDLModule module = mod.mod;
    const VHDLInstParameter *inputsIt = mod.inputs.begin();
    const VHDLInstParameter *outputsIt = mod.outputs.begin();
    llvm::outs() << "signal " << mod.instanceName << "_clk : "
                 << "std_logic;\n";
    llvm::outs() << "signal " << mod.instanceName << "_rst : "
                 << "std_logic;\n";
    // inputs
    getSignalsPorts(module.concrInputs, inputsIt, mod);
    // outputs
    getSignalsPorts(module.concrOutputs, outputsIt, mod);
    llvm::outs() << "\n";
  }
}

/// Get signals' wiring for instances from instanceLib
static void processInstance(mlir::Operation *&op, mlir::Operation *&i,
                            const VHDLInstParameter *&opIt,
                            VHDLInstanceLibrary::iterator &it,
                            llvm::StringMap<std::string> &d) {
  // The operation exists in instanceLib
  VHDLInstance inst = it->second;
  // Iterator through inputs of each successor
  const VHDLInstParameter *acIt = inst.inputs.begin();
  for (Value opr : i->getOperands()) {
    Operation *defOp = opr.getDefiningOp();
    if (defOp && defOp == op && opIt->value == acIt->value) {
      // If our initial operation is the predeccessor of an input of the
      // successor
      d.insert(std::pair(acIt->name, opIt->name));
      if (acIt->type == VHDLInstParameter::Type::CHANNEL) {
        llvm::outs() << acIt->name << "_valid <= " << opIt->name << "_valid;\n";
        llvm::outs() << opIt->name << "_ready <= " << acIt->name << "_ready;\n";
      }
      llvm::outs() << acIt->name << " <= " << opIt->name << ";\n";

      break;
    }
    ++acIt;
  }
}

/// Get signals' wiring for the output
static void processOutput(mlir::Operation *&op, mlir::Operation *&i,
                          const VHDLInstParameter *&opIt,
                          std::pair<Operation *, VHDLInstance> &hwOut,
                          llvm::StringMap<std::string> &d) {
  // Output module
  VHDLInstance inst = hwOut.second;
  const VHDLInstParameter *acIt = inst.inputs.begin();
  for (Value opr : i->getOperands()) {
    Operation *defOp = opr.getDefiningOp();
    if (defOp && defOp == op) {
      // if our initial operation is the predeccessor of an input of the
      // successor
      if (d.find(acIt->name) == d.end()) {
        d.insert(std::pair(acIt->name, opIt->name));
        if (acIt->type == VHDLInstParameter::Type::CHANNEL) {
          llvm::outs() << acIt->name << "_valid <= " << opIt->name
                       << "_valid;\n";
          llvm::outs() << opIt->name << "_ready <= " << acIt->name
                       << "_ready;\n";
        }
        llvm::outs() << acIt->name << " <= " << opIt->name << ";\n";
        break;
      }
    }
    ++acIt;
  }
}

/// Get connections between signals
static void getWiring(VHDLInstanceLibrary &instanceLib,
                      std::pair<Operation *, VHDLInstance> &hwOut) {
  // we process every component line from the input IR (== every instance
  // from instanceLib)
  for (auto &[op, mod] : instanceLib) {
    llvm::outs() << mod.instanceName << "_clk <= clock;\n";
    llvm::outs() << mod.instanceName << "_rst <= reset;\n";
    // Iterator through outputs of each instance
    const VHDLInstParameter *opIt = mod.outputs.begin();
    llvm::StringMap<std::string> d;
    for (mlir::Operation *i : op->getUsers()) {
      // Get an operation - successor of the instance
      VHDLInstanceLibrary::iterator it = instanceLib.find(i);
      if (it != instanceLib.end())
        processInstance(op, i, opIt, it, d);
      else if (i == hwOut.first)
        processOutput(op, i, opIt, hwOut, d);
      else {
        llvm::errs() << "Error in MLIR: it must be an output!\n";
        exit(1);
      }
      ++opIt;
    }
    llvm::outs() << "\n";
  }
}

/// Get modules instantiations, i.e. assignments between template
/// VHDLModuleDescription iputs and outputs and signals
static void getModulesInstantiation(VHDLInstanceLibrary &instanceLib) {
  for (auto &[op, mod] : instanceLib)
    llvm::outs() << mod.instanceText << "\n";
}

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

int main(int argc, char **argv) {
  // Initialize LLVM and parse command line arguments
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "VHDL exporter\n\n"
      "This tool prints on stdout the VHDL design corresponding to the "
      "input"
      "netlist-level MLIR representation of a dataflow circuit.\n");

  // Read the input IR in memory
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from
  // high(er) level dialects or parsers. Allow unregistered dialects to
  // not fail in these cases
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::esi::ESIDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module in memory
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return 1;
  mlir::ModuleOp modOp = *module;

  // We only support one HW module per MLIR module
  auto ops = modOp.getOps<hw::HWModuleOp>();
  if (std::distance(ops.begin(), ops.end()) != 1) {
    llvm::errs() << "The tool only supports a single top-level module in the "
                    "netlist.\n";
    return 1;
  }
  hw::HWModuleOp hwModOp = *ops.begin();
  std::string hwModName = hwModOp.getName().str();

  // Get all necessary structures
  VHDLComponentLibrary m = parseJSON();
  VHDLModuleLibrary modLib = parseExternOps(modOp, m);
  VHDLInstanceLibrary instanceLib = parseInstanceOps(module, modLib);
  std::pair<mlir::Operation *, VHDLInstance> hwOut = parseOut(hwModOp);
  std::pair<mlir::Operation *, VHDLInstance> hwMod = parseModule(hwModOp);

  // Generate VHDL
  getSupportFiles();
  llvm::outs() << "\n";
  getModulesArchitectures(modLib);
  llvm::outs() << "\n";
  llvm::outs() << "-- "
                  "============================================================"
                  "==\nlibrary IEEE;\n use IEEE.std_logic_1164.all;\n use "
                  "IEEE.numeric_std.all;\n use work.customTypes.all;\n-- "
                  "============================================================"
                  "==\nentity "
               << hwModName << " is\nport (\n";
  getEntityDeclaration(hwMod);
  llvm::outs() << "end;\n\n";
  llvm::outs() << "architecture behavioral of " << hwModName << " is\n\n";

  getSignalsDeclaration(instanceLib);
  llvm::outs() << "\nbegin\n\n";
  getWiring(instanceLib, hwOut);
  llvm::outs() << "\n";
  getModulesInstantiation(instanceLib);
  llvm::outs() << "end behavioral;\n\n";
  llvm::outs() << "\n";

  return 0;
}
