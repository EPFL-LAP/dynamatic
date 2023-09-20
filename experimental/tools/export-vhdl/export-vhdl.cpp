//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Experimental tool that exports VHDL from a netlist-level IR expressed in a
// combination of the HW and ESI dialects. The result is produced on standart
// llvm output.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <fstream>
#include <string>

#define LIBRARY_PATH "experimental/tools/export-vhdl/library.json"

using namespace llvm;
using namespace mlir;
using namespace circt;

struct VHDLDescParameter;
struct VHDLModParameter;
struct VHDLInstParameter;
struct VHDLModuleDescription;
struct VHDLModule;
struct VHDLInstance;
/// The library that proves to be a cpp representation of library.json - a file
/// that contains all the components we can meet, with their inputs, outputs,
/// pathes to declaration files and so on.
typedef llvm::StringMap<VHDLModuleDescription> VHDLComponentLibrary;
/// After we constructed the "template" VHDLComponentLibrary we need to
/// concretize it, that is to choose modules, only the components existing in
/// netlist.mlir - a representation from which we want to export VHDL. This
/// step, after processing extern module operations in netlist.mlir, results in
/// VHDLModuleLibrary.
typedef llvm::StringMap<VHDLModule> VHDLModuleLibrary;
/// We obtained components with exact parameters, i.e. forks with 2 or 3 outputs
/// or buffers with different bitwidthes. Now we should process the situation
/// when more then one of these "concretized templates" exist: get instances,
/// different members of the same module structures. VHDLInstanceLibrary
/// contains all these instances with mlir::operation * as keys
typedef llvm::MapVector<Operation *, VHDLInstance> VHDLInstanceLibrary;

// ============================================================================
// Helper functions
// ============================================================================

/// Split the string with discriminating parameters into string vector for
/// convenience
llvm::SmallVector<std::string>
parseDiscriminatingParameters(std::string &modParameters) {
  llvm::SmallVector<std::string> s{};
  std::stringstream str(modParameters);
  std::string temp;
  while (str.good()) {
    std::getline(str, temp, '_');
    s.push_back(temp);
  }
  return s;
}

/// Check if the given string is numerical
bool is_number(const std::string &str) {
  auto it = str.begin();
  while (it != str.end() && std::isdigit(*it))
    ++it;
  return !str.empty() && it == str.end();
}

/// Extract an integer from mlir::type
std::pair<std::string, size_t> extractType(mlir::Type &t) {
  if (t.isIntOrFloat()) {
    auto type = t.getIntOrFloatBitWidth();
    return std::pair("value", type > 0 ? type : 1);
  } else if (auto ch = t.dyn_cast<esi::ChannelType>()) {
    // In case of !esi.channel<i...>
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    t.print(rso);
    std::stringstream str;
    str << rso.str();
    typeStr.clear();
    typeStr = str.str();
    std::string newTypeStr;
    for (auto &i : typeStr)
      if (isdigit(i))
        newTypeStr.push_back(i);

    return std::pair("channel", atoi(newTypeStr.c_str()));
  } else {
    llvm::errs() << "Wrong type!\n";
    return {};
  }
}

/// extName consists of modName and modParameters (e.g. handshake_fork_3_32)
/// and this function splits these parameters into 2 strings (in our example
/// handshake_fork and 3_32 respectively)
std::pair<std::string, std::string> splitExtName(StringRef extName) {
  size_t first_ = extName.find('.');
  std::string modName = extName.substr(0, first_).str();
  std::string modParameters = extName.substr(first_ + 1).str();
  return std::pair(modName, modParameters);
}

// ============================================================================
// Necessary structures
// ============================================================================

/// A helper structure to describe parameters inside VHDLComponentLibrary
/// generics
struct VHDLGenericParam {
  VHDLGenericParam(std::string tempName = "", std::string tempType = "")
      : name{tempName}, type{tempType} {}
  /// Name of the parameter, e.g. "BITWIDTH"
  std::string getName() const { return name; }
  /// Type of the parameter, e.g. "integer"
  std::string getType() const { return type; }

private:
  std::string name;
  std::string type;
};

/// A helper structure to describe parameters among VHDLComponentLibrary
/// input and output ports
struct VHDLDescParameter {
  VHDLDescParameter(std::string tempName = "", std::string tempType = "",
                    std::string tempSize = "", std::string tempBitwidth = "")
      : name{tempName}, type{tempType}, size{tempSize},
        bitwidth{tempBitwidth} {};
  /// Name of the parameter, e.g. "ins"
  std::string getName() const { return name; }
  /// Type of the parameter. There're three possible options:
  /// 1. "dataflow", when we have a data signal as well as valid and ready
  /// signals
  /// 2. "data", when we have onle a data signal without valid and ready
  /// signals
  /// 3. "control", when we don't have a data signal, only valid and ready
  /// signals
  std::string getType() const { return type; }
  /// Size of the parameter, that is the length of the corresponding array. It
  /// can be either a string ("LOAD_COUNT"), which will be updated with exact
  /// data later, or a number ("32") in case it is always the same
  std::string getSize() const {
    if (size.empty())
      return "-1";
    else
      return size;
  }
  /// Bitwidth of the parameter. It can be either a string ("BITWIDTH"), which
  /// will be updated with exact data later, or a number ("32") in case it is
  /// always the same
  std::string getBitwidth() const {
    if (bitwidth.empty())
      return "-1";
    else
      return bitwidth;
  }

private:
  std::string name;
  std::string type;
  std::string size;
  std::string bitwidth;
};

// A helper structure to describe components' inputs and outputs inside
// VHDLModuleLibrary. Each VHDLModParameter corresponds to either a signal or
// an "empty" signal
struct VHDLModParameter {
  VHDLModParameter(bool tempFlag = false, std::string tempName = "",
                   std::string tempType = "", std::string tempAmount = "",
                   size_t tempSize = 0, size_t tempBitwidth = 0)
      : flag{tempFlag}, name{tempName}, type{tempType}, amount{tempAmount},
        size{tempSize}, bitwidth{tempBitwidth} {};
  /// Value that shows if the parameter exists inside signals' inputs / outputs.
  /// True, if exists, false, if not. It's useful in components' instantiation
  /// and so called empty signals, signals needed only as VHDL components'
  /// outputs but not participated in wiring
  bool getFlag() const { return flag; }
  /// Name of the parameter, e.g. "ins"
  std::string getName() const { return name; }
  /// Type of the parameter. There're two possible options:
  /// 1. "std_logic", when bitwidth = 1
  /// 2. "std_logic_vector", when bitwidth > 1
  std::string getType() const { return type; }
  /// Amount of the parameter. There're two possible options:
  /// 1. "value", when the length of the corresponding array is 1
  /// 2. "array", when the length of the corresponding array is more than 1
  std::string getAmount() const { return amount; }
  /// Size of the parameter, that is the length of the corresponding array.
  size_t getSize() const { return size; }
  /// Bitwidth of the parameter
  size_t getBitwidth() const { return bitwidth; }

private:
  bool flag;
  std::string name;
  std::string type;
  std::string amount;
  size_t size;
  size_t bitwidth;
};

// A helper structure to describe components' inputs and outputs inside
// VHDLInstanceLibrary. Each VHDLInstParameter is a signal
struct VHDLInstParameter {
  VHDLInstParameter(mlir::Value tempValue = {}, std::string tempName = "",
                    std::string tempType = "", size_t tempBitwidth = (size_t)0)
      : value{tempValue}, name{tempName}, type{tempType},
        bitwidth{tempBitwidth} {};
  /// mlir::value of the input / output (for identification)
  mlir::Value getValue() const { return value; }
  /// Name of the parameter, e.g. "extsi1_in0"
  std::string getName() const { return name; }
  /// Type of the parameter. There're two possible options:
  /// 1. "value", when there's only data signal
  /// 2. "channel", when there're also valid and control signals additionally to
  /// data signal
  std::string getType() const { return type; }
  /// Bitwidth of the parameter
  size_t getBitwidth() const { return bitwidth; }

private:
  mlir::Value value;
  std::string name;
  std::string type;
  size_t bitwidth;
};

/// Description of the component in the library.json. Obtained after parsing
/// library.json.
struct VHDLModuleDescription {
  VHDLModuleDescription(
      std::string tempPath = {}, std::string tempConcrMethod = {},
      llvm::SmallVector<std::string> tempGenerators = {},
      llvm::SmallVector<VHDLGenericParam> tempGenerics = {},
      llvm::SmallVector<VHDLDescParameter> tempInputPorts = {},
      llvm::SmallVector<VHDLDescParameter> tempOutputPorts = {})
      : path(tempPath), concretizationMethod(tempConcrMethod),
        generators(tempGenerators), generics(tempGenerics),
        inputPorts(tempInputPorts), outputPorts(tempOutputPorts) {}
  /// Path to a file with VHDL component's achitecture.
  /// Either .vhd or binary, depends on concretization method
  std::string getPath() const { return path; }
  /// Method of concretization, that is the way we get the component's
  /// architecture. There're two possible options:
  /// 1. "GENERIC". Means that we simply obtain an architecture from a file
  /// 2. "GENERATOR". Means that the architecture needs further concretization
  /// and it's necessary to get some parameters for it from netlist.mlir
  std::string getConcretizationMethod() const { return concretizationMethod; };
  /// Parameters' names that will be used in generation
  const llvm::SmallVector<std::string> &getGenerators() const {
    return generators;
  }
  /// Parameters' names that are used in VHDL components' declarations (that is
  /// VHDL generics)
  const llvm::SmallVector<VHDLGenericParam> &getGenerics() const {
    return generics;
  }
  /// Input ports as they exist in library.json
  const llvm::SmallVector<VHDLDescParameter> &getInputPorts() const {
    return inputPorts;
  }
  /// Output ports as they exist in library.json
  const llvm::SmallVector<VHDLDescParameter> &getOutputPorts() const {
    return outputPorts;
  }
  /// Function that concretizes a library component, that is gets a module with
  /// exact values (BITWIDTH, SIZE and so on). It uses modParameters string,
  /// obtained from processing netlist.mlir, to get all required information.
  VHDLModule concretize(std::string modName, std::string modParameters) const;
  /// Function that declares all the components (i.e. gets "component" in VHDL
  /// syntax)
  std::string declare() const;

private:
  std::string path;
  std::string concretizationMethod;
  llvm::SmallVector<std::string> generators;
  llvm::SmallVector<VHDLGenericParam> generics;
  llvm::SmallVector<VHDLDescParameter> inputPorts;
  llvm::SmallVector<VHDLDescParameter> outputPorts;
};

/// VHDL module, that is VHDLModuleDescription + extern module operations from
/// netlist.mlir
struct VHDLModule {
  VHDLModule(std::string tempModName = "", std::string tempMtext = "",
             llvm::SmallVector<std::string> tempModParameters = {},
             llvm::SmallVector<VHDLModParameter> tempInputs = {},
             llvm::SmallVector<VHDLModParameter> tempOutputs = {},
             const VHDLModuleDescription &tempModDesc = {})
      : modName(tempModName), modText(tempMtext),
        modParameters(tempModParameters), concrInputs(tempInputs),
        concrOutputs(tempOutputs), modDesc(tempModDesc) {}
  /// Concretized module's name, i.g. "fork"
  std::string getModName() const { return modName; }
  /// Component's definition and architecture, obtained from "concretize"
  /// function
  const std::string &getModText() const { return modText; }
  /// Discriminating parameters, i.e. parameters string parsed into separate
  /// parts
  const llvm::SmallVector<std::string> &getModParameters() const {
    return modParameters;
  }
  /// Concretized module's inputs
  const llvm::SmallVector<VHDLModParameter> &getInputs() const {
    return concrInputs;
  }
  /// Concretized module's outputs
  const llvm::SmallVector<VHDLModParameter> &getOutputs() const {
    return concrOutputs;
  }
  /// Reference to the corresponding template in VHDLComponentLibrary
  const VHDLModuleDescription &getModDesc() const { return modDesc; }
  std::string getComponentsDeclaration() {
    std::string res = "component " + modName + " " + modDesc.declare();
    return res;
  }
  VHDLInstance instantiate(std::string instName,
                           circt::hw::InstanceOp &innerOp) const;

private:
  std::string modName;
  std::string modText;
  llvm::SmallVector<std::string> modParameters;
  llvm::SmallVector<VHDLModParameter> concrInputs;
  llvm::SmallVector<VHDLModParameter> concrOutputs;
  const VHDLModuleDescription &modDesc;
};

VHDLModule VHDLModuleDescription::concretize(std::string modName,
                                             std::string modParameters) const {
  // Split the given modParameters string into separate parts for convenience
  llvm::SmallVector<std::string> modParametersVec =
      parseDiscriminatingParameters(modParameters);
  // Fill the map that links between themselves template and exact names.
  // Template names - parameters that are contained in generators and generics
  // arrays
  llvm::StringMap<std::string> params{};
  auto it = modParametersVec.begin();
  for (auto &i : generators) {
    params.insert(std::pair(i, *it));
    ++it;
  }
  for (auto &i : generics) {
    params.insert(std::pair(i.getName(), *it));
    ++it;
  }
  // Future VHDLModule inputs
  llvm::SmallVector<VHDLModParameter> inputs;
  for (auto &i : inputPorts) {
    size_t mod_size;
    std::string mod_amount;
    if (i.getSize() == "-1") {
      // If parameter "size" for the input doesn't exist in library.json it
      // means that input is one-dimentional, "value"
      mod_size = 1;
      mod_amount = "value";
    } else {
      // Otherwise it's an "array", and we have the length of it in params
      auto s = params.find(i.getSize());
      mod_size = std::atoi(s->second.c_str());
      mod_amount = "array";
    }
    std::string mod_type;
    size_t mod_bitwidth;
    if (i.getBitwidth() == "-1") {
      // If parameter "bitwidth" for the input doesn't exist in library.json it
      // means that input is 1-bit, "std_logic"
      mod_type = "std_logic";
      mod_bitwidth = 1;
    } else {
      // Otherwise the concrete bitwidth is either given or should also be found
      // in params
      mod_type = "std_logic_vector";
      if (is_number(i.getBitwidth()))
        mod_bitwidth = std::atoi(i.getBitwidth().c_str());
      else {
        auto b = params.find(i.getBitwidth());
        mod_bitwidth = std::atoi(b->second.c_str());
      }
    }
    // If input doesn't exist among the signals (0 - size), it's an empty signal
    if (mod_size)
      inputs.push_back(VHDLModParameter(true, i.getName(), mod_type, mod_amount,
                                        mod_size, mod_bitwidth));
    else
      inputs.push_back(VHDLModParameter(false, i.getName(), mod_type,
                                        mod_amount, mod_size, mod_bitwidth));
  }
  // Future VHDLModule outputs
  llvm::SmallVector<VHDLModParameter> outputs;
  for (auto &i : outputPorts) {
    size_t mod_size;
    std::string mod_amount;
    if (i.getSize() != "-1") {
      // If parameter "size" for the output doesn't exist in library.json it
      // means that input is one-dimentional, "value"
      auto s = params.find(i.getSize());
      mod_size = std::atoi(s->second.c_str());
      mod_amount = "array";
    } else {
      // Otherwise it's an "array", and we have the length of it in params
      mod_size = 1;
      mod_amount = "value";
    }
    std::string mod_type;
    size_t mod_bitwidth;
    if (i.getBitwidth() == "-1") {
      // If parameter "bitwidth" for the output doesn't exist in library.json it
      // means that output is 1-bit, "std_logic"
      mod_type = "std_logic";
      mod_bitwidth = 1;
    } else {
      // Otherwise the concrete bitwidth is either given or should also be found
      // in params
      mod_type = "std_logic_vector";
      if (is_number(i.getBitwidth()))
        mod_bitwidth = std::atoi(i.getBitwidth().c_str());
      else {
        auto b = params.find(i.getBitwidth());
        mod_bitwidth = std::atoi(b->second.c_str());
      }
    }
    // If output doesn't exist among the signals (0 - size), it's an empty
    // signal
    if (mod_size)
      outputs.push_back(VHDLModParameter(true, i.getName(), mod_type,
                                         mod_amount, mod_size, mod_bitwidth));
    else
      outputs.push_back(VHDLModParameter(false, i.getName(), mod_type,
                                         mod_amount, mod_size, mod_bitwidth));
  }
  std::string modText;
  // Open a file with component concretization data
  std::ifstream file;
  // Read as file
  std::stringstream buffer;
  if (concretizationMethod == "GENERATOR") {
    // In case of the generator we're looking for binary
    std::string commandLineArguments;
    auto i = modParametersVec.begin();
    size_t k = generators.size() + generics.size();
    size_t m = generators.size();
    // Collecting discriminating params for command line
    while (k > 0) {
      commandLineArguments += " " + (*i);
      ++i;
      --k;
    }
    i = modParametersVec.begin();
    while (m > 0) {
      modName += "_" + (*i);
      ++i;
      --m;
    }
    // Create a temporary text.txt
    std::string resultPath = path + commandLineArguments + " > test.txt";
    std::system(resultPath.c_str());
    file.open("test.txt");
    // ... and delete it
    std::system("rm test.txt");
  } else if (concretizationMethod == "GENERIC")
    // In case of generic we're looking for ordinary file
    file.open(path);
  else
    // Error
    llvm::errs() << "Wrong concredization method";
  buffer << file.rdbuf();
  modText = buffer.str();
  // Obtain VHDLModule::modParameters vector
  llvm::SmallVector<std::string> resultParamArr{};
  for (size_t i = generators.size(); i < generators.size() + generics.size();
       ++i) {
    resultParamArr.push_back(modParametersVec[i]);
  }
  // Get rid of "handshake" and "arith" prefixes in module name
  modName = modName.substr(modName.find('_') + 1);
  return VHDLModule(modName, modText, resultParamArr, inputs, outputs, *this);
}

std::string VHDLModuleDescription::declare() const {
  std::string result;
  result += " is\ngeneric(\n";
  for (auto &i : generics) {
    result += i.getName() + " : " + i.getType() + ";\n";
  }
  result[result.size() - 2] = ')';
  result[result.size() - 1] = ';';
  result += "\nport(\nclk : in std_logic;\nrst : in std_logic;\n";
  // explore inputPorts
  for (auto &i : inputPorts) {
    std::string resultType;
    std::string resultControlType;
    std::string resultSize;
    std::string resultBitwidth;
    std::string mod_amount;
    // get the description of the input, types of its size and bitwidth...
    if (i.getSize() != "-1") {
      mod_amount = "array";
    } else {
      mod_amount = "value";
    }
    std::string mod_type;
    if (i.getBitwidth() == "-1") {
      mod_type = "std_logic";
    } else {
      mod_type = "std_logic_vector";
    }
    // ... and corresponding actual values
    std::string s, b;
    if (is_number(i.getSize())) {
      s = std::to_string(std::atoi(i.getSize().c_str()) - 1);
    } else {
      s = i.getSize() + " - 1";
    }
    if (is_number(i.getBitwidth())) {
      b = std::to_string(std::atoi(i.getBitwidth().c_str()) - 1);
    } else {
      b = i.getBitwidth() + " - 1";
    }
    // According to previous investigation construct the input's declaration
    if (mod_amount == "array" && mod_type == "std_logic_vector") {
      resultType = "data_array";
      resultControlType = "std_logic_vector";
      resultSize = "(" + s + " downto 0)";
      resultBitwidth = "(" + b + " downto 0)";
    } else if (mod_amount == "value" && mod_type == "std_logic") {
      resultType = "std_logic";
      resultControlType = "std_logic";
    } else if (mod_amount == "value" && mod_type == "std_logic_vector") {
      resultType = "std_logic_vector";
      resultControlType = "std_logic";
      resultBitwidth = "(" + b + " downto 0)";
    } else {
      resultType = "std_logic_vector";
      resultControlType = "std_logic";
      resultSize = "(" + s + " downto 0)";
    }

    if (i.getType() == "dataflow" || i.getType() == "data") {
      result += i.getName() + " : in " + resultType + resultSize +
                resultBitwidth + ";\n";
    }
    // control signals if we need them
    if (i.getType() == "dataflow" || i.getType() == "control") {
      result +=
          i.getName() + "_valid : in " + resultControlType + resultSize + ";\n";
      result += i.getName() + "_ready : out " + resultControlType + resultSize +
                ";\n";
    }
  }
  // explore outputPorts
  for (auto &i : outputPorts) {
    std::string resultType;
    std::string resultSize;
    std::string resultBitwidth;
    std::string resultControlType;
    std::string mod_amount;
    // Get the description of the input, types of its size and bitwidth...
    if (i.getSize() != "-1") {
      mod_amount = "array";
    } else {
      mod_amount = "value";
    }
    std::string mod_type;
    if (i.getBitwidth() == "-1") {
      mod_type = "std_logic";
    } else {
      mod_type = "std_logic_vector";
    }
    // ... and corresponding actual values
    std::string s, b;
    if (is_number(i.getSize())) {
      s = std::to_string(std::atoi(i.getSize().c_str()) - 1);
    } else {
      s = i.getSize() + " - 1";
    }
    if (is_number(i.getBitwidth())) {
      b = std::to_string(std::atoi(i.getBitwidth().c_str()) - 1);
    } else {
      b = i.getBitwidth() + " - 1";
    }
    // According to previous investigation construct the output's declaration
    if (mod_amount == "array" && mod_type == "std_logic_vector") {
      resultType = "data_array";
      resultControlType = "std_logic_vector";
      resultSize = "(" + s + " downto 0)";
      resultBitwidth = "(" + b + " downto 0)";
    } else if (mod_amount == "value" && mod_type == "std_logic") {
      resultType = "std_logic";
      resultControlType = "std_logic";
    } else if (mod_amount == "value" && mod_type == "std_logic_vector") {
      resultType = "std_logic_vector";
      resultControlType = "std_logic";
      resultBitwidth = "(" + b + " downto 0)";
    } else {
      resultType = "std_logic_vector";
      resultControlType = "std_logic";
      resultSize = "(" + s + " downto 0)";
    }

    if (i.getType() == "dataflow" || i.getType() == "data") {
      result += i.getName() + " : out " + resultType + resultSize +
                resultBitwidth + ";\n";
    }
    // control signals if we need them
    if (i.getType() == "dataflow" || i.getType() == "control") {
      result += i.getName() + "_valid : out " + resultControlType + resultSize +
                ";\n";
      result +=
          i.getName() + "_ready : in " + resultControlType + resultSize + ";\n";
    }
  }
  result[result.size() - 2] = ')';
  result[result.size() - 1] = ';';
  result += "\nend component;\n";
  return result;
}

/// VHDL instance, that is VHDLModule + inner module operations from
/// netlist.mlir
struct VHDLInstance {
  VHDLInstance(std::string tempInstanceName, std::string tempItext,
               llvm::SmallVector<VHDLInstParameter> tempInputs,
               llvm::SmallVector<VHDLInstParameter> tempOutputs,
               const VHDLModule &tempMod)
      : instanceName(tempInstanceName), instanceText(tempItext),
        inputs(tempInputs), outputs(tempOutputs), mod(tempMod) {}
  /// Instance name, for instance "fork0"
  std::string getInstName() const { return instanceName; }
  /// Instantiation of the component, i.e. "port mapping", assighnment between
  /// template component's inputs / outputs and other signals
  std::string getInstText() const { return instanceText; }
  /// Instance inputs, i.e. signals
  const llvm::SmallVector<VHDLInstParameter> &getInputs() const {
    return inputs;
  }
  /// Instance outputs, i.e. signals
  const llvm::SmallVector<VHDLInstParameter> &getOutputs() const {
    return outputs;
  }
  /// Reference to the corresponding module in VHDLModuleLibrary
  const VHDLModule &getConcrModule() const { return mod; }

private:
  std::string instanceName;
  std::string instanceText;
  llvm::SmallVector<VHDLInstParameter> inputs;
  llvm::SmallVector<VHDLInstParameter> outputs;
  const VHDLModule &mod;
};

/// Function that instantiates a module component, that is gets an instance with
/// an exact number. It uses innerOp operation, obtained from processing
/// netlist.mlir, to get all required information.
VHDLInstance VHDLModule::instantiate(std::string instName,
                                     circt::hw::InstanceOp &innerOp) const {
  // Shorten the name
  instName = instName.substr(instName.find('_') + 1);
  // Counter for innerOp argumentss or results array
  size_t num = 0;
  // Process the input signal
  llvm::SmallVector<VHDLInstParameter> inputs{};
  for (Value opr : innerOp.getOperands()) {
    std::string name = instName + "_" + innerOp.getArgumentName(num).str();
    auto t = opr.getType();
    std::string type = extractType(t).first;
    size_t bitwidth = extractType(t).second;
    // Both clock and reset always exist. There's no need in them among inputs.
    if (innerOp.getArgumentName(num).str() != "clock" &&
        innerOp.getArgumentName(num).str() != "reset")
      inputs.push_back(VHDLInstParameter(opr, name, type, bitwidth));
    ++num;
  }
  num = 0;
  // Process the output signal
  llvm::SmallVector<VHDLInstParameter> outputs{};
  for (Value opr : innerOp.getResults()) {
    std::string name = instName + "_" + innerOp.getResultName(num).str();
    auto t = opr.getType();
    std::string type = extractType(t).first;
    size_t bitwidth = extractType(t).second;
    outputs.push_back(VHDLInstParameter(opr, name, type, bitwidth));
    ++num;
  }

  std::string compName = modName.substr(0, modName.find("."));
  std::string instText = instName + " : " + compName + " generic map(";
  // "0", i.e. none, types are not allowed, so always replace them with 1
  for (auto &i : modParameters)
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
  auto concrInpIt = getInputs().begin();
  auto inpIt = inputs.begin();
  auto descInpIt = getModDesc().getInputPorts().begin();
  // We process the list of template inputPorts (inputs from library.json) and
  // assign them to signals
  while (descInpIt != getModDesc().getInputPorts().end()) {
    auto descName = descInpIt->getName();
    auto channelType = descInpIt->getType();
    if (concrInpIt->getFlag()) {
      // If not an empty signal
      auto count = concrInpIt->getAmount();
      auto bitLen = concrInpIt->getType();
      if (count == "value" && bitLen == "std_logic") {
        auto insName = inpIt->getName();
        // std_logic
        if (channelType == "dataflow" || channelType == "data")
          instText += descName + " => " + insName + ",\n";

        if (channelType == "dataflow" || channelType == "control") {
          instText += descName + "_valid => " + insName + "_valid,\n";
          instText += descName + "_ready => " + insName + "_ready,\n";
        }
        ++inpIt;
      } else if (count == "array" && bitLen == "std_logic") {
        // std_logic_vector(COUNT)
        auto prevIt = inpIt;
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          if (channelType == "dataflow" || channelType == "data")
            instText +=
                descName + "(" + std::to_string(k) + ") => " + insName + ",\n";

          ++inpIt;
        }
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_valid(" + std::to_string(k) + ") => " +
                        insName + "_valid,\n";

          ++inpIt;
        }
        // According to VHDL compiler's rules ports of the same logic should go
        // in a row, so cycles for ready and valid signals are splitted up
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_ready(" + std::to_string(k) + ") => " +
                        insName + "_ready,\n";

          ++inpIt;
        }
      } else if (count == "value" && bitLen == "std_logic_vector") {
        // std_logic_vector(BITWIDTH)
        auto insName = inpIt->getName();
        std::string z;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        if (concrInpIt->getBitwidth() <= 1)
          z = "(0)";

        if (channelType == "dataflow" || channelType == "data")
          instText += descName + z + " => " + insName + ",\n";

        if (channelType == "dataflow" || channelType == "control") {
          instText += descName + "_valid" + " => " + insName + "_valid,\n";
          instText += descName + "_ready" + " => " + insName + "_ready,\n";
        }
        ++inpIt;
      } else if (count == "array" && bitLen == "std_logic_vector") {
        // data_array(BITWIDTH)(COUNT)
        auto prevIt = inpIt;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          std::string z;
          if (concrInpIt->getBitwidth() <= 1)
            z = "(0)";
          if (channelType == "dataflow" || channelType == "data")
            instText += descName + "(" + std::to_string(k) + ")" + z + " => " +
                        insName + ",\n";
          ++inpIt;
        }
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_valid(" + std::to_string(k) + ")" +
                        " => " + insName + "_valid,\n";
          ++inpIt;
        }
        // The same about order of valid and ready signals as above
        inpIt = prevIt;
        for (size_t k = 0; k < concrInpIt->getSize(); ++k) {
          auto insName = inpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_ready(" + std::to_string(k) + ")" +
                        " => " + insName + "_ready,\n";
          ++inpIt;
        }
      }
    } else {
      // If the input doesn't exist among signals in netlist.mlir it's empty
      std::string extraSignalName = instName + "_x" + descName;
      if (channelType == "dataflow" || channelType == "data")
        instText += descName + "(0) => " + extraSignalName + ",\n";
      if (channelType == "dataflow" || channelType == "control") {
        instText += descName + "_valid(0) => " + extraSignalName + "_valid,\n";
        instText += descName + "_ready(0) => " + extraSignalName + "_ready,\n";
      }
    }
    ++concrInpIt;
    ++descInpIt;
  }

  // Outputs
  auto concrOutpIt = getOutputs().begin();
  auto outpIt = outputs.begin();
  auto descOutpIt = getModDesc().getOutputPorts().begin();
  // We process the list of template inputPorts (inputs from library.json) and
  // assign them to signals
  while (descOutpIt != getModDesc().getOutputPorts().end()) {
    auto descName = descOutpIt->getName();
    auto channelType = descOutpIt->getType();
    if (concrOutpIt->getFlag()) {
      // If not an empty signal
      auto count = concrOutpIt->getAmount();
      auto bitLen = concrOutpIt->getType();
      if (count == "value" && bitLen == "std_logic") {
        auto insName = outpIt->getName();
        // std_logic
        if (channelType == "dataflow" || channelType == "data")
          instText += descName + " => " + insName + ",\n";
        if (channelType == "dataflow" || channelType == "control") {
          instText += descName + "_valid => " + insName + "_valid,\n";
          instText += descName + "_ready => " + insName + "_ready,\n";
        }
        ++outpIt;
      } else if (count == "array" && bitLen == "std_logic") {
        // std_logic_vector(COUNT)
        auto prevIt = outpIt;
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          if (channelType == "dataflow" || channelType == "data")
            instText +=
                descName + "(" + std::to_string(k) + ") => " + insName + ",\n";
          ++outpIt;
        }
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_valid(" + std::to_string(k) + ") => " +
                        insName + "_valid,\n";
          ++outpIt;
        }
        // According to VHDL compiler's rules ports of the same logic should go
        // in a row, so cycles for ready and valid signals are splitted up
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_ready(" + std::to_string(k) + ") => " +
                        insName + "_ready,\n";
          ++outpIt;
        }
      } else if (count == "value" && bitLen == "std_logic_vector") {
        // std_logic_vector(BITWIDTH)
        auto insName = outpIt->getName();
        std::string z;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        if (concrOutpIt->getBitwidth() <= 1)
          z = "(0)";
        if (channelType == "dataflow" || channelType == "data")
          instText += descName + z + " => " + insName + ",\n";
        if (channelType == "dataflow" || channelType == "control") {
          instText += descName + "_valid" + " => " + insName + "_valid,\n";
          instText += descName + "_ready" + " => " + insName + "_ready,\n";
        }
        ++outpIt;
      } else if (count == "array" && bitLen == "std_logic_vector") {
        // data_array(BITWIDTH)(COUNT)
        auto prevIt = outpIt;
        // This if is for wiring between parameters with (0 downto 0) and
        // std_logic types. (0 downto 0) automatically get (0) index.
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          std::string z;
          if (concrOutpIt->getBitwidth() <= 1)
            z = "(0)";
          if (channelType == "dataflow" || channelType == "data")
            instText += descName + "(" + std::to_string(k) + ")" + z + " => " +
                        insName + ",\n";
          ++outpIt;
        }
        outpIt = prevIt;
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_valid(" + std::to_string(k) + ")" +
                        " => " + insName + "_valid,\n";
          ++outpIt;
        }
        outpIt = prevIt;
        // The same about order of valid and ready signals as above
        for (size_t k = 0; k < concrOutpIt->getSize(); ++k) {
          auto insName = outpIt->getName();
          if (channelType == "dataflow" || channelType == "control")
            instText += descName + "_ready(" + std::to_string(k) + ")" +
                        " => " + insName + "_ready,\n";
          ++outpIt;
        }
      }
    } else {
      // If the output doesn't exist among signals in netlist.mlir it's empty
      std::string extraSignalName = instName + "_y" + descName;
      if (channelType == "dataflow" || channelType == "data")
        instText += descName + "(0) => " + extraSignalName + ",\n";
      if (channelType == "dataflow" || channelType == "control") {
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

// ============================================================================
// Wrappers
// ============================================================================

/// Get a module
VHDLModule getMod(StringRef extName, VHDLComponentLibrary &jsonLib) {
  // extern Module Name = name + parameters with '.' as delimiter
  auto p = splitExtName(extName);
  std::string modName = p.first;
  std::string modParameters = p.second;
  // find external module in VHDLComponentLibrary
  llvm::StringMapIterator<VHDLModuleDescription> comp = jsonLib.find(modName);

  if (comp == jsonLib.end()) {
    llvm::errs() << "Unable to find the element in the components' library\n";
    return VHDLModule({}, {}, {}, {}, {}, {});
  }
  const VHDLModuleDescription &desc = (*comp).second;
  auto mod = desc.concretize(modName, modParameters);
  return mod;
};

/// Get an instance
VHDLInstance getInstance(StringRef extName, StringRef name,
                         VHDLModuleLibrary &modLib,
                         circt::hw::InstanceOp innerOp) {
  // find external module in VHDLModuleLibrary
  auto comp = modLib.find(extName);
  if (comp == modLib.end()) {
    llvm::errs() << "Unable to find the element in the instances' library\n";
    return VHDLInstance({}, {}, {}, {}, {});
  }
  const VHDLModule &desc = (*comp).second;
  auto inst = desc.instantiate(name.str(), innerOp);
  return inst;
}

// ============================================================================
// Functions that parse extern files
// ============================================================================

/// Get a cpp representation for given library.json file, i.e.
/// VHDLComponentLibrary
VHDLComponentLibrary parseJSON() {
  // Load JSON library
  std::ifstream lib;
  lib.open(LIBRARY_PATH);

  VHDLComponentLibrary m{};
  if (!lib.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    return m;
  }
  // Read as file
  std::stringstream buffer;
  buffer << lib.rdbuf();
  std::string jsonStr = buffer.str();

  // Parse the library
  auto jsonLib = llvm::json::parse(StringRef(jsonStr));

  if (!jsonLib) {
    llvm::errs() << "Library JSON could not be parsed"
                 << "\n";
    return m;
  }

  if (!jsonLib->getAsObject()) {
    llvm::errs() << "Library JSON is not a valid JSON"
                 << "\n";
    return m;
  }
  // Parse elements in json
  for (auto item : *jsonLib->getAsObject()) {
    // ModuleName is either "arith" or "handshake"
    std::string moduleName = item.getFirst().str();
    auto moduleArray = item.getSecond().getAsArray();
    for (auto c = moduleArray->begin(); c != moduleArray->end(); ++c) {
      // c is iterator, which points on a specific component's scheme inside
      // arith / handshake class
      auto obj = c->getAsObject();
      auto jsonComponents = obj->get("components")->getAsArray();
      auto jsonConcretizationMethod =
          obj->get("concretization_method")->getAsString();
      auto jsonGenerators = obj->get("generators")->getAsArray();
      auto jsonGenerics = obj->get("generics")->getAsArray();
      auto jsonPorts = obj->get("ports")->getAsObject();
      // Creating corresponding VHDLModuleDescription variables
      std::string concretizationMethod = jsonConcretizationMethod.value().str();

      llvm::SmallVector<std::string> generators{};
      for (auto i = jsonGenerators->begin(); i != jsonGenerators->end(); ++i)
        generators.push_back(i->getAsString().value().str());

      llvm::SmallVector<VHDLGenericParam> generics{};
      for (auto i = jsonGenerics->begin(); i != jsonGenerics->end(); ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto type = ob->get("type")->getAsString().value().str();
        generics.push_back(VHDLGenericParam(name, type));
      }
      // Get input ports
      llvm::SmallVector<VHDLDescParameter> inputPorts{};
      auto jsonInputPorts = jsonPorts->get("in")->getAsArray();
      for (auto i = jsonInputPorts->begin(); i != jsonInputPorts->end(); ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto type = ob->get("type")->getAsString().value().str();
        std::string size;
        if (ob->find("size") != ob->end())
          size = ob->get("size")->getAsString().value().str();
        std::string bitwidth;
        if (ob->find("bitwidth") != ob->end())
          bitwidth = ob->get("bitwidth")->getAsString().value().str();
        inputPorts.push_back(VHDLDescParameter(name, type, size, bitwidth));
      }
      // Get output ports
      llvm::SmallVector<VHDLDescParameter> outputPorts{};
      auto jsonOutputPorts = jsonPorts->get("out")->getAsArray();
      for (auto i = jsonOutputPorts->begin(); i != jsonOutputPorts->end();
           ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto type = ob->get("type")->getAsString().value().str();
        std::string size;
        if (ob->find("size") != ob->end())
          size = ob->get("size")->getAsString().value().str();
        std::string bitwidth;
        if (ob->find("bitwidth") != ob->end())
          bitwidth = ob->get("bitwidth")->getAsString().value().str();
        outputPorts.push_back(VHDLDescParameter(name, type, size, bitwidth));
      }

      for (auto i = jsonComponents->begin(); i != jsonComponents->end(); ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto path = ob->get("path")->getAsString().value().str();
        std::string key_name = moduleName + "_" + name;
        // Inserting our component into library
        m.insert(std::pair(key_name, VHDLModuleDescription(
                                         path, concretizationMethod, generators,
                                         generics, inputPorts, outputPorts)));
      }
    }
  }
  lib.close();
  return m;
}

/// Get modules from extern operations in netlist.mlir
VHDLModuleLibrary parseExternOps(mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 VHDLComponentLibrary &m) {
  VHDLModuleLibrary modLib{};
  for (auto modOp : module->getOps<hw::HWModuleExternOp>()) {
    auto extName = modOp.getName();
    auto i = getMod(extName, m);
    modLib.insert(std::pair(extName, i));
  }
  return modLib;
}

/// Get instances from extern operations in netlist.mlir
VHDLInstanceLibrary parseInnerOp(mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 VHDLModuleLibrary &modLib) {
  VHDLInstanceLibrary instLib{};
  for (auto modOp : module->getOps<hw::HWModuleOp>()) {
    for (auto innerOp : modOp.getOps<hw::InstanceOp>()) {
      auto extName = innerOp.getReferencedModuleName();
      auto name = innerOp.getInstanceName();
      auto i = getInstance(extName, name, modLib, innerOp);
      instLib.insert(std::pair(innerOp.getOperation(), i));
    }
  }
  return instLib;
}

/// Get the description of the head instance, "hw.module"
std::pair<Operation *, VHDLInstance>
parseModule(mlir::OwningOpRef<mlir::ModuleOp> &module) {
  for (auto modOp : module->getOps<hw::HWModuleOp>()) {
    auto inputs = modOp.getPorts().inputs;
    auto outputs = modOp.getPorts().outputs;
    SmallVector<VHDLInstParameter> ins{};
    for (auto i : inputs) {
      auto t = i.type;
      auto name = i.getName().str();
      auto type = extractType(t).first;
      auto bitwidth = extractType(t).second;
      if (name != "clock" && name != "reset")
        ins.push_back(VHDLInstParameter({}, name, type, bitwidth));
    }
    SmallVector<VHDLInstParameter> outs{};
    for (auto i : outputs) {
      auto t = i.type;
      auto name = i.getName().str();
      auto type = extractType(t).first;
      auto bitwidth = extractType(t).second;
      outs.push_back(VHDLInstParameter({}, name, type, bitwidth));
    }
    return std::pair(modOp.getOperation(),
                     VHDLInstance(modOp.getName().str(), {}, ins, outs, {}));
  }
}

/// Get the description of the output instance, "hw.output".
/// Names are obtained from hw.module
std::pair<Operation *, VHDLInstance>
parseOut(mlir::OwningOpRef<mlir::ModuleOp> &module) {
  for (auto modOp : module->getOps<hw::HWModuleOp>()) {
    auto outputs = modOp.getPorts().outputs;
    SmallVector<VHDLInstParameter> outs{};
    for (auto i : outputs) {
      auto t = i.type;
      auto name = i.getName().str();
      auto type = extractType(t).first;
      auto bitwidth = extractType(t).second;
      outs.push_back(VHDLInstParameter({}, name, type, bitwidth));
    }
    for (auto outOp : modOp.getOps<hw::OutputOp>()) {
      return std::pair(outOp.getOperation(),
                       VHDLInstance(modOp.getName().str(), {}, outs, {}, {}));
    }
  }
}

// ============================================================================
// Export functions
// ============================================================================

/// Get supporting .vhd components that are used in ours, i.g. join
void getSupportFiles() {
  std::ifstream file;
  file.open("experimental/data/vhdl/supportfiles.vhd");
  std::stringstream buffer;
  buffer << file.rdbuf();
  llvm::outs() << buffer.str() << "\n";
}

/// Get existing components' architectures
void getModulesArchitectures(VHDLModuleLibrary &modLib) {
  // module arches
  llvm::StringMap<std::string> modulesArchitectures{};
  for (auto &i : modLib) {
    if (modulesArchitectures.find(i.getValue().getModName()) ==
        modulesArchitectures.end()) {
      modulesArchitectures.insert(std::pair(i.getValue().getModName(), ""));
      llvm::outs() << i.getValue().getModText() << "\n";
    }
  }
}

/// Get the declaration of head instance
void getEntityDeclaration(std::pair<Operation *, VHDLInstance> &hwmod) {
  auto inst = hwmod.second;
  std::string res;
  res += "clock : in std_logic;\n";
  res += "reset : in std_logic;\n";
  for (auto &i : inst.getInputs()) {
    res += i.getName();
    if (i.getBitwidth() > 1)
      res += " : in std_logic_vector (" + std::to_string(i.getBitwidth() - 1) +
             " downto 0);\n";
    else
      res += " : in std_logic;\n";
    if (i.getType() == "channel") {
      res += i.getName() + "_valid : in std_logic;\n";
      res += i.getName() + "_ready : out std_logic;\n";
    }
  }

  for (auto &i : inst.getOutputs()) {
    res += i.getName();
    if (i.getBitwidth() > 1)
      res += " : out std_logic_vector (" + std::to_string(i.getBitwidth() - 1) +
             " downto 0);\n";
    else
      res += " : out std_logic;\n";
    if (i.getType() == "channel") {
      res += i.getName() + "_valid : out std_logic;\n";
      res += i.getName() + "_ready : in std_logic;\n";
    }
  }

  res[res.length() - 2] = ')';
  res[res.length() - 1] = ';';
  res += "\n";
  llvm::outs() << res;
}

/// Get the declaration of other components
void getComponentsDeclaration(VHDLModuleLibrary &modLib) {
  llvm::StringMap<std::string> componentsDeclaration{};
  for (auto &i : modLib) {
    if (componentsDeclaration.find(i.getValue().getModName()) ==
        componentsDeclaration.end()) {
      componentsDeclaration.insert(std::pair(i.getValue().getModName(), ""));
      llvm::outs() << i.getValue().getComponentsDeclaration() << "\n";
    }
  }
}

/// Get the signals' declaration
void getSignalsDeclaration(VHDLInstanceLibrary &instanceLib) {
  for (auto &[op, mod] : instanceLib) {
    auto module = mod.getConcrModule();
    auto inputsIt = mod.getInputs().begin();
    auto outputsIt = mod.getOutputs().begin();
    llvm::outs() << "signal " << mod.getInstName() << "_clk : "
                 << "std_logic;\n";
    llvm::outs() << "signal " << mod.getInstName() << "_rst : "
                 << "std_logic;\n";
    // inputs
    for (auto &i : module.getInputs()) {
      if (i.getFlag()) {
        // If it's not an empty signal
        for (size_t j = 0; j < i.getSize(); ++j) {
          llvm::outs() << "signal " << inputsIt->getName() << " : ";
          if (i.getType() == "std_logic")
            llvm::outs() << i.getType() << ";\n";
          else if (i.getType() == "std_logic_vector")
            if (inputsIt->getBitwidth() <= 1)
              llvm::outs() << "std_logic;\n";
            else {
              size_t p = inputsIt->getBitwidth() - 1;
              llvm::outs() << i.getType() << "(" << p << " downto 0);\n";
            }
          else {
            llvm::errs() << "Wrong module type!\n";
            return;
          }
          if (inputsIt->getType() == "channel") {
            llvm::outs() << "signal " << inputsIt->getName() << "_valid : "
                         << "std_logic;\n";
            llvm::outs() << "signal " << inputsIt->getName() << "_ready : "
                         << "std_logic;\n";
          }
          ++inputsIt;
        }
      } else {
        // Empty signals also should be declared
        llvm::outs() << "signal " << mod.getInstName() + "_x" + i.getName()
                     << " : ";
        if (i.getType() == "std_logic")
          llvm::outs() << i.getType() << ";\n";
        else if (i.getType() == "std_logic_vector")
          if (i.getBitwidth() <= 1)
            llvm::outs() << "std_logic;\n";
          else {
            size_t p = i.getBitwidth() - 1;
            llvm::outs() << i.getType() << "(" << p << " downto 0);\n";
          }
        else {
          llvm::errs() << "Wrong module type!\n";
          return;
        }
        llvm::outs() << "signal " << mod.getInstName() + "_x" + i.getName()
                     << "_valid : std_logic;\n";
        llvm::outs() << "signal " << mod.getInstName() + "_x" + i.getName()
                     << "_ready : std_logic;\n";
      }
    }
    // outputs
    for (auto &i : module.getOutputs()) {
      if (i.getFlag()) {
        // If it's not an empty signal
        for (size_t j = 0; j < i.getSize(); ++j) {
          llvm::outs() << "signal " << outputsIt->getName() << " : ";
          if (i.getType() == "std_logic")
            llvm::outs() << i.getType() << ";\n";
          else if (i.getType() == "std_logic_vector")
            if (outputsIt->getBitwidth() <= 1)
              llvm::outs() << "std_logic;\n";
            else {
              size_t p = outputsIt->getBitwidth() - 1;
              llvm::outs() << i.getType() << "(" << p << " downto 0);\n";
            }
          else {
            llvm::errs() << "Wrong module type!\n";
            return;
          }
          if (outputsIt->getType() == "channel") {
            llvm::outs() << "signal " << outputsIt->getName() << "_valid : "
                         << "std_logic;\n";
            llvm::outs() << "signal " << outputsIt->getName() << "_ready : "
                         << "std_logic;\n";
          }
          ++outputsIt;
        }
      } else {
        // Empty signals also should be declared
        llvm::outs() << "signal " << mod.getInstName() + "_y" + i.getName()
                     << " : ";
        if (i.getType() == "std_logic")
          llvm::outs() << i.getType() << ";\n";
        else if (i.getType() == "std_logic_vector")
          if (i.getBitwidth() <= 1)
            llvm::outs() << "std_logic;\n";
          else {
            size_t p = i.getBitwidth() - 1;
            llvm::outs() << i.getType() << "(" << p << " downto 0);\n";
          }
        else {
          llvm::errs() << "Wrong module type!\n";
          return;
        }
        llvm::outs() << "signal " << mod.getInstName() + "_y" + i.getName()
                     << "_valid : std_logic;\n";
        llvm::outs() << "signal " << mod.getInstName() + "_y" + i.getName()
                     << "_ready : std_logic;\n";
      }
    }
    llvm::outs() << "\n";
  }
}

/// Get connections between signals
void getWiring(VHDLInstanceLibrary &instanceLib,
               std::pair<Operation *, VHDLInstance> &hwOut) {
  // we process every component line from netlist.mlir (== every instance
  // from instanceLib)
  for (auto &[op, mod] : instanceLib) {
    llvm::outs() << mod.getInstName() << "_clk <= clock;\n";
    llvm::outs() << mod.getInstName() << "_rst <= reset;\n";
    // Iterator through outputs of each instance
    auto opIt = mod.getOutputs().begin();
    llvm::StringMap<std::string> d;
    for (auto i : op->getUsers()) {
      // Get an operation - successor of the instance
      auto it = instanceLib.find(i);
      if (it != instanceLib.end()) {
        // The operation exists in instanceLib
        auto inst = it->second;
        // Iterator through inputs of each successor
        auto acIt = inst.getInputs().begin();
        for (Value opr : i->getOperands()) {
          Operation *defOp = opr.getDefiningOp();
          if (defOp && defOp == op && opIt->getValue() == acIt->getValue()) {
            // If our initial operation is the predeccessor of an input of the
            // successor
            d.insert(std::pair(acIt->getName(), opIt->getName()));
            if (acIt->getType() == "channel") {
              llvm::outs() << acIt->getName() << "_valid <= " << opIt->getName()
                           << "_valid;\n";
              llvm::outs() << opIt->getName() << "_ready <= " << acIt->getName()
                           << "_ready;\n";
            }
            if (acIt->getBitwidth() != opIt->getBitwidth())
              if (opIt->getBitwidth())
                llvm::outs() << acIt->getName()
                             << " <= std_logic_vector (resize(unsigned("
                             << opIt->getName() << "), " << acIt->getName()
                             << "\'length));\n";
              else
                llvm::outs()
                    << acIt->getName()
                    << " <= std_logic_vector (resize(unsigned(\'0\'&"
                    << opIt->getName() << "), " << acIt->getName() << "));\n";

            else
              llvm::outs() << acIt->getName() << " <= " << opIt->getName()
                           << ";\n";

            break;
          }
          ++acIt;
        }
      } else if (i == hwOut.first) {
        // Output module
        auto inst = hwOut.second;
        auto acIt = inst.getInputs().begin();
        for (Value opr : i->getOperands()) {
          Operation *defOp = opr.getDefiningOp();
          if (defOp && defOp == op) {
            // if our initial operation is the predeccessor of an input of the
            // successor
            if (d.find(acIt->getName()) == d.end()) {
              d.insert(std::pair(acIt->getName(), opIt->getName()));
              if (acIt->getType() == "channel") {
                llvm::outs() << acIt->getName()
                             << "_valid <= " << opIt->getName() << "_valid;\n";
                llvm::outs() << opIt->getName()
                             << "_ready <= " << acIt->getName() << "_ready;\n";
              }
              llvm::outs() << acIt->getName() << " <= " << opIt->getName()
                           << ";\n";
              break;
            }
          }
          ++acIt;
        }
      } else
        llvm::errs() << "Error in MLIR: it must be an output!\n";

      ++opIt;
    }
    llvm::outs() << "\n";
  }
}

/// Get modules instantiations, i.e. assignments between template
/// VHDLModuleDescription iputs and outputs and signals
void getModulesInstantiation(VHDLInstanceLibrary &instanceLib) {
  for (auto &[op, mod] : instanceLib) {
    llvm::outs() << mod.getInstText() << "\n";
  }
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

  // Get all necessary structures
  auto m = parseJSON();
  auto modLib = parseExternOps(module, m);
  auto instanceLib = parseInnerOp(module, modLib);
  auto hwOut = parseOut(module);
  auto hwMod = parseModule(module);

  // Obtain the module name
  std::string modName;
  for (auto modOp : module->getOps<hw::HWModuleOp>()) {
    modName = modOp.getName();
  }

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
               << modName << " is\nport (\n";
  getEntityDeclaration(hwMod);
  llvm::outs() << "end;\n\n";
  llvm::outs() << "architecture behavioral of " << modName << " is\n\n";
  getComponentsDeclaration(modLib);
  llvm::outs() << "\n";
  getSignalsDeclaration(instanceLib);
  llvm::outs() << "\nbegin\n\n";
  getWiring(instanceLib, hwOut);
  llvm::outs() << "\n";
  getModulesInstantiation(instanceLib);
  llvm::outs() << "end behavioral;\n\n";
  llvm::outs() << "\n";

  return 0;
}
