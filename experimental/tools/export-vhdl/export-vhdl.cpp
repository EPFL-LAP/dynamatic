//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Experimental tool that exports VHDL from a netlist-level IR expressed in a
// combination of the HW and ESI dialects.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/type_traits.h"

#include <any>
#include <cstdio>
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define LIBRARY_PATH "experimental/tools/export-vhdl/library.json"

using namespace llvm;
using namespace mlir;
using namespace circt;

struct VHDLParameter;
struct VHDLModuleDescription;
struct VHDLModule;
typedef llvm::StringMap<VHDLModuleDescription> VHDLComponentLibrary;
typedef llvm::StringMap<size_t> StoreComponentNumbers;
typedef llvm::SmallVector<std::string> string_arr;
typedef llvm::SmallVector<int> integer_arr;

std::string extractModName(StringRef extName);
std::string extractModGroup(std::string modName);
std::string extractModOwnName(std::string modName);
std::string extractModParameters(StringRef extName);

/*---------------------------------
             GENERAL
-----------------------------------*/

// Just an inner description in VHDLComponentLibrary lib
struct VHDLParameter {
  VHDLParameter(std::string tempName = "", std::string tempType = "")
      : name{tempName}, type{tempType} {};
  std::string getName() const { return name; }
  std::string getType() const { return type; }

private:
  std::string name;
  std::string type;
};

// Description of the component in the lib
struct VHDLModuleDescription {

  VHDLModuleDescription(std::string tempPath = {},
                        std::string tempConcrMethod = {},
                        llvm::SmallVector<VHDLParameter> tempParameters = {})
      : path(tempPath), concretizationMethod(tempConcrMethod),
        parameters(tempParameters) {}
  std::string getPath() const { return path; }
  std::string getConcretizationMethod() const { return concretizationMethod; };
  const llvm::SmallVector<VHDLParameter> &getParameters() const {
    return parameters;
  }
  VHDLModule concretize(std::string modName, std::string exactModName,
                        string_arr inputPorts, string_arr outputPorts,
                        std::string modParameters) const;

private:
  std::string path;
  std::string concretizationMethod;
  llvm::SmallVector<VHDLParameter> parameters;
};

// VHDL module, that is VHDLModuleDescription + actual parameters data
struct VHDLModule {
  VHDLModule(std::string tempModName,
             llvm::SmallVector<std::string> tempInputPorts,
             llvm::SmallVector<std::string> tempOutputPorts,
             std::string tempMtext,
             llvm::StringMap<std::string> tempModParameters,
             const VHDLModuleDescription &tempModDesc)
      : modName(tempModName), inputPorts(tempInputPorts),
        outputPorts(tempOutputPorts), modText(tempMtext),
        modParameters(tempModParameters), modDesc(tempModDesc) {}
  std::string getModName() const { return modName; }
  const std::string &getModText() const { return modText; }
  const llvm::StringMap<std::string> &getModParameters() const {
    return modParameters;
  }
  const VHDLModuleDescription &getModDesc() const { return modDesc; }
  const llvm::SmallVector<std::string> &getInputPorts() const {
    return inputPorts;
  }
  const llvm::SmallVector<std::string> &getOutputPorts() const {
    return outputPorts;
  }

private:
  std::string modName;
  llvm::SmallVector<std::string> inputPorts;
  llvm::SmallVector<std::string> outputPorts;
  std::string modText;
  llvm::StringMap<std::string> modParameters;
  const VHDLModuleDescription &modDesc;
};

// delete extra spaces from a string
void deleteSpaces(std::string &str) {
  str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
}

// extract a number from the middle of the given string and move it to it's end
std::string moveNumber(std::string const &str) {
  auto copystr = str;
  std::string dig{};
  auto ind = std::find_if(str.begin(), str.end(), ::isdigit);
  if (ind != str.end()) {
    while (isdigit(*ind)) {
      dig += (*ind);
      ++ind;
    }
    auto i = copystr.find(dig);
    copystr = copystr.substr(0, i) + copystr.substr(i + dig.length());
    copystr += '@' + dig;
  }
  return copystr;
}

// Get a module corresponding the given component and data
VHDLModule VHDLModuleDescription::concretize(std::string modName,
                                             std::string exactModName,
                                             string_arr inputPorts,
                                             string_arr outputPorts,
                                             std::string modParameters) const {
  // parse the string with parameters to get a vector for convenience
  llvm::StringMap<std::string> modParametersVec;
  std::stringstream ss(modParameters);
  for (auto &i : parameters) {
    std::string type = i.getType();
    std::string name{};
    std::getline(ss, name, '_');
    deleteSpaces(name);
    if (type == "string_arr") {
      std::string testName = "smth";
      while (ss.good()) {
        std::getline(ss, testName, '_');
        name += testName + '_';
      }
    }
    modParametersVec.insert(std::pair(i.getName(), name));
  }

  // open a file with component concretization data
  std::ifstream file;
  if (modName != exactModName) {
    std::string newPath{};

    if (exactModName == "predicate") {
      newPath =
          path + "/" + modParametersVec.find("PREDICATE")->getValue() + ".vhd";
      modName += '_' + modParametersVec.find("PREDICATE")->getValue();
    } else {
      auto libNameLast = extractModOwnName(modName);
      int ind = path.find(libNameLast);
      newPath = path.substr(0, ind) + extractModOwnName(exactModName) + ".vhd";
      modName = exactModName;
    }
    file.open(newPath);
  } else {
    file.open(path);
  }

  if (!file.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    file.close();
    return VHDLModule(modName, inputPorts, outputPorts, {}, modParametersVec,
                      *this);
  }
  // Read as file
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string BufferStr = buffer.str();
  file.close();
  // build a module
  return VHDLModule(modName, inputPorts, outputPorts, BufferStr,
                    modParametersVec, *this);
}

// Get a cpp representation for given .json file
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
  // parse elements in json
  for (auto item : *jsonLib->getAsObject()) {
    auto key_name = item.first.str();
    auto path = item.second.getAsObject()
                    ->find("path")
                    ->second.getAsString()
                    .value()
                    .str();
    auto concretizationMethod = item.second.getAsObject()
                                    ->find("concretization_method")
                                    ->second.getAsString()
                                    .value()
                                    .str();
    auto parameters =
        item.getSecond().getAsObject()->get("parameters")->getAsArray();

    llvm::SmallVector<VHDLParameter> components{};
    for (auto i = parameters->begin(); i != parameters->end(); ++i) {
      auto obj = i->getAsObject();
      auto name = obj->get("name")->getAsString().value().str();
      auto type = obj->get("type")->getAsString().value().str();
      components.push_back(VHDLParameter(name, type));
    }

    m.insert(std::pair(key_name, VHDLModuleDescription(
                                     path, concretizationMethod, components)));
  }
  lib.close();

  return m;
}

/*---------------------------------
             CONCRETIZATION
-----------------------------------*/

// Get component main name (e.g. handshake.fork)
std::string extractModName(StringRef extName) {
  size_t first_ = extName.find('_', 0);
  size_t second_ = extName.find('_', first_ + 1);
  std::string firstPart =
      extName.substr(first_ + 1, second_ - first_ - 1).str();
  if (firstPart == "lazy" || firstPart == "control" || firstPart == "cond" ||
      firstPart == "d" || firstPart == "mem") {
    second_ = extName.find('_', second_ + 1);
  }
  std::string mod_name = extName.substr(0, second_).str();
  mod_name[first_] = '.';
  return mod_name;
}

// Get component's group (either handshake or arith)
std::string extractModGroup(std::string modName) {
  size_t firstDot = modName.find('.', 0);
  std::string modGroup = modName.substr(0, firstDot);
  return modGroup;
}

// Get component's own name (fork, addi etc)
std::string extractModOwnName(std::string modName) {
  size_t firstDot = modName.find('.', 0);
  std::string modOwn =
      modName.substr(firstDot + 1, modName.size() - firstDot - 1);
  return modOwn;
}

// Get corresponding parameters (e.g. 2_32 for a fork)
std::string extractModParameters(StringRef extName) {
  std::string modName = extractModName(extName);
  std::string modParameters = extName.substr(modName.size() + 1).str();
  return modParameters;
}

// get .vhd module description
VHDLModule getMod(circt::hw::HWModuleExternOp &extModOp,
                  VHDLComponentLibrary &jsonLib) {
  StringRef extName = extModOp.getModuleName();

  // extract external module name
  std::string modName = extractModName(extName);

  // extract module group
  std::string modGroup = extractModGroup(modName);

  // extract external module parameters
  std::string modParameters = extractModParameters(extName);

  // find external module in VHDLComponentLibrary
  llvm::StringMapIterator<VHDLModuleDescription> comp;
  if (modGroup == "arith") {
    // just 3 templates for all arithmetic operations to save memory
    if (modName == "arith.extsi" || modName == "arith.extui" ||
        modName == "arith.trunci") {
      comp = jsonLib.find("arith.extsi");
    } else if (modName == "arith.cmpf" || modName == "arith.cmpi") {
      comp = jsonLib.find(modName);
    } else {
      comp = jsonLib.find("arith.addf");
    }
  } else {
    comp = jsonLib.find(modName);
  }

  if (comp == jsonLib.end()) {
    llvm::errs() << "Unable to find the element in the library\n";
    return VHDLModule({}, {}, {}, {}, {}, {});
  }
  const VHDLModuleDescription &desc = (*comp).second;

  llvm::SmallVector<std::string> inputPorts{};
  llvm::SmallVector<std::string> outputPorts{};
  // fill input ports
  for (auto &k : extModOp.getPorts().inputs) {
    inputPorts.push_back(moveNumber(k.getName().str()));
  }

  // fill output ports
  for (auto &k : extModOp.getPorts().outputs) {
    outputPorts.push_back(moveNumber(k.getName().str()));
  }

  // cmpi && cmpf are defined with a predicate also, which exists inside
  // discriminating parameters
  if (modName == "arith.cmpf" || modName == "arith.cmpi") {
    modName = "predicate";
  }
  auto mod = desc.concretize((*comp).getKey().str(), modName, inputPorts,
                             outputPorts, modParameters);

  return mod;
};

/*---------------------------------
             INSTANTIATION
-----------------------------------*/
// For component's name on instantiation: get a number of next similar component
// (e.g fork) or add a new component to SoreComponentNumbers library (numeration
// starts with 0)
size_t getModNumber(std::string modName, StoreComponentNumbers &n) {
  auto it = n.find(modName);
  if (it == n.end()) {
    n.insert(std::pair(modName, 0));
    return 0;
  } else {
    ++it->second;
    return it->second;
  }
}

// Get list of parameters in brackets on instantiation.
// That's important that parameters in both vectors, modParameters (with
// parameters) and substr (name in brackets in modText file) have the same
// order.
std::string getExactValue(std::string substr, VHDLModule &mod) {
  std::stringstream strStr(substr);
  auto modParameters = mod.getModParameters();
  std::string name;
  std::string type;
  std::string result{};

  while (strStr.good()) {
    std::getline(strStr, name, ':');
    deleteSpaces(name);
    std::getline(strStr, type, ';');
    deleteSpaces(type);
    if (modParameters.find(name) != modParameters.end()) {
      result += modParameters.find(StringRef(name))->getValue();
    } else {
      llvm::errs() << "This argument neither parameter nor constant!\n";
      continue;
    }
    if (strStr.good()) {
      result += ", ";
    }
  }
  return result;
}

// Get full description = instantiation = for a given module
// To Do: change arrays from in_0... to in[0]...
std::string getEntityDeclaration(VHDLModule &mod, size_t modNumber) {
  std::string modTextInstance{};
  std::string modNameFull = mod.getModName();
  // get an own component's name (e g fork)
  std::string modName = extractModOwnName(modNameFull);
  if (modName == "end" || modName == "start") {
    modName += "_node";
  }
  modTextInstance += modName + "_" + std::to_string(modNumber) +
                     ": entity work." + modName + "(arch) generic map(";

  // find a substring with parameters inside modText
  std::string modText = mod.getModText();
  auto tempName = " " + modName + " ";
  auto ind = modText.find(tempName);
  auto firstBr = modText.find('(', ind);
  auto lastBr = modText.find(')', ind);
  auto substr = modText.substr(firstBr + 1, lastBr - firstBr - 1);
  deleteSpaces(substr);
  // get corresponding values
  auto exactValueStr = getExactValue(substr, mod);
  modTextInstance += exactValueStr + ")\nport map(\n";
  // list of arrays & signals
  modTextInstance +=
      "clk => " + modName + "_" + std::to_string(modNumber) + "_clk,\n";
  modTextInstance +=
      "rst => " + modName + "_" + std::to_string(modNumber) + "_rst,\n";
  auto inputs = mod.getInputPorts();
  auto outputs = mod.getOutputPorts();
  for (auto &j : inputs) {
    if (j != "clock" && j != "reset") {
      // special format implementation for ... => ...
      auto ind = j.find('@', 0);
      if (ind < j.length()) {
        j[ind] = '(';
        modTextInstance +=
            j + ") => " + modName + "_" + std::to_string(modNumber);
        j[ind] = '_';
        modTextInstance += "_" + j + ",\n";
      } else {
        modTextInstance += j + " => " + modName + "_" +
                           std::to_string(modNumber) + "_" + j + ",\n";
      }
    }
  }
  for (auto &j : outputs) {
    if (j != "clock" && j != "reset") {
      // special format implementation for ... => ...
      auto ind = j.find('@', 0);
      if (ind < j.length()) {
        j[ind] = '(';
        modTextInstance +=
            j + ") => " + modName + "_" + std::to_string(modNumber);
        j[ind] = '_';
        modTextInstance += "_" + j + ",\n";
      } else {
        modTextInstance += j + " => " + modName + "_" +
                           std::to_string(modNumber) + "_" + j + ",\n";
      }
    }
  }
  modTextInstance += ");\n";
  return modTextInstance;
}

/*---------------------------------
             TESTING
-----------------------------------
*/

// Check if library is correct
void testLib(VHDLComponentLibrary &m) {
  for (auto &[keyl, val] : m) {
    std::ifstream file;
    file.open(val.getPath());

    if (!file.is_open()) {
      errs() << "Filepath is uncorrect\n";
      file.close();
      return;
    }
    file.close();
    llvm::outs() << "---\n"
                 << keyl << " "
                 << "\npath: " << val.getPath()
                 << "\nconcr_method: " << val.getConcretizationMethod()
                 << "\nparameters:\n";
    for (auto &i : val.getParameters()) {
      llvm::outs() << "[" << i.getName() << "," << i.getType() << "]\n";
    }
  }
}

// Test how modules are printed on concretization phase
void testModulesConcretization(mlir::OwningOpRef<mlir::ModuleOp> &module,
                               VHDLComponentLibrary &m) {
  StoreComponentNumbers comp{};
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {
    auto extName = extModOp.getModuleName();
    llvm::outs() << "---\nOfficial module name: " << extName << "\n";
    auto i = getMod(extModOp, m);

    if (i.getModText().empty()) {
      llvm::outs() << "Still doesn't exist in the lib\n";
      continue;
    } else {
      llvm::outs() << "Mod_name: " << i.getModName() << "\n"
                   << "Mod_text:\n"
                   << i.getModText() << "\n"
                   << "Path: " << i.getModDesc().getPath() << "\n";
      llvm::outs() << "Concretization_method: "
                   << i.getModDesc().getConcretizationMethod() << "\n"
                   << "Parameters:\n";
      for (auto &j : i.getModParameters()) {
        llvm::outs() << j.getKey() << " " << j.getValue() << "\n";
      }
      llvm::outs() << "Inputs:\n";
      for (auto &j : i.getInputPorts()) {
        llvm::outs() << j << " ";
      }
      llvm::outs() << "\n";
      llvm::outs() << "Outputs:\n";

      for (auto &j : i.getOutputPorts()) {
        llvm::outs() << j << " ";
      }
      llvm::outs() << "\n";

      llvm::outs() << "Parameters of template:\n";
      for (auto &j : i.getModDesc().getParameters()) {
        llvm::outs() << "[" << j.getName() << ";" << j.getType() << "]\n";
      }
    }
  }
}

// Test modules instantiation description
void testModulesInstantiation(mlir::OwningOpRef<mlir::ModuleOp> &module,
                              VHDLComponentLibrary &m) {
  StoreComponentNumbers comp{};
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {
    llvm::outs() << "---------------\n";
    auto i = getMod(extModOp, m);

    if (i.getModText().empty()) {
      llvm::outs() << extModOp.getModuleName()
                   << " still doesn't exist in the lib\n";
      continue;
    } else {
      auto instance =
          getEntityDeclaration(i, getModNumber(i.getModName(), comp));
      llvm::outs() << instance << "\n";
    }
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
      "This tool prints on stdout the VHDL design corresponding to the input"
      "netlist-level MLIR representation of a dataflow circuit.\n");

  // Read the input IR in memory
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er)
  // level dialects or parsers. Allow unregistered dialects to not fail in
  // these cases
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

  //////////////////////////////////////
  auto m = parseJSON();
  // testLib(m);
  // testModulesConcretization(module, m);

  // Instantiations go to standart llvm output
  testModulesInstantiation(module, m);
  return 0;
}
