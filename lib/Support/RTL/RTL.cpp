//===- RTL.cpp - RTL support ------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JSON-parsing logic for the RTL configuration file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <regex>
#include <string>

#define DEBUG_TYPE "support-rtl"

using namespace llvm;
using namespace dynamatic;
using namespace dynamatic::json;

namespace ljson = llvm::json;

/// Recognized keys in RTL configuration files.
static constexpr StringLiteral KEY_PARAMETERS("parameters"),
    KEY_MODELS("models"), KEY_GENERIC("generic"), KEY_GENERATOR("generator"),
    KEY_NAME("name"), KEY_PATH("path"), KEY_CONSTRAINTS("constraints"),
    KEY_PARAMETER("parameter"), KEY_DEPENDENCIES("dependencies"),
    KEY_MODULE_NAME("module-name"), KEY_ARCH_NAME("arch-name"), KEY_HDL("hdl"),
    KEY_USE_JSON_CONFIG("use-json-config"), KEY_IO_KIND("io-kind"),
    KEY_IO_MAP("io-map"), KEY_IO_SIGNALS("io-signals");

/// JSON path errors.
static constexpr StringLiteral
    ERR_MISSING_CONCRETIZATION("missing concretization method, either "
                               "\"generic\" or \"generator\" key must exist"),
    ERR_MULTIPLE_CONCRETIZATION(
        "multiple concretization methods provided, only one of "
        "\"generic\" or \"generator\" key must exist"),
    ERR_UNKNOWN_PARAM("unknown parameter name"),
    ERR_DUPLICATE_NAME("duplicated parameter name"),
    ERR_RESERVED_NAME("this is a reserved parameter name"),
    ERR_INVALID_HDL(R"(unknown hdl: options are "vhdl", "verilog", or "smv)"),
    ERR_INVALID_IO_STYLE(
        R"(unknown IO style: options are "hierarchical" or "flat")");

/// Reserved parameter names. No user-provided parameter can have any of those
/// names in the RTL configuration files.
static const mlir::DenseSet<StringRef> RESERVED_PARAMETER_NAMES{
    RTLParameter::DYNAMATIC, RTLParameter::OUTPUT_DIR,
    RTLParameter::MODULE_NAME};

StringRef dynamatic::getHDLExtension(HDL hdl) {
  switch (hdl) {
  case HDL::VHDL:
    return "vhd";
  case HDL::VERILOG:
    return "v";
  case HDL::SMV:
    return "smv";
  }
}
std::string dynamatic::replaceRegexes(
    StringRef input, const std::map<std::string, std::string> &replacements) {
  std::string result(input);
  for (auto &[from, to] : replacements)
    result = std::regex_replace(result, std::regex(from), to);
  return result;
}

std::string dynamatic::substituteParams(StringRef input,
                                        const ParameterMappings &parameters) {
  std::map<std::string, std::string> replacements;
  for (auto &[name, value] : parameters)
    replacements["\\$" + name.str()] = value;
  return replaceRegexes(input, replacements);
}

RTLRequestFromOp::RTLRequestFromOp(Operation *op, const llvm::Twine &name)
    : RTLRequest(op->getLoc()), name(name.str()), op(op),
      parameters(op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME)) {
      };

Attribute RTLRequestFromOp::getParameter(const RTLParameter &param) const {
  if (!parameters)
    return nullptr;
  return parameters.get(param.getName());
}

RTLMatch *RTLRequestFromOp::tryToMatch(const RTLComponent &component) const {
  ParameterMappings mappings;
  if (failed(areParametersCompatible(component, mappings)))
    return nullptr;

  mappings[RTLParameter::MODULE_NAME] = "";
  return new RTLMatch(component, mappings);
}

LogicalResult
RTLRequestFromOp::areParametersCompatible(const RTLComponent &component,
                                          ParameterMappings &mappings) const {
  LLVM_DEBUG(llvm::dbgs() << "Attempting match with RTL component "
                          << component.getName() << "\n";);
  if (name != component.getName()) {
    LLVM_DEBUG(llvm::dbgs() << "\t-> Names do not match.\n");
    return failure();
  }

  // Keep track of the set of parameters we parse from the dictionnary
  // attribute. We never have duplicate parameters due to the dictionary's
  // keys uniqueness invariant, but we need to make sure all known parameters
  // are present
  DenseSet<StringRef> parsedParams;
  SmallVector<StringRef> ignoredParams;

  for (const RTLParameter *parameter : component.getParameters()) {
    ParamMatch paramMatch = matchParameter(*parameter);
    StringRef paramName = parameter->getName();
    LLVM_DEBUG({
      switch (paramMatch.state) {
      case ParamMatch::State::DOES_NOT_EXIST:
        llvm::dbgs() << "\t-> Parameter \"" << paramName
                     << "\" does not exist\n";
        break;
      case ParamMatch::State::FAILED_VERIFICATION:
        llvm::dbgs() << "\t-> Failed constraint checking for parameter \""
                     << paramName << "\"\n";
        break;
      case ParamMatch::State::FAILED_SERIALIZATION:
        llvm::dbgs() << "\t-> Failed serialization for parameter \""
                     << paramName << "\"\n";
        break;
      case ParamMatch::State::SUCCESS:
        llvm::dbgs() << "\t-> Matched parameter \"" << paramName << "\"\n";
        break;
      }
    });
    if (paramMatch.state != ParamMatch::SUCCESS)
      return failure();
    mappings[paramName] = paramMatch.serialized;
  }
  LLVM_DEBUG(llvm::dbgs() << "\t-> Matched!\n");
  return success();
}

ParamMatch RTLRequestFromOp::matchParameter(const RTLParameter &param) const {
  Attribute attr = getParameter(param);
  if (!attr)
    return ParamMatch::doesNotExist();
  if (!param.getType().verify(attr))
    return ParamMatch::failedVerification();
  std::string serialized = param.getType().serialize(attr);
  if (serialized.empty())
    return ParamMatch::failedSerialization();
  return ParamMatch::success(serialized);
}

LogicalResult
RTLRequestFromOp::paramsToJSON(const llvm::Twine &filepath) const {
  return serializeToJSON(parameters, filepath.str(), loc);
};

RTLRequestFromHWModule::RTLRequestFromHWModule(hw::HWModuleExternOp modOp)
    : RTLRequestFromOp(modOp, getName(modOp)) {}

RTLMatch *
RTLRequestFromHWModule::tryToMatch(const RTLComponent &component) const {
  ParameterMappings mappings;
  if (failed(areParametersCompatible(component, mappings)))
    return nullptr;

  mappings[RTLParameter::MODULE_NAME] =
      component.isGeneric()
          ? substituteParams(component.getModuleName(), mappings)
          : cast<hw::HWModuleExternOp>(op).getSymName();
  return new RTLMatch(component, mappings);
}

std::string RTLRequestFromHWModule::getName(hw::HWModuleExternOp modOp) {
  if (auto nameAttr = modOp->getAttrOfType<StringAttr>(RTL_NAME_ATTR_NAME))
    return nameAttr.str();
  return "";
}

RTLDependencyRequest::RTLDependencyRequest(const Twine &moduleName,
                                           Location loc)
    : RTLRequest(loc), moduleName(moduleName.str()) {}

RTLMatch *
RTLDependencyRequest::tryToMatch(const RTLComponent &component) const {
  LLVM_DEBUG(
      llvm::dbgs() << "Attempting dependency match between request for \""
                   << moduleName << "\" and RTL component "
                   << component.getName() << "\n\t-> ";);

  if (!component.isGeneric()) {
    LLVM_DEBUG(llvm::dbgs() << "Component is not generic\n");
    return nullptr;
  }
  if (component.getModuleName() != moduleName) {
    LLVM_DEBUG(llvm::dbgs() << "Component has incorrect module name \""
                            << component.getModuleName() << "\"\n");
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "Matched!\n");
  ParameterMappings mappings;
  mappings[RTLParameter::MODULE_NAME] = moduleName;
  return new RTLMatch(component, mappings);
}

RTLMatch::RTLMatch(const RTLComponent &component,
                   const ParameterMappings &serializedParams)
    : component(&component),
      moduleName(substituteParams(component.moduleName, serializedParams)),
      archName(substituteParams(component.archName, serializedParams)),
      serializedParams(serializedParams) {}

MapVector<StringRef, StringRef> RTLMatch::getGenericParameterValues() const {
  MapVector<StringRef, StringRef> values;
  for (const RTLParameter *param : component->getGenericParameters()) {
    auto valueIt = serializedParams.find(param->getName());
    assert(valueIt != serializedParams.end() && "missing parameter value");
    values.insert({param->getName(), valueIt->second});
  }
  return values;
}

/// Serializes the module's "port_types", which includes the types of all ports
/// (operands and results) of the original operation. This is passed to the RTL
/// generator to help it generate the correct port types. e.g., '{"lhs":
/// "!handshake.channel<i32, [spec: i1]>",
// "rhs": "!handshake.channel<i32, [spec: i1]>",
// "result": "!handshake.channel<i1, [spec: i1]>"}'
static std::string serializePortTypes(hw::ModuleType &mod) {
  // Prepare a string stream to serialize the port types
  std::string portTypesValue;
  llvm::raw_string_ostream portTypes(portTypesValue);

  // Wrap in single quotes for easier passing as a generator argument.
  portTypes << "'{"; // Start of the JSON object

  bool first = true;
  for (const hw::ModulePort &port : mod.getPorts()) {
    // Skip the clock and reset ports
    if (port.name == "clk" || port.name == "rst")
      continue;

    if (!first)
      portTypes << ", ";
    first = false;

    portTypes << "\"" << port.name.str() << "\": \"";
    // TODO: Escape "" in the port type (if needed)
    port.type.print(portTypes);
    portTypes << "\"";
  }
  portTypes << "}'"; // End of the JSON object

  return portTypes.str();
}

static std::string serializeExtraSignalsInner(const Type &type) {
  assert(type.isa<handshake::ExtraSignalsTypeInterface>() &&
         "type should be ChannelType or ControlType");

  handshake::ExtraSignalsTypeInterface extraSignalsType =
      type.cast<handshake::ExtraSignalsTypeInterface>();

  std::string extraSignalsValue;
  llvm::raw_string_ostream extraSignals(extraSignalsValue);

  extraSignals << "{";
  bool first = true;
  for (const handshake::ExtraSignal &extraSignal :
       extraSignalsType.getExtraSignals()) {
    if (!first)
      extraSignals << ", ";
    first = false;

    extraSignals << "\"" << extraSignal.name << "\": ";
    extraSignals << extraSignal.getBitWidth();
  }
  extraSignals << "}";

  return extraSignals.str();
}

static std::string serializeExtraSignals(const Type &type) {
  return "'" + serializeExtraSignalsInner(type) + "'";
}

/// Returns the bitwidth of the type as string.
/// If the type is a control type, returns "0".
static std::string getBitwidthString(Type type) {
  return std::to_string(handshake::getHandshakeTypeBitWidth(type));
}

void RTLMatch::registerParameters(hw::HWModuleExternOp &modOp) {
  auto modName =
      modOp->template getAttrOfType<StringAttr>(RTL_NAME_ATTR_NAME).getValue();
  auto modType = modOp.getModuleType();

  registerPortTypesParameter(modOp, modName, modType);
  registerBitwidthParameter(modOp, modName, modType);
  registerTransparentParameter(modOp, modName, modType);
  registerExtraSignalParameters(modOp, modName, modType);
  registerSelectedDelayParameter(modOp, modName, modType);
}

void RTLMatch::registerPortTypesParameter(hw::HWModuleExternOp &modOp,
                                          llvm::StringRef modName,
                                          hw::ModuleType &modType) {
  serializedParams["PORT_TYPES"] = serializePortTypes(modType);
}

void RTLMatch::registerSelectedDelayParameter(hw::HWModuleExternOp &modOp,
                                              llvm::StringRef modName,
                                              hw::ModuleType &modType) {
  // Look for INTERNAL_DELAY in the module's parameters.
  if (auto paramsAttr = modOp->getAttrOfType<DictionaryAttr>("hw.parameters")) {
    if (auto selectedDelay = paramsAttr.get("INTERNAL_DELAY")) {
      if (auto stringAttr = selectedDelay.dyn_cast<StringAttr>()) {
        std::string delayStr = stringAttr.getValue().str();
        serializedParams["INTERNAL_DELAY"] = delayStr;
        return;
      }
    }
  } else {
    // default case for graceful handling of units without an internal delay set
    serializedParams["INTERNAL_DELAY"] = "0.0";
  }
}

void RTLMatch::registerBitwidthParameter(hw::HWModuleExternOp &modOp,
                                         llvm::StringRef modName,
                                         hw::ModuleType &modType) {
  if (
      // default (All(Data)TypesMatch)
      modName == "handshake.addi" || modName == "handshake.andi" ||
      modName == "handshake.buffer" || modName == "handshake.cmpi" ||
      modName == "handshake.fork" || modName == "handshake.merge" ||
      modName == "handshake.muli" || modName == "handshake.sink" ||
      modName == "handshake.subi" || modName == "handshake.shli" ||
      modName == "handshake.blocker" || modName == "handshake.sitofp" ||
      modName == "handshake.fptosi" ||
      // the first input has data bitwidth
      modName == "handshake.speculator" || modName == "handshake.spec_commit" ||
      modName == "handshake.spec_save_commit" ||
      modName == "handshake.non_spec") {
    // Default
    serializedParams["BITWIDTH"] = getBitwidthString(modType.getInputType(0));
  } else if (modName == "handshake.cond_br" || modName == "handshake.select") {
    serializedParams["BITWIDTH"] = getBitwidthString(modType.getInputType(1));
  } else if (modName == "handshake.constant") {
    serializedParams["BITWIDTH"] = getBitwidthString(modType.getOutputType(0));
  } else if (modName == "handshake.control_merge") {
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["INDEX_BITWIDTH"] =
        getBitwidthString(modType.getOutputType(1));
  } else if (modName == "handshake.extsi" || modName == "handshake.trunci" ||
             modName == "handshake.extui") {
    serializedParams["INPUT_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["OUTPUT_BITWIDTH"] =
        getBitwidthString(modType.getOutputType(0));
  } else if (modName == "handshake.load") {
    serializedParams["ADDR_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getOutputType(1));
  } else if (modName == "handshake.mux") {
    serializedParams["INDEX_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(1));
  } else if (modName == "handshake.store") {
    serializedParams["ADDR_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(1));
  } else if (modName == "handshake.speculating_branch") {
    serializedParams["SPEC_TAG_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(1));
  } else if (modName == "handshake.mem_controller") {
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(0));
    // Warning: Ports differ from instance to instance.
    // Therefore, mod.getNumOutputs() is also variable.
    serializedParams["ADDR_BITWIDTH"] =
        getBitwidthString(modType.getOutputType(modType.getNumOutputs() - 2));
  } else if (modName == "mem_to_bram") {
    serializedParams["ADDR_BITWIDTH"] =
        getBitwidthString(modType.getInputType(1));
    serializedParams["DATA_BITWIDTH"] =
        getBitwidthString(modType.getInputType(4));
  } else if (modName == "handshake.addf" || modName == "handshake.cmpf" ||
             modName == "handshake.mulf" || modName == "handshake.subf") {
    int bitwidth = handshake::getHandshakeTypeBitWidth(modType.getInputType(0));
    serializedParams["IS_DOUBLE"] = bitwidth == 64 ? "True" : "False";
  } else if (modName == "handshake.source" || modName == "mem_controller") {
    // Skip
  }
}

void RTLMatch::registerTransparentParameter(hw::HWModuleExternOp &modOp,
                                            llvm::StringRef modName,
                                            hw::ModuleType &modType) {
  if (modName == "handshake.buffer") {
    auto params =
        modOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      auto info = timing.getInfo();
      if (info == handshake::TimingInfo::break_r() ||
          info == handshake::TimingInfo::break_none()) {
        serializedParams["TRANSPARENT"] = "True";
      } else if (info == handshake::TimingInfo::break_dv() ||
                 info == handshake::TimingInfo::break_dvr()) {
        serializedParams["TRANSPARENT"] = "False";
      } else {
        llvm_unreachable("Unknown timing info");
      }
    } else {
      llvm_unreachable("Unknown timing attr");
    }
  }
}

void RTLMatch::registerExtraSignalParameters(hw::HWModuleExternOp &modOp,
                                             llvm::StringRef modName,
                                             hw::ModuleType &modType) {
  if (
      // default (AllExtraSignalsMatch)
      modName == "handshake.addf" || modName == "handshake.addi" ||
      modName == "handshake.andi" || modName == "handshake.buffer" ||
      modName == "handshake.cmpf" || modName == "handshake.cmpi" ||
      modName == "handshake.cond_br" || modName == "handshake.constant" ||
      modName == "handshake.extsi" || modName == "handshake.fork" ||
      modName == "handshake.merge" || modName == "handshake.mulf" ||
      modName == "handshake.muli" || modName == "handshake.select" ||
      modName == "handshake.sink" || modName == "handshake.subf" ||
      modName == "handshake.extui" || modName == "handshake.shli" ||
      modName == "handshake.subi" || modName == "handshake.spec_save_commit" ||
      modName == "handshake.speculator" || modName == "handshake.trunci" ||
      modName == "handshake.mux" || modName == "handshake.control_merge" ||
      modName == "handshake.blocker" || modName == "handshake.sitofp" ||
      modName == "handshake.fptosi" ||
      // the first input has extra signals
      modName == "handshake.load" || modName == "handshake.store" ||
      modName == "handshake.spec_commit" ||
      modName == "handshake.speculating_branch") {
    serializedParams["EXTRA_SIGNALS"] =
        serializeExtraSignals(modType.getInputType(0));
  } else if (modName == "handshake.source" || modName == "handshake.non_spec") {
    serializedParams["EXTRA_SIGNALS"] =
        serializeExtraSignals(modType.getOutputType(0));
  } else if (modName == "handshake.mem_controller" ||
             modName == "mem_to_bram") {
    // Skip
  }
}

LogicalResult RTLMatch::concretize(const RTLRequest &request,
                                   StringRef dynamaticPath,
                                   StringRef outputDir) const {
  // Consolidate reserved and regular parameters in a single map to perform
  // text substitutions
  ParameterMappings allParams(serializedParams);
  allParams[RTLParameter::DYNAMATIC] = dynamaticPath;
  allParams[RTLParameter::OUTPUT_DIR] = outputDir;

  if (component->isGeneric()) {
    std::string inputFile = substituteParams(component->generic, allParams);
    HDL hdl = component->hdl;
    std::string outputFile = outputDir.str() +
                             sys::path::get_separator().str() + moduleName +
                             "." + getHDLExtension(hdl).str();

    // Just copy the file to the output location
    if (auto ec = sys::fs::copy_file(inputFile, outputFile); ec.value() != 0) {
      return emitError(request.loc)
             << "Failed to copy generic RTL implementation from \"" << inputFile
             << "\" to \"" << outputFile << "\"\n"
             << ec.message();
    }
    return success();
  }
  assert(!component->generator.empty() && "generator is empty");

  if (component->jsonConfig && failed(request.paramsToJSON(substituteParams(
                                   *(component->jsonConfig), allParams))))
    return failure();

  // The implementation needs to be generated
  std::string cmd = substituteParams(component->generator, allParams);
  if (int ret = std::system(cmd.c_str()); ret != 0) {
    return emitError(request.loc)
           << "Failed to generate component, generator failed with status "
           << ret << ": " << cmd << "\n";
  }
  return success();
}

bool RTLParameter::fromJSON(const ljson::Value &value, ljson::Path path) {
  ljson::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) ||
      !mapper.map(KEY_GENERIC, useAsGeneric))
    return false;

  if (RESERVED_PARAMETER_NAMES.contains(name)) {
    path.field(KEY_NAME).report(ERR_RESERVED_NAME);
    return false;
  }

  return type.fromJSON(value, path);
}

RTLComponent::Model::~Model() {
  for (auto &[_, constraints] : addConstraints)
    delete constraints;
}

/// Counts the number of times the wildcard character ('*') appears in the
/// input.
static inline size_t countWildcards(StringRef input) {
  return std::count_if(input.begin(), input.end(),
                       [](char c) { return c == '*'; });
}

bool RTLComponent::fromJSON(const ljson::Value &value, ljson::Path path) {
  ljson::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.mapOptional(KEY_NAME, name) ||
      !mapper.mapOptional(KEY_PARAMETERS, parameters) ||
      !mapper.mapOptional(KEY_GENERIC, generic) ||
      !mapper.mapOptional(KEY_GENERATOR, generator) ||
      !mapper.mapOptional(KEY_DEPENDENCIES, dependencies) ||
      !mapper.mapOptional(KEY_MODULE_NAME, moduleName) ||
      !mapper.mapOptional(KEY_ARCH_NAME, archName) ||
      !mapper.mapOptional(KEY_HDL, hdl) ||
      !mapper.mapOptional(KEY_USE_JSON_CONFIG, jsonConfig) ||
      !mapper.mapOptional(KEY_IO_KIND, ioKind) ||
      !mapper.mapOptional(KEY_IO_MAP, ioMap) ||
      !mapper.mapOptional(KEY_IO_SIGNALS, ioSignals)) {
    return false;
  }

  if (!checkValidAndSetDefaults(path))
    return false;

  // Timing models need access to the component's RTL parameters to verify
  // the sanity of constraints, so they are parsed separately and somewhat
  // differently

  // The mapper ensures that the object is valid
  const ljson::Value *jsonModelsValue = value.getAsObject()->get(KEY_MODELS);
  if (!jsonModelsValue)
    return true;

  const ljson::Array *jsonModelsArray = jsonModelsValue->getAsArray();
  ljson::Path modelsPath = path.field(KEY_MODELS);
  if (!jsonModelsArray) {
    modelsPath.report(ERR_EXPECTED_ARRAY);
    return false;
  }

  for (auto [modIdx, jsonModel] : llvm::enumerate(*jsonModelsArray)) {
    ljson::Path modPath = modelsPath.index(modIdx);

    Model &model = models.emplace_back();
    ljson::ObjectMapper modelMapper(jsonModel, modPath);
    if (!modelMapper || !modelMapper.map(KEY_PATH, model.path))
      return false;

    // The mapper ensures that this object is valid
    auto *jsonConstraints = jsonModel.getAsObject()->get(KEY_CONSTRAINTS);
    if (!jsonConstraints) {
      // This model has no constraints
      continue;
    }

    const ljson::Array *jsonConstraintsArray = jsonConstraints->getAsArray();
    ljson::Path constraintsPath = modPath.field(KEY_CONSTRAINTS);
    if (!jsonConstraintsArray) {
      constraintsPath.report(ERR_EXPECTED_ARRAY);
      return false;
    }

    for (auto [constIdx, jsonConst] : llvm::enumerate(*jsonConstraintsArray)) {
      ljson::Path constPath = constraintsPath.index(constIdx);

      // Retrieve the name of the parameter the constraint applies on
      ljson::ObjectMapper constMapper(jsonConst, constPath);
      std::string paramName;
      if (!constMapper || !constMapper.map(KEY_NAME, paramName))
        return false;

      // Add a new parameter/constraint vector pair to the model's list.
      auto &[param, constraints] = model.addConstraints.emplace_back();

      // Retrieve the parameter with this name, if it exists
      if (!(param = getParameter(paramName))) {
        constPath.field(KEY_NAME).report(ERR_UNKNOWN_PARAM);
        return false;
      }

      if (!param->type.constraintsFromJSON(jsonConst, constraints, constPath))
        return false;
    }
  }
  return true;
}

RTLParameter *RTLComponent::getParameter(StringRef name) const {
  auto paramIt = nameToParam.find(name);
  if (paramIt != nameToParam.end())
    return paramIt->second;
  return nullptr;
}

SmallVector<const RTLParameter *> RTLComponent::getParameters() const {
  SmallVector<const RTLParameter *> genericParams;
  for (const RTLParameter &param : parameters)
    genericParams.push_back(&param);
  return genericParams;
}

SmallVector<const RTLParameter *> RTLComponent::getGenericParameters() const {
  SmallVector<const RTLParameter *> genericParams;
  bool componentIsGeneric = isGeneric();
  for (const RTLParameter &param : parameters) {
    if (componentIsGeneric) {
      // Component generic, need explicit notice to NOT use parameter as generic
      if (!param.useAsGeneric.value_or(true))
        continue;
    } else {
      // Component generated, need explicit notice to use parameter as generic
      if (!param.useAsGeneric.value_or(false))
        continue;
    }
    genericParams.push_back(&param);
  }
  return genericParams;
}

const RTLComponent::Model *
RTLComponent::getModel(const RTLRequest &request) const {
  RTLMatch *match = request.tryToMatch(*this);
  if (!match)
    return nullptr;
  delete match;

  for (const Model &model : models) {
    /// Returns true when all additional constraints on the parameters are
    /// satisfied.
    auto satisfied = [&](const Model::AddConstraints &addConstraints) -> bool {
      auto &[param, constraints] = addConstraints;
      return constraints->verify(request.getParameter(*param));
    };

    // The model matches if all additional parameter constraints are satsified
    if (llvm::all_of(model.addConstraints, satisfied))
      return &model;
  }

  // No matching model
  return nullptr;
}

std::string RTLComponent::portRemap(StringRef mlirPortName) const {
  for (const auto &[rtlPortName, mappedRTLPortName] : ioMap) {

    size_t wildcardIdx = rtlPortName.find('*');
    if (wildcardIdx == std::string::npos) {
      // Only an exact match will work
      if (mlirPortName != rtlPortName)
        continue;
      return mappedRTLPortName;
    }

    // Check whether we can match the MLIR port name to the RTl port name
    // template; we must identify whether any part of the MLIR port name matches
    // the wildcard, then replace the potential wildcard in the remapped port
    // name with that part of the MLIR port name
    StringRef refRTlPortName(rtlPortName);

    // Characters before the wildcard must match between the MLIR port name and
    // RTL source port name
    if (mlirPortName.size() < wildcardIdx ||
        mlirPortName.take_front(wildcardIdx) !=
            refRTlPortName.take_front(wildcardIdx))
      continue;

    StringRef wildcardMatch;
    size_t afterWildcardSize = rtlPortName.size() - wildcardIdx - 1;
    if (afterWildcardSize > 0) {
      // Characters after the wildcard must match between the MLIR port name and
      // source RTL port name
      if (mlirPortName.size() < afterWildcardSize ||
          mlirPortName.take_back(afterWildcardSize) !=
              refRTlPortName.take_back(afterWildcardSize))
        continue;
      wildcardMatch = mlirPortName.slice(wildcardIdx, mlirPortName.size() -
                                                          afterWildcardSize);
    } else {
      wildcardMatch = mlirPortName.drop_front(wildcardIdx);
    }

    // Replace a potential wildcard in the remapped name with the part of the
    // MLIR port name that matched the wildcard in the source port name
    if (size_t idx = mappedRTLPortName.find('*'); idx != std::string::npos)
      return std::string{mappedRTLPortName}.replace(idx, 1, wildcardMatch);
    return mappedRTLPortName;
  }

  // When no source port name in the map matched the MLIR port name, just return
  // it unmodified
  return mlirPortName.str();
}

bool RTLComponent::portNameIsIndexed(StringRef portName, StringRef &baseName,
                                     size_t &arrayIdx) const {
  // IO kind must be hierarchical and port name must contain an underscore to
  // separate a base name from an index
  if (ioKind == IOKind::FLAT)
    return false;
  size_t idx = portName.rfind("_");
  if (idx == std::string::npos)
    return false;

  StringRef maybeNumber = portName.substr(idx + 1);
  if (!StringRef{maybeNumber}.getAsInteger(10, arrayIdx)) {
    baseName = portName.substr(0, idx);
    return true;
  }
  return false;
}

std::pair<std::string, bool>
RTLComponent::getRTLPortName(StringRef mlirPortName, HDL hdl) const {
  std::string remappedName = portRemap(mlirPortName);
  StringRef baseName;
  size_t arrayIdx;
  if (portNameIsIndexed(remappedName, baseName, arrayIdx))
    return {baseName.str(), true};
  return {remappedName, false};
}

std::pair<std::string, bool>
RTLComponent::getRTLPortName(StringRef mlirPortName, SignalType signalType,
                             HDL hdl) const {
  auto portName = getRTLPortName(mlirPortName, hdl);
  return {portName.first + ioSignals.at(signalType), portName.second};
}

bool RTLComponent::checkValidAndSetDefaults(llvm::json::Path path) {
  // Make sure all parameter names are unique, and store name to parameter
  // associations in the map for easy access later on
  for (RTLParameter &param : parameters) {
    StringRef name = param.getName();
    if (nameToParam.contains(name)) {
      path.field(KEY_PARAMETER).report(ERR_DUPLICATE_NAME);
      return false;
    }
    nameToParam[name] = &param;
  }

  // Check that at least one concretization method was provided
  if (generic.empty() && generator.empty()) {
    path.report(ERR_MISSING_CONCRETIZATION);
    return false;
  }
  if (!generic.empty() && !generator.empty()) {
    path.report(ERR_MULTIPLE_CONCRETIZATION);
    return false;
  }

  // Derive the module name if none was provided
  if (moduleName.empty()) {
    if (isGeneric()) {
      // Get the filename and remove the extension
      std::string filename = llvm::sys::path::filename(generic).str();
      if (size_t idx = filename.find('.'); idx != std::string::npos)
        filename = filename.substr(0, idx);
      moduleName = filename;
    } else {
      // Component is generated, by default the name is the one provided during
      // generation
      moduleName = "$" + RTLParameter::MODULE_NAME.str();
    }
  }

  /// Defines default signal type suffixes if they were not overriden
  auto setDefaultSignalSuffix = [&](SignalType signalType, StringRef suffix) {
    if (ioSignals.find(signalType) == ioSignals.end())
      ioSignals[signalType] = suffix;
  };
  setDefaultSignalSuffix(SignalType::DATA, "");
  setDefaultSignalSuffix(SignalType::VALID, "_valid");
  setDefaultSignalSuffix(SignalType::READY, "_ready");

  // Make sure the IO map makes sense
  return llvm::all_of(ioMap, [&](std::pair<std::string, std::string> &mapping) {
    auto &[from, to] = mapping;
    ljson::Path fromPath = path.field(KEY_IO_MAP).field(from);

    // At most one wildcard in the key
    size_t keyNumWild = countWildcards(from);
    if (keyNumWild > 1) {
      fromPath.report("At most one wildcard is allowed in the key");
      return false;
    }

    // At most one wildcard in the value, and no more than in the key
    size_t valudNumWild = countWildcards(to);
    if (valudNumWild > 1) {
      fromPath.report("At most one wildcard is allowed in the value");
      return false;
    }
    if (keyNumWild == 0 && valudNumWild == 1) {
      fromPath.report(
          "Value has wildcard but key does not, this is not allowed");
      return false;
    }
    return true;
  });
}

inline bool ljson::fromJSON(const json::Value &value,
                            std::pair<std::string, std::string> &stringPair,
                            json::Path path) {
  const json::Object *object = value.getAsObject();
  if (!object) {
    path.report(ERR_EXPECTED_OBJECT);
    return false;
  }

  // We expect a single key-value pair
  if (object->size() != 1) {
    path.report("expected single key-value pair mapping");
    return false;
  }

  // The JSON value in the object must be a string
  const auto &[jsonKey, jsonValue] = *object->begin();
  std::string mappedRTLPortName;
  if (!fromJSON(jsonValue, mappedRTLPortName, path.field(jsonKey)))
    return false;

  stringPair.first = jsonKey.str();
  stringPair.second = mappedRTLPortName;
  return true;
}

inline bool dynamatic::fromJSON(const ljson::Value &value,
                                std::map<SignalType, std::string> &ioChannels,
                                ljson::Path path) {
  const ljson::Object *object = value.getAsObject();
  if (!object) {
    path.report(ERR_EXPECTED_OBJECT);
    return false;
  }

  ioChannels.clear();
  for (const auto &[signalStr, jsonSuffix] : *object) {
    // Deserialize the signal type
    SignalType signalType;
    if (signalStr == "data") {
      signalType = SignalType::DATA;
    } else if (signalStr == "valid") {
      signalType = SignalType::VALID;
    } else if (signalStr == "ready") {
      signalType = SignalType::READY;
    } else {
      path.field(signalStr).report("unknown channel signal type: possible keys "
                                   "are 'data', 'valid', or 'ready'");
      return false;
    }

    // Deserialize the suffix (just a string)
    if (!fromJSON(jsonSuffix, ioChannels[signalType], path.field(signalStr)))
      return false;
  }

  return true;
}

inline bool dynamatic::fromJSON(const ljson::Value &value, HDL &hdl,
                                ljson::Path path) {
  std::optional<StringRef> hdlStr = value.getAsString();
  if (!hdlStr) {
    path.report(ERR_EXPECTED_STRING);
    return false;
  }
  if (hdlStr == "verilog") {
    hdl = HDL::VERILOG;
  } else if (hdlStr == "vhdl") {
    hdl = HDL::VHDL;
  } else if (hdlStr == "smv") {
    hdl = HDL::SMV;
  } else {
    path.report(ERR_INVALID_HDL);
    return false;
  }
  return true;
}

inline bool dynamatic::fromJSON(const ljson::Value &value,
                                RTLComponent::IOKind &io, ljson::Path path) {
  std::optional<StringRef> hdlStr = value.getAsString();
  if (!hdlStr) {
    path.report(ERR_EXPECTED_STRING);
    return false;
  }
  if (hdlStr == "hierarchical") {
    io = RTLComponent::IOKind::HIERARCICAL;
  } else if (hdlStr == "flat") {
    io = RTLComponent::IOKind::FLAT;
  } else {
    path.report(ERR_INVALID_IO_STYLE);
    return false;
  }
  return true;
}

LogicalResult RTLConfiguration::addComponentsFromJSON(StringRef filepath) {
  // Open the RTL configuration file
  std::ifstream inputFile(filepath.str());
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open RTL configuration file @ \"" << filepath
                 << "\"\n";
    return failure();
  }

  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<ljson::Value> value = ljson::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse RTL configuration file @ \"" << filepath
                 << "\" as JSON.\n-> " << toString(value.takeError()) << "\n";
    return failure();
  }

  ljson::Path::Root jsonRoot(filepath);
  ljson::Path jsonPath(jsonRoot);

  ljson::Array *jsonComponents = value->getAsArray();
  if (!jsonComponents) {
    jsonPath.report(ERR_EXPECTED_ARRAY);
    jsonRoot.printErrorContext(*value, llvm::errs());
    return failure();
  }

  for (auto [idx, jsonComponent] : llvm::enumerate(*jsonComponents)) {
    RTLComponent &component = components.emplace_back();
    if (!fromJSON(jsonComponent, component, jsonPath.index(idx))) {
      jsonRoot.printErrorContext(*value, llvm::errs());
      return failure();
    }
  }

  return success();
}

static inline void notifyRequest(const RTLRequest &request) {
  LLVM_DEBUG(llvm::dbgs()
             << "Attempting to find compatible component for RTL request at "
             << request.loc << "\n");
}

bool RTLConfiguration::hasMatchingComponent(const RTLRequest &request) {
  notifyRequest(request);
  return llvm::any_of(components, [&](const RTLComponent &component) {
    if (RTLMatch *match = request.tryToMatch(component)) {
      delete match;
      return true;
    }
    return false;
  });
}

RTLMatch *RTLConfiguration::getMatchingComponent(const RTLRequest &request) {
  notifyRequest(request);
  std::vector<RTLMatch> matches;
  for (const RTLComponent &component : components) {
    if (RTLMatch *match = request.tryToMatch(component))
      return match;
  }
  return nullptr;
}

void RTLConfiguration::findMatchingComponents(
    const RTLRequest &request, std::vector<RTLMatch *> &matches) const {
  notifyRequest(request);
  for (const RTLComponent &component : components) {
    if (RTLMatch *match = request.tryToMatch(component))
      matches.push_back(match);
  }
  LLVM_DEBUG(llvm::dbgs() << matches.size()
                          << " compatible components found\n");
}

const RTLComponent::Model *
RTLConfiguration::getModel(const RTLRequest &request) const {
  for (const RTLComponent &component : components) {
    if (const RTLComponent::Model *model = component.getModel(request))
      return model;
  }
  return nullptr;
}