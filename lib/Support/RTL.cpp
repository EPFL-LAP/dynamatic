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

#include "dynamatic/Support/RTL.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include <fstream>
#include <regex>
#include <string>

#define DEBUG_TYPE "support-rtl"

using namespace llvm;
using namespace dynamatic;

/// Recognized keys in RTL configuration files.
static constexpr StringLiteral KEY_PARAMETERS("parameters"),
    KEY_MODELS("models"), KEY_GENERIC("generic"), KEY_GENERATOR("generator"),
    KEY_NAME("name"), KEY_TYPE("type"), KEY_PATH("path"),
    KEY_CONSTRAINTS("constraints"), KEY_PARAMETER("parameter"),
    KEY_DEPENDENCIES("dependencies"), KEY_MODULE_NAME("module-name"),
    KEY_ARCH_NAME("arch-name"), KEY_HDL("hdl"),
    KEY_USE_JSON_CONFIG("use-json-config"), KEY_IO_KIND("io-kind"),
    KEY_IO_MAP("io-map"), KEY_IO_SIGNALS("io-signals");

/// JSON path errors.
static constexpr StringLiteral ERR_EXPECTED_OBJECT("expected object"),
    ERR_EXPECTED_ARRAY("expected array"),
    ERR_MISSING_CONCRETIZATION("missing concretization method, either "
                               "\"generic\" or \"generator\" key must exist"),
    ERR_MULTIPLE_CONCRETIZATION(
        "multiple concretization methods provided, only one of "
        "\"generic\" or \"generator\" key must exist"),
    ERR_EXPECTED_STRING("expected string"),
    ERR_UNKNOWN_PARAM("unknown parameter name"),
    ERR_DUPLICATE_NAME("duplicated parameter name"),
    ERR_UNKNOWN_TYPE(
        R"(unknown parameter type: options are "unsigned" or "string")"),
    ERR_RESERVED_NAME("this is a reserved parameter name"),
    ERR_INVALID_HDL(R"(unknown hdl: options are "vhdl" or "verilog")"),
    ERR_INVALID_IO_STYLE(
        R"(unknown IO style: options are "hierarchical" or "flat")");

/// Reserved JSON keys when deserializing type constraints, should be ignored.
static const mlir::DenseSet<StringRef> RESERVED_KEYS{
    KEY_NAME, KEY_TYPE, KEY_PARAMETER, KEY_GENERIC};

/// Reserved parameter names. No user-provided parameter can have any of those
/// names in the RTL configuration files.
static const mlir::DenseSet<StringRef> RESERVED_PARAMETER_NAMES{
    RTLParameter::DYNAMATIC, RTLParameter::OUTPUT_DIR,
    RTLParameter::MODULE_NAME};

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

RTLType::~RTLType() {
  if (constraints)
    delete constraints;
}

bool RTLUnsignedType::UnsignedConstraints::verify(Attribute attr) const {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr || !intAttr.getType().isUnsignedInteger())
    return false;

  // Check all constraints
  unsigned value = intAttr.getUInt();
  return (!lb || lb <= value) && (!ub || value <= ub) && (!eq || value == eq) &&
         (!ne || value != ne);
}

bool RTLUnsignedType::constraintsFromJSON(const json::Object &object,
                                          Constraints *&constraints,
                                          json::Path path) {
  auto boundFromJSON = [&](StringRef kw, StringLiteral err,
                           const llvm::json::Value &value,
                           std::optional<unsigned> &bound) -> bool {
    if (bound) {
      // The bound may be set by the "range" key or the dedicated bound key,
      // make sure there is no conflict
      path.report(err);
      return false;
    }
    return json::fromJSON(value, bound, path);
  };

  // Allocate the constraint object
  UnsignedConstraints *cons = new UnsignedConstraints;
  constraints = cons;

  return llvm::all_of(object, [&](auto &keyAndVal) {
    auto &[jsonKey, val] = keyAndVal;
    std::string key = jsonKey.str();

    if (RESERVED_KEYS.contains(key))
      return true;
    if (key == LB)
      return boundFromJSON(LB, ERR_LB, val, cons->lb);
    if (key == UB)
      return boundFromJSON(UB, ERR_UB, val, cons->ub);
    if (key == RANGE) {
      const json::Array *array = val.getAsArray();
      if (!array) {
        path.report(ERR_EXPECTED_ARRAY);
        return false;
      }
      if (array->size() != 2) {
        path.report(ERR_ARRAY_FORMAT);
        return false;
      }
      return boundFromJSON(LB, ERR_LB, (*array)[0], cons->lb) &&
             boundFromJSON(UB, ERR_UB, (*array)[1], cons->ub);
    }
    if (key == EQ)
      return json::fromJSON(val, cons->eq, path);
    if (key == NE)
      return json::fromJSON(val, cons->eq, path);
    path.report(ERR_UNSUPPORTED);
    return false;
  });
}

std::string RTLUnsignedType::serialize(Attribute attr) const {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr)
    return "";
  return std::to_string(intAttr.getUInt());
}

bool RTLStringType::StringConstraints::verify(Attribute attr) const {
  StringAttr stringAttr = dyn_cast_if_present<StringAttr>(attr);
  if (!stringAttr)
    return false;
  return (!eq || stringAttr == eq) && (!ne || stringAttr != ne);
}

bool RTLStringType::constraintsFromJSON(const json::Object &object,
                                        Constraints *&constraints,
                                        json::Path path) {
  // Allocate the constraint object
  StringConstraints *cons = new StringConstraints;
  constraints = cons;

  return llvm::all_of(object, [&](auto &keyAndVal) {
    auto &[jsonKey, val] = keyAndVal;
    std::string key = jsonKey.str();

    if (RESERVED_KEYS.contains(key))
      return true;
    if (key == EQ)
      return json::fromJSON(val, cons->eq, path);
    if (key == NE)
      return json::fromJSON(val, cons->ne, path);
    path.report(ERR_UNSUPPORTED);
    return false;
  });
}

std::string RTLStringType::serialize(Attribute attr) const {
  StringAttr stringAttr = dyn_cast_if_present<StringAttr>(attr);
  if (!stringAttr)
    return "";
  return stringAttr.str();
}

bool dynamatic::fromJSON(const llvm::json::Value &value, RTLType *&type,
                         llvm::json::Path path) {
  std::optional<StringRef> strType = value.getAsString();
  if (!strType) {
    path.report(ERR_EXPECTED_STRING);
    return false;
  }
  if (*strType == "unsigned") {
    type = new RTLUnsignedType;
  } else if (*strType == "string") {
    type = new RTLStringType;
  } else {
    path.report(ERR_UNKNOWN_TYPE);
    return false;
  }
  return true;
}

RTLMatch RTLRequest::setMatch(const RTLComponent &component,
                              ParameterMappings &serializedParams) const {
  serializedParams[RTLParameter::MODULE_NAME] = "";
  return RTLMatch(component, serializedParams);
}

RTLRequestFromOp::RTLRequestFromOp(Operation *op, const llvm::Twine &name)
    : RTLRequest(name, op->getLoc()), op(op),
      parameters(
          op->getAttrOfType<DictionaryAttr>(RTLRequest::PARAMETERS_ATTR)){};

Attribute RTLRequestFromOp::getParameter(const RTLParameter &param) const {
  if (!parameters)
    return nullptr;
  return parameters.get(param.getName());
}

ParamMatch RTLRequest::matchParameter(const RTLParameter &param) const {
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
    : RTLRequestFromOp(modOp, getName(modOp)){};

RTLMatch
RTLRequestFromHWModule::setMatch(const RTLComponent &component,
                                 ParameterMappings &serializedParams) const {
  serializedParams[RTLParameter::MODULE_NAME] =
      component.isGeneric()
          ? substituteParams(component.getModuleName(), serializedParams)
          : cast<hw::HWModuleExternOp>(op).getSymName();
  return RTLMatch(component, serializedParams);
}

std::string RTLRequestFromHWModule::getName(hw::HWModuleExternOp modOp) {
  if (StringAttr nameAttr =
          modOp->getAttrOfType<StringAttr>(RTLRequest::NAME_ATTR))
    return nameAttr.str();
  return "";
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
    std::string outputFile = outputDir.str() +
                             sys::path::get_separator().str() + moduleName +
                             ".vhd";

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

bool RTLParameter::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) || !mapper.map(KEY_TYPE, type) ||
      !mapper.map(KEY_GENERIC, useAsGeneric))
    return false;

  if (RESERVED_PARAMETER_NAMES.contains(name)) {
    path.field(KEY_NAME).report(ERR_RESERVED_NAME);
    return false;
  }

  // The mapper ensures that this object is valid
  const json::Object &object = *value.getAsObject();
  return type->constraintsFromJSON(object, type->constraints, path);
}

RTLParameter::~RTLParameter() {
  if (type)
    delete type;
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

bool RTLComponent::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) ||
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
  auto setDefaultSignalSuffix = [&](SignalType type, StringRef suffix) {
    if (ioSignals.find(type) == ioSignals.end())
      ioSignals[type] = suffix;
  };
  setDefaultSignalSuffix(SignalType::DATA, "");
  setDefaultSignalSuffix(SignalType::VALID, "_valid");
  setDefaultSignalSuffix(SignalType::READY, "_ready");

  // Make sure the IO map makes sense
  for (auto &[from, to] : ioMap) {
    // At most one wildcard in the key
    size_t keyNumWild = countWildcards(from);
    if (keyNumWild > 1) {
      path.field(KEY_IO_MAP)
          .field(from)
          .report("At most one wildcard is allowed in the key");
      return false;
    }

    // At most one wildcard in the value, and no more than in the key
    size_t valudNumWild = countWildcards(to);
    if (valudNumWild > 1) {
      path.field(KEY_IO_MAP)
          .field(from)
          .report("At most one wildcard is allowed in the value");
      return false;
    }
    if (keyNumWild == 0 && valudNumWild == 1) {
      path.field(KEY_IO_MAP)
          .field(from)
          .report("Value has wildcard but key does not, this is not allowed");
      return false;
    }
  }

  // Timing models need access to the component's RTL parameters to verify
  // the sanity of constraints, so they are parsed separately and somewhat
  // differently

  // The mapper ensures that the object is valid
  const json::Value *jsonModelsValue = value.getAsObject()->get(KEY_MODELS);
  json::Path modelsPath = path.field(KEY_MODELS);
  if (!jsonModelsValue)
    return true;

  const json::Array *jsonModelsArray = jsonModelsValue->getAsArray();
  if (!jsonModelsArray) {
    modelsPath.report(ERR_EXPECTED_ARRAY);
    return false;
  }

  DenseSet<StringRef> skipped{KEY_PARAMETER};
  for (auto [modIdx, jsonModel] : llvm::enumerate(*jsonModelsArray)) {
    Model &model = models.emplace_back();
    json::Path modPath = modelsPath.index(modIdx);

    json::ObjectMapper modelMapper(jsonModel, modPath);
    if (!modelMapper || !modelMapper.map(KEY_PATH, model.path))
      return false;

    // The mapper ensures that this object is valid
    const json::Object &jsonModelObject = *value.getAsObject();

    const json::Value *jsonConstraints = jsonModelObject.get(KEY_CONSTRAINTS);
    json::Path constraintsPath = modPath.field(KEY_CONSTRAINTS);
    if (!jsonConstraints) {
      // Fallback model without constraints
      continue;
    }

    const json::Array *jsonConstraintsArray = jsonConstraints->getAsArray();
    if (!jsonConstraintsArray) {
      constraintsPath.report(ERR_EXPECTED_ARRAY);
      return false;
    }

    for (auto [constIdx, jsonConst] : llvm::enumerate(*jsonConstraintsArray)) {
      json::Path constPath = constraintsPath.index(constIdx);

      // Retrieve the name of the parameter the constraint applies on
      json::ObjectMapper constMapper(jsonConst, constPath);
      std::string paramName;
      if (!constMapper || !modelMapper.map(KEY_PARAMETER, paramName))
        return false;

      // Add a new parameter/constraint vector pair to the model's list.
      auto &[param, constraints] = model.addConstraints.emplace_back();

      // Retrieve the parameter with this name, if it exists
      param = getParameter(paramName);
      if (!param) {
        constPath.field(KEY_NAME).report(ERR_UNKNOWN_PARAM);
        return false;
      }

      // The mapper ensures that this object is valid
      const json::Object &object = *value.getAsObject();
      if (param->type->constraintsFromJSON(object, constraints, path))
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

bool RTLComponent::isCompatible(const RTLRequest &request,
                                std::vector<RTLMatch> *matches) const {
  LLVM_DEBUG(llvm::dbgs() << "Attempting match with RTL component " << name
                          << "\n\t->";);
  if (request.name != name) {
    LLVM_DEBUG(llvm::dbgs() << "Names to do match.\n");
    return false;
  }

  // Keep track of the set of parameters we parse from the dictionnary
  // attribute. We never have duplicate parameters due to the dictionary's
  // keys uniqueness invariant, but we need to make sure all known parameters
  // are present
  DenseSet<StringRef> parsedParams;
  SmallVector<StringRef> ignoredParams;

  ParameterMappings serializedParams;

  for (const RTLParameter &parameter : parameters) {
    ParamMatch paramMatch = request.matchParameter(parameter);
    LLVM_DEBUG({
      switch (paramMatch.state) {
      case ParamMatch::State::DOES_NOT_EXIST:
        llvm::dbgs() << "Parameter \"" << parameter.getName()
                     << "\" does not exist\n";
        break;
      case ParamMatch::State::FAILED_VERIFICATION:
        llvm::dbgs() << "Failed constraint checking for parameter \""
                     << parameter.getName() << "\"\n";
        break;
      case ParamMatch::State::FAILED_SERIALIZATION:
        llvm::dbgs() << "Failed serialization for parameter \""
                     << parameter.getName() << "\"\n";
        break;
      case ParamMatch::State::SUCCESS:
        llvm::dbgs() << "Matched parameter \"" << parameter.getName() << "\"\n";
        break;
      }
    });
    if (paramMatch.state != ParamMatch::SUCCESS)
      return false;
    serializedParams[parameter.name] = paramMatch.serialized;
  }

  if (matches)
    matches->push_back(request.setMatch(*this, serializedParams));
  LLVM_DEBUG(llvm::dbgs() << "Matched!\n");
  return true;
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
  if (!isCompatible(request))
    return nullptr;

  for (const Model &model : models) {
    /// Returns true when all additional constraints on the parameters are
    /// satisfied.
    auto satisifed = [&](const Model::AddConstraints &addConstraints) -> bool {
      auto &[param, constraints] = addConstraints;
      return constraints->verify(request.getParameter(*param));
    };

    // The model matches if all additional parameter constraints are satsified
    if (llvm::all_of(model.addConstraints, satisifed))
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

/// Returns the indexed version of a vector-like RTL signal for a specfic HDL.
static std::string getIndexedName(const Twine &name, size_t arrayIdx,
                                  RTLComponent::HDL hdl) {
  switch (hdl) {
  case RTLComponent::HDL::VHDL:
    return name.str() + "(" + std::to_string(arrayIdx) + ")";
  case RTLComponent::HDL::VERILOG:
    return name.str() + "[" + std::to_string(arrayIdx) + "]";
  }
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

std::string RTLComponent::getRTLPortName(StringRef mlirPortName,
                                         HDL hdl) const {
  std::string remappedName = portRemap(mlirPortName);
  StringRef baseName;
  size_t arrayIdx;
  if (!portNameIsIndexed(remappedName, baseName, arrayIdx))
    return remappedName;
  return getIndexedName(baseName, arrayIdx, hdl);
}

std::string RTLComponent::getRTLPortName(StringRef mlirPortName,
                                         SignalType type, HDL hdl) const {
  std::string signalSuffix = ioSignals.at(type);
  std::string remappedName = portRemap(mlirPortName);
  StringRef baseName;
  size_t arrayIdx;
  if (!portNameIsIndexed(remappedName, baseName, arrayIdx))
    return remappedName + signalSuffix;
  return getIndexedName(baseName + signalSuffix, arrayIdx, hdl);
}

inline bool
llvm::json::fromJSON(const json::Value &value,
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

inline bool dynamatic::fromJSON(const json::Value &value,
                                std::map<SignalType, std::string> &ioChannels,
                                json::Path path) {
  const json::Object *object = value.getAsObject();
  if (!object) {
    path.report(ERR_EXPECTED_OBJECT);
    return false;
  }

  ioChannels.clear();
  for (const auto &[signalStr, jsonSuffix] : *object) {
    // Deserialize the signal type
    SignalType type;
    if (signalStr == "data") {
      type = SignalType::DATA;
    } else if (signalStr == "valid") {
      type = SignalType::VALID;
    } else if (signalStr == "ready") {
      type = SignalType::READY;
    } else {
      path.field(signalStr).report("unknown channel signal type: possible keys "
                                   "are 'data', 'valid', or 'ready'");
      return false;
    }

    // Deserialize the suffix (just a string)
    if (!fromJSON(jsonSuffix, ioChannels[type], path.field(signalStr)))
      return false;
  }

  return true;
}

inline bool dynamatic::fromJSON(const llvm::json::Value &value,
                                RTLComponent::HDL &hdl, llvm::json::Path path) {
  std::optional<StringRef> hdlStr = value.getAsString();
  if (!hdlStr) {
    path.report(ERR_EXPECTED_STRING);
    return false;
  }
  if (hdlStr == "verilog") {
    hdl = RTLComponent::HDL::VERILOG;
  } else if (hdlStr == "vhdl") {
    hdl = RTLComponent::HDL::VHDL;
  } else {
    path.report(ERR_INVALID_HDL);
    return false;
  }
  return true;
}

inline bool dynamatic::fromJSON(const llvm::json::Value &value,
                                RTLComponent::IOKind &io,
                                llvm::json::Path path) {
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
  llvm::Expected<json::Value> value = json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse RTL configuration file @ \"" << filepath
                 << "\" as JSON.\n-> " << toString(value.takeError()) << "\n";
    return failure();
  }

  json::Path::Root jsonRoot(filepath);
  json::Path jsonPath(jsonRoot);

  json::Array *jsonComponents = value->getAsArray();
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

bool RTLConfiguration::hasMatchingComponent(const RTLRequest &request) {
  LLVM_DEBUG(
      llvm::dbgs() << "Attempting to find compatible component for request \""
                   << request.name << "\"\n");
  return llvm::any_of(components, [&](const RTLComponent &component) {
    return component.isCompatible(request);
  });
}

std::optional<RTLMatch>
RTLConfiguration::getMatchingComponent(const RTLRequest &request) {
  LLVM_DEBUG(
      llvm::dbgs() << "Attempting to find compatible component for request \""
                   << request.name << "\"\n");
  std::vector<RTLMatch> matches;
  for (const RTLComponent &component : components) {
    if (component.isCompatible(request, &matches))
      return matches.front();
  }
  return std::nullopt;
}

void RTLConfiguration::findMatchingComponents(
    const RTLRequest &request, std::vector<RTLMatch> &matches) const {
  LLVM_DEBUG(llvm::dbgs() << "Finding compatible components for request \""
                          << request.name << "\"\n");
  for (const RTLComponent &component : components)
    component.isCompatible(request, &matches);
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
