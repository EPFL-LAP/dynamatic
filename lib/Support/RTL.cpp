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
#include "dynamatic/Support/TimingModels.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <regex>

#define DEBUG_TYPE "support-rtl"

using namespace llvm;
using namespace dynamatic;

/// Recognized keys in RTL configuration files.
static constexpr StringLiteral KEY_PARAMETERS("parameters"),
    KEY_MODELS("models"), KEY_GENERIC("generic"), KEY_GENERATOR("generator"),
    KEY_NAME("name"), KEY_TYPE("type"), KEY_PATH("path"),
    KEY_CONSTRAINTS("constraints"), KEY_PARAMETER("parameter"),
    KEY_DEPENDENCIES("dependencies");

/// JSON path errors.
static constexpr StringLiteral ERR_EXPECTED_ARRAY("expected array"),
    ERR_MISSING_CONCRETIZATION("missing concretization method, either "
                               "\"generic\" or \"generator\" key must exist"),
    ERR_MULTIPLE_CONCRETIZATION(
        "multiple concretization methods provided, only one of "
        "\"generic\" or \"generator\" key must exist"),
    ERR_EXPECTED_STRING("expected string"),
    ERR_UNKNOWN_PARAM("unknown parameter name"),
    ERR_DUPLICATE_NAME("duplicated parameter name"),
    ERR_UNKNOWN_TYPE(
        R"(unknown parameter type: options are "unsigned" or "string")");

/// Reserved JSON keys when deserializing type constraints, should be ignored.
static const mlir::DenseSet<StringRef> RESERVED_KEYS{
    KEY_NAME, KEY_TYPE, KEY_PARAMETER, KEY_GENERIC};

RTLType::~RTLType() {
  if (constraints)
    delete constraints;
}

bool RTLUnsignedType::UnsignedConstraints::verify(StringRef paramValue) const {
  // Decode the unsigned value from the string, if possible
  std::optional optValue = RTLUnsignedType::decode(paramValue);
  if (!optValue)
    return false;
  unsigned value = *optValue;

  // Check all constraints
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

bool RTLStringType::StringConstraints::verify(StringRef paramValue) const {
  return (!eq || paramValue == eq) && (!ne || paramValue != ne);
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

RTLMatch::RTLMatch(hw::HWModuleExternOp modOp) : op(modOp) {
  StringAttr nameAttr = modOp->getAttrOfType<StringAttr>(NAME_ATTR);
  if (!nameAttr)
    return;
  name = nameAttr.str();

  auto paramAttr = modOp->getAttrOfType<DictionaryAttr>(PARAMETERS_ATTR);
  if (!paramAttr)
    return;

  for (const NamedAttribute &namedParameter : paramAttr) {
    // Must be a string attribute
    StringAttr paramValue = dyn_cast<StringAttr>(namedParameter.getValue());
    if (!paramValue)
      return;
    parameters[namedParameter.getName()] = paramValue.str();
  }

  invalid = false;
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

bool RTLParameter::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) || !mapper.map(KEY_TYPE, type) ||
      !mapper.map(KEY_GENERIC, useAsGeneric))
    return false;
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

bool RTLComponent::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) ||
      !mapper.mapOptional(KEY_PARAMETERS, parameters) ||
      !mapper.mapOptional(KEY_GENERIC, generic) ||
      !mapper.mapOptional(KEY_GENERATOR, generator) ||
      !mapper.mapOptional(KEY_DEPENDENCIES, dependencies)) {
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
  // Check that at most one concretization method was provided
  if (!generic.empty() && !generator.empty()) {
    path.report(ERR_MULTIPLE_CONCRETIZATION);
    return false;
  }

  // Timing models need access to the component's RTL parameters to verify the
  // sanity of constraints, so they are parsed separately

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

bool RTLComponent::isCompatible(const RTLMatch &match) const {
  LLVM_DEBUG(llvm::dbgs() << "Attempting match with RTL component " << name
                          << "\n";);
  // Component must be valid and have matching name
  if (match.invalid) {
    LLVM_DEBUG(llvm::dbgs() << "-> Match is invalid\n");
    return false;
  }
  if (match.name != name) {
    LLVM_DEBUG(llvm::dbgs() << "-> Names to do match.\n");
    return false;
  }

  // Keep track of the set of parameters we parse from the dictionnary
  // attribute. We never have duplicate parameters due to the dictionary's
  // keys uniqueness invariant, but we need to make sure all known parameters
  // are present
  DenseSet<StringRef> parsedParams;
  SmallVector<StringRef> ignoredParams;

  for (auto &[paramName, paramValue] : match.parameters) {
    RTLParameter *parameter = getParameter(paramName);
    if (!parameter) {
      // Skip over unknown parameters (the external module may just be "more
      // specialized" than the RTL support)
      ignoredParams.push_back(paramName);
      continue;
    }

    // Must pass type constraints
    if (!parameter->verifyConstraints(paramValue)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "-> Failed constraint checking for parameter \""
                 << paramName << "\"\n");
      return false;
    }
    parsedParams.insert(paramName);
  }

  // Due to key uniqueness, if we parsed the same number of parameters as the
  // number of known RTL parameters, then it means we have parsed them all
  if (parsedParams.size() != parameters.size()) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Missing " << parameters.size() - parsedParams.size()
                   << " parameters in match";);
    return false;
  }

  // We have a successful match, warn user of any ignored parameter
  for (StringRef paramName : ignoredParams) {
    match.op->emitWarning()
        << "Ignoring parameter \"" << paramName << "\n not found in match.";
  }
  return true;
}

const RTLComponent::Model *RTLComponent::getModel(const RTLMatch &match) const {
  if (!isCompatible(match))
    return nullptr;

  for (const Model &model : models) {
    /// Returns true when all additional constraints on the parameters are
    /// satisfied.
    auto constraintsSatisfied = [&](const auto &paramAndType) -> bool {
      auto &[param, type] = paramAndType;
      // Guaranteed to exist since the component matches
      auto paramValue = match.parameters.find(param->getName());
      assert(paramValue != match.parameters.end() && "parameter must exist");
      return type->verify(paramValue->second);
    };

    // The model matches if all additional parameter constraints are satsified
    if (llvm::all_of(model.addConstraints, constraintsSatisfied))
      return &model;
  }

  // No matching model
  return nullptr;
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
                 << "\" as JSON\n";
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

const RTLComponent *
RTLConfiguration::getComponent(const RTLMatch &match) const {
  LLVM_DEBUG({
    llvm::dbgs() << "Attempting to match " << match.name
                 << " with parameters:\n";
    for (const auto &[name, value] : match.parameters)
      llvm::dbgs() << "\t" << name << ": " << value << "\n";
  });
  if (match.invalid)
    return nullptr;

  for (const RTLComponent &component : components) {
    if (component.isCompatible(match))
      return &component;
  }
  return nullptr;
}

const RTLComponent::Model *
RTLConfiguration::getModel(const RTLMatch &match) const {
  if (match.invalid)
    return nullptr;
  for (const RTLComponent &component : components) {
    if (const RTLComponent::Model *model = component.getModel(match))
      return model;
  }
  return nullptr;
}

std::string dynamatic::replaceRegexes(
    StringRef input, const std::map<std::string, std::string> &replacements) {
  std::string result(input);
  for (auto &[from, to] : replacements)
    result = std::regex_replace(result, std::regex(from), to);
  return result;
}