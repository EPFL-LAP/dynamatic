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
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <fstream>

#define DEBUG_TYPE "support-rtl"

using namespace llvm;
using namespace dynamatic;

/// Recognized keys in RTL configuration files.
static constexpr StringLiteral KEY_COMPONENT("component"),
    KEY_PARAMETERS("parameters"), KEY_MODELS("models"), KEY_GENERIC("generic"),
    KEY_GENERATOR("generator"), KEY_NAME("name"), KEY_TYPE("type"),
    KEY_PATH("path"), KEY_CONSTRAINTS("constraints"),
    KEY_PARAMETER("parameter");

/// JSON path errors.
static constexpr StringLiteral ERR_MISSING_VALUE("missing value"),
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
        R"(unknown parameter type: options are "unsigned" or "string")");

/// Attempts to deserialize the template parameter constraint from the JSON
/// value if the JSON key matches the constraint's expected key. Sets `knownKey`
/// to true if the key matches, and returns whether the constraint was able to
/// be deserialized from JSON. If it was, adds the constraint to the constraint
/// vector.
template <typename Constraint>
static bool fromJSON(StringRef jsonKey, const llvm::json::Value &value,
                     llvm::json::Path path, ConstraintVector &consVec,
                     bool &knownKey) {
  if (Constraint::getKey() != jsonKey)
    return false;
  knownKey = true;
  Constraint *cons = new Constraint;
  if (!cons->fromJSON(value, path.field(jsonKey))) {
    delete cons;
    return false;
  }

  consVec.constraints.push_back(cons);
  return true;
}

/// Attempts to deserialize any of the template parameter constraints from the
/// JSON value, using the JSON key to decide which (if any) to match. Sets
/// `knownKey` to true if the key matches, and returns whether the constraint
/// was able to be deserialized from JSON. If it was, adds the constraint to the
/// constraint vector.
template <typename FirstConstraint, typename SecondConstraint,
          typename... OtherConstraints>
static bool fromJSON(StringRef jsonKey, const llvm::json::Value &value,
                     llvm::json::Path path, ConstraintVector &consVec,
                     bool &knownKey) {
  if (fromJSON<FirstConstraint>(jsonKey, value, path, consVec, knownKey) ||
      knownKey)
    return true;
  return fromJSON<SecondConstraint, OtherConstraints...>(jsonKey, value, path,
                                                         consVec, knownKey);
}

/// Attempts to deserialize any of the template parameter constraints from the
/// JSON value, using the JSON key to decide which (if any) to match. Returns
/// whether the constraint was able to be deserialized from JSON. If it was,
/// adds the constraint to the constraint vector.
template <typename FirstConstraint, typename... OtherConstraints>
static bool fromJSON(StringRef jsonKey, const llvm::json::Value &value,
                     llvm::json::Path path, StringLiteral err,
                     ConstraintVector &consVec) {
  bool knownKey = false;
  if (fromJSON<FirstConstraint, OtherConstraints...>(jsonKey, value, path,
                                                     consVec, knownKey))
    return true;
  if (!knownKey)
    path.field(jsonKey).report(err);
  return false;
}

/// Attempts to deserialize parameter constraints from the JSON object based on
/// the provided RTL parameter type. Reports an error if any key in the object
/// cannot be matched to a known constraint, unless the key is in the set of
/// skipped keys. Returns whether constraints were able to be parsed.
static bool parseConstraints(const json::Object &object,
                             RTLParameter::Type type, llvm::json::Path path,
                             const mlir::DenseSet<StringRef> &skippedKeys,
                             ConstraintVector &consVec) {
  for (auto &[jsonKey, value] : object) {
    std::string key = jsonKey.str();
    // Skip any known key
    if (skippedKeys.contains(key))
      continue;

    // Parse optional constraint, depending on the parameter type
    bool ret;
    switch (type) {
    case RTLParameter::Type::UNSIGNED:
      ret = fromJSON<UnsignedLowerBound, UnsignedUpperBound, UnsignedRange,
                     UnsignedEqual, UnsignedDifferent>(
          key, value, path, UnsignedConstraint::ERR_UNSUPPORTED, consVec);
      break;
    case RTLParameter::Type::STRING:
      ret = fromJSON<StringEqual, StringDifferent>(
          key, value, path, StringConstraint::ERR_UNSUPPORTED, consVec);
      break;
    }
    if (!ret)
      return false;
  }
  return true;
}

bool UnsignedConstraint::apply(StringRef paramValue) {
  if (llvm::any_of(paramValue.str(), [](char c) { return !isdigit(c); }))
    return false;
  return apply(std::stoi(paramValue.str()));
}

bool UnsignedRange::fromJSON(const json::Value &value, json::Path path) {
  const json::Array *array = value.getAsArray();
  if (!array) {
    path.report(ERR_EXPECTED_ARRAY);
    return false;
  }
  if (array->size() != 2) {
    path.report(ERR_ARRAY_FORMAT);
    return false;
  }
  return llvm::json::fromJSON((*array)[0], lb, path) &&
         llvm::json::fromJSON((*array)[1], ub, path);
}

bool ConstraintVector::verifyConstraints(StringRef paramValue) const {
  return llvm::all_of(constraints, [&](ParameterConstraint *cons) {
    return cons->apply(paramValue);
  });
}

RTLMatch::RTLMatch(hw::HWModuleExternOp modOp) : op(modOp) {
  StringAttr nameAttr = modOp->getAttrOfType<StringAttr>("hw.name");
  if (!nameAttr)
    return;
  name = nameAttr.str();

  auto paramAttr = modOp->getAttrOfType<DictionaryAttr>("hw.parameters");
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

bool RTLParameter::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_NAME, name) || !mapper.map(KEY_TYPE, type))
    return false;

  // The mapper ensures that this object is valid
  const json::Object &object = *value.getAsObject();
  DenseSet<StringRef> skipped{KEY_NAME, KEY_TYPE};
  return parseConstraints(object, type, path, skipped, constraints);
}

bool dynamatic::fromJSON(const llvm::json::Value &value,
                         RTLParameter::Type &type, llvm::json::Path path) {
  std::optional<StringRef> strType = value.getAsString();
  if (!strType) {
    path.report(ERR_EXPECTED_STRING);
    return false;
  }
  if (*strType == "unsigned") {
    type = RTLParameter::Type::UNSIGNED;
  } else if (*strType == "string") {
    type = RTLParameter::Type::STRING;
  } else {
    path.report(ERR_UNKNOWN_TYPE);
    return false;
  }
  return true;
}

bool dynamatic::fromJSON(const llvm::json::Value &value,
                         SmallVector<RTLParameter> &parameters,
                         llvm::json::Path path) {
  const json::Array *jsonParametersArray = value.getAsArray();
  if (!jsonParametersArray) {
    path.report(ERR_EXPECTED_ARRAY);
    return false;
  }

  for (auto [idx, jsonParam] : llvm::enumerate(*jsonParametersArray)) {
    RTLParameter *param = &parameters.emplace_back();
    if (!param->fromJSON(jsonParam, path.index(idx)))
      return false;
  }
  return true;
}

bool RTLComponent::fromJSON(const llvm::json::Value &value,
                            llvm::json::Path path) {
  json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(KEY_COMPONENT, component) ||
      !mapper.map(KEY_PARAMETERS, parameters) ||
      !mapper.mapOptional(KEY_GENERIC, generic) ||
      !mapper.mapOptional(KEY_GENERATOR, generator)) {
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
  if (!jsonModelsValue) {
    modelsPath.report(ERR_MISSING_VALUE);
    return false;
  }
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
      auto &[param, constraints] = model.constraints.emplace_back();

      // Retrieve the parameter with this name, if it exists
      param = getParameter(paramName);
      if (!param) {
        constPath.field(KEY_NAME).report(ERR_UNKNOWN_PARAM);
        return false;
      }

      // Parse the constraints for the parameter, checking that they make sense
      // with the type
      if (parseConstraints(*jsonConst.getAsObject(), param->getType(),
                           constPath, skipped, constraints))
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
  LLVM_DEBUG(llvm::dbgs() << "Attempting match with RTL component " << component
                          << "\n";);
  // Component must be valid and have matching name
  if (match.invalid) {
    LLVM_DEBUG(llvm::dbgs() << "-> Match is invalid\n");
    return false;
  }
  if (match.name != component) {
    LLVM_DEBUG(llvm::dbgs() << "-> Names to do match.\n");
    return false;
  }

  // Keep track of the set of parameters we parse from the dictionnary
  // attribute. We never have duplicate parameters due to the dictionary's keys
  // uniqueness invariant, but we need to make sure all known parameters are
  // present
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
    auto constraintsSatisfied = [&](const auto &paramAndConstraints) -> bool {
      auto &[param, constraints] = paramAndConstraints;
      // Guaranteed to exist since the component matches
      auto paramValue = match.parameters.find(param->getName());
      assert(paramValue != match.parameters.end() && "parameter must exist");
      return constraints.verifyConstraints(paramValue->second);
    };

    // The model matches if all additional parameter constraints are satsified
    if (llvm::all_of(model.constraints, constraintsSatisfied))
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
