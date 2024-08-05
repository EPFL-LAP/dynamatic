//===- RTLTypes.cpp - All supported RTL types -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements all supported RTL types.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL/RTLTypes.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

/// Recognized keys in RTL configuration files.
static constexpr StringLiteral KEY_GENERIC("generic"), KEY_NAME("name"),
    KEY_TYPE("type"), KEY_PARAMETER("parameter");

/// Reserved JSON keys when deserializing type constraints, should be ignored.
static const mlir::DenseSet<StringRef> RESERVED_KEYS{
    KEY_NAME, KEY_TYPE, KEY_PARAMETER, KEY_GENERIC};

static constexpr StringLiteral ERR_UNKNOWN_TYPE(
    R"(unknown parameter type: options are "unsigned" or "string")");

RTLType::~RTLType() {
  if (constraints)
    delete constraints;
}

bool RTLBooleanType::BoolConstraints::verify(Attribute attr) const {
  auto boolAttr = dyn_cast_if_present<BoolAttr>(attr);
  if (!boolAttr)
    return false;

  // Check all constraints
  bool value = boolAttr.getValue();
  return (!eq || value == eq) && (!ne || value != ne);
}

bool RTLBooleanType::constraintsFromJSON(const json::Object &object,
                                         Constraints *&constraints,
                                         json::Path path) {
  // Allocate the constraint object
  BoolConstraints *cons = new BoolConstraints;
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

std::string RTLBooleanType::serialize(Attribute attr) const {
  BoolAttr boolAttr = dyn_cast_if_present<BoolAttr>(attr);
  if (!boolAttr)
    return "";
  return std::to_string(boolAttr.getValue());
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
  } else if (*strType == "boolean") {
    type = new RTLBooleanType;
  } else {
    path.report(ERR_UNKNOWN_TYPE);
    return false;
  }
  return true;
}
