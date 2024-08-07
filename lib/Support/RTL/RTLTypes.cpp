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

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::json;

namespace ljson = llvm::json;

static constexpr StringLiteral KEY_TYPE("type");

/// Reserved JSON keys when deserializing type constraints, should be ignored.
static const mlir::DenseSet<StringRef> RESERVED_KEYS{"name", KEY_TYPE,
                                                     "generic"};

static constexpr StringLiteral ERR_UNKNOWN_TYPE(
    R"(unknown parameter type: options are "boolean", "unsigned", or "string")");

bool RTLType::fromJSON(const ljson::Value &value, ljson::Path path) {
  if (typeConcept)
    delete typeConcept;

  std::string paramType;
  ObjectDeserializer mapper(value, path);
  if (!mapper.map(KEY_TYPE, paramType).valid()) {
    path.field(KEY_TYPE).report(ERR_MISSING_VALUE);
    return false;
  }
  if (!allocIf<RTLBooleanType, RTLUnsignedType, RTLStringType>(paramType)) {
    path.field(KEY_TYPE).report(ERR_UNKNOWN_TYPE);
    return false;
  }

  return typeConcept->fromJSON(value, path);
}

bool BooleanConstraints::verify(Attribute attr) const {
  auto boolAttr = dyn_cast_if_present<BoolAttr>(attr);
  if (!boolAttr)
    return false;

  // Check all constraints
  bool value = boolAttr.getValue();
  return (!eq || value == eq) && (!ne || value != ne);
}

bool dynamatic::fromJSON(const ljson::Value &value, BooleanConstraints &cons,
                         ljson::Path path) {
  return ObjectDeserializer(value, path)
      .map("eq", cons.eq)
      .map("ne", cons.ne)
      .exhausted(RESERVED_KEYS);
}

std::string RTLBooleanType::serialize(Attribute attr) {
  BoolAttr boolAttr = dyn_cast_if_present<BoolAttr>(attr);
  if (!boolAttr)
    return "";
  return std::to_string(boolAttr.getValue());
}

bool UnsignedConstraints::verify(Attribute attr) const {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr || !intAttr.getType().isUnsignedInteger())
    return false;

  // Check all constraints
  unsigned value = intAttr.getUInt();
  return (!lb || lb <= value) && (!ub || value <= ub) && (!eq || value == eq) &&
         (!ne || value != ne);
}

/// Deserialization errors for unsigned constraints.
static constexpr llvm::StringLiteral
    ERR_ARRAY_FORMAT = "expected array to have [lb, ub] format",
    ERR_LB = "lower bound already set", ERR_UB = "upper bound already set";

bool dynamatic::fromJSON(const ljson::Value &value, UnsignedConstraints &cons,
                         ljson::Path path) {
  auto boundFromJSON = [&](StringLiteral err, const ljson::Value &value,
                           std::optional<unsigned> &bound,
                           ljson::Path keyPath) -> bool {
    if (bound) {
      // The bound may be set by the "range" key or the dedicated bound key,
      // make sure there is no conflict
      path.report(err);
      return false;
    }
    return ljson::fromJSON(value, bound, keyPath);
  };

  return ObjectDeserializer(value, path)
      .map("eq", cons.eq)
      .map("ne", cons.ne)
      .mapOptional("lb",
                   [&](auto &val, auto path) {
                     return boundFromJSON(ERR_LB, val, cons.lb, path);
                   })
      .mapOptional("ub",
                   [&](auto &val, auto path) {
                     return boundFromJSON(ERR_UB, val, cons.ub, path);
                   })
      .mapOptional("range",
                   [&](auto &val, auto path) {
                     const ljson::Array *array = val.getAsArray();
                     if (!array) {
                       path.report(ERR_EXPECTED_ARRAY);
                       return false;
                     }
                     if (array->size() != 2) {
                       path.report(ERR_ARRAY_FORMAT);
                       return false;
                     }
                     return boundFromJSON(ERR_LB, (*array)[0], cons.lb, path) &&
                            boundFromJSON(ERR_UB, (*array)[1], cons.ub, path);
                   })
      .exhausted(RESERVED_KEYS);
}

std::string RTLUnsignedType::serialize(Attribute attr) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr)
    return "";
  return std::to_string(intAttr.getUInt());
}

bool StringConstraints::verify(Attribute attr) const {
  StringAttr stringAttr = dyn_cast_if_present<StringAttr>(attr);
  if (!stringAttr)
    return false;
  return (!eq || stringAttr == eq) && (!ne || stringAttr != ne);
}

bool dynamatic::fromJSON(const ljson::Value &value, StringConstraints &cons,
                         ljson::Path path) {
  return ObjectDeserializer(value, path)
      .map("eq", cons.eq)
      .map("ne", cons.ne)
      .exhausted(RESERVED_KEYS);
}

std::string RTLStringType::serialize(Attribute attr) {
  StringAttr stringAttr = dyn_cast_if_present<StringAttr>(attr);
  if (!stringAttr)
    return "";
  return stringAttr.str();
}
