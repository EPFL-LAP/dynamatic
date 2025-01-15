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
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/Utils/Utils.h"
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
    R"(unknown parameter type: options are "boolean", "unsigned", "string", )"
    R"("dataflow", or "timing")");

bool RTLType::fromJSON(const ljson::Value &value, ljson::Path path) {
  if (typeConcept)
    delete typeConcept;

  std::string paramType;
  ObjectDeserializer mapper(value, path);
  if (!mapper.map(KEY_TYPE, paramType).valid()) {
    path.field(KEY_TYPE).report(ERR_MISSING_VALUE);
    return false;
  }
  if (!allocIf<RTLBooleanType, RTLUnsignedType, RTLStringType, RTLDataflowType,
               RTLTimingType>(paramType)) {
    path.field(KEY_TYPE).report(ERR_UNKNOWN_TYPE);
    return false;
  }

  return typeConcept->fromJSON(value, path);
}

//===----------------------------------------------------------------------===//
// BooleanConstraints / RTLBooleanType
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// UnsignedConstraints / RTLUnsignedType
//===----------------------------------------------------------------------===//

bool UnsignedConstraints::verify(Attribute attr) const {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr || !intAttr.getType().isUnsignedInteger())
    return false;
  return verify(intAttr.getUInt());
}

bool UnsignedConstraints::verify(unsigned value) const {
  return (!lb || lb <= value) && (!ub || value <= ub) && (!eq || value == eq) &&
         (!ne || value != ne);
}

bool UnsignedConstraints::unconstrained() const {
  return !lb && !ub && !eq && !ne;
}

/// Deserialization errors for unsigned constraints.
static constexpr llvm::StringLiteral
    ERR_ARRAY_FORMAT = "expected array to have [lb, ub] format",
    ERR_LB = "lower bound already set", ERR_UB = "upper bound already set";

json::ObjectDeserializer &
UnsignedConstraints::deserialize(json::ObjectDeserializer &deserial,
                                 StringRef keyPrefix) {
  auto boundFromJSON = [&](StringLiteral err, const ljson::Value &value,
                           std::optional<unsigned> &bound,
                           ljson::Path keyPath) -> bool {
    if (bound) {
      // The bound may be set by the "range" key or the dedicated bound key,
      // make sure there is no conflict
      keyPath.report(err);
      return false;
    }
    return ljson::fromJSON(value, bound, keyPath);
  };

  deserial.map(keyPrefix + "eq", eq)
      .map(keyPrefix + "ne", ne)
      .mapOptional(keyPrefix + "lb",
                   [&](auto &val, auto path) {
                     return boundFromJSON(ERR_LB, val, lb, path);
                   })
      .mapOptional(keyPrefix + "ub",
                   [&](auto &val, auto path) {
                     return boundFromJSON(ERR_UB, val, ub, path);
                   })
      .mapOptional(keyPrefix + "range", [&](auto &val, auto path) {
        const ljson::Array *array = val.getAsArray();
        if (!array) {
          path.report(ERR_EXPECTED_ARRAY);
          return false;
        }
        if (array->size() != 2) {
          path.report(ERR_ARRAY_FORMAT);
          return false;
        }
        return boundFromJSON(ERR_LB, (*array)[0], lb, path) &&
               boundFromJSON(ERR_UB, (*array)[1], ub, path);
      });

  return deserial;
}

bool dynamatic::fromJSON(const ljson::Value &value, UnsignedConstraints &cons,
                         ljson::Path path) {
  ObjectDeserializer deserial(value, path);
  return cons.deserialize(deserial).exhausted(RESERVED_KEYS);
}

std::string RTLUnsignedType::serialize(Attribute attr) {
  auto intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!intAttr)
    return "";
  return std::to_string(intAttr.getUInt());
}

//===----------------------------------------------------------------------===//
// StringConstraints / RTLStringType
//===----------------------------------------------------------------------===//

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
  auto stringAttr = dyn_cast_if_present<StringAttr>(attr);
  if (!stringAttr)
    return "";
  return stringAttr.str();
}

//===----------------------------------------------------------------------===//
// DataflowConstraints / RTLDataflowType
//===----------------------------------------------------------------------===//

bool DataflowConstraints::verify(Attribute attr) const {
  auto typeAttr = dyn_cast_if_present<TypeAttr>(attr);
  if (!typeAttr)
    return false;
  Type ty = typeAttr.getValue();
  if (auto channelType = dyn_cast<handshake::ChannelType>(ty)) {
    return dataWidth.verify(channelType.getDataBitWidth()) &&
           numExtras.verify(channelType.getNumExtraSignals()) &&
           numDownstreams.verify(channelType.getNumDownstreamExtraSignals()) &&
           numUpstreams.verify(channelType.getNumUpstreamExtraSignals());
  }
  if (isa<handshake::ControlType>(ty)) {
    return dataWidth.verify(0) && numExtras.verify(0) &&
           numDownstreams.verify(0) && numUpstreams.verify(0);
  }
  return false;
}

bool dynamatic::fromJSON(const ljson::Value &value, DataflowConstraints &cons,
                         ljson::Path path) {
  ObjectDeserializer deserial(value, path);
  cons.dataWidth.deserialize(deserial, "data-");
  cons.numExtras.deserialize(deserial, "extra-");
  cons.numDownstreams.deserialize(deserial, "down-");
  cons.numUpstreams.deserialize(deserial, "up-");
  return deserial.exhausted(RESERVED_KEYS);
}

std::string RTLDataflowType::serialize(Attribute attr) {
  auto typeAttr = dyn_cast_if_present<TypeAttr>(attr);
  if (!typeAttr)
    return "";
  if (auto ty = dyn_cast<handshake::ChannelType>(typeAttr.getValue())) {
    std::stringstream ss;
    ss << ty.getDataBitWidth();
    for (const handshake::ExtraSignal &extra : ty.getExtraSignals()) {
      ss << "-" << extra.name.str() << "-" << extra.getBitWidth()
         << (extra.downstream ? "-D" : "-U");
    }
    return ss.str();
  }
  if (auto ty = dyn_cast<handshake::ControlType>(typeAttr.getValue())) {
    std::stringstream ss;
    ss << "0";
    for (const handshake::ExtraSignal &extra : ty.getExtraSignals()) {
      ss << "-" << extra.name.str() << "-" << extra.getBitWidth()
         << (extra.downstream ? "-D" : "-U");
    }
    return ss.str();
  }
  return "";
}

//===----------------------------------------------------------------------===//
// TimingConstraints / RTLTimingType
//===----------------------------------------------------------------------===//

TimingConstraints::TimingConstraints() {
  for (SignalType type : getSignalTypes())
    latencies.emplace(type, UnsignedConstraints{});
}

bool TimingConstraints::verify(Attribute attr) const {
  auto timingAttr = dyn_cast_if_present<handshake::TimingAttr>(attr);
  if (!timingAttr)
    return false;

  handshake::TimingInfo info = timingAttr.getInfo();
  for (SignalType type : getSignalTypes()) {
    std::optional<unsigned> latency = info.getLatency(type);
    const UnsignedConstraints &cons = latencies.at(type);
    if (latency) {
      if (!cons.verify(*latency))
        return false;
    } else {
      if (!cons.unconstrained())
        return false;
    }
  }
  return true;
}

static const std::map<SignalType, StringRef> SIGNAL_TYPE_NAMES = {
    {SignalType::DATA, "data"},
    {SignalType::VALID, "valid"},
    {SignalType::READY, "ready"},
};

bool dynamatic::fromJSON(const ljson::Value &value, TimingConstraints &cons,
                         ljson::Path path) {
  ObjectDeserializer deserial(value, path);

  std::string latSuffix = RTLTimingType::LATENCY.str() + "-";
  for (SignalType type : getSignalTypes()) {
    std::string key = SIGNAL_TYPE_NAMES.at(type).str() + latSuffix;
    cons.latencies.at(type).deserialize(deserial, key);
  }
  return deserial.exhausted(RESERVED_KEYS);
}
