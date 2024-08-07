//===- JSON.h - JSON-related helpers ----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of helper library to work with JSON files in Dynamatic.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dynamatic::json;

bool llvm::json::fromJSON(const json::Value &value, unsigned &number,
                          json::Path path) {
  std::optional<uint64_t> opt = value.getAsUINT64();
  if (!opt.has_value()) {
    path.report("expected unsigned number");
    return false;
  }
  number = opt.value();
  return true;
}

namespace {

/// Serializes MLIR attributes to JSON. Only supports a restricted number of
/// attribute types at the moment but can be easily extended.
class AttributeJSONSerializer {
public:
  /// Constructs the serializer from an intialized output stream and a location
  /// to report errors from.
  AttributeJSONSerializer(llvm::raw_fd_ostream &filestream, Location loc)
      : os(filestream), loc(loc) {};

  /// Attempts to serialize an attribute. If the key name is non-empty, uses it
  /// on failure to create a nicer error message denoting where the problem
  /// occured.
  LogicalResult serializeAttr(Attribute attr, StringRef keyName = "") {
    return llvm::TypeSwitch<Attribute, LogicalResult>(attr)
        .Case<StringAttr>([&](StringAttr stringAttr) {
          os.value(stringAttr.strref());
          return success();
        })
        .Case<BoolAttr>([&](BoolAttr boolAttr) {
          os.value(boolAttr.getValue());
          return success();
        })
        .Case<IntegerAttr>([&](IntegerAttr intAttr) {
          Type intType = intAttr.getType();
          if (intType.isUnsignedInteger())
            os.value(intAttr.getUInt());
          else if (intType.isSignedInteger())
            os.value(intAttr.getSInt());
          else
            os.value(intAttr.getInt());
          return success();
        })
        .Case<ArrayAttr>([&](ArrayAttr arrayAttr) {
          os.arrayBegin();
          for (Attribute nestedAttr : arrayAttr) {
            if (failed(serializeAttr(nestedAttr, keyName))) {
              os.arrayEnd();
              return failure();
            }
          }
          os.arrayEnd();
          return success();
        })
        .Case<DictionaryAttr>([&](DictionaryAttr dictAttr) {
          os.objectBegin();
          for (NamedAttribute nestedNamedAttr : dictAttr) {
            if (failed(serializeNamedAttr(nestedNamedAttr))) {
              os.objectEnd();
              return failure();
            }
          }
          os.objectEnd();
          return success();
        })
        .Default([&](auto) {
          if (keyName.empty())
            return emitError(loc) << "Unsupported attribute type";
          return emitError(loc)
                 << "Unsupported attribute type nested within attribute \""
                 << keyName << "\"";
        });
  }

  /// Attempts to serialize a named attribute. Similar internal logic as for
  /// non-named attribute, but additionally creates a JSON key at the current
  /// level to associate the attribute's name to its value's serialized
  /// representation.
  LogicalResult serializeNamedAttr(NamedAttribute namedAttr) {
    StringRef attrName = namedAttr.getName();
    Attribute attr = namedAttr.getValue();
    return llvm::TypeSwitch<Attribute, LogicalResult>(attr)
        .Case<StringAttr>([&](StringAttr stringAttr) {
          os.attribute(attrName, stringAttr.strref());
          return success();
        })
        .Case<BoolAttr>([&](BoolAttr boolAttr) {
          os.attribute(attrName, boolAttr.getValue());
          return success();
        })
        .Case<IntegerAttr>([&](IntegerAttr intAttr) {
          Type intType = intAttr.getType();
          if (intType.isUnsignedInteger())
            os.attribute(attrName, intAttr.getUInt());
          else if (intType.isSignedInteger())
            os.attribute(attrName, intAttr.getSInt());
          else
            os.attribute(attrName, intAttr.getInt());
          return success();
        })
        .Case<ArrayAttr>([&](ArrayAttr arrayAttr) {
          os.attributeBegin(attrName);
          os.arrayBegin();
          for (Attribute nestedAttr : arrayAttr) {
            if (failed(serializeAttr(nestedAttr, attrName))) {
              os.arrayEnd();
              os.attributeEnd();
              return failure();
            }
          }
          os.arrayEnd();
          os.attributeEnd();
          return success();
        })
        .Case<DictionaryAttr>([&](DictionaryAttr dictAttr) {
          os.attributeBegin(attrName);
          os.objectBegin();
          for (const NamedAttribute &nestedNamedAttr : dictAttr) {
            if (failed(serializeNamedAttr(nestedNamedAttr))) {
              os.objectEnd();
              os.attributeEnd();
              return failure();
            }
          }
          os.objectEnd();
          os.attributeEnd();
          return success();
        })
        .Default([&](auto) {
          return emitError(loc)
                 << "Unsupported attribute type associated with attribute \""
                 << attrName << "\"";
        });
  }

private:
  /// Stream to the output file.
  llvm::json::OStream os;
  /// Location at which to report errors.
  Location loc;
};
} // namespace

LogicalResult dynamatic::json::serializeToJSON(Attribute attr,
                                               StringRef filepath,
                                               Location loc) {
  if (!attr)
    return emitError(loc) << "The attribute is null; cannot serialize to JSON.";

  std::error_code ec;
  llvm::raw_fd_ostream filestream(filepath, ec);
  if (ec.value() != 0) {
    return emitError(loc) << "Failed to create JSON file @ \"" << filepath
                          << "\"";
  }
  AttributeJSONSerializer serializer(filestream, loc);
  return serializer.serializeAttr(attr);
}

ObjectDeserializer::ObjectDeserializer(const llvm::json::Value &val,
                                       llvm::json::Path path)
    : obj(val.getAsObject()), path(path) {
  if (!obj)
    path.report(ERR_EXPECTED_OBJECT);
}

ObjectDeserializer::ObjectDeserializer(const llvm::json::Object &obj,
                                       llvm::json::Path path)
    : obj(&obj), path(path) {}

ObjectDeserializer &ObjectDeserializer::map(const Twine &key, const MapFn &fn) {
  if (!mapValid || !obj)
    return *this;
  std::string keyStr = key.str();
  if (const llvm::json::Value *val = obj->get(keyStr)) {
    if (auto [_, newKey] = mappedKeys.insert(keyStr); !newKey) {
      path.field(keyStr).report(ERR_DUP_KEY);
      mapValid = false;
    } else {
      mapValid = fn(*val, path.field(keyStr));
    }
    return *this;
  }
  mapValid = false;
  return *this;
}

ObjectDeserializer &ObjectDeserializer::mapOptional(const Twine &key,
                                                    const MapFn &fn) {
  if (!mapValid || !obj)
    return *this;
  std::string keyStr = key.str();
  if (const llvm::json::Value *val = obj->get(keyStr)) {
    if (auto [_, newKey] = mappedKeys.insert(keyStr); !newKey) {
      path.field(keyStr).report(ERR_DUP_KEY);
      mapValid = false;
      return *this;
    }
    mapValid = fn(*val, path.field(keyStr));
  }
  return *this;
}

bool ObjectDeserializer::exhausted(const DenseSet<StringRef> &allowUnmapped) {
  if (!mapValid || !obj)
    return false;
  return llvm::all_of(*obj, [&](auto &keyAndVal) {
    std::string key = keyAndVal.first.str();
    if ((mappedKeys.find(key) != mappedKeys.end()) ||
        allowUnmapped.contains(key))
      return true;
    path.field(key).report("unmapped key in object");
    return false;
  });
}