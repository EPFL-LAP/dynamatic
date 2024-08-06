//===- JSON.h - JSON-related helpers ----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines some helpers on top of LLVM's standard JSON library to work with
// JSON files in Dynamatic
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_JSON_JSON_H
#define DYNAMATIC_SUPPORT_JSON_JSON_H

#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"

namespace llvm {
namespace json {
/// Deserializes a JSON value into an unsigned number. This function is placed
/// inside of the ::llvm::json namespace since the deserialization target type
/// is a standard type. See ::llvm::json::Value's documentation for a longer
/// description of this function's behavior.
bool fromJSON(const llvm::json::Value &value, unsigned &number,
              llvm::json::Path path);
} // namespace json
} // namespace llvm

namespace dynamatic {

namespace json {

/// Standard JSON errors.
static constexpr StringLiteral ERR_EXPECTED_OBJECT("expected object"),
    ERR_EXPECTED_ARRAY("expected array"),
    ERR_EXPECTED_STRING("expected string"),
    ERR_EXPECTED_NUMBER("expected number"),
    ERR_EXPECTED_BOOLEAN("expected boolean"),
    ERR_MISSING_VALUE("missing value");

/// Attempts to serialize an MLIR attribute into a JSON file, which is
/// created at the provided filepath. Succeeds when the attribute was of a
/// supported type; otherwise fails and reports an error at the given
/// location. Note that the top-level attribute must either be an array or
/// dictionnary attribute for serialization to have a change at succeeding.
LogicalResult serializeToJSON(Attribute attr, StringRef filepath, Location loc);

/// Helper for deserializing JSON objects into arbitrary data. Similar to
/// `llvm::json::ObjectMapper` in purpose but
/// 1. all keys are considered optional,
/// 2. keys can be associated to a callback instead of a reference to a value,
/// 3. the mapper tracks which objet keys have been read throuhgout the object's
/// lifetime, and can detect whether all keys have been mapped.
///
/// Example:
/// \code
///   bool fromJSON(const Object &o, MyStruct &s, Path p) {
///     return OptionalObjectMapper(o, p)
///         .map("a_number", s.number)
///         .map("an_optional_number", s.optionalNumber)
///         .map("a_callback", [&](const Value &val, Path path) -> bool {
///             ...
///         })
///         .exhaustedObject();
///   }
/// \endcode
class OptionalObjectMapper {
  using MapFn =
      std::function<bool(const llvm::json::Value &, llvm::json::Path)>;

public:
  /// Creates the mapper from a JSON object and the current path. A set of
  /// potential object keys can be delcared "ignored" (they will be considered
  /// as "already mapped", even if they do not exist).
  OptionalObjectMapper(const llvm::json::Object &obj, llvm::json::Path path,
                       const llvm::DenseSet<StringRef> &ignoredKeys = {});

  /// If the key exists, invoke the callback with its value and updated path.
  /// Returns the receiver object to allow chaining. A missing key is not
  /// considered a mapping failure.
  OptionalObjectMapper &map(StringRef key, const MapFn &fn);

  /// If the key exists, attempts to deserialize it into a value of a given type
  /// by looking for a matching standard `fromJSON` function. Returns the
  /// receiver object to allow chaining. A missing key is not considered a
  /// mapping failure.
  template <typename T>
  OptionalObjectMapper &map(StringRef key, T &out) {
    if (!mapValid)
      return *this;
    if (const llvm::json::Value *val = obj.get(key)) {
      if (auto [_, newKey] = mappedKeys.insert(key); !newKey) {
        path.field(key).report(ERR_DUP_KEY);
        mapValid = false;
        return *this;
      }
      mapValid = fromJSON(*val, out, path.field(key));
    }
    return *this;
  }

  /// Terminates a sequence of mappings. Returns true if all mappings succeeded
  /// and if all keys in the object were mapped; returns false otherwise.
  bool exhausted();

  /// Terminates a sequence of mappings. Returns whether all mappings succeeded.
  bool valid() { return mapValid; }

private:
  /// The JSON object to map on.
  const llvm::json::Object &obj;
  /// The JSON object's path.
  llvm::json::Path path;

  /// The set of keys that have already been mapped.
  llvm::DenseSet<StringRef> mappedKeys;
  /// Whether all mappings so far were successful. If `map` is invoked and this
  /// is false then the method will not even look for the key in the object.
  bool mapValid = true;

  static constexpr llvm::StringLiteral ERR_DUP_KEY =
      "key mapped multiple times";
};

} // namespace json
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_JSON_JSON_H