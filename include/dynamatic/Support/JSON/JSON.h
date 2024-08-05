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
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_JSON_JSON_H