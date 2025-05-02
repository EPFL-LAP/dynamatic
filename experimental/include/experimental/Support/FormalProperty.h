//===- FormalProperty.h -----------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JSON-parsing logic for the formal properties' database.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/JSON.h"
#include <fstream>

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE { AOB, VEQ };

  /// Attempts to deserialize a propertyfrom a JSON value. Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  unsigned long getId() const { return id; }
  llvm::json::Value getInfo() const { return info; }

  std::optional<TYPE> typeFromStr(const std::string &s);
  std::optional<TAG> tagFromStr(const std::string &s);

  FormalProperty() : info(nullptr) {}

private:
  unsigned long id;
  TYPE type;
  TAG tag;
  std::string check;
  llvm::json::Value info;
};

class FormalPropertyTable {
public:
  FormalPropertyTable() = default;

  LogicalResult addPropertiesFromJSON(StringRef filepath);

  const std::vector<FormalProperty> &getProperties() const {
    return properties;
  }

  inline bool fromJSON(const llvm::json::Value &value, FormalProperty &property,
                       llvm::json::Path path) {
    return property.fromJSON(value, path);
  }

private:
  /// List of properties.
  std::vector<FormalProperty> properties;
};

} // namespace dynamatic