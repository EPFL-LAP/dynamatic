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
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <memory>

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE { AOB, VEQ };

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  unsigned long getId() const { return id; }

  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string tagToStr(TAG t);

  inline virtual llvm::json::Object toJsonObj() const {
    return llvm::json::Object();
  };

  std::unique_ptr<FormalProperty> static fromJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  FormalProperty() = default;
  FormalProperty(unsigned long id, TAG tag, TYPE type)
      : id(id), tag(tag), type(type), check("unchecked") {}
  virtual ~FormalProperty() = default;

  static bool classof(const FormalProperty *fp) { return true; }

protected:
  unsigned long id;
  TAG tag;
  TYPE type;
  std::string check;
};

class AOBProperty : public FormalProperty {
public:
  std::string getOwner() { return owner; }
  std::string getUser() { return user; }
  int getOwnerIndex() { return ownerIndex; }
  int getUserIndex() { return userIndex; }
  std::string getOwnerChannel() { return ownerChannel; }
  std::string getUserChannel() { return userChannel; }

  llvm::json::Object toJsonObj() const override;
  /// Attempts to deserialize a propertyfrom a JSON value.
  static std::unique_ptr<AOBProperty> fromJSON(const llvm::json::Value &value,
                                               llvm::json::Path path);

  AOBProperty() = default;
  AOBProperty(unsigned long id, TAG tag, const OpResult &res);
  ~AOBProperty() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::AOB;
  }

private:
  std::string owner;
  std::string user;
  int ownerIndex;
  int userIndex;
  std::string ownerChannel;
  std::string userChannel;
};

class VEQProperty : public FormalProperty {
public:
  std::string getOwner() { return owner; }
  std::string getTarget() { return target; }
  int getOwnerIndex() { return ownerIndex; }
  int getTargetIndex() { return targetIndex; }
  std::string getOwnerChannel() { return ownerChannel; }
  std::string getTargetChannel() { return targetChannel; }

  llvm::json::Object toJsonObj() const override;
  /// Attempts to deserialize a propertyfrom a JSON value.
  static std::unique_ptr<VEQProperty> fromJSON(const llvm::json::Value &value,
                                               llvm::json::Path path);

  VEQProperty() = default;
  VEQProperty(unsigned long id, TAG tag, const OpResult &res1,
              const OpResult &res2);
  ~VEQProperty() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::VEQ;
  }

private:
  std::string owner;
  std::string target;
  int ownerIndex;
  int targetIndex;
  std::string ownerChannel;
  std::string targetChannel;
};

class FormalPropertyTable {
public:
  FormalPropertyTable() = default;

  LogicalResult addPropertiesFromJSON(StringRef filepath);

  const std::vector<std::unique_ptr<FormalProperty>> &getProperties() const {
    return properties;
  }

  inline bool fromJSON(const llvm::json::Value &value,
                       std::unique_ptr<FormalProperty> &property,
                       llvm::json::Path path) {
    // fromJson internally allocates the correct space for the class with
    // make_unique and returns a pointer
    property = FormalProperty::fromJSON(value, path);

    return true;
  }

private:
  /// List of properties.
  std::vector<std::unique_ptr<FormalProperty>> properties;
};

} // namespace dynamatic