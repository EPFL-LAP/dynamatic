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
  std::string getCheck() const { return check; }

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);
  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string tagToStr(TAG t);

  llvm::json::Value toJSON() const;

  inline virtual llvm::json::Value extraInfoToJSON() const { return nullptr; };

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

  llvm::json::Value parseBaseAndExtractInfo(const llvm::json::Value &value,
                                            llvm::json::Path path);
};

struct SignalName {
  std::string operationName;
  std::string name;
  unsigned index;
};

class AbsenceOfBackpressure : public FormalProperty {
public:
  std::string getOwner() { return ownerChannel.operationName; }
  std::string getUser() { return userChannel.operationName; }
  int getOwnerIndex() { return ownerChannel.index; }
  int getUserIndex() { return userChannel.index; }
  std::string getOwnerChannel() { return ownerChannel.name; }
  std::string getUserChannel() { return userChannel.name; }

  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<AbsenceOfBackpressure>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  AbsenceOfBackpressure() = default;
  AbsenceOfBackpressure(unsigned long id, TAG tag, const OpResult &res);
  ~AbsenceOfBackpressure() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::VEQ;
  }

private:
  SignalName ownerChannel;
  SignalName userChannel;
};

class ValidEquivalence : public FormalProperty {
public:
  std::string getOwner() { return ownerChannel.operationName; }
  std::string getTarget() { return targetChannel.operationName; }
  int getOwnerIndex() { return ownerChannel.index; }
  int getTargetIndex() { return targetChannel.index; }
  std::string getOwnerChannel() { return ownerChannel.name; }
  std::string getTargetChannel() { return targetChannel.name; }

  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<ValidEquivalence>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  ValidEquivalence() = default;
  ValidEquivalence(unsigned long id, TAG tag, const OpResult &res1,
                   const OpResult &res2);
  ~ValidEquivalence() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::VEQ;
  }

private:
  SignalName ownerChannel;
  SignalName targetChannel;
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