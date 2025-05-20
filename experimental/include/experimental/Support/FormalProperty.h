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

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);
  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string tagToStr(TAG t);

  llvm::json::Value toJSON() const;

  inline virtual llvm::json::Value extraInfoToJSON() const { return nullptr; };

  FormalProperty() = default;
  FormalProperty(unsigned long id, TAG tag, TYPE type)
      : id(id), tag(tag), type(type), check("unchecked") {}
  virtual ~FormalProperty() = default;

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

  llvm::json::Value extraInfoToJSON() const override;

  AOBProperty() = default;
  AOBProperty(unsigned long id, TAG tag, const OpResult &res);
  ~AOBProperty() = default;

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

  llvm::json::Value extraInfoToJSON() const override;

  VEQProperty() = default;
  VEQProperty(unsigned long id, TAG tag, const OpResult &res1,
              const OpResult &res2);
  ~VEQProperty() = default;

private:
  std::string owner;
  std::string target;
  int ownerIndex;
  int targetIndex;
  std::string ownerChannel;
  std::string targetChannel;
};

} // namespace dynamatic