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

struct Channel {
  std::string operationName;
  std::string name;
  unsigned index;
};

class AbsenceOfBackpressure : public FormalProperty {
public:
  std::string getOwner() { return owner.operationName; }
  std::string getUser() { return user.operationName; }
  int getOwnerIndex() { return owner.index; }
  int getUserIndex() { return owner.index; }
  std::string getOwnerChannel() { return owner.name; }
  std::string getUserChannel() { return user.name; }

  llvm::json::Value extraInfoToJSON() const override;

  AbsenceOfBackpressure() = default;
  AbsenceOfBackpressure(unsigned long id, TAG tag, const OpResult &res);
  ~AbsenceOfBackpressure() = default;

private:
  Channel owner;
  Channel user;
};

class ValidEquivalence : public FormalProperty {
public:
  std::string getOwner() { return owner.operationName; }
  std::string getTarget() { return target.operationName; }
  int getOwnerIndex() { return owner.index; }
  int getTargetIndex() { return target.index; }
  std::string getOwnerChannel() { return owner.name; }
  std::string getTargetChannel() { return target.name; }

  llvm::json::Value extraInfoToJSON() const override;

  ValidEquivalence() = default;
  ValidEquivalence(unsigned long id, TAG tag, const OpResult &res1,
                   const OpResult &res2);
  ~ValidEquivalence() = default;

private:
  Channel owner;
  Channel target;
};

} // namespace dynamatic