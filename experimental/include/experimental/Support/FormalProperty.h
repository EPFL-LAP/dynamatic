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

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE { AOB, VEQ };

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  unsigned long getId() const { return id; }
  llvm::json::Value getInfo() const { return info; }

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);
  static std::string tagToStr(TAG t);

  llvm::json::Object toJsonObj() const;

  FormalProperty() : info(nullptr) {}
  FormalProperty(unsigned long id, const std::string &type,
                 const std::string &tag, llvm::json::Value info)
      : id(id), type(*typeFromStr(type)), tag(*tagFromStr(tag)), info(info) {}
  FormalProperty(unsigned long id, TYPE type, TAG tag, llvm::json::Value info)
      : id(id), type(type), tag(tag), info(info) {}

  static llvm::json::Object AOBInfo(const OpResult &res);
  static llvm::json::Object VEQInfo(const OpResult &res1, const OpResult &res2);

private:
  unsigned long id;
  TYPE type;
  TAG tag;
  std::string check;
  llvm::json::Value info;
};

} // namespace dynamatic