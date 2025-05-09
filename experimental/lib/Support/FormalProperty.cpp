//===- FormalProperty.cpp ---------------------------------------*- C++ -*-===//
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
#include "experimental/Support/FormalProperty.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/JSON/JSON.h"
#include <optional>
#include <string>

namespace dynamatic {

std::optional<FormalProperty::TYPE>
FormalProperty::typeFromStr(const std::string &s) {

  auto toLower = [](const std::string &s) {
    std::string tmp(s);
    for (auto &c : tmp)
      c = tolower(c);
    return tmp;
  };

  if (toLower(s) == "aob")
    return FormalProperty::TYPE::AOB;
  if (toLower(s) == "veq")
    return FormalProperty::TYPE::VEQ;

  return std::nullopt;
}

std::optional<FormalProperty::TAG>
FormalProperty::tagFromStr(const std::string &s) {

  auto toLower = [](const std::string &s) {
    std::string tmp(s);
    for (auto &c : tmp)
      c = tolower(c);
    return tmp;
  };

  if (toLower(s) == "opt")
    return FormalProperty::TAG::OPT;
  if (toLower(s) == "invar")
    return FormalProperty::TAG::INVAR;
  if (toLower(s) == "error")
    return FormalProperty::TAG::ERROR;

  return std::nullopt;
}

std::string FormalProperty::typeToStr(TYPE t) {
  switch (t) {
  case TYPE::AOB:
    return "AOB";
  case TYPE::VEQ:
    return "VEQ";
  }
}

std::string FormalProperty::tagToStr(TAG t) {
  switch (t) {
  case TAG::OPT:
    return "OPT";
  case TAG::INVAR:
    return "INVAR";
  case TAG::ERROR:
    return "ERRR";
  }
}

llvm::json::Object FormalProperty::toJsonObj() const {
  return llvm::json::Object{{"id", id},
                            {"info", info},
                            {"tag", tagToStr(tag)},
                            {"type", typeToStr(type)},
                            {"check", check}};
}

llvm::json::Object FormalProperty::AOBInfo(const OpResult &res) {
  Operation *ownerOp = res.getOwner();
  Operation *userOp = *res.getUsers().begin();

  handshake::PortNamer ownerNamer(ownerOp);
  handshake::PortNamer userNamer(userOp);

  unsigned long operandIndex = userOp->getNumOperands();
  for (auto [j, arg] : llvm::enumerate(userOp->getOperands())) {
    if (arg == res) {
      operandIndex = j;
      break;
    }
  }
  assert(operandIndex < userOp->getNumOperands());

  return llvm::json::Object{
      {"owner", getUniqueName(ownerOp).str()},
      {"user", getUniqueName(userOp).str()},
      {"owner_index", res.getResultNumber()},
      {"user_index", operandIndex},
      {"owner_channel", ownerNamer.getOutputName(res.getResultNumber()).str()},
      {"user_channel", userNamer.getInputName(operandIndex).str()}};
}

llvm::json::Object FormalProperty::VEQInfo(const OpResult &res1,
                                           const OpResult &res2) {
  Operation *op1 = res1.getOwner();
  unsigned int i = res1.getResultNumber();
  handshake::PortNamer namer1(op1);

  Operation *op2 = res2.getOwner();
  unsigned int j = res2.getResultNumber();
  handshake::PortNamer namer2(op2);

  return llvm::json::Object(
      {{"owner", getUniqueName(op1).str()},
       {"target", getUniqueName(op2).str()},
       {"owner_index", i},
       {"target_index", j},
       {"owner_channel", namer1.getOutputName(i).str()},
       {"target_channel", namer2.getOutputName(j).str()}});
}

} // namespace dynamatic