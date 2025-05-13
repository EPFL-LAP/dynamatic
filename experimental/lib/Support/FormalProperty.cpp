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

// Absence of Backpressure

AOBProperty::AOBProperty(unsigned long id, TAG tag, const OpResult &res)
    : FormalProperty(id, tag) {
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

  owner = getUniqueName(ownerOp).str();
  user = getUniqueName(userOp).str();
  ownerIndex = res.getResultNumber();
  userIndex = operandIndex;
  ownerChannel = ownerNamer.getOutputName(res.getResultNumber()).str();
  userChannel = userNamer.getInputName(operandIndex).str();
}

llvm::json::Object AOBProperty::toJsonObj() const {
  return llvm::json::Object({{"id", id},
                             {"type", "AOB"},
                             {"tag", tagToStr(tag)},
                             {"check", check},
                             {"owner", owner},
                             {"user", user},
                             {"owner_index", ownerIndex},
                             {"user_index", userIndex},
                             {"owner_channel", ownerChannel},
                             {"user_channel", userChannel}});
}

// Valid Equivalence

VEQProperty::VEQProperty(unsigned long id, TAG tag, const OpResult &res1,
                         const OpResult &res2)
    : FormalProperty(id, tag) {
  Operation *op1 = res1.getOwner();
  unsigned int i = res1.getResultNumber();
  handshake::PortNamer namer1(op1);

  Operation *op2 = res2.getOwner();
  unsigned int j = res2.getResultNumber();
  handshake::PortNamer namer2(op2);

  owner = getUniqueName(op1).str();
  target = getUniqueName(op2).str();
  ownerIndex = i;
  targetIndex = j;
  ownerChannel = namer1.getOutputName(i).str();
  targetChannel = namer2.getOutputName(j).str();
}

llvm::json::Object VEQProperty::toJsonObj() const {
  return llvm::json::Object({{"id", id},
                             {"type", "VEQ"},
                             {"tag", tagToStr(tag)},
                             {"check", check},
                             {"owner", owner},
                             {"target", target},
                             {"owner_index", ownerIndex},
                             {"target_index", targetIndex},
                             {"owner_channel", ownerChannel},
                             {"target_channel", targetChannel}});
}
} // namespace dynamatic