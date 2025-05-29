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
#include "llvm/Support/JSON.h"
#include <memory>
#include <optional>
#include <string>

namespace dynamatic {

std::optional<FormalProperty::TYPE>
FormalProperty::typeFromStr(const std::string &s) {

  if (s == "AOB")
    return FormalProperty::TYPE::AOB;
  if (s == "VEQ")
    return FormalProperty::TYPE::VEQ;

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

std::optional<FormalProperty::TAG>
FormalProperty::tagFromStr(const std::string &s) {

  if (s == "OPT")
    return FormalProperty::TAG::OPT;
  if (s == "INVAR")
    return FormalProperty::TAG::INVAR;
  if (s == "ERROR")
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
    return "ERROR";
  }
}

llvm::json::Value FormalProperty::toJSON() const {
  return llvm::json::Object({{"id", id},
                             {"type", typeToStr(type)},
                             {"tag", tagToStr(tag)},
                             {"check", check},
                             {"info", extraInfoToJSON()}});
}

// Absence of Backpressure

AbsenceOfBackpressure::AbsenceOfBackpressure(unsigned long id, TAG tag,
                                             const OpResult &res)
    : FormalProperty(id, tag, TYPE::AOB) {
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

  ownerChannel.operationName = getUniqueName(ownerOp).str();
  userChannel.operationName = getUniqueName(userOp).str();
  ownerChannel.channelIndex = res.getResultNumber();
  userChannel.channelIndex = operandIndex;
  ownerChannel.channelName =
      ownerNamer.getOutputName(res.getResultNumber()).str();
  userChannel.channelName = userNamer.getInputName(operandIndex).str();
}

llvm::json::Value AbsenceOfBackpressure::extraInfoToJSON() const {
  return llvm::json::Object({{"owner", ownerChannel.operationName},
                             {"user", userChannel.operationName},
                             {"owner_index", ownerChannel.channelIndex},
                             {"user_index", userChannel.channelIndex},
                             {"owner_channel", ownerChannel.channelName},
                             {"user_channel", userChannel.channelName}});
}

// Valid Equivalence

ValidEquivalence::ValidEquivalence(unsigned long id, TAG tag,
                                   const OpResult &res1, const OpResult &res2)
    : FormalProperty(id, tag, TYPE::VEQ) {
  Operation *op1 = res1.getOwner();
  unsigned int i = res1.getResultNumber();
  handshake::PortNamer namer1(op1);

  Operation *op2 = res2.getOwner();
  unsigned int j = res2.getResultNumber();
  handshake::PortNamer namer2(op2);

  ownerChannel.operationName = getUniqueName(op1).str();
  targetChannel.operationName = getUniqueName(op2).str();
  ownerChannel.channelIndex = i;
  targetChannel.channelIndex = j;
  ownerChannel.channelName = namer1.getOutputName(i).str();
  targetChannel.channelName = namer2.getOutputName(j).str();
}

llvm::json::Value ValidEquivalence::extraInfoToJSON() const {
  return llvm::json::Object({{"owner", ownerChannel.operationName},
                             {"target", targetChannel.operationName},
                             {"owner_index", ownerChannel.channelIndex},
                             {"target_index", targetChannel.channelIndex},
                             {"owner_channel", ownerChannel.channelName},
                             {"target_channel", targetChannel.channelName}});
}

} // namespace dynamatic