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

llvm::json::Value FormalProperty::toJSON() const {
  return llvm::json::Object({{"id", id},
                             {"type", typeToStr(type)},
                             {"tag", tagToStr(tag)},
                             {"check", check},
                             {"info", extraInfoToJSON()}});
}

// Factory implementation
std::unique_ptr<FormalProperty>
FormalProperty::fromJSON(const llvm::json::Value &value,
                         llvm::json::Path path) {
  std::string typeStr;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.mapOptional("type", typeStr))
    return nullptr;

  auto typeOpt = typeFromStr(typeStr);
  if (!typeOpt)
    return nullptr;
  TYPE type = *typeOpt;

  switch (type) {
  case TYPE::AOB:
    return AbsenceOfBackpressure::fromJSON(value, path);
  case TYPE::VEQ:
    return ValidEquivalence::fromJSON(value, path);
  }
}

llvm::json::Value
FormalProperty::parseBaseAndExtractInfo(const llvm::json::Value &value,
                                        llvm::json::Path path) {
  std::string typeStr, tagStr;
  llvm::json::ObjectMapper mapper(value, path);

  if (!mapper || !mapper.mapOptional("id", id) ||
      !mapper.mapOptional("type", typeStr) ||
      !mapper.mapOptional("tag", tagStr) || !mapper.mapOptional("check", check))
    return nullptr;

  auto typeOpt = typeFromStr(typeStr);
  if (!typeOpt)
    return nullptr;
  type = *typeOpt;

  auto tagOpt = tagFromStr(tagStr);
  if (!tagOpt)
    return nullptr;
  tag = *tagOpt;

  if (const auto *obj = value.getAsObject()) {
    auto it = obj->find("info");
    if (it != obj->end())
      return it->second;
  }
  return nullptr;
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
  ownerChannel.index = res.getResultNumber();
  userChannel.index = operandIndex;
  ownerChannel.name = ownerNamer.getOutputName(res.getResultNumber()).str();
  userChannel.name = userNamer.getInputName(operandIndex).str();
}

llvm::json::Value AbsenceOfBackpressure::extraInfoToJSON() const {
  return llvm::json::Object({{"owner", ownerChannel.operationName},
                             {"user", userChannel.operationName},
                             {"owner_index", ownerChannel.index},
                             {"user_index", userChannel.index},
                             {"owner_channel", ownerChannel.name},
                             {"user_channel", userChannel.name}});
}

std::unique_ptr<AbsenceOfBackpressure>
AbsenceOfBackpressure::fromJSON(const llvm::json::Value &value,
                                llvm::json::Path path) {
  auto prop = std::make_unique<AbsenceOfBackpressure>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  if (!mapper ||
      !mapper.mapOptional("owner", prop->ownerChannel.operationName) ||
      !mapper.mapOptional("user", prop->ownerChannel.operationName) ||
      !mapper.mapOptional("owner_index", prop->ownerChannel.index) ||
      !mapper.mapOptional("user_index", prop->ownerChannel.name) ||
      !mapper.mapOptional("owner_channel", prop->ownerChannel.name) ||
      !mapper.mapOptional("user_channel", prop->userChannel.name))
    return nullptr;

  return prop;
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
  ownerChannel.index = i;
  targetChannel.index = j;
  ownerChannel.name = namer1.getOutputName(i).str();
  targetChannel.name = namer2.getOutputName(j).str();
}

llvm::json::Value ValidEquivalence::extraInfoToJSON() const {
  return llvm::json::Object({{"owner", ownerChannel.operationName},
                             {"target", targetChannel.operationName},
                             {"owner_index", ownerChannel.index},
                             {"target_index", targetChannel.index},
                             {"owner_channel", ownerChannel.name},
                             {"target_channel", targetChannel.name}});
}

std::unique_ptr<ValidEquivalence>
ValidEquivalence::fromJSON(const llvm::json::Value &value,
                           llvm::json::Path path) {
  auto prop = std::make_unique<ValidEquivalence>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  if (!mapper ||
      !mapper.mapOptional("owner", prop->ownerChannel.operationName) ||
      !mapper.mapOptional("target", prop->targetChannel.operationName) ||
      !mapper.mapOptional("owner_index", prop->ownerChannel.index) ||
      !mapper.mapOptional("target_index", prop->targetChannel.index) ||
      !mapper.mapOptional("owner_channel", prop->ownerChannel.name) ||
      !mapper.mapOptional("target_channel", prop->targetChannel.name))
    return nullptr;

  return prop;
}

LogicalResult FormalPropertyTable::addPropertiesFromJSON(StringRef filepath) {
  // Open the properties' database
  std::ifstream inputFile(filepath.str());
  if (!inputFile.is_open()) {
    llvm::errs() << "[WARNING] Failed to open property database file @ \""
                 << filepath << "\"\n";
    return failure();
  }

  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse property table @ \"" << filepath
                 << "\" as JSON.\n-> " << toString(value.takeError()) << "\n";
    return failure();
  }

  llvm::json::Path::Root jsonRoot(filepath);
  llvm::json::Path jsonPath(jsonRoot);

  llvm::json::Array *jsonComponents = value->getAsArray();
  if (!jsonComponents) {
    jsonPath.report(json::ERR_EXPECTED_ARRAY);
    jsonRoot.printErrorContext(*value, llvm::errs());
    return failure();
  }

  for (auto [idx, jsonComponent] : llvm::enumerate(*jsonComponents)) {
    std::unique_ptr<FormalProperty> &property = properties.emplace_back();
    if (!fromJSON(jsonComponent, property, jsonPath.index(idx))) {
      jsonRoot.printErrorContext(*value, llvm::errs());
      return failure();
    }
  }

  return success();
}
} // namespace dynamatic