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
  return llvm::json::Object({{ID_LIT, id},
                             {TYPE_LIT, typeToStr(type)},
                             {TAG_LIT, tagToStr(tag)},
                             {CHECK_LIT, check},
                             {INFO_LIT, extraInfoToJSON()}});
}

std::unique_ptr<FormalProperty>
FormalProperty::fromJSON(const llvm::json::Value &value,
                         llvm::json::Path path) {
  std::string typeStr;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(TYPE_LIT, typeStr))
    return nullptr;

  auto typeOpt = typeFromStr(typeStr);
  if (!typeOpt)
    return nullptr;
  TYPE type = *typeOpt;

  switch (type) {
  case TYPE::AOB:
    return AbsenceOfBackpressure::fromJSON(value, path.field(INFO_LIT));
  case TYPE::VEQ:
    return ValidEquivalence::fromJSON(value, path.field(INFO_LIT));
  }
}

llvm::json::Value
FormalProperty::parseBaseAndExtractInfo(const llvm::json::Value &value,
                                        llvm::json::Path path) {
  std::string typeStr, tagStr;
  llvm::json::ObjectMapper mapper(value, path);

  if (!mapper || !mapper.map(ID_LIT, id) || !mapper.map(TYPE_LIT, typeStr) ||
      !mapper.map(TAG_LIT, tagStr) || !mapper.map(CHECK_LIT, check))
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
    auto it = obj->find(INFO_LIT);
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
  ownerChannel.channelIndex = res.getResultNumber();
  userChannel.channelIndex = operandIndex;
  ownerChannel.channelName =
      ownerNamer.getOutputName(res.getResultNumber()).str();
  userChannel.channelName = userNamer.getInputName(operandIndex).str();
}

llvm::json::Value AbsenceOfBackpressure::extraInfoToJSON() const {
  return llvm::json::Object({{OWNER_OP_LIT, ownerChannel.operationName},
                             {USER_OP_LIT, userChannel.operationName},
                             {OWNER_INDEX_LIT, ownerChannel.channelIndex},
                             {USER_INDEX_LIT, userChannel.channelIndex},
                             {OWNER_CHANNEL_LIT, ownerChannel.channelName},
                             {USER_CHANNEL_LIT, userChannel.channelName}});
}

std::unique_ptr<AbsenceOfBackpressure>
AbsenceOfBackpressure::fromJSON(const llvm::json::Value &value,
                                llvm::json::Path path) {
  auto prop = std::make_unique<AbsenceOfBackpressure>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  if (!mapper || !mapper.map(OWNER_OP_LIT, prop->ownerChannel.operationName) ||
      !mapper.map(USER_OP_LIT, prop->userChannel.operationName) ||
      !mapper.map(OWNER_INDEX_LIT, prop->ownerChannel.channelIndex) ||
      !mapper.map(USER_INDEX_LIT, prop->userChannel.channelIndex) ||
      !mapper.map(OWNER_CHANNEL_LIT, prop->ownerChannel.channelName) ||
      !mapper.map(USER_CHANNEL_LIT, prop->userChannel.channelName))
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
  ownerChannel.channelIndex = i;
  targetChannel.channelIndex = j;
  ownerChannel.channelName = namer1.getOutputName(i).str();
  targetChannel.channelName = namer2.getOutputName(j).str();
}

llvm::json::Value ValidEquivalence::extraInfoToJSON() const {
  return llvm::json::Object({{OWNER_OP_LIT, ownerChannel.operationName},
                             {TARGET_OP_LIT, targetChannel.operationName},
                             {OWNER_INDEX_LIT, ownerChannel.channelIndex},
                             {TARGET_INDEX_LIT, targetChannel.channelIndex},
                             {OWNER_CHANNEL_LIT, ownerChannel.channelName},
                             {TARGET_CHANNEL_LIT, targetChannel.channelName}});
}

std::unique_ptr<ValidEquivalence>
ValidEquivalence::fromJSON(const llvm::json::Value &value,
                           llvm::json::Path path) {
  auto prop = std::make_unique<ValidEquivalence>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  if (!mapper || !mapper.map(OWNER_OP_LIT, prop->ownerChannel.operationName) ||
      !mapper.map(TARGET_OP_LIT, prop->targetChannel.operationName) ||
      !mapper.map(OWNER_INDEX_LIT, prop->ownerChannel.channelIndex) ||
      !mapper.map(TARGET_INDEX_LIT, prop->targetChannel.channelIndex) ||
      !mapper.map(OWNER_CHANNEL_LIT, prop->ownerChannel.channelName) ||
      !mapper.map(TARGET_CHANNEL_LIT, prop->targetChannel.channelName))
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

  // Retrieve formal properties (see
  // https://github.com/EPFL-LAP/dynamatic/blob/main/docs/Specs/FormalProperties.md)
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