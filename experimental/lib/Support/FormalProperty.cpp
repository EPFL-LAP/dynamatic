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
#include <memory>
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

// Factory implementation
std::unique_ptr<FormalProperty>
FormalProperty::fromJSON(const llvm::json::Value &value,
                         llvm::json::Path path) {
  std::string propType;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.mapOptional("type", propType))
    return nullptr;

  if (propType == "AOB") {
    return AOBProperty::fromJson(value, path);
  } else if (propType == "VEQ") {
    return VEQProperty::fromJson(value, path);
  }
  return nullptr;
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

std::unique_ptr<AOBProperty>
AOBProperty::fromJSON(const llvm::json::Value &value, llvm::json::Path path) {
  auto prop = std::make_unique<AOBProperty>();

  llvm::json::ObjectMapper mapper(value, path);
  std::string typeStr, tagStr;
  if (!mapper || !mapper.mapOptional("id", prop->id) ||
      !mapper.mapOptional("type", typeStr) ||
      !mapper.mapOptional("tag", tagStr) ||
      !mapper.mapOptional("check", prop->check) ||
      !mapper.mapOptional("owner", prop->owner) ||
      !mapper.mapOptional("user", prop->user) ||
      !mapper.mapOptional("owner_index", prop->ownerIndex) ||
      !mapper.mapOptional("user_index", prop->userIndex) ||
      !mapper.mapOptional("owner_channel", prop->ownerChannel) ||
      !mapper.mapOptional("user_channel", prop->userChannel))
    return nullptr;

  if (typeStr != "AOB")
    return nullptr;

  auto tagOpt = tagFromStr(tagStr);
  if (!tagOpt)
    return nullptr;

  prop->tag = *tagOpt;

  return prop;
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

std::unique_ptr<VEQProperty>
VEQProperty::fromJSON(const llvm::json::Value &value, llvm::json::Path path) {
  auto prop = std::make_unique<VEQProperty>();
  llvm::json::ObjectMapper mapper(value, path);
  std::string typeStr, tagStr;
  if (!mapper || !mapper.mapOptional("id", prop->id) ||
      !mapper.mapOptional("type", typeStr) ||
      !mapper.mapOptional("tag", tagStr) ||
      !mapper.mapOptional("check", prop->check) ||
      !mapper.mapOptional("owner", prop->owner) ||
      !mapper.mapOptional("target", prop->target) ||
      !mapper.mapOptional("owner_index", prop->ownerIndex) ||
      !mapper.mapOptional("target_index", prop->targetIndex) ||
      !mapper.mapOptional("owner_channel", prop->ownerChannel) ||
      !mapper.mapOptional("target_channel", prop->targetChannel))
    return nullptr;

  if (typeStr != "VEQ")
    return nullptr;

  auto tagOpt = tagFromStr(tagStr);
  if (!tagOpt)
    return nullptr;

  prop->tag = *tagOpt;

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