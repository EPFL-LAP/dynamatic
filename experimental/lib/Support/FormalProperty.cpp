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

  if (s == "AbsenceOfBackpressure")
    return FormalProperty::TYPE::AbsenceOfBackpressure;
  if (s == "ValidEquivalence")
    return FormalProperty::TYPE::ValidEquivalence;
  if (s == "EagerForkNotAllOutputSent")
    return FormalProperty::TYPE::EagerForkNotAllOutputSent;
  if (s == "CopiedSlotsOfActiveForksAreFull")
    return FormalProperty::TYPE::CopiedSlotsOfActiveForksAreFull;
  if (s == "ReconvergentPathFlow")
    return FormalProperty::TYPE::ReconvergentPathFlow;
  if (s == "IOGSingleToken")
    return FormalProperty::TYPE::IOGSingleToken;
  if (s == "IOGConsecutiveTokens")
    return FormalProperty::TYPE::IOGConsecutiveTokens;

  return std::nullopt;
}

std::string FormalProperty::typeToStr(TYPE t) {
  switch (t) {
  case TYPE::AbsenceOfBackpressure:
    return "AbsenceOfBackpressure";
  case TYPE::ValidEquivalence:
    return "ValidEquivalence";
  case TYPE::EagerForkNotAllOutputSent:
    return "EagerForkNotAllOutputSent";
  case TYPE::CopiedSlotsOfActiveForksAreFull:
    return "CopiedSlotsOfActiveForksAreFull";
  case TYPE::ReconvergentPathFlow:
    return "ReconvergentPathFlow";
  case TYPE::IOGSingleToken:
    return "IOGSingleToken";
  case TYPE::IOGConsecutiveTokens:
    return "IOGConsecutiveTokens";
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
  case TYPE::AbsenceOfBackpressure:
    return AbsenceOfBackpressure::fromJSON(value, path.field(INFO_LIT));
  case TYPE::ValidEquivalence:
    return ValidEquivalence::fromJSON(value, path.field(INFO_LIT));
  case TYPE::EagerForkNotAllOutputSent:
    return EagerForkNotAllOutputSent::fromJSON(value, path.field(INFO_LIT));
  case TYPE::CopiedSlotsOfActiveForksAreFull:
    return CopiedSlotsOfActiveForkAreFull::fromJSON(value,
                                                    path.field(INFO_LIT));
  case TYPE::ReconvergentPathFlow:
    return ReconvergentPathFlow::fromJSON(value, path.field(INFO_LIT));
  case TYPE::IOGSingleToken:
    return IOGSingleToken::fromJSON(value, path.field(INFO_LIT));
  case TYPE::IOGConsecutiveTokens:
    return IOGConsecutiveTokens::fromJSON(value, path.field(INFO_LIT));
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

AbsenceOfBackpressure::AbsenceOfBackpressure(uint64_t id, TAG tag,
                                             const OpResult &res)
    : FormalProperty(id, tag, TYPE::AbsenceOfBackpressure) {
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

ValidEquivalence::ValidEquivalence(uint64_t id, TAG tag, const OpResult &res1,
                                   const OpResult &res2)
    : FormalProperty(id, tag, TYPE::ValidEquivalence) {
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

EagerForkNotAllOutputSent::EagerForkNotAllOutputSent(
    uint64_t id, TAG tag, handshake::EagerForkLikeOpInterface &forkOp)
    : FormalProperty(id, tag, TYPE::EagerForkNotAllOutputSent) {
  sentStateNamers = forkOp.getInternalSentStateNamers();
}

llvm::json::Value EagerForkNotAllOutputSent::extraInfoToJSON() const {
  std::vector<llvm::json::Value> channels{};
  // std::string opName = sentStateNamers[0].opName;
  for (auto [i, state] : llvm::enumerate(sentStateNamers)) {
    channels.push_back(state.toInnerJSON());
  }
  // Example JSON:
  // [
  //   {
  //     "channel_name": "outs_0",
  //     "channel_size": 0,
  //     "operation": "fork0",
  //   },
  //   {
  //     "channel_name": "outs_1",
  //     "channel_size": 0,
  //     "operation": "fork0",
  //   },
  //   {
  //     "channel_name": "outs_2",
  //     "channel_size": 0,
  //     "operation": "fork0",
  //   },
  //   {
  //     "channel_name": "outs_3",
  //     "channel_size": 0,
  //     "operation": "fork0",
  //   }
  // ]
  return llvm::json::Array(channels);
}

std::unique_ptr<EagerForkNotAllOutputSent>
EagerForkNotAllOutputSent::fromJSON(const llvm::json::Value &value,
                                    llvm::json::Path path) {
  auto prop = std::make_unique<EagerForkNotAllOutputSent>();

  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::Array *array = info.getAsArray();
  assert(array &&
         "expected info of EFNAO to be an array of eager fork outputs");
  for (auto &stateJSON : *array) {
    auto sentStateNamer =
        handshake::EagerForkSentNamer::fromInnerJSON(stateJSON, path);
    prop->sentStateNamers.push_back(*sentStateNamer);
  }
  return prop;
}

// Invariant 2 -- see https://ieeexplore.ieee.org/document/10323796

CopiedSlotsOfActiveForkAreFull::CopiedSlotsOfActiveForkAreFull(
    uint64_t id, TAG tag, handshake::BufferLikeOpInterface &bufferOpI,
    handshake::EagerForkLikeOpInterface &forkOpI)
    : FormalProperty(id, tag, TYPE::CopiedSlotsOfActiveForksAreFull) {
  sentStateNamers = forkOpI.getInternalSentStateNamers();
  auto slots = bufferOpI.getInternalSlotStateNamers();
  // last slot is the copied slot!
  copiedSlot = std::make_unique<BufferSlotFullNamer>(slots[slots.size() - 1]);
}

CopiedSlotsOfActiveForkAreFull::CopiedSlotsOfActiveForkAreFull(
    uint64_t id, TAG tag, handshake::LatencyInterface &latencyOpI,
    handshake::EagerForkLikeOpInterface &forkOpI)
    : FormalProperty(id, tag, TYPE::CopiedSlotsOfActiveForksAreFull) {
  sentStateNamers = forkOpI.getInternalSentStateNamers();
  auto slots = latencyOpI.getPipelineSlots();
  // last slot is the copied slot!
  copiedSlot = std::make_unique<PipelineSlotNamer>(slots[slots.size() - 1]);
}

llvm::json::Value CopiedSlotsOfActiveForkAreFull::extraInfoToJSON() const {
  std::vector<llvm::json::Value> channels{};
  for (auto [i, state] : llvm::enumerate(sentStateNamers)) {
    channels.push_back(state.toInnerJSON());
  }
  return llvm::json::Object(
      {{FORK_CHANNELS_LIT, channels}, {COPIED_SLOT_LIT, copiedSlot->toJSON()}});
}

std::unique_ptr<CopiedSlotsOfActiveForkAreFull>
CopiedSlotsOfActiveForkAreFull::fromJSON(const llvm::json::Value &value,
                                         llvm::json::Path path) {
  auto prop = std::make_unique<CopiedSlotsOfActiveForkAreFull>();

  auto info = prop->parseBaseAndExtractInfo(value, path);

  const llvm::json::Object *obj = info.getAsObject();
  assert(obj && "CSOAFAF json info not an object");

  const llvm::json::Value *channelNameJSON = obj->get(FORK_CHANNELS_LIT);
  assert(channelNameJSON && "missing FORK_CHANNELS_LIT in CSOAFAF info");
  const llvm::json::Array *channelNameJSONs = channelNameJSON->getAsArray();
  assert(channelNameJSONs && "FORK_CHANNELS_LIT in CSOAFAF is not an array");
  for (auto &sentJSON : *channelNameJSONs) {
    prop->sentStateNamers.push_back(
        *handshake::EagerForkSentNamer::fromInnerJSON(sentJSON, path));
  }

  const llvm::json::Value *bufferSlotJSON = obj->get(COPIED_SLOT_LIT);
  assert(bufferSlotJSON && "missing COPIED_SLOT_LIT in CSOAFAF json");
  prop->copiedSlot = InternalStateNamer::fromJSON(*bufferSlotJSON, path);

  return prop;
}

// Reconvergent path flow

ReconvergentPathFlow::ReconvergentPathFlow(unsigned long id, TAG tag)
    : FormalProperty(id, tag, TYPE::ReconvergentPathFlow) {}

llvm::json::Value ReconvergentPathFlow::extraInfoToJSON() const {
  std::vector<llvm::json::Value> jsonEqs{};
  jsonEqs.reserve(equations.size());

  for (auto &eq : equations) {
    jsonEqs.push_back(eq.toJSON());
  }
  return llvm::json::Array(jsonEqs);
}

std::unique_ptr<ReconvergentPathFlow>
ReconvergentPathFlow::fromJSON(const llvm::json::Value &value,
                               llvm::json::Path path) {
  auto prop = std::make_unique<ReconvergentPathFlow>();

  llvm::json::Value info = prop->parseBaseAndExtractInfo(value, path);
  const llvm::json::Array *arr = info.getAsArray();
  if (!arr)
    return nullptr;

  for (const llvm::json::Value &eq : *arr) {
    prop->equations.push_back(FlowExpression::fromJSON(eq, path));
  }

  return prop;
}

// IOGSingleToken

llvm::json::Value IOGSingleToken::extraInfoToJSON() const {
  std::vector<llvm::json::Value> slotsJSON;
  slotsJSON.reserve(slots.size());
  for (auto &namer : slots) {
    slotsJSON.push_back(namer->toJSON());
  }
  llvm::json::Value slotsValue = slotsJSON;

  std::vector<llvm::json::Value> forksJSON;
  forksJSON.reserve(forks.size());
  for (auto &sent : forks) {
    forksJSON.push_back(sent.toInnerJSON());
  }
  llvm::json::Value forksValue = forksJSON;

  return llvm::json::Object({{SLOTS_LIT, slotsValue}, {FORKS_LIT, forksValue}});
}

std::unique_ptr<IOGSingleToken>
IOGSingleToken::fromJSON(const llvm::json::Value &value,
                         llvm::json::Path path) {
  auto prop = std::make_unique<IOGSingleToken>();
  llvm::json::Value info = prop->parseBaseAndExtractInfo(value, path);

  llvm::json::Object *obj = info.getAsObject();
  assert(obj);
  if (auto iter = obj->find(SLOTS_LIT); iter != obj->end()) {
    llvm::json::Array *slotsArray = iter->second.getAsArray();
    assert(slotsArray);
    prop->slots.reserve(slotsArray->size());
    for (const llvm::json::Value &sentValue : *slotsArray) {
      auto json = InternalStateNamer::fromJSON(sentValue, path);
      assert(json);
      prop->slots.push_back(std::move(json));
    }
  } else {
    path.report(json::ERR_MISSING_VALUE);
    return nullptr;
  }
  if (auto iter = obj->find(FORKS_LIT); iter != obj->end()) {
    llvm::json::Array *forksArray = iter->second.getAsArray();
    assert(forksArray);
    prop->forks.reserve(forksArray->size());
    for (const llvm::json::Value &sentValue : *forksArray) {
      auto innerJSON = EagerForkSentNamer::fromInnerJSON(sentValue, path);
      assert(innerJSON);
      prop->forks.push_back(*innerJSON);
    }
  } else {
    path.report(json::ERR_MISSING_VALUE);
    return nullptr;
  }
  return prop;
}

// IOGConsecutiveTokens

llvm::json::Value IOGConsecutiveTokens::extraInfoToJSON() const {
  std::vector<llvm::json::Value> sentsJSON;
  sentsJSON.reserve(sents.size());
  for (auto &sent : sents) {
    sentsJSON.push_back(sent.toInnerJSON());
  }
  llvm::json::Value sentsValue = sentsJSON;

  return llvm::json::Object({{SLOT1_LIT, slot1->toJSON()},
                             {SLOT2_LIT, slot2->toJSON()},
                             {SENTS_LIT, sentsValue}});
}

std::unique_ptr<IOGConsecutiveTokens>
IOGConsecutiveTokens::fromJSON(const llvm::json::Value &value,
                               llvm::json::Path path) {
  auto prop = std::make_unique<IOGConsecutiveTokens>();
  llvm::json::Value info = prop->parseBaseAndExtractInfo(value, path);

  llvm::json::Object *obj = info.getAsObject();
  assert(obj);
  if (auto iter = obj->find(SLOT1_LIT); iter != obj->end()) {
    prop->slot1 = InternalStateNamer::fromJSON(iter->second, path);
  } else {
    path.report(json::ERR_MISSING_VALUE);
    return nullptr;
  }

  if (auto iter = obj->find(SLOT2_LIT); iter != obj->end()) {
    prop->slot2 = InternalStateNamer::fromJSON(iter->second, path);
  } else {
    path.report(json::ERR_MISSING_VALUE);
    return nullptr;
  }

  if (auto iter = obj->find(SENTS_LIT); iter != obj->end()) {
    llvm::json::Array *sentsArray = iter->second.getAsArray();
    assert(sentsArray);
    prop->sents.reserve(sentsArray->size());
    for (const llvm::json::Value &sentValue : *sentsArray) {
      auto innerJSON = EagerForkSentNamer::fromInnerJSON(sentValue, path);
      assert(innerJSON);
      prop->sents.push_back(*innerJSON);
    }
  } else {
    path.report(json::ERR_MISSING_VALUE);
  }

  /*
  auto prop = std::make_unique<IOGConsecutiveTokens>();
  auto info = prop->parseBaseAndExtractInfo(value, path);

  const llvm::json::Object *obj = info.getAsObject();
  if (!obj)
    return nullptr;
  assert(false && "TODO");
  */
  return prop;
}

IOGConsecutiveTokens::IOGConsecutiveTokens(
    unsigned long id, TAG tag, std::shared_ptr<InternalStateNamer> slot1,
    std::shared_ptr<InternalStateNamer> slot2,
    std::vector<EagerForkSentNamer> sents)
    : FormalProperty(id, tag, TYPE::IOGConsecutiveTokens),
      slot1(std::move(slot1)), slot2(std::move(slot2)),
      sents(std::move(sents)) {}

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
