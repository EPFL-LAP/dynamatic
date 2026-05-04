#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "llvm/ADT/StringExtras.h"

namespace dynamatic {
namespace handshake {
std::optional<InternalStateNamer::TYPE>
InternalStateNamer::typeFromStr(const std::string &s) {
  if (s == EAGER_FORK_SENT)
    return TYPE::EagerForkSent;
  if (s == BUFFER_SLOT_FULL)
    return TYPE::BufferSlotFull;
  if (s == PIPELINE_SLOT)
    return TYPE::PipelineSlot;
  if (s == CONSTRAINED)
    return TYPE::Constrained;
  if (s == MEMORY_CONTROLLER_SLOT)
    return TYPE::MemoryControllerSlot;
  if (s == EFFECTIVE_SLOT)
    return TYPE::EffectiveSlot;
  return std::nullopt;
}

std::string InternalStateNamer::typeToStr(TYPE t) {
  switch (t) {
  case TYPE::EagerForkSent:
    return EAGER_FORK_SENT.str();
  case TYPE::BufferSlotFull:
    return BUFFER_SLOT_FULL.str();
  case TYPE::PipelineSlot:
    return PIPELINE_SLOT.str();
  case TYPE::Constrained:
    return CONSTRAINED.str();
  case TYPE::MemoryControllerSlot:
    return MEMORY_CONTROLLER_SLOT.str();
  case TYPE::EffectiveSlot:
    return EFFECTIVE_SLOT.str();
  }
}

std::unique_ptr<InternalStateNamer>
InternalStateNamer::fromJSON(const llvm::json::Value &value,
                             llvm::json::Path path) {
  std::string typeStr;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(TYPE_LIT, typeStr))
    return nullptr;

  auto typeOpt = typeFromStr(typeStr);
  if (!typeOpt)
    return nullptr;
  TYPE type = *typeOpt;
  llvm::json::Value inner = nullptr;
  if (const auto *obj = value.getAsObject()) {
    auto it = obj->find(INNER_LIT);
    if (it != obj->end())
      inner = it->second;
  }
  std::unique_ptr<InternalStateNamer> prop = nullptr;
  switch (type) {
  case TYPE::EagerForkSent:
    prop = EagerForkSentNamer::fromInnerJSON(inner, path);
    assert(prop && "inner eager fork failed");
    break;
  case TYPE::BufferSlotFull:
    prop = BufferSlotFullNamer::fromInnerJSON(inner, path);
    assert(prop && "inner buffer slot failed");
    break;
  case TYPE::PipelineSlot:
    prop = PipelineSlotNamer::fromInnerJSON(inner, path);
    assert(prop && "inner latency slot failed");
    break;
  case TYPE::Constrained:
    assert(false && "todo");
    break;
  case TYPE::MemoryControllerSlot:
    prop = MemoryControllerSlotNamer::fromInnerJSON(inner, path);
    assert(prop && "mc slot failed");
    break;
  case TYPE::EffectiveSlot:
    prop = EffectiveSlotNamer::fromInnerJSON(inner, path);
    assert(prop && "effective slot failed");
    break;
  }
  prop->type = type;
  return prop;
}

std::unique_ptr<InternalStateNamer>
InternalStateNamer::tryConstrain(int32_t value) const {
  if (auto *namer = dyn_cast<EagerForkSentNamer>(this)) {
    return std::make_unique<ConstrainedEagerForkSentNamer>(
        namer->constrain(value));
  }
  if (auto *namer = dyn_cast<BufferSlotFullNamer>(this)) {
    return std::make_unique<ConstrainedBufferSlotFullNamer>(
        namer->constrain(value));
  }
  if (auto *namer = dyn_cast<EffectiveSlotNamer>(this)) {
    return namer->constrain(value);
  }

  return nullptr;
}

std::unique_ptr<EagerForkSentNamer>
EagerForkSentNamer::fromInnerJSON(const llvm::json::Value &value,
                                  llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<EagerForkSentNamer>();
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName) ||
      !mapper.map(CHANNEL_NAME_LIT, prop->channelName) ||
      !mapper.map(CHANNEL_SIZE_LIT, prop->channelSize))
    return nullptr;
  return prop;
}

ConstrainedEagerForkSentNamer
EagerForkSentNamer::constrain(int32_t value) const {
  ConstrainedEagerForkSentNamer p(*this, value);
  return p;
}

std::unique_ptr<BufferSlotFullNamer>
BufferSlotFullNamer::fromInnerJSON(const llvm::json::Value &value,
                                   llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<BufferSlotFullNamer>();
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName) ||
      !mapper.map(SLOT_NAME_LIT, prop->slotName) ||
      !mapper.map(SLOT_SIZE_LIT, prop->slotSize))
    return nullptr;
  return prop;
}

ConstrainedBufferSlotFullNamer
BufferSlotFullNamer::constrain(int32_t value) const {
  ConstrainedBufferSlotFullNamer p(*this, value);
  return p;
}

std::unique_ptr<PipelineSlotNamer>
PipelineSlotNamer::fromInnerJSON(const llvm::json::Value &value,
                                 llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<PipelineSlotNamer>();
  int index;
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName) ||
      !mapper.map(SLOT_INDEX_LIT, index))
    return nullptr;
  prop->slotIndex = index;
  return prop;
}

std::unique_ptr<InternalStateNamer>
ConstrainedNamer::fromInnerJSON(const llvm::json::Value &value,
                                llvm::json::Path path) {
  auto namer = InternalStateNamer::fromJSON(value, path);
  assert(namer && "inner json must be an internal state");
  auto mapper = llvm::json::ObjectMapper(value, path);
  int32_t val;
  if (!mapper || !mapper.map(CONSTRAINT_VALUE, val))
    return nullptr;

  if (auto eagerFork = dyn_cast<EagerForkSentNamer>(namer)) {
    return std::make_unique<ConstrainedEagerForkSentNamer>(
        eagerFork->constrain(val));
  }
  return nullptr;
}

std::unique_ptr<MemoryControllerSlotNamer>
MemoryControllerSlotNamer::fromInnerJSON(const llvm::json::Value &value,
                                         llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<MemoryControllerSlotNamer>();
  int t;
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName) ||
      !mapper.map(SLOT_INDEX_LIT, prop->slotIndex) ||
      !mapper.map(PORT_TYPE_LIT, t) ||
      !mapper.map(LOADLESS_LIT, prop->loadless))
    return nullptr;
  prop->portType = (PortType)t;
  return prop;
}

std::string EffectiveSlotNamer::getSMVName() const {
  if (copiedSents.empty()) {
    return slot->getSMVName();
  }

  std::vector<std::string> sentNames;
  sentNames.reserve(copiedSents.size());
  for (auto &sent : copiedSents) {
    sentNames.push_back(llvm::formatv("!{0}", sent.getSMVName()));
  }
  return llvm::formatv("({0} & {1})", slot->getSMVName(),
                       llvm::join(sentNames, " & "));
}
llvm::json::Value EffectiveSlotNamer::toInnerJSON() const {
  std::vector<llvm::json::Value> copiedSentsJSON;
  copiedSentsJSON.reserve(copiedSents.size());
  for (auto &sent : copiedSents) {
    copiedSentsJSON.push_back(sent.toInnerJSON());
  }

  return llvm::json::Object(
      {{SLOT_LIT, slot->toJSON()}, {COPIED_SENTS_LIT, copiedSentsJSON}});
}

std::unique_ptr<EffectiveSlotNamer>
EffectiveSlotNamer::fromInnerJSON(const llvm::json::Value &value,
                                  llvm::json::Path path) {
  auto prop = std::make_unique<EffectiveSlotNamer>();
  auto *obj = value.getAsObject();
  assert(obj);
  auto *slotJSON = obj->get(SLOT_LIT);
  assert(slotJSON);
  prop->slot = InternalStateNamer::fromJSON(*slotJSON, path);

  const llvm::json::Value *copiedSentsJSON = obj->get(COPIED_SENTS_LIT);
  assert(copiedSentsJSON);
  auto *array = copiedSentsJSON->getAsArray();
  assert(array);
  for (const llvm::json::Value &sentJSON : *array) {
    prop->copiedSents.push_back(
        *EagerForkSentNamer::fromInnerJSON(sentJSON, path));
  }
  return prop;
}

} // namespace handshake
} // namespace dynamatic
