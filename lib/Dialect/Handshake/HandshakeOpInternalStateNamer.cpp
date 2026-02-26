#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"

namespace dynamatic {
namespace handshake {
std::optional<InternalStateNamer::TYPE>
InternalStateNamer::typeFromStr(const std::string &s) {
  if (s == EAGER_FORK_SENT)
    return TYPE::EagerForkSent;
  if (s == BUFFER_SLOT_FULL)
    return TYPE::BufferSlotFull;
  if (s == LATENCY_INDUCED_SLOT)
    return TYPE::LatencyInducedSlot;
  return std::nullopt;
}

std::string InternalStateNamer::typeToStr(TYPE t) {
  switch (t) {
  case TYPE::EagerForkSent:
    return EAGER_FORK_SENT.str();
  case TYPE::BufferSlotFull:
    return BUFFER_SLOT_FULL.str();
  case TYPE::LatencyInducedSlot:
    return LATENCY_INDUCED_SLOT.str();
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
    break;
  case TYPE::BufferSlotFull:
    prop = BufferSlotFullNamer::fromInnerJSON(inner, path);
    break;
  case TYPE::LatencyInducedSlot:
    assert(false && "not yet implemented");
  }
  prop->type = type;
  return prop;
}

std::unique_ptr<EagerForkSentNamer>
EagerForkSentNamer::fromInnerJSON(const llvm::json::Value &value,
                                  llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<EagerForkSentNamer>();
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName) ||
      !mapper.map(CHANNEL_NAME_LIT, prop->channelName))
    return nullptr;
  return prop;
}

ConstrainedEagerForkSentNamer EagerForkSentNamer::constrained(int32_t value) {
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

} // namespace handshake
} // namespace dynamatic
