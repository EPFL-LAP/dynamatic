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
  if (s == CONSTRAINED)
    return TYPE::Constrained;
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
  case TYPE::Constrained:
    return CONSTRAINED.str();
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
  case TYPE::LatencyInducedSlot:
    prop = LatencyInducedSlotNamer::fromInnerJSON(inner, path);
    assert(prop && "inner latency slot failed");
    break;
  case TYPE::Constrained:
    assert(false && "todo");
  }
  prop->type = type;
  return prop;
}

std::unique_ptr<ConstrainedNamer>
InternalStateNamer::tryConstrain(int32_t value) {
  if (auto *namer = dyn_cast<EagerForkSentNamer>(this)) {
    return std::make_unique<ConstrainedEagerForkSentNamer>(
        namer->constrain(value));
  }
  if (auto *namer = dyn_cast<BufferSlotFullNamer>(this)) {
    return std::make_unique<ConstrainedBufferSlotFullNamer>(
        namer->constrain(value));
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

ConstrainedEagerForkSentNamer EagerForkSentNamer::constrain(int32_t value) {
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

ConstrainedBufferSlotFullNamer BufferSlotFullNamer::constrain(int32_t value) {
  ConstrainedBufferSlotFullNamer p(*this, value);
  return p;
}

std::unique_ptr<LatencyInducedSlotNamer>
LatencyInducedSlotNamer::fromInnerJSON(const llvm::json::Value &value,
                                       llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<LatencyInducedSlotNamer>();
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
} // namespace handshake
} // namespace dynamatic
