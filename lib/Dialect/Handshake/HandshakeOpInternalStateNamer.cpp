#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

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
  if (s == ENTRY_SLOT)
    return TYPE::EntrySlot;
  if (s == TOKEN_COUNT)
    return TYPE::TokenCount;
  if (s == PIPELINE_TOKEN_COUNT)
    return TYPE::PipelineTokenCount;
  llvm::errs() << "unknown type\n";
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
  case TYPE::EntrySlot:
    return ENTRY_SLOT.str();
  case TYPE::TokenCount:
    return TOKEN_COUNT.str();
  case TYPE::PipelineTokenCount:
    return PIPELINE_TOKEN_COUNT.str();
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
  case TYPE::TokenCount:
    prop = TokenCountNamer::fromInnerJSON(inner, path);
    assert(prop && "inner token count failed");
    break;
  case TYPE::PipelineTokenCount:
    prop = PipelineTokenCountNamer::fromInnerJSON(inner, path);
    assert(prop && "pipeline token count failed");
    break;
  case TYPE::MemoryControllerSlot:
    prop = MemoryControllerSlotNamer::fromInnerJSON(inner, path);
    assert(prop && "mc slot failed");
    break;
  case TYPE::EntrySlot:
    prop = EntrySlotNamer::fromInnerJSON(inner, path);
    assert(prop && "entry slot failed");
    break;
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

std::unique_ptr<PipelineTokenCountNamer>
PipelineTokenCountNamer::fromInnerJSON(const llvm::json::Value &value,
                                       llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<PipelineTokenCountNamer>();
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName))
    return nullptr;
  return prop;
}

std::unique_ptr<TokenCountNamer>
TokenCountNamer::fromInnerJSON(const llvm::json::Value &value,
                               llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<TokenCountNamer>();
  if (!mapper || !mapper.map(OPERATION_LIT, prop->opName))
    return nullptr;
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

std::unique_ptr<EntrySlotNamer>
EntrySlotNamer::fromInnerJSON(const llvm::json::Value &value,
                              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  auto prop = std::make_unique<EntrySlotNamer>();
  if (!mapper || !mapper.map(ARG_NAME_LIT, prop->argName))
    return nullptr;
  return prop;
}

std::vector<std::unique_ptr<InternalStateNamer>>
getAllSlotsOfOperation(Operation *op) {
  std::vector<std::unique_ptr<InternalStateNamer>> ret;
  if (auto latencyOp = dyn_cast<LatencyInterface>(op)) {
    auto slots = latencyOp.getPipelineSlots();
    for (auto &slot : slots) {
      ret.push_back(std::make_unique<PipelineSlotNamer>(slot));
    }
  }
  if (auto sinkOp = dyn_cast<SinkOp>(op)) {
    if (sinkOp.terminatesIOG()) {
      ret.push_back(
          std::make_unique<BufferSlotFullNamer>(sinkOp.getTerminatingSlot()));
    }
    return ret;
  }

  if (auto endOp = dyn_cast<EndOp>(op)) {
    ret.push_back(std::make_unique<BufferSlotFullNamer>("testbench", "end", 0));
  }
  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    // TODO: Handle LoadOp for MC slot
    auto slots = loadOp.getInternalSlotStateNamers();
    auto *mcOp = loadOp.getAddressResult().getUses().begin()->getOwner();
    assert(mcOp);
    auto mc = dyn_cast<MemoryControllerOp>(mcOp);
    assert(mc);
    size_t nLoads = mc.getNumLoadPorts();
    std::optional<MemoryControllerSlotNamer> mcSlot;
    for (size_t i = 0; i < nLoads; ++i) {
      if (mc.getLoadPort(i)->getLoadOp() == loadOp) {
        assert(!mcSlot.has_value());
        mcSlot = mc.getLoadPortSlotNamer(i);
      }
    }
    assert(mcSlot);
    ret.push_back(std::make_unique<BufferSlotFullNamer>(slots[0]));
    ret.push_back(std::make_unique<MemoryControllerSlotNamer>(*mcSlot));
    ret.push_back(std::make_unique<BufferSlotFullNamer>(slots[1]));
    return ret;
  }

  if (auto bufferOp = dyn_cast<BufferLikeOpInterface>(op)) {
    auto slots = bufferOp.getInternalSlotStateNamers();
    for (auto &slot : slots) {
      ret.push_back(std::make_unique<BufferSlotFullNamer>(slot));
    }
  }
  return ret;
}

std::optional<std::unique_ptr<InternalStateNamer>>
getTokenCountNamerOfOperation(Operation *op) {
  if (isa<LatencyInterface>(op) && isa<BufferLikeOpInterface>(op)) {
    assert(false &&
           "cannot handle token count of operations with latency and slots");
    return std::nullopt;
  }

  if (auto latencyOp = dyn_cast<LatencyInterface>(op)) {
    return std::make_unique<PipelineTokenCountNamer>(getUniqueName(op).str());
  }
  if (auto bufferOp = dyn_cast<BufferLikeOpInterface>(op)) {
    return std::make_unique<TokenCountNamer>(getUniqueName(op).str());
  }
  return std::nullopt;
}

} // namespace handshake
} // namespace dynamatic
