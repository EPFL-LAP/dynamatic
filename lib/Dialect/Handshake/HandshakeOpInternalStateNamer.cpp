#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "llvm/Support/Casting.h"

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
  }
}

llvm::json::Value toJSON(const std::unique_ptr<InternalStateNamer> &namer) {
  // Example:
  // {
  //   "type": "EagerForkSent",
  //   "inner": {
  //     "operation": "fork1",
  //     "channel_name": "outs_1",
  //     "channel_size": 2
  //   }
  // }
  return llvm::json::Object({
      {InternalStateNamer::TYPE_LIT,
       InternalStateNamer::typeToStr(namer->type)},
      {InternalStateNamer::INNER_LIT, namer->toInnerJSON()},
  });
}

llvm::json::Value toJSON(const std::shared_ptr<InternalStateNamer> &namer) {
  return llvm::json::Object({
      {InternalStateNamer::TYPE_LIT,
       InternalStateNamer::typeToStr(namer->type)},
      {InternalStateNamer::INNER_LIT, namer->toInnerJSON()},
  });
}

bool fromJSON(const llvm::json::Value &value,
              std::unique_ptr<InternalStateNamer> &namer,
              llvm::json::Path path) {
  std::string typeStr;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(InternalStateNamer::TYPE_LIT, typeStr))
    return false;

  auto typeOpt = InternalStateNamer::typeFromStr(typeStr);
  if (!typeOpt)
    return false;
  InternalStateNamer::TYPE type = *typeOpt;
  namer = nullptr;
  EagerForkSentNamer ef;
  BufferSlotFullNamer bs;
  PipelineSlotNamer ps;
  MemoryControllerSlotNamer mc;
  switch (type) {
  case InternalStateNamer::TYPE::EagerForkSent:
    ef = EagerForkSentNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, ef))
      return false;
    namer = std::make_unique<EagerForkSentNamer>(std::move(ef));
    break;
  case InternalStateNamer::TYPE::BufferSlotFull:
    bs = BufferSlotFullNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, bs))
      return false;
    namer = std::make_unique<BufferSlotFullNamer>(std::move(bs));
    break;
  case InternalStateNamer::TYPE::PipelineSlot:
    ps = PipelineSlotNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, ps))
      return false;
    namer = std::make_unique<PipelineSlotNamer>(std::move(ps));
    break;
  case InternalStateNamer::TYPE::Constrained:
    assert(false && "todo");
    break;
  case InternalStateNamer::TYPE::MemoryControllerSlot:
    mc = MemoryControllerSlotNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, mc))
      return false;
    namer = std::make_unique<MemoryControllerSlotNamer>(std::move(mc));
    break;
  }
  namer->type = type;
  return true;
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

bool fromJSON(const llvm::json::Value &value, EagerForkSentNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper &&
         mapper.map(EagerForkSentNamer::OPERATION_LIT, namer.opName) &&
         mapper.map(EagerForkSentNamer::CHANNEL_NAME_LIT, namer.channelName) &&
         mapper.map(EagerForkSentNamer::CHANNEL_SIZE_LIT, namer.channelSize);
}

ConstrainedEagerForkSentNamer EagerForkSentNamer::constrain(int32_t value) {
  ConstrainedEagerForkSentNamer p(*this, value);
  return p;
}

bool fromJSON(const llvm::json::Value &value, BufferSlotFullNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper &&
         mapper.map(BufferSlotFullNamer::OPERATION_LIT, namer.opName) &&
         mapper.map(BufferSlotFullNamer::SLOT_NAME_LIT, namer.slotName) &&
         mapper.map(BufferSlotFullNamer::DATA_NAME_LIT, namer.dataName) &&
         mapper.map(BufferSlotFullNamer::SLOT_SIZE_LIT, namer.slotSize);
}

ConstrainedBufferSlotFullNamer BufferSlotFullNamer::constrain(int32_t value) {
  ConstrainedBufferSlotFullNamer p(*this, value);
  return p;
}

bool fromJSON(const llvm::json::Value &value, PipelineSlotNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  int index;
  if (!mapper || !mapper.map(PipelineSlotNamer::OPERATION_LIT, namer.opName) ||
      !mapper.map(PipelineSlotNamer::SLOT_INDEX_LIT, index))
    return false;
  namer.slotIndex = index;
  return true;
}

bool fromJSON(const llvm::json::Value &value, ConstrainedNamer &namer,
              llvm::json::Path path) {
  auto mapper = llvm::json::ObjectMapper(value, path);
  std::unique_ptr<InternalStateNamer> inner;
  int32_t val;
  if (!mapper || !mapper.map(ConstrainedNamer::CONSTRAINT_VALUE_LIT, val) ||
      !mapper.map(ConstrainedNamer::INNER_LIT, inner))
    return false;
  namer = *inner->tryConstrain(val);
  return true;
}

bool fromJSON(const llvm::json::Value &value, MemoryControllerSlotNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  int t;
  if (!mapper ||
      !mapper.map(MemoryControllerSlotNamer::OPERATION_LIT, namer.opName) ||
      !mapper.map(MemoryControllerSlotNamer::SLOT_INDEX_LIT, namer.slotIndex) ||
      !mapper.map(MemoryControllerSlotNamer::PORT_TYPE_LIT, t) ||
      !mapper.map(MemoryControllerSlotNamer::LOADLESS_LIT, namer.loadless))
    return false;
  namer.portType = (MemoryControllerSlotNamer::PortType)t;
  return true;
}

} // namespace handshake
} // namespace dynamatic
