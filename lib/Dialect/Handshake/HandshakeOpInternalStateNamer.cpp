#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "llvm/ADT/StringExtras.h"
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
  if (s == EFFECTIVE_SLOT)
    return TYPE::EffectiveSlot;
  if (s == ENTRY_SLOT)
    return TYPE::EntrySlot;
  if (s == TOKEN_COUNT)
    return TYPE::TokenCount;
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
  case TYPE::EffectiveSlot:
    return EFFECTIVE_SLOT.str();
  case TYPE::EntrySlot:
    return ENTRY_SLOT.str();
  case TYPE::TokenCount:
    return TOKEN_COUNT.str();
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
              std::shared_ptr<InternalStateNamer> &namer,
              llvm::json::Path path) {
  std::unique_ptr<InternalStateNamer> unique;

  if (!fromJSON(value, unique, path))
    return false;
  namer = std::move(unique);
  return true;
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
  EffectiveSlotNamer es;
  TokenCountNamer tc;
  EntrySlotNamer en;
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
  case InternalStateNamer::TYPE::EffectiveSlot:
    if (!mapper.map(InternalStateNamer::INNER_LIT, es))
      return false;
    namer = std::make_unique<EffectiveSlotNamer>(std::move(es));
    break;
  case InternalStateNamer::TYPE::TokenCount:
    tc = TokenCountNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, tc))
      return false;
    namer = std::make_unique<TokenCountNamer>(std::move(tc));
    break;
  case InternalStateNamer::TYPE::EntrySlot:
    en = EntrySlotNamer();
    if (!mapper.map(InternalStateNamer::INNER_LIT, en))
      return false;
    namer = std::make_unique<EntrySlotNamer>(std::move(en));
    break;
  }
  namer->type = type;
  return true;
}

std::unique_ptr<ConstrainedNamer>
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

bool fromJSON(const llvm::json::Value &value, EagerForkSentNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper &&
         mapper.map(EagerForkSentNamer::OPERATION_LIT, namer.opName) &&
         mapper.map(EagerForkSentNamer::CHANNEL_NAME_LIT, namer.channelName) &&
         mapper.map(EagerForkSentNamer::CHANNEL_SIZE_LIT, namer.channelSize);
}

ConstrainedEagerForkSentNamer
EagerForkSentNamer::constrain(int32_t value) const {
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

ConstrainedBufferSlotFullNamer
BufferSlotFullNamer::constrain(int32_t value) const {
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

std::string TokenCountNamer::getSMVName() const {
  if (slots->empty()) {
    return "0";
  }
  std::vector<std::string> names;
  names.reserve(slots->size());

  for (const auto &x : *slots) {
    names.push_back(x->getSMVName());
  }

  return llvm::formatv("count({0})", llvm::join(names, ", "));
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
  return llvm::json::Object(
      {{SLOT_LIT, slot}, {COPIED_SENTS_LIT, copiedSents}});
}

bool fromJSON(const llvm::json::Value &value, EffectiveSlotNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper && mapper.map(EffectiveSlotNamer::SLOT_LIT, namer.slot) &&
         mapper.map(EffectiveSlotNamer::COPIED_SENTS_LIT, namer.copiedSents);
}

std::unique_ptr<ConstrainedNamer>
EffectiveSlotNamer::constrain(int32_t value) const {
  return std::make_unique<ConstrainedEffectiveSlotNamer>(*this, value);
}

EntrySlotNamer::EntrySlotNamer(BlockArgument arg)
    : InternalStateNamer(TYPE::EntrySlot) {
  Operation *op = arg.getOwner()->getParentOp();
  auto nameAttr = op->getAttrOfType<ArrayAttr>("argNames")[arg.getArgNumber()];
  std::string name = dyn_cast<StringAttr>(nameAttr).str();
  argName = name;
}

bool fromJSON(const llvm::json::Value &value, EntrySlotNamer &namer,
              llvm::json::Path path) {
  llvm::json::ObjectMapper mapper(value, path);
  return mapper && mapper.map(EntrySlotNamer::ARG_NAME_LIT, namer.argName);
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

  if (auto endOp = dyn_cast<EndOp>(op)) {
    ret.push_back(
        std::make_unique<BufferSlotFullNamer>("testbench", "end_full", "", 0));
  }
  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    auto slots = loadOp.getInternalSlotStateNamers();
    auto *mcOp = loadOp.getAddressResult().getUses().begin()->getOwner();
    auto mc = dyn_cast<MemoryControllerOp>(mcOp);
    if (!mc) {
      op->emitError("Cannot get slot of LoadOp that is not connected to Memory "
                    "Controller Op");
      llvm::report_fatal_error("Unhandled LoadOp");
    }
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

std::optional<TokenCountNamer> getTokenCountNamerOfOperation(Operation *op) {
  auto tokens = getAllSlotsOfOperation(op);
  if (tokens.empty()) {
    return std::nullopt;
  }
  return TokenCountNamer(std::move(tokens));
}

} // namespace handshake
} // namespace dynamatic
