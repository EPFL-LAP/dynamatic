#include "dynamatic/Dialect/Handshake/HandshakeOpInternalState.h"
#include "dynamatic/Support/JSON/JSON.h"

namespace dynamatic {
namespace handshake {

std::optional<HandshakeOpInternalState::TYPE>
HandshakeOpInternalState::typeFromStr(const std::string &s) {
  if (s == "EagerForkSent")
    return TYPE::EagerForkSent;
  if (s == "BufferSlotFull")
    return TYPE::BufferSlotFull;

  return std::nullopt;
}

std::string HandshakeOpInternalState::typeToStr(TYPE t) {
  switch (t) {
  case TYPE::EagerForkSent:
    return "EagerForkSent";
  case TYPE::BufferSlotFull:
    return "BufferSlotFull";
  }
}

llvm::json::Value HandshakeOpInternalState::toJSON() const {
  return llvm::json::Object(
      {{TYPE_LIT, typeToStr(type)}, {INFO_LIT, extraInfoToJSON()}});
}

std::unique_ptr<HandshakeOpInternalState>
HandshakeOpInternalState::fromJSON(NameAnalysis &nameAnalysis,
                                   const llvm::json::Value &value,
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
  case TYPE::EagerForkSent:
    return EagerForkSent::fromJSON(nameAnalysis, value, path.field(INFO_LIT));
  case TYPE::BufferSlotFull:
    return BufferSlotFull::fromJSON(nameAnalysis, value, path.field(INFO_LIT));
  }
}

llvm::json::Value HandshakeOpInternalState::parseBaseAndExtractInfo(
    const llvm::json::Value &value, llvm::json::Path path) {
  std::string typeStr;
  llvm::json::ObjectMapper mapper(value, path);
  if (!mapper || !mapper.map(TYPE_LIT, typeStr))
    return nullptr;

  auto typeOpt = typeFromStr(typeStr);
  if (!typeOpt)
    return nullptr;
  type = *typeOpt;

  // Find extra info object
  if (const auto *obj = value.getAsObject()) {
    auto it = obj->find(INFO_LIT);
    if (it != obj->end())
      return it->second;
  }

  return nullptr;
}

// EagerForkSent
llvm::json::Value EagerForkSent::extraInfoToJSON() const {
  return llvm::json::Object(
      {{OP_LIT, getUniqueName(op)}, {OUTPUT_INDEX_LIT, outputIndex}});
}

std::string EagerForkSent::writeSmv() const {
  return llvm::formatv("{0}.full_{1}", getUniqueName(op), outputIndex).str();
}

std::unique_ptr<EagerForkSent>
EagerForkSent::fromJSON(NameAnalysis &nameAnalysis,
                        const llvm::json::Value &value, llvm::json::Path path) {
  auto prop = std::make_unique<EagerForkSent>();
  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  std::string opName;
  if (!mapper || !mapper.map(OP_LIT, opName) ||
      !mapper.map(OUTPUT_INDEX_LIT, prop->outputIndex))
    return nullptr;

  Operation *op = nameAnalysis.getOp(opName);

  if (op == nullptr)
    return nullptr;

  // TODO: verify that op is a fork-like operation with enough inputs

  prop->op = op;
  return prop;
}

// EagerForkSent::EagerForkSent(mlir::Operation *op, unsigned outputIndex) :
// HandshakeOpInternalState(TYPE::EagerForkSent), op(op),
// outputIndex(outputIndex)  {}

// BufferSlotFull
std::unique_ptr<BufferSlotFull>
BufferSlotFull::fromJSON(NameAnalysis &nameAnalysis,
                         const llvm::json::Value &value,
                         llvm::json::Path path) {
  auto prop = std::make_unique<BufferSlotFull>();
  auto info = prop->parseBaseAndExtractInfo(value, path);
  llvm::json::ObjectMapper mapper(info, path);

  std::string opName;
  if (!mapper || !mapper.map(OP_LIT, opName) ||
      !mapper.map(SLOT_INDEX_LIT, prop->slotIndex))
    return nullptr;

  Operation *op = nameAnalysis.getOp(opName);

  if (op == nullptr)
    return nullptr;

  // TODO: verify that op is a buffer-like operation with enough slots

  prop->op = op;
  return prop;
}

BufferSlotFull::BufferSlotFull(mlir::Operation *op, unsigned slotIndex)
    : HandshakeOpInternalState(TYPE::BufferSlotFull), op(op),
      slotIndex(slotIndex) {}
} // namespace handshake
} // namespace dynamatic
