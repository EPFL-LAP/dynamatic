#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"

namespace dynamatic {
namespace handshake {

struct HandshakeOpInternalState {
  enum class TYPE {
    EagerForkSent,
    BufferSlotFull,
  };
  static std::optional<TYPE> typeFromStr(const std::string &s) {
    if (s == "EagerForkSent")
      return TYPE::EagerForkSent;
    if (s == "BufferSlotFull")
      return TYPE::BufferSlotFull;
    return std::nullopt;
  }

  static std::string typeToStr(TYPE t) {
    switch (t) {
    case TYPE::EagerForkSent:
      return "EagerForkSent";
    case TYPE::BufferSlotFull:
      return "BufferSlotFull";
    }
  }

  HandshakeOpInternalState() = default;
  HandshakeOpInternalState(TYPE type) : type(type) {}
  virtual ~HandshakeOpInternalState() = default;
  TYPE type;
};

struct EagerForkSent : public HandshakeOpInternalState {
  EagerForkSent() = default;
  EagerForkSent(const std::string &opName, const std::string &channelName)
      : opName(opName), channelName(channelName) {}
  ~EagerForkSent() = default;

  std::string opName;
  std::string channelName;
};

struct BufferSlotFull : public HandshakeOpInternalState {
  BufferSlotFull() = default;
  BufferSlotFull(const std::string &opName, const std::string &slotName)
      : HandshakeOpInternalState(TYPE::BufferSlotFull), opName(opName),
        slotName(slotName) {}
  ~BufferSlotFull() = default;

  std::string opName;
  std::string slotName;
};

} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H
