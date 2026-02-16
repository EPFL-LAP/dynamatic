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
    if (s == EAGER_FORK_SENT)
      return TYPE::EagerForkSent;
    if (s == BUFFER_SLOT_FULL)
      return TYPE::BufferSlotFull;
    return std::nullopt;
  }

  static std::string typeToStr(TYPE t) {
    switch (t) {
    case TYPE::EagerForkSent:
      return EAGER_FORK_SENT.str();
    case TYPE::BufferSlotFull:
      return BUFFER_SLOT_FULL.str();
    }
  }

  HandshakeOpInternalState() = default;
  HandshakeOpInternalState(TYPE type) : type(type) {}
  virtual ~HandshakeOpInternalState() = default;
  TYPE type;
  static constexpr llvm::StringLiteral EAGER_FORK_SENT = "EagerForkSent";
  static constexpr llvm::StringLiteral BUFFER_SLOT_FULL = "BufferSlotFull";
};

// To define a `sent` state of an eager fork, the exact channel that contains
// this `sent` state needs to be identified. For this, the operation is
// identified through `opName` (e.g. "fork1"), and the output of this operations
// is defined by `channelName` (e.g. "out1")
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
