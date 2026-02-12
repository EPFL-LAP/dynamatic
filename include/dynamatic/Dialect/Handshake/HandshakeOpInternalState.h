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
  EagerForkSent(mlir::OpResult channel)
      : HandshakeOpInternalState(TYPE::EagerForkSent), channel(channel) {}
  EagerForkSent(mlir::Operation *op, unsigned outputIndex)
      : HandshakeOpInternalState(TYPE::EagerForkSent),
        channel(op->getResults()[outputIndex]) {
    assert(op && "Fork operation has to exist");
    // TODO: assert(isa<EagerForkLikeOpInterface>(op));
  }
  ~EagerForkSent() = default;

  mlir::OpResult channel;
};

struct BufferSlotFull : public HandshakeOpInternalState {
  BufferSlotFull() = default;
  BufferSlotFull(mlir::Operation *op, unsigned slotIndex)
      : HandshakeOpInternalState(TYPE::BufferSlotFull), op(op),
        slotIndex(slotIndex) {}
  ~BufferSlotFull() = default;

  mlir::Operation *op;
  unsigned slotIndex;
};

} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H
