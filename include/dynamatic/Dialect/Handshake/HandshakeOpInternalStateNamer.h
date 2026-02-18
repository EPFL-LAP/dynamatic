#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"

namespace dynamatic {
namespace handshake {

struct InternalStateNamer {
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

  InternalStateNamer() = default;
  InternalStateNamer(TYPE type) : type(type) {}
  virtual ~InternalStateNamer() = default;
  TYPE type;
  static constexpr llvm::StringLiteral EAGER_FORK_SENT = "EagerForkSent";
  static constexpr llvm::StringLiteral BUFFER_SLOT_FULL = "BufferSlotFull";
};

// To define a `sent` state of an eager fork, the exact channel that contains
// this `sent` state needs to be identified. For this, the operation is
// identified through `opName` (e.g. "fork1"), and the output of this operations
// is defined by `channelName` (e.g. "out1")
struct EagerForkSentNamer : public InternalStateNamer {
  EagerForkSentNamer() = default;
  EagerForkSentNamer(const std::string &opName, const std::string &channelName)
      : opName(opName), channelName(channelName) {}
  ~EagerForkSentNamer() = default;

  std::string opName;
  std::string channelName;
};

struct BufferSlotFullNamer : public InternalStateNamer {
  BufferSlotFullNamer() = default;
  BufferSlotFullNamer(const std::string &opName, const std::string &slotName)
      : InternalStateNamer(TYPE::BufferSlotFull), opName(opName),
        slotName(slotName) {}
  ~BufferSlotFullNamer() = default;

  std::string opName;
  std::string slotName;
};

} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
