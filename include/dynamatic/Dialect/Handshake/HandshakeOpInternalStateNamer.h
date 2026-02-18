#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"

namespace dynamatic {
namespace handshake {

// A general structure for an operation is assumed:
// in1, in2, ... -> Join/Merge/Mux
// -> Latency Slots
// -> Slots
// -> Fork/Branch -> out1, out2, ...
//
// Some operations do not follow this structure, and should be handled
// separately to avoid making false assumptions.
struct InternalStateNamer {
  enum class TYPE {
    EagerForkSent,
    BufferSlotFull,
    LatencyInducedSlot,
  };
  static std::optional<TYPE> typeFromStr(const std::string &s) {
    if (s == EAGER_FORK_SENT)
      return TYPE::EagerForkSent;
    if (s == BUFFER_SLOT_FULL)
      return TYPE::BufferSlotFull;
    if (s == LATENCY_INDUCED_SLOT)
      return TYPE::LatencyInducedSlot;
    return std::nullopt;
  }

  static std::string typeToStr(TYPE t) {
    switch (t) {
    case TYPE::EagerForkSent:
      return EAGER_FORK_SENT.str();
    case TYPE::BufferSlotFull:
      return BUFFER_SLOT_FULL.str();
    case TYPE::LatencyInducedSlot:
      return LATENCY_INDUCED_SLOT.str();
    }
  }

  virtual std::string getSMVName() = 0;

  InternalStateNamer() = default;
  InternalStateNamer(TYPE type, const std::string &opName)
      : type(type), opName(opName) {}
  virtual ~InternalStateNamer() = default;

  TYPE type;
  std::string opName;
  static constexpr llvm::StringLiteral EAGER_FORK_SENT = "EagerForkSent";
  static constexpr llvm::StringLiteral BUFFER_SLOT_FULL = "BufferSlotFull";
  static constexpr llvm::StringLiteral LATENCY_INDUCED_SLOT =
      "LatencyInducedSlot";
};

// To define a `sent` state of an eager fork, the exact channel that contains
// this `sent` state needs to be identified. The base class defines the
// operation by its name, and the eager fork class identifies the output by its
// port name (see NamedIOInterface)
struct EagerForkSentNamer : public InternalStateNamer {
  EagerForkSentNamer() = default;
  EagerForkSentNamer(const std::string &opName, const std::string &channelName)
      : InternalStateNamer(TYPE::EagerForkSent, opName),
        channelName(channelName) {}
  ~EagerForkSentNamer() = default;

  inline std::string getSMVName() override {
    return llvm::formatv("{0}.{1}_sent", opName, channelName).str();
  }

  std::string channelName;
};

struct BufferSlotFullNamer : public InternalStateNamer {
  BufferSlotFullNamer() = default;
  BufferSlotFullNamer(const std::string &opName, const std::string &slotName)
      : InternalStateNamer(TYPE::BufferSlotFull, opName), slotName(slotName) {}
  ~BufferSlotFullNamer() = default;

  inline std::string getSMVName() override {
    return llvm::formatv("{0}.{1}_full", opName, slotName).str();
  }
  std::string slotName;
};

struct LatencyInducedSlotNamer : public InternalStateNamer {
  LatencyInducedSlotNamer() = default;
  LatencyInducedSlotNamer(const std::string &opName, unsigned slotIndex)
      : InternalStateNamer(TYPE::BufferSlotFull, opName), slotIndex(slotIndex) {
  }
  ~LatencyInducedSlotNamer() = default;

  inline std::string getSMVName() override {
    return llvm::formatv("{0}.v{1}", opName, slotIndex).str();
  }

  unsigned slotIndex;
};

} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
