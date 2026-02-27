#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

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
  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);

  virtual std::string getSMVName() const = 0;
  virtual llvm::json::Value toInnerJSON() const = 0;

  inline llvm::json::Value toJSON() const {
    return llvm::json::Object({
        {TYPE_LIT, typeToStr(type)},
        {INNER_LIT, toInnerJSON()},
    });
  }

  std::unique_ptr<InternalStateNamer> static fromJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  InternalStateNamer() = default;
  InternalStateNamer(TYPE type) : type(type) {}
  virtual ~InternalStateNamer() = default;

  TYPE type;
  static constexpr llvm::StringLiteral TYPE_LIT = "type";
  static constexpr llvm::StringLiteral EAGER_FORK_SENT = "EagerForkSent";
  static constexpr llvm::StringLiteral BUFFER_SLOT_FULL = "BufferSlotFull";
  static constexpr llvm::StringLiteral LATENCY_INDUCED_SLOT =
      "LatencyInducedSlot";
  static constexpr llvm::StringLiteral INNER_LIT = "inner";
};

struct ConstrainedEagerForkSentNamer;
// To define a `sent` state of an eager fork, the exact channel that contains
// this `sent` state needs to be identified. The base class defines the
// operation by its name, and the eager fork class identifies the output by its
// port name (see NamedIOInterface)
struct EagerForkSentNamer : public InternalStateNamer {
  EagerForkSentNamer() = default;
  EagerForkSentNamer(const std::string &opName, const std::string &channelName,
                     size_t channelSize)
      : InternalStateNamer(TYPE::EagerForkSent), opName(opName),
        channelName(channelName), channelSize(channelSize) {}
  ~EagerForkSentNamer() = default;

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.{1}_sent", opName, channelName).str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object(
        {{OPERATION_LIT, opName}, {CHANNEL_NAME_LIT, channelName}});
  }

  std::unique_ptr<EagerForkSentNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  std::string opName;
  std::string channelName;
  size_t channelSize;

  ConstrainedEagerForkSentNamer constrained(int32_t value);

  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral CHANNEL_NAME_LIT = "channel_name";
  static constexpr llvm::StringLiteral CHANNEL_SIZE_LIT = "channel_name";
};

inline std::string smvValue(size_t channelSize, size_t value) {
  assert(channelSize > 0 && "can only values for channels with >=1 bit");
  if (channelSize == 1) {
    switch (value) {
    case 0:
      return "FALSE";
    case 1:
      return "TRUE";
    default:
      assert(false && "value outside channel size");
    }
  } else {
    return llvm::formatv("0ud{0}_{1}", channelSize, value).str();
  }
}

struct ConstrainedEagerForkSentNamer : public InternalStateNamer {
  ConstrainedEagerForkSentNamer() = default;
  ConstrainedEagerForkSentNamer(const EagerForkSentNamer &base, int32_t value)
      : InternalStateNamer(TYPE::EagerForkSent), base(base), value(value) {}
  ~ConstrainedEagerForkSentNamer() = default;
  inline std::string getSMVName() const override {
    return llvm::formatv("{0} & ({1}.ins = {2})", base.getSMVName(),
                         base.opName, smvValue(base.channelSize, value))
        .str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    llvm::json::Value obj = base.toInnerJSON();
    return llvm::json::Object({{BASE_LIT, obj}, {VALUE_LIT, value}});
  }

  EagerForkSentNamer base;
  int32_t value;
  static constexpr llvm::StringLiteral BASE_LIT = "base";
  static constexpr llvm::StringLiteral VALUE_LIT = "value";
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
};

struct BufferSlotFullNamer : public InternalStateNamer {
  BufferSlotFullNamer() = default;
  BufferSlotFullNamer(const std::string &opName, const std::string &slotName,
                      size_t slotSize)
      : InternalStateNamer(TYPE::BufferSlotFull), opName(opName),
        slotName(slotName), slotSize(slotSize) {}
  ~BufferSlotFullNamer() = default;

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.{1}_full", opName, slotName).str();
  }
  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object({
        {OPERATION_LIT, opName},
        {SLOT_NAME_LIT, slotName},
        {SLOT_SIZE_LIT, slotSize},
    });
  }

  std::unique_ptr<BufferSlotFullNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  std::string opName;
  std::string slotName;
  size_t slotSize;
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral SLOT_NAME_LIT = "slot_name";
  static constexpr llvm::StringLiteral SLOT_SIZE_LIT = "slot_size";
};

struct LatencyInducedSlotNamer : public InternalStateNamer {
  LatencyInducedSlotNamer() = default;
  LatencyInducedSlotNamer(const std::string &opName, unsigned slotIndex)
      : InternalStateNamer(TYPE::BufferSlotFull), opName(opName),
        slotIndex(slotIndex) {}
  ~LatencyInducedSlotNamer() = default;

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.inner_handshake_manager.inner_delay_buffer.v{1}",
                         opName, slotIndex)
        .str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object({{SLOT_INDEX_LIT, slotIndex}});
  }
  std::string opName;
  unsigned slotIndex;
  static constexpr llvm::StringLiteral SLOT_INDEX_LIT = "slot_index";
};
} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
