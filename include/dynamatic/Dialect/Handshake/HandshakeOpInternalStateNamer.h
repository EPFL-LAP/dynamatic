#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

namespace dynamatic {
namespace handshake {
struct InternalStateNamer;
struct EagerForkSentNamer;
struct BufferSlotFullNamer;
struct PipelineSlotNamer;
struct ConstrainedNamer;
struct ConstrainedEagerForkSentNamer;
struct ConstrainedBufferSlotFullNamer;
struct MemoryControllerSlotNamer;

// A general structure for an operation is assumed:
// in1, in2, ... -> Join/Merge/Mux
// -> Pipeline Slots
// -> Slots
// -> Fork/Branch -> out1, out2, ...
//
// Some operations do not follow this structure, and should be handled
// separately to avoid making false assumptions.
struct InternalStateNamer {
  enum class TYPE {
    EagerForkSent,
    BufferSlotFull,
    PipelineSlot,
    Constrained,
    MemoryControllerSlot,
  };
  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);

  virtual std::string getSMVName() const = 0;
  virtual llvm::json::Value toInnerJSON() const = 0;

  inline llvm::json::Value toJSON() const {
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
        {TYPE_LIT, typeToStr(type)},
        {INNER_LIT, toInnerJSON()},
    });
  }

  std::unique_ptr<InternalStateNamer> static fromJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  InternalStateNamer() = default;
  InternalStateNamer(TYPE type) : type(type) {}
  virtual ~InternalStateNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) { return true; }

  std::unique_ptr<ConstrainedNamer> tryConstrain(int32_t value);

  TYPE type;
  static constexpr llvm::StringLiteral TYPE_LIT = "type";
  static constexpr llvm::StringLiteral EAGER_FORK_SENT = "EagerForkSent";
  static constexpr llvm::StringLiteral BUFFER_SLOT_FULL = "BufferSlotFull";
  static constexpr llvm::StringLiteral PIPELINE_SLOT = "PipelineSlot";
  static constexpr llvm::StringLiteral CONSTRAINED = "Constrained";
  static constexpr llvm::StringLiteral MEMORY_CONTROLLER_SLOT =
      "MemoryControllerSlot";
  static constexpr llvm::StringLiteral INNER_LIT = "inner";
};

struct ConstrainedNamer : InternalStateNamer {
  ConstrainedNamer() = default;
  ConstrainedNamer(TYPE type, int32_t value)
      : InternalStateNamer(TYPE::Constrained), value(value) {}
  virtual ~ConstrainedNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) {
    return fp->type == TYPE::Constrained;
  }

  virtual std::unique_ptr<InternalStateNamer> getUnconstrained() const = 0;

  inline llvm::json::Value toInnerJSON() const override {
    // This assumes the internal state being named is represented as an object.
    // Example for fork8 output 0 constrained to 0:
    // {
    //   "channel_name": "outs_0",
    //   "channel_size": 1,
    //   "operation": "fork8",
    //   "value": 0
    // }
    llvm::json::Object *objP = getUnconstrained()->toJSON().getAsObject();
    assert(objP && "internal state namer is a json object");
    llvm::json::Object &obj = *objP;
    obj[CONSTRAINT_VALUE] = value;

    return llvm::json::Object(obj);
  }

  std::unique_ptr<InternalStateNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  int32_t value;
  static constexpr llvm::StringLiteral CONSTRAINT_VALUE = "value";
};

// To define a `sent` state of an eager fork, the exact channel that contains
// this `sent` state needs to be identified. The base class defines the
// operation by its name, and the eager fork class identifies the output by its
// port name (see NamedIOInterface)
struct EagerForkSentNamer : InternalStateNamer {
  EagerForkSentNamer() = default;
  EagerForkSentNamer(const std::string &opName, const std::string &channelName,
                     size_t channelSize)
      : InternalStateNamer(TYPE::EagerForkSent), opName(opName),
        channelName(channelName), channelSize(channelSize) {}
  ~EagerForkSentNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) {
    return fp->type == TYPE::EagerForkSent;
  }

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.{1}_sent", opName, channelName).str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object({{OPERATION_LIT, opName},
                               {CHANNEL_NAME_LIT, channelName},
                               {CHANNEL_SIZE_LIT, channelSize}});
  }

  std::unique_ptr<EagerForkSentNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  std::string opName;
  std::string channelName;
  size_t channelSize;

  ConstrainedEagerForkSentNamer constrain(int32_t value);

  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral CHANNEL_NAME_LIT = "channel_name";
  static constexpr llvm::StringLiteral CHANNEL_SIZE_LIT = "channel_size";
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
      llvm::report_fatal_error("value outside channel size");
    }
  } else {
    return llvm::formatv("0ud{0}_{1}", channelSize, value).str();
  }
}

struct ConstrainedEagerForkSentNamer : ConstrainedNamer {
  ConstrainedEagerForkSentNamer() = default;
  ConstrainedEagerForkSentNamer(const EagerForkSentNamer &base, int32_t value)
      : ConstrainedNamer(TYPE::EagerForkSent, value), base(base) {}
  ~ConstrainedEagerForkSentNamer() = default;

  inline std::string getSMVName() const override {
    return llvm::formatv("{0} & ({1}.ins = {2})", base.getSMVName(),
                         base.opName, smvValue(base.channelSize, value))
        .str();
  }

  inline std::unique_ptr<InternalStateNamer> getUnconstrained() const override {
    return std::make_unique<EagerForkSentNamer>(base);
  }

  EagerForkSentNamer base;
  static constexpr llvm::StringLiteral BASE_LIT = "base";
  static constexpr llvm::StringLiteral VALUE_LIT = "value";
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
};

struct BufferSlotFullNamer : InternalStateNamer {
  BufferSlotFullNamer() = default;
  BufferSlotFullNamer(const std::string &opName, const std::string &slotName,
                      const std::string &dataName, size_t slotSize)
      : InternalStateNamer(TYPE::BufferSlotFull), opName(opName),
        slotName(slotName), dataName(dataName), slotSize(slotSize) {}
  ~BufferSlotFullNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) {
    return fp->type == TYPE::BufferSlotFull;
  }

  ConstrainedBufferSlotFullNamer constrain(int32_t value);

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.{1}", opName, slotName).str();
  }
  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object({
        {OPERATION_LIT, opName},
        {SLOT_NAME_LIT, slotName},
        {DATA_NAME_LIT, dataName},
        {SLOT_SIZE_LIT, slotSize},
    });
  }

  std::unique_ptr<BufferSlotFullNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  std::string opName;
  std::string slotName;
  std::string dataName;
  size_t slotSize;
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral SLOT_NAME_LIT = "slot_name";
  static constexpr llvm::StringLiteral DATA_NAME_LIT = "data_name";
  static constexpr llvm::StringLiteral SLOT_SIZE_LIT = "slot_size";
};

struct ConstrainedBufferSlotFullNamer : ConstrainedNamer {
  ConstrainedBufferSlotFullNamer() = default;
  ConstrainedBufferSlotFullNamer(const BufferSlotFullNamer &base, int32_t value)
      : ConstrainedNamer(), base(base), value(value) {}
  ~ConstrainedBufferSlotFullNamer() = default;

  inline std::string getSMVName() const override {
    // Assuming buffer1 contains a 32bit slot:
    // buffer1.slot_full & (buffer1.slot_data = 0ud32_1)
    return llvm::formatv("{0} & ({1}.{2} = {3})", base.getSMVName(),
                         base.opName, base.dataName,
                         smvValue(base.slotSize, value))
        .str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    llvm::json::Value obj = base.toInnerJSON();
    return llvm::json::Object({{BASE_LIT, obj}, {VALUE_LIT, value}});
  }

  inline std::unique_ptr<InternalStateNamer> getUnconstrained() const override {
    return std::make_unique<BufferSlotFullNamer>(base);
  }

  BufferSlotFullNamer base;
  int32_t value;
  static constexpr llvm::StringLiteral BASE_LIT = "base";
  static constexpr llvm::StringLiteral VALUE_LIT = "value";
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
};

struct PipelineSlotNamer : InternalStateNamer {
  PipelineSlotNamer() = default;
  PipelineSlotNamer(const std::string &opName, unsigned slotIndex)
      : InternalStateNamer(TYPE::PipelineSlot), opName(opName),
        slotIndex(slotIndex) {}
  ~PipelineSlotNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) {
    return fp->type == TYPE::PipelineSlot;
  }

  inline std::string getSMVName() const override {
    return llvm::formatv("{0}.inner_handshake_manager.inner_delay_buffer.v{1}",
                         opName, slotIndex)
        .str();
  }

  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object(
        {{OPERATION_LIT, opName}, {SLOT_INDEX_LIT, slotIndex}});
  }

  std::unique_ptr<PipelineSlotNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);
  std::string opName;
  unsigned slotIndex;
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral SLOT_INDEX_LIT = "pipeline_index";
};

struct MemoryControllerSlotNamer : InternalStateNamer {
  enum PortType {
    Load,
    Store,
    Control,
  };
  MemoryControllerSlotNamer() = default;
  MemoryControllerSlotNamer(PortType portType, const std::string &name,
                            size_t slotIndex)
      : InternalStateNamer(TYPE::MemoryControllerSlot), opName(name),
        slotIndex(slotIndex), portType(portType), loadless(false) {}
  ~MemoryControllerSlotNamer() = default;

  static inline bool classof(const InternalStateNamer *fp) {
    return fp->type == TYPE::MemoryControllerSlot;
  }

  inline std::string getSMVName() const override {
    switch (portType) {
    case Load:
      return llvm::formatv("{0}.inner_arbiter.valid_{1}", opName, slotIndex);
    case Store:
      return llvm::formatv("{0}.inner_mc_loadless.inner_arbiter.valid_{1}",
                           opName, slotIndex);
    case Control:
      assert(false && "todo");
      return nullptr;
    }
  }

  inline llvm::json::Value toInnerJSON() const override {
    return llvm::json::Object({{OPERATION_LIT, opName},
                               {SLOT_INDEX_LIT, slotIndex},
                               {PORT_TYPE_LIT, (int)portType},
                               {LOADLESS_LIT, loadless}});
  }
  std::unique_ptr<MemoryControllerSlotNamer> static fromInnerJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  std::string opName;
  size_t slotIndex;
  PortType portType;
  bool loadless;
  static constexpr llvm::StringLiteral OPERATION_LIT = "operation";
  static constexpr llvm::StringLiteral SLOT_INDEX_LIT = "slot_index";
  static constexpr llvm::StringLiteral PORT_TYPE_LIT = "port_type";
  static constexpr llvm::StringLiteral LOADLESS_LIT = "loadless";
};
} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_NAMER_H
