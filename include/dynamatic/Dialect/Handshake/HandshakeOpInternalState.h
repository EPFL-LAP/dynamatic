#ifndef DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H
#define DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace dynamatic {
namespace handshake {

class HandshakeOpInternalState {
public:
  enum class TYPE {
    EagerForkSent,
    BufferSlotFull,
  };
  TYPE getType() const { return type; }

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);

  // Serializes the state into JSON format.
  llvm::json::Value toJSON() const;
  // Serializes the extra info to JSON format.
  inline virtual llvm::json::Value extraInfoToJSON() const { return nullptr; };

  // Deserializes state from JSON
  std::unique_ptr<HandshakeOpInternalState> static fromJSON(
      NameAnalysis &nameAnalysis, const llvm::json::Value &value,
      llvm::json::Path path);

  // Serializes the state into SMV format.
  inline virtual std::string writeSmv() const = 0;

  HandshakeOpInternalState() = default;
  HandshakeOpInternalState(TYPE type) : type(type) {}
  virtual ~HandshakeOpInternalState() = default;
  // static virtual std::unique_ptr<HandshakeOpInternalState> parse();
protected:
  TYPE type;
  llvm::json::Value parseBaseAndExtractInfo(const llvm::json::Value &value,
                                            llvm::json::Path path);

private:
  inline static const StringLiteral TYPE_LIT = "type";
  inline static const StringLiteral INFO_LIT = "type";
};

struct EagerForkSent : public HandshakeOpInternalState {
  /*
  inline llvm::json::Value extraInfoToJSON() const override {
    return llvm::json::Object({{OP_LIT, getUniqueName(op)},
        {OUTPUT_INDEX_LIT, outputIndex}});
  }
  inline std::string writeSmv() const override {
    return llvm::formatv("{0}.full_{1}", getUniqueName(op), outputIndex).str();
  }
  */
  llvm::json::Value extraInfoToJSON() const override;
  std::string writeSmv() const override;
  static std::unique_ptr<EagerForkSent> fromJSON(NameAnalysis &nameAnalysis,
                                                 const llvm::json::Value &value,
                                                 llvm::json::Path path);
  EagerForkSent() = default;
  EagerForkSent(mlir::Operation *op, unsigned outputIndex)
      : HandshakeOpInternalState(TYPE::EagerForkSent), op(op),
        outputIndex(outputIndex) {}
  ~EagerForkSent() = default;

  inline OpResult getChannel() { return op->getResults()[outputIndex]; }

  mlir::Operation *op;
  unsigned outputIndex;
  inline static const StringLiteral OP_LIT = "operation";
  inline static const StringLiteral OUTPUT_INDEX_LIT = "output_index";
};

struct BufferSlotFull : public HandshakeOpInternalState {
  inline llvm::json::Value extraInfoToJSON() const override {
    return llvm::json::Object(
        {{OP_LIT, getUniqueName(op)}, {SLOT_INDEX_LIT, slotIndex}});
  }
  inline std::string writeSmv() const override {
    return llvm::formatv("{0}.slot_{1}", getUniqueName(op), slotIndex).str();
  }
  static std::unique_ptr<BufferSlotFull>
  fromJSON(NameAnalysis &nameAnalysis, const llvm::json::Value &value,
           llvm::json::Path path);
  BufferSlotFull() = default;
  BufferSlotFull(mlir::Operation *op, unsigned slotIndex);
  ~BufferSlotFull() = default;

  mlir::Operation *op;
  unsigned slotIndex;
  inline static const StringLiteral OP_LIT = "operation";
  inline static const StringLiteral SLOT_INDEX_LIT = "slot_index";
};

} // namespace handshake
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HANDSHAKE_OP_INTERNAL_STATE_H
