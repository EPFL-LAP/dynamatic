//===- FormalProperty.h -----------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JSON-parsing logic for the formal properties' database.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <memory>
#include <optional>

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE {
    AOB /* Absence Of Backpressure */,
    VEQ /* Valid EQuivalence */,
    EFNAO /* Eager Fork Not All Output sent */,
    CSOAFAF, /* Copied Slots Of Active Forks Are Full */
  };

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  unsigned long getId() const { return id; }
  std::optional<bool> getCheck() const { return check; }

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);
  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string tagToStr(TAG t);

  // Serializes the formal property into JSON format
  llvm::json::Value toJSON() const;
  // Serializes the extra info to JSON fomrat. For the base class this is empty.
  // Derived class are supposed to overrde this with their method to serialize
  // extra info
  inline virtual llvm::json::Value extraInfoToJSON() const { return nullptr; };

  // Deserializes a formal property from JSON. The return type can be casted to
  // the derived classes to access the extra info
  std::unique_ptr<FormalProperty> static fromJSON(
      const llvm::json::Value &value, llvm::json::Path path);

  FormalProperty() = default;
  FormalProperty(unsigned long id, TAG tag, TYPE type)
      : id(id), tag(tag), type(type), check(std::nullopt) {}
  virtual ~FormalProperty() = default;

  static bool classof(const FormalProperty *fp) { return true; }

protected:
  unsigned long id;
  TAG tag;
  TYPE type;
  std::optional<bool> check;

  llvm::json::Value parseBaseAndExtractInfo(const llvm::json::Value &value,
                                            llvm::json::Path path);

private:
  inline static const StringLiteral ID_LIT = "id";
  inline static const StringLiteral TYPE_LIT = "type";
  inline static const StringLiteral TAG_LIT = "tag";
  inline static const StringLiteral INFO_LIT = "info";
  inline static const StringLiteral CHECK_LIT = "check";
};

struct SignalName {
  std::string operationName;
  std::string channelName;
  unsigned channelIndex;
};

class AbsenceOfBackpressure : public FormalProperty {
public:
  std::string getOwner() { return ownerChannel.operationName; }
  std::string getUser() { return userChannel.operationName; }
  int getOwnerIndex() { return ownerChannel.channelIndex; }
  int getUserIndex() { return userChannel.channelIndex; }
  std::string getOwnerChannel() { return ownerChannel.channelName; }
  std::string getUserChannel() { return userChannel.channelName; }

  // Overriding the serilization of extra info with the new fields added for
  // absence of backpressure
  llvm::json::Value extraInfoToJSON() const override;
  // Deserializes absence of backpressure form JSON
  static std::unique_ptr<AbsenceOfBackpressure>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  AbsenceOfBackpressure() = default;
  AbsenceOfBackpressure(unsigned long id, TAG tag, const OpResult &res);
  ~AbsenceOfBackpressure() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::AOB;
  }

private:
  SignalName ownerChannel;
  SignalName userChannel;
  inline static const StringLiteral OWNER_OP_LIT = "owner_op";
  inline static const StringLiteral USER_OP_LIT = "user_op";
  inline static const StringLiteral OWNER_CHANNEL_LIT = "owner_channel";
  inline static const StringLiteral USER_CHANNEL_LIT = "user_channel";
  inline static const StringLiteral OWNER_INDEX_LIT = "owner_index";
  inline static const StringLiteral USER_INDEX_LIT = "user_index";
};

class ValidEquivalence : public FormalProperty {
public:
  std::string getOwner() { return ownerChannel.operationName; }
  std::string getTarget() { return targetChannel.operationName; }
  int getOwnerIndex() { return ownerChannel.channelIndex; }
  int getTargetIndex() { return targetChannel.channelIndex; }
  std::string getOwnerChannel() { return ownerChannel.channelName; }
  std::string getTargetChannel() { return targetChannel.channelName; }

  // Overriding the serilization of extra info with the new fields added for
  // absence of backpressure
  llvm::json::Value extraInfoToJSON() const override;
  // Deserializes absence of backpressure form JSON
  static std::unique_ptr<ValidEquivalence>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  ValidEquivalence() = default;
  ValidEquivalence(unsigned long id, TAG tag, const OpResult &res1,
                   const OpResult &res2);
  ~ValidEquivalence() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::VEQ;
  }

private:
  SignalName ownerChannel;
  SignalName targetChannel;
  inline static const StringLiteral OWNER_OP_LIT = "owner_op";
  inline static const StringLiteral TARGET_OP_LIT = "target_op";
  inline static const StringLiteral OWNER_CHANNEL_LIT = "owner_channel";
  inline static const StringLiteral TARGET_CHANNEL_LIT = "target_channel";
  inline static const StringLiteral OWNER_INDEX_LIT = "owner_index";
  inline static const StringLiteral TARGET_INDEX_LIT = "target_index";
};

// An eager fork propagates an incoming token to each output as soon as the
// output is ready, and keeps track of which outputs already have a token sent
// across them through the `sent` state. When the token has been sent to all
// outputs, the token at the input is consumed and the states of the fork are
// reset. The state where all outputs are in the `sent` state simultaneously is
// unreachable, as the fork resets as soon as this state would be reached. See
// invariant 1 of https://ieeexplore.ieee.org/document/10323796 for more
// details.
class EagerForkNotAllOutputSent : public FormalProperty {
public:
  std::string getOwner() { return ownerOp; }
  unsigned getNumEagerForkOutputs() { return numEagerForkOutputs; }

  llvm::json::Value extraInfoToJSON() const override;

  static std::unique_ptr<EagerForkNotAllOutputSent>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  EagerForkNotAllOutputSent() = default;
  EagerForkNotAllOutputSent(unsigned long id, TAG tag,
                            handshake::EagerForkLikeOpInterface &op);
  ~EagerForkNotAllOutputSent() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::EFNAO;
  }

private:
  std::string ownerOp;
  unsigned numEagerForkOutputs;
  inline static const StringLiteral OWNER_OP_LIT = "owner_op";
  inline static const StringLiteral NUM_EAGER_OUTPUTS_LIT = "num_eager_outputs";
};

// When an eager fork is `sent` state for at least one of its outputs, it is
// considered `active`. When transitioning to the `active` state, the `ready`
// signal is false, and the incoming token is blocked. Because of this, all
// slots immediately before the fork (i.e. copied slots) must be full. More
// formally, a copied slot of a fork is defined as a slot that has a path
// towards the fork without any other slots on it. See invariant 2 of
// https://ieeexplore.ieee.org/document/10323796 for more details
class CopiedSlotsOfActiveForkAreFull : public FormalProperty {
public:
  std::string getForkOp() { return forkOp; }
  unsigned getNumEagerForkOutputs() { return numEagerForkOutputs; }
  std::string getBufferOp() { return bufferOp; }
  unsigned getBufferSlot() { return bufferSlot; }

  llvm::json::Value extraInfoToJSON() const override;

  static std::unique_ptr<CopiedSlotsOfActiveForkAreFull>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  CopiedSlotsOfActiveForkAreFull() = default;
  CopiedSlotsOfActiveForkAreFull(unsigned long id, TAG tag,
                                 handshake::BufferLikeOpInterface &bufferOp,
                                 handshake::EagerForkLikeOpInterface &forkOp);
  ~CopiedSlotsOfActiveForkAreFull() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::CSOAFAF;
  }

private:
  std::string forkOp;
  unsigned numEagerForkOutputs;
  std::string bufferOp;
  unsigned bufferSlot;
  inline static const StringLiteral FORK_OP_LIT = "fork_op";
  inline static const StringLiteral NUM_EAGER_OUTPUTS_LIT = "num_eager_outputs";
  inline static const StringLiteral BUFFER_OP_LIT = "buffer_op";
  inline static const StringLiteral BUFFER_SLOT_LIT = "buffer_slot";
};

class PathSingleSentForkOutput : public FormalProperty {
public:
  std::vector<std::string> getForkOps() { return forkOps; }
  std::vector<unsigned> getOutputIdxs() { return outputIdxs; }

  llvm::json::Value extraInfoToJSON() const override;

  static std::unique_ptr<PathSingleSentForkOutput>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  PathSingleSentForkOutput() = default;
  PathSingleSentForkOutput(unsigned long id, TAG tag,
                           std::vector<std::string> &forkOps,
                           std::vector<unsigned> &outputIdxs);
  ~PathSingleSentForkOutput() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::CSOAFAF;
  }

private:
  std::vector<std::string> forkOps;
  std::vector<unsigned> outputIdxs;
  inline static const StringLiteral FORK_OPS_LIT = "fork_ops";
  inline static const StringLiteral OUTPUT_IDXS_LIT = "output_idxs";
};

class FormalPropertyTable {
public:
  FormalPropertyTable() = default;

  LogicalResult addPropertiesFromJSON(StringRef filepath);

  const std::vector<std::unique_ptr<FormalProperty>> &getProperties() const {
    return properties;
  }

  inline bool fromJSON(const llvm::json::Value &value,
                       std::unique_ptr<FormalProperty> &property,
                       llvm::json::Path path) {
    // fromJson internally allocates the correct space for the class with
    // make_unique and returns a pointer
    property = FormalProperty::fromJSON(value, path);

    return property != nullptr;
  }

private:
  /// List of properties.
  std::vector<std::unique_ptr<FormalProperty>> properties;
};

} // namespace dynamatic
