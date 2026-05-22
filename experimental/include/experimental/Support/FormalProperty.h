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

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/FlowExpression.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE {
    AbsenceOfBackpressure,
    ValidEquivalence,
    EagerForkNotAllOutputSent,
    CopiedSlotsOfActiveForksAreFull,
    ReconvergentPathFlow,
    IOGSingleToken,
    IOGConsecutiveTokens,
    EntryTokenOrder,
    SingleEntryToken,
  };

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  uint64_t getId() const { return id; }
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
  // Use uint64_t instead of unsigned long because LLVM JSON provides
  // fromJSON(uint64_t&) but not fromJSON(unsigned long&) on some platforms
  // (ex. arm64 macos).
  FormalProperty(uint64_t id, TAG tag, TYPE type)
      : id(id), tag(tag), type(type), check(std::nullopt) {}
  virtual ~FormalProperty() = default;

  static bool classof(const FormalProperty *fp) { return true; }

protected:
  uint64_t id;
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
  AbsenceOfBackpressure(uint64_t id, TAG tag, const OpResult &res);
  ~AbsenceOfBackpressure() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::AbsenceOfBackpressure;
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
  ValidEquivalence(uint64_t id, TAG tag, const OpResult &res1,
                   const OpResult &res2);
  ~ValidEquivalence() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::ValidEquivalence;
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
  std::vector<handshake::EagerForkSentNamer> getSentStateNamers() {
    return sentStateNamers;
  }

  llvm::json::Value extraInfoToJSON() const override;

  static std::unique_ptr<EagerForkNotAllOutputSent>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  EagerForkNotAllOutputSent() = default;
  EagerForkNotAllOutputSent(uint64_t id, TAG tag,
                            handshake::EagerForkLikeOpInterface &op);
  ~EagerForkNotAllOutputSent() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::EagerForkNotAllOutputSent;
  }

private:
  // The `sent` states that cannot be active at the same time
  std::vector<handshake::EagerForkSentNamer> sentStateNamers;
  inline static const StringLiteral OWNER_OP_LIT = "owner_op";
  inline static const StringLiteral CHANNELS_LIT = "channels";
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
  std::vector<handshake::EagerForkSentNamer> getSentStateNamers() {
    return sentStateNamers;
  }
  const handshake::InternalStateNamer &getCopiedSlot() { return *copiedSlot; }

  llvm::json::Value extraInfoToJSON() const override;

  static std::unique_ptr<CopiedSlotsOfActiveForkAreFull>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  CopiedSlotsOfActiveForkAreFull() = default;
  CopiedSlotsOfActiveForkAreFull(uint64_t id, TAG tag,
                                 handshake::BufferLikeOpInterface &bufferOpI,
                                 handshake::EagerForkLikeOpInterface &forkOp);
  CopiedSlotsOfActiveForkAreFull(uint64_t id, TAG tag,
                                 handshake::LatencyInterface &latencyOpI,
                                 handshake::EagerForkLikeOpInterface &forkOp);
  ~CopiedSlotsOfActiveForkAreFull() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::CopiedSlotsOfActiveForksAreFull;
  }

private:
  std::vector<handshake::EagerForkSentNamer> sentStateNamers;
  std::unique_ptr<handshake::InternalStateNamer> copiedSlot;
  inline static const StringLiteral FORK_CHANNELS_LIT = "fork_channels";
  inline static const StringLiteral COPIED_SLOT_LIT = "copied_slot";
};

// A pair of two paths is called reconvergent if they split at the same fork,
// and later reconverge at some join. Both of these paths will contain the same
// number of tokens (although one needs to account for eager forks "generating"
// new tokens when eagerly forwarding a token). Rather than starting at each
// fork and following each path until they reconverge, these reconvergent paths
// are annotated using Gaussian elimination: Each operation describes a local
// equation about the number of tokens that have arrived at each operand, the
// number of tokens that have left at each result, and the number of tokens
// stored within internal state (e.g. a buffer slot or eager fork sent). If, for
// every operation, these local equations are put into a matrix and Gaussian
// elimination is performed, many variables can be eliminated, leaving a few
// equations relating only the internal states. These equations correspond
// exactly to the reconvergent paths.
// See https://ieeexplore.ieee.org/document/10323796 Invariants from
// Reconvergent Paths
class ReconvergentPathFlow : public FormalProperty {
public:
  std::vector<FlowExpression> getEquations() { return equations; }
  void addEquation(const FlowExpression &expr) { equations.push_back(expr); }
  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<ReconvergentPathFlow>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  ReconvergentPathFlow() = default;
  ReconvergentPathFlow(unsigned long id, TAG tag);
  ~ReconvergentPathFlow() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::ReconvergentPathFlow;
  }

private:
  std::vector<FlowExpression> equations;
  inline static const StringLiteral EQUATIONS_LIT = "equations";
};

// An IOG contains a single token at the start (at the entry), and no other
// tokens will ever enter or leave. Tokens can be duplicated by eager forks, but
// they keep track of this within their `sent` state. Because of this, the
// number of occupied slots within an IOG is equal to one plus the number of
// duplicated tokens by fork.
class IOGSingleToken : public FormalProperty {
public:
  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<IOGSingleToken>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  IOGSingleToken() = default;
  IOGSingleToken(unsigned long id, TAG tag,
                 std::vector<std::unique_ptr<InternalStateNamer>> slots,
                 std::vector<EagerForkSentNamer> forks)
      : FormalProperty(id, tag, TYPE::IOGSingleToken), slots(std::move(slots)),
        forks(std::move(forks)) {};
  ~IOGSingleToken() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::IOGSingleToken;
  }

  std::vector<std::unique_ptr<InternalStateNamer>> slots;
  std::vector<EagerForkSentNamer> forks;

private:
  inline static const StringLiteral SLOTS_LIT = "slots";
  inline static const StringLiteral FORKS_LIT = "forks";
};

// Within an IOG, multiple slots can be occupied when forks duplicate tokens.
// Whereas the previous invariant only states that there needs to be an active
// fork somewhere, this invariant determines where the active fork could be: If
// any two slots are occupied, there must be at least one path connecting them,
// and for some path p, there must be an active fork along p with the start of p
// as the copied slot (i.e. no other slots between the fork and the starting
// slot)
class IOGConsecutiveTokens : public FormalProperty {
public:
  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<IOGConsecutiveTokens>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  IOGConsecutiveTokens() = default;
  IOGConsecutiveTokens(unsigned long id, TAG tag, const TokenCountNamer &slot1,
                       const TokenCountNamer &slot2,
                       std::vector<EagerForkSentNamer> sents);
  ~IOGConsecutiveTokens() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::IOGConsecutiveTokens;
  }

  TokenCountNamer slot1;
  TokenCountNamer slot2;
  std::vector<EagerForkSentNamer> sents;

private:
  inline static const StringLiteral SLOT1_LIT = "slot1";
  inline static const StringLiteral SLOT2_LIT = "slot2";
  inline static const StringLiteral SENTS_LIT = "sents";
};

// EntryTokenOrder describes the invariant that, for a path between an entry
// control merge and a following mux operation, only the last effective token
// along that path can be an entry. Intuitively, this is because the entry token
// will only appear a single time when the loop starts, and after that all
// tokens along the path will be looping tokens.
// This invariant is defined by the effective slots along the path from control
// merge to mux `slots`, and by the value of the entry token `entryValue` (which
// corresponds to the index of the control merge's input connected to the
// entry path).
// e.g. if the path from start -> cmerge arrives at the 0th input of cmerge,
// `entryValue` will be 0
//
// clang-format off
//
// Example: fir (using fpl22)
//
// %0:3 = fork [3] %arg4 {...} : <>
// %result, %index = control_merge [%0#2, %trueResult_6]  {...} : [<>, <>] to <>, <i1>
// %16:2 = fork [2] %index {...} : <i1>
// %10 = buffer %16#1
// %11 = buffer %10
// %12 = buffer %11
// %15 = mux %12 [%3#1, %14] {...} : <i1>, [<i32>, <i32>] to <i32>
//
// Annotates the path consisting of effective slots:
// slot (copied sent)
//
// control_merge0.slot_full (fork3.outs_1_sent)
// buffer4.outs_valid_i
// buffer5.full
// buffer6.full
//
// The entry value is 0, as %0#2 comes from the entry.
//
// clang-format on
class EntryTokenOrder : public FormalProperty {
public:
  const std::vector<EffectiveSlotNamer> &getSlots() const { return slots; }
  int32_t getValue() const { return entryValue; }
  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<EntryTokenOrder>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);
  EntryTokenOrder() = default;
  EntryTokenOrder(unsigned long id, TAG tag,
                  std::vector<EffectiveSlotNamer> slots, int32_t value)
      : FormalProperty(id, tag, TYPE::EntryTokenOrder), slots(std::move(slots)),
        entryValue(value) {}
  ~EntryTokenOrder() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::EntryTokenOrder;
  }

private:
  std::vector<EffectiveSlotNamer> slots;
  int32_t entryValue;
  inline static const StringLiteral SLOTS_LIT = "slots";
  inline static const StringLiteral ENTRY_VALUE_LIT = "entry_value";
};

// SingleEntryToken describes the invariant that only a single token is emitted
// from the entry unit. This means that, if any path between an entry cmerge and
// a subsequent mux operation contains a token (regardless of whether it is an
// entry token or a loop token), the path between the entry and cmerge must be
// empty. This is because both entry tokens and loop tokens only exist after a
// token has propagated from entry to cmerge.
// This invariant is represented by the path from entry to cmerge `ec`, and the
// path from cmerge to mux `cm`. Effective slots are used to ensure that
// duplicated tokens are not counted double
//
// clang-format off
//
// Example: fir (using fpl22)
//
// %0:3 = fork [3] %arg4 {...} : <>
// %result, %index = control_merge [%0#2, %trueResult_6]  {...} : [<>, <>] to <>, <i1>
// %16:2 = fork [2] %index {...} : <i1>
// %10 = buffer %16#1
// %11 = buffer %10
// %12 = buffer %11
// %15 = mux %12 [%3#1, %14] {...} : <i1>, [<i32>, <i32>] to <i32>
//
// Annotates the following paths consisting of effective slots
// slot (copied sent)
//
// cmerge -> mux path:
//
// control_merge0.slot_full (fork3.outs_1_sent)
// buffer4.outs_valid_i
// buffer5.full
// buffer6.full
//
//
// entry -> cmerge path:
//
// start_valid (fork0.outs_2_sent)
//
// clang-format on
class SingleEntryToken : public FormalProperty {
public:
  const std::vector<EffectiveSlotNamer> &getEcPath() const { return ec; }
  const std::vector<EffectiveSlotNamer> &getCmPath() const { return cm; }

  llvm::json::Value extraInfoToJSON() const override;
  static std::unique_ptr<SingleEntryToken>
  fromJSON(const llvm::json::Value &value, llvm::json::Path path);
  SingleEntryToken() = default;
  SingleEntryToken(unsigned long id, TAG tag,
                   std::vector<EffectiveSlotNamer> ec,
                   std::vector<EffectiveSlotNamer> cm)
      : FormalProperty(id, tag, TYPE::SingleEntryToken), ec(std::move(ec)),
        cm(std::move(cm)) {}
  ~SingleEntryToken() = default;

  static bool classof(const FormalProperty *fp) {
    return fp->getType() == TYPE::SingleEntryToken;
  }

private:
  std::vector<EffectiveSlotNamer> ec;
  std::vector<EffectiveSlotNamer> cm;
  inline static const StringLiteral PATH_EC_LIT = "entry_cmerge_path";
  inline static const StringLiteral PATH_CM_LIT = "cmerge_mux_path";
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
