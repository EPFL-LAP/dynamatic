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

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <memory>

namespace dynamatic {

class FormalProperty {

public:
  enum class TAG { OPT, INVAR, ERROR };
  enum class TYPE {
    AOB /* Absence Of Backpressure */,
    VEQ /* Valid EQuivalence */
  };

  TAG getTag() const { return tag; }
  TYPE getType() const { return type; }
  unsigned long getId() const { return id; }

  static std::optional<TYPE> typeFromStr(const std::string &s);
  static std::string typeToStr(TYPE t);
  static std::optional<TAG> tagFromStr(const std::string &s);
  static std::string tagToStr(TAG t);

  llvm::json::Value toJSON() const;

  inline virtual llvm::json::Value extraInfoToJSON() const { return nullptr; };

  FormalProperty() = default;
  FormalProperty(unsigned long id, TAG tag, TYPE type)
      : id(id), tag(tag), type(type), check("unchecked") {}
  virtual ~FormalProperty() = default;

protected:
  unsigned long id;
  TAG tag;
  TYPE type;
  std::string check;
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

  llvm::json::Value extraInfoToJSON() const override;

  AbsenceOfBackpressure() = default;
  AbsenceOfBackpressure(unsigned long id, TAG tag, const OpResult &res);
  ~AbsenceOfBackpressure() = default;

private:
  SignalName ownerChannel;
  SignalName userChannel;
};

class ValidEquivalence : public FormalProperty {
public:
  std::string getOwner() { return ownerChannel.operationName; }
  std::string getTarget() { return targetChannel.operationName; }
  int getOwnerIndex() { return ownerChannel.channelIndex; }
  int getTargetIndex() { return targetChannel.channelIndex; }
  std::string getOwnerChannel() { return ownerChannel.channelName; }
  std::string getTargetChannel() { return targetChannel.channelName; }

  llvm::json::Value extraInfoToJSON() const override;

  ValidEquivalence() = default;
  ValidEquivalence(unsigned long id, TAG tag, const OpResult &res1,
                   const OpResult &res2);
  ~ValidEquivalence() = default;

private:
  SignalName ownerChannel;
  SignalName targetChannel;
};

} // namespace dynamatic