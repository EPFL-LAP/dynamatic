//===- FormalProperty.h -----------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the formal properties of a circuit and their JSON database.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_FORMALPROPERTY_H
#define EXPERIMENTAL_SUPPORT_FORMALPROPERTY_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/PropertyExpr.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dynamatic {

/// A formal property of a circuit: the expression `expr` over the state and
/// signals of the circuit, which either is assumed to hold (an invariant), or
/// is to be checked (e.g., to find out whether a signal can be removed).
struct FormalProperty {
  enum class TAG {
    // A property whose outcome decides whether the circuit can be optimized
    OPT,
    // An invariant, which holds by construction and is assumed when checking
    // the other properties
    INVAR,
    ERROR
  };

  // Kinds of the properties whose outcome is acted upon (by rigidification).
  // The channels they are about are those of the valid signals on result ports
  // in their expression (see `getResultValidSignals`).
  /// Absence of backpressure: the channel is never stalled.
  static constexpr StringLiteral ABSENCE_OF_BACKPRESSURE =
      "AbsenceOfBackpressure";
  /// Valid equivalence: the two channels are always valid in the same cycles.
  static constexpr StringLiteral VALID_EQUIVALENCE = "ValidEquivalence";

  // Kinds of the invariants, which hold by construction (see the documentation
  // of formal properties for their description)
  static constexpr StringLiteral EAGER_FORK_NOT_ALL_OUTPUT_SENT =
      "EagerForkNotAllOutputSent";
  static constexpr StringLiteral COPIED_SLOTS_OF_ACTIVE_FORKS_ARE_FULL =
      "CopiedSlotsOfActiveForksAreFull";
  static constexpr StringLiteral EAGER_FORK_PATH_TOKEN_COPIED_MAXIMUM_ONCE =
      "EagerForkPathTokenCopiedMaximumOnce";
  static constexpr StringLiteral RECONVERGENT_PATH_FLOW =
      "ReconvergentPathFlow";
  static constexpr StringLiteral IOG_SINGLE_TOKEN = "IOGSingleToken";
  static constexpr StringLiteral IOG_CONSECUTIVE_TOKENS =
      "IOGConsecutiveTokens";
  static constexpr StringLiteral ENTRY_TOKEN_ORDER = "EntryTokenOrder";
  static constexpr StringLiteral SINGLE_ENTRY_TOKEN = "SingleEntryToken";
  static constexpr StringLiteral EXIT_TOKEN_ORDER = "ExitTokenOrder";
  static constexpr StringLiteral EXIT_TOKEN_NO_ANCESTORS =
      "ExitTokenNoAncestors";

  /// Unique identifier, which also names the property in the model.
  uint64_t id;
  /// What the property describes (e.g., "AbsenceOfBackpressure").
  std::string kind;
  TAG tag;
  /// Outcome of the model checker, if checked.
  std::optional<bool> check;
  PropertyExpr expr;

  /// The valid signals on result ports in the expression, in pre-order. Each
  /// identifies a channel by the unit producing it and the index of the result.
  std::vector<Signal> getResultValidSignals() const;

  static std::optional<TAG> tagFromStr(StringRef s);
  static StringRef tagToStr(TAG t);
};

llvm::json::Value toJSON(const FormalProperty &property);
bool fromJSON(const llvm::json::Value &value, FormalProperty &property,
              llvm::json::Path path);

class FormalPropertyTable {
public:
  FormalPropertyTable() = default;

  LogicalResult addPropertiesFromJSON(StringRef filepath);

  const std::vector<FormalProperty> &getProperties() const {
    return properties;
  }

private:
  /// List of properties.
  std::vector<FormalProperty> properties;
};

} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_FORMALPROPERTY_H
