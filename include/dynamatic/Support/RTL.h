//===- RTL.h - RTL support --------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the data structures and logic related to the JSON-parsing logic
// for the RTL configuration file. See the formal specification for the JSON
// file in `docs/Specs/RTLConfiguration.md`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_RTL_H
#define DYNAMATIC_SUPPORT_RTL_H

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace dynamatic {

/// An abstract constraint on an RTL parameter. Children should override the
/// pure virtual `apply` and `fromJSON` methods to define a custom constraint.
struct ParameterConstraint {
  /// Returns whether the constraint is satisfied for a specific parameter vale
  /// (encoded as a string).
  virtual bool apply(StringRef paramValue) = 0;

  /// JSON deserializer for the constraint. Same semantics as the LLVM-standard
  /// `fromJSON(const json::Value&, T&, Path) -> bool` functions, but defined on
  /// the constraint object itself so that we can make polymorphic calls to the
  /// method.
  virtual bool fromJSON(const llvm::json::Value &value,
                        llvm::json::Path path) = 0;

  virtual ~ParameterConstraint() = default;
};

/// JSON deserializer for a generic parameter constraint, calling the
/// corresponding virtual deserializer of the actual constraint type.
inline bool fromJSON(const llvm::json::Value &value,
                     ParameterConstraint &constraint, llvm::json::Path path) {
  return constraint.fromJSON(value, path);
}

/// An abstract constraint on an RTL parameter of `unsigned` type.
struct UnsignedConstraint : public ParameterConstraint {
  /// Error message to print when the constraint's keyword cannot be recognized.
  static constexpr StringLiteral ERR_UNSUPPORTED =
      StringLiteral("unknown unsigned constraint: options are \"lb\", \"ub\", "
                    "\"range\", \"eq\", or \"ne\"");

  bool apply(StringRef paramValue) override;

protected:
  /// Overloads of parent's `apply` method that takes as input an unsigned
  /// number rather than a string, so that concrete unsigned constraints can
  /// directly operate on an unsigned number.
  virtual bool apply(unsigned paramValue) = 0;

  virtual ~UnsignedConstraint() = default;
};

/// Lower bound (inclusive) constraint on an RTL parameter of unsigned type.
struct UnsignedLowerBound : public UnsignedConstraint {
  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, lb, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "lb"; }

protected:
  bool apply(unsigned paramValue) override { return paramValue >= lb; }

private:
  /// The lower bound.
  unsigned lb;
};

/// Upper bound (inclusive) constraint on an RTL parameter of unsigned type.
struct UnsignedUpperBound : public UnsignedConstraint {
  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, ub, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "ub"; }

protected:
  bool apply(unsigned paramValue) override { return paramValue <= ub; }

private:
  /// The upper bound.
  unsigned ub;
};

/// Range (inclusive on both sides) constraint on an RTL parameter of unsigned
/// type.
struct UnsignedRange : public UnsignedConstraint {

  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path) override;

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "range"; }

protected:
  bool apply(unsigned paramValue) override {
    return lb <= paramValue && paramValue <= ub;
  }

private:
  /// The lower bound.
  unsigned lb;
  /// The upper bound.
  unsigned ub;

  static constexpr StringLiteral ERR_ARRAY_FORMAT =
      StringLiteral("expected array to have [lb, ub] format");
};

/// Equality constraint on an RTL parameter of unsigned type.
struct UnsignedEqual : public UnsignedConstraint {
  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, val, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "eq"; }

protected:
  bool apply(unsigned paramValue) override { return paramValue == val; }

private:
  /// The value to compare to.
  unsigned val;
};

/// Difference constraint on an RTL parameter of unsigned type.
struct UnsignedDifferent : public UnsignedConstraint {
  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, val, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "ne"; }

protected:
  bool apply(unsigned paramValue) override { return paramValue != val; }

private:
  /// The value to compare to.
  unsigned val;
};

/// An abstract constraint on an RTL parameter of `string` type.
struct StringConstraint : public ParameterConstraint {
  static constexpr StringLiteral ERR_UNSUPPORTED = StringLiteral(
      R"(unknown string constraint: options are "unsigned" or "string")");
};

/// Equality constraint on an RTL parameter of string type.
struct StringEqual : public StringConstraint {
  bool apply(StringRef paramValue) override { return paramValue == str; }

  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, str, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "eq"; }

private:
  /// The string to compare to.
  std::string str;
};

/// Difference constraint on an RTL parameter of string type.
struct StringDifferent : public StringConstraint {
public:
  bool apply(StringRef paramValue) override { return paramValue != str; }

  bool fromJSON(const llvm::json::Value &value,
                llvm::json::Path path) override {
    return llvm::json::fromJSON(value, str, path);
  }

  /// Returns the JSON key associated to the constraint.
  static StringRef getKey() { return "ne"; }

private:
  /// The string to compare to.
  std::string str;
};

/// A vector of parameter constraints. Every constraint added the the vector
/// should be dynamically allocated; they will be freed when the object goes out
/// of scope.
struct ConstraintVector {
  /// Constraint vector, stored as pointers because the class is abstract.
  SmallVector<ParameterConstraint *> constraints;

  /// Default constructor.
  ConstraintVector() = default;

  /// Verifies whether the parameter value satisifies all the constraints.
  bool verifyConstraints(StringRef paramValue) const;

  /// Object can't be copied due to memory allocation.
  ConstraintVector(const ConstraintVector &) = delete;
  /// Object can't be copy-assigned due to memory allocation.
  ConstraintVector &operator=(const ConstraintVector &) = delete;

  ConstraintVector(ConstraintVector &&other) noexcept = default;
  ConstraintVector &operator=(ConstraintVector &&other) noexcept = default;

  /// Deletes all constraints stored in the vector.
  ~ConstraintVector() {
    for (ParameterConstraint *param : constraints)
      delete param;
  }
};

/// Represents a named RTL parameter of a specific type, with optional
/// constraints on allowed values for it. This can be moved but not copied due
/// to underlying dynamatic memory allocation.
class RTLParameter {
public:
  /// RTL parameter type.
  enum class Type { UNSIGNED, STRING };

  /// Default constructor.
  RTLParameter() = default;

  /// Attempts to deserialize the parameter's state from a JSON value. Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  /// Verifies whether the parameter value satisifies all the parameter's
  /// constraints.
  bool verifyConstraints(StringRef paramValue) const {
    return constraints.verifyConstraints(paramValue);
  }

  /// Return's the parameter's name.
  StringRef getName() const { return name; }

  /// Return's the parameter's type.
  RTLParameter::Type getType() const { return type; }

  RTLParameter(RTLParameter &&) noexcept = default;
  RTLParameter &operator=(RTLParameter &&) noexcept = default;

private:
  /// The parameter's name.
  std::string name;
  /// The parameter's type.
  RTLParameter::Type type;
  /// A list of optional constraints for the parameter.
  ConstraintVector constraints;
};

/// ADL-findable LLVM-standard JSON deserializer for an RTL parameter's type.
bool fromJSON(const llvm::json::Value &value, RTLParameter::Type &type,
              llvm::json::Path path);

/// ADL-findable LLVM-standard JSON deserializer for an RTL parameter.
inline bool fromJSON(const llvm::json::Value &value, RTLParameter &parameter,
                     llvm::json::Path path) {
  return parameter.fromJSON(value, path);
}

/// ADL-findable LLVM-standard JSON deserializer for a vector of RTL parameters.
bool fromJSON(const llvm::json::Value &value,
              SmallVector<RTLParameter> &parameters, llvm::json::Path path);

/// Helper data-structure used to qeury for a component/model match between an
/// MLIR operation and an entity parsed from the RTL configuration file. Should
/// rarely be instantiated on its own; instead, its one-argument constructor
/// can implicitly perform the conversion when calling methods expecting an
/// instance of this struct.
struct RTLMatch {
  /// The MLIR operation we are trying to match an RTL component/model for.
  Operation *op;
  /// The RTL component's name to look for.
  std::string name;
  /// Maps RTL component's parameter names to their respective value.
  llvm::StringMap<std::string> parameters;
  /// Whether the match is possible altohgether. If this is `true`, all matching
  /// methods should "fail" with this object.
  bool invalid = true;

  /// Constructs a match from an external module operation from the HW dialect.
  RTLMatch(dynamatic::hw::HWModuleExternOp modOp);
};

/// Represents an RTL component i.e., a top-level entry in the RTL configuration
/// file. A component maps to an MLIR operation identified by its canonical name
/// and has a set of named, typed, and  optionally constrained RTL parameters. A
/// component has optional timing models attached to it, whose applicability can
/// be restricted by parameter constraints too.
class RTLComponent {
public:
  /// A timing model for the component, optionally constrained.
  struct Model {
    /// Path of the timing model on disk.
    std::string path;
    /// Additional parameter constraints under which the timing model is
    /// applicable.
    SmallVector<std::pair<RTLParameter *, ConstraintVector>> constraints;

    /// Default constructor.
    Model() = default;

    Model(Model &&other) noexcept = default;
    Model &operator=(Model &&other) noexcept = default;
  };

  /// Default constructor.
  RTLComponent() = default;

  /// Attempts to deserialize the component's state from a JSON value. Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  /// Determines whether the RTL component is compatible with the match object.
  bool isCompatible(const RTLMatch &match) const;

  /// Determines whether the RTL component has any timing model compatible with
  /// the match object. Returns the first compatible model (in model list
  /// order), if any exists.
  const RTLComponent::Model *getModel(const RTLMatch &match) const;

  RTLComponent(RTLComponent &&) noexcept = default;
  RTLComponent &operator=(RTLComponent &&) noexcept = default;

private:
  /// Canonical name of the MLIR operation corresponding to this component.
  std::string component;
  /// The component's RTL parameters.
  SmallVector<RTLParameter> parameters;

  /// Maps each unique parameter name to its parameter object for easy access
  /// when concretizing components.
  llvm::StringMap<RTLParameter *> nameToParam;

  /// The component's timing models.
  SmallVector<Model> models;

  /// Path to the generic implementation of the component.
  std::string generic;
  /// Opaque command to issue when generating when concretizing the component
  /// for a specific set of parameter values.
  std::string generator;

  /// Returns a pointer to the RTL parameter with a specific name, if it exists.
  RTLParameter *getParameter(StringRef name) const;
};

/// ADL-findable LLVM-standard JSON deserializer for an RTL component.
inline bool fromJSON(const llvm::json::Value &value, RTLComponent &component,
                     llvm::json::Path path) {
  return component.fromJSON(value, path);
}

/// Represents the content of one or more RTL configuration files; essentially a
/// list of RTL component descriptions along with a matching logic to select a
/// description (if any) that can concretize a specific component configuration.
class RTLConfiguration {
public:
  /// Default constructor.
  RTLConfiguration() = default;

  /// Attempts to deserialize the JSON-formatted RTL configuration file at the
  /// provided location and add all its component descriptions to the existing
  /// list.
  LogicalResult addComponentsFromJSON(StringRef filepath);

  /// Determines whether the RTL configuration has any compatible component with
  /// the match object. Returns the first compatible component (in component
  /// list order), if any exists.
  const RTLComponent *getComponent(const RTLMatch &match) const;

  /// Determines whether the RTL configuration has any component with a timing
  /// model compatible with the match object. Returns the first compatible model
  /// (in component and model list order), if any exists.
  const RTLComponent::Model *getModel(const RTLMatch &match) const;

  RTLConfiguration(RTLConfiguration &&) noexcept = default;
  RTLConfiguration &operator=(RTLConfiguration &&) noexcept = default;

private:
  /// List of RTL component descriptions parsed from one or many RTL
  /// configuration files.
  std::vector<RTLComponent> components;
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTL_H
