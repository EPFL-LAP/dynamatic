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
#include "llvm/Support/JSON.h"
#include <map>
#include <string>

namespace dynamatic {

class RTLParameter;
class RTLComponent;

/// Abstract base class for RTL parameter types, with optional and customizable
/// type constraints.
class RTLType {
public:
  /// Abstract base class for RTL parameter type constraints. Each concrete RTL
  /// parameter type should subclass if it wants to define custom constraints.
  struct Constraints {
    /// Determines whether constraints are satisified with a specific parameter
    /// value encoded as a string.
    virtual bool verify(StringRef paramValue) const { return true; };

    /// Necessary because of virtual function.
    virtual ~Constraints() = default;
  };

  /// Default constructor.
  RTLType() = default;

  /// Allocates the constraints object and attemps to deserialize its content
  /// from the JSON object. Returns whether constraints were able to be
  /// deserialized from the JSON data.
  virtual bool constraintsFromJSON(const llvm::json::Object &object,
                                   Constraints *&constraints,
                                   llvm::json::Path path) = 0;

  /// Prohibit copy due to dynamic allocation of constraints.
  RTLType(const RTLType &) = delete;
  /// Prohibit copy due to dynamic allocation of constraints.
  RTLType operator=(const RTLType &) = delete;

  RTLType(RTLType &&other) noexcept
      : constraints(std::exchange(other.constraints, nullptr)) {}

  RTLType &operator=(RTLType &&other) noexcept {
    constraints = std::exchange(other.constraints, nullptr);
    return *this;
  }

  /// Deallocates the constraints if they are not `nullptr`.
  virtual ~RTLType();

  /// RTL parameters need mutable access to their type constraints to initialize
  /// them during JSON deserialization.
  friend RTLParameter;

protected:
  /// Type constraints (dynamically allocated).
  Constraints *constraints = nullptr;
};

/// Unsigned RTL type, mappable to an `unsigned` in C++.
class RTLUnsignedType : public RTLType {
  using RTLType::RTLType;

public:
  /// Unsigned type constraints.
  struct UnsignedConstraints : public Constraints {
    /// Lower bound (inclusive).
    std::optional<unsigned> lb;
    /// Upper bound (inclusive).
    std::optional<unsigned> ub;
    /// Equality constraint.
    std::optional<unsigned> eq;
    /// Difference constraint.
    std::optional<unsigned> ne;

    bool verify(StringRef paramValue) const override;
  };

  bool constraintsFromJSON(const llvm::json::Object &object,
                           Constraints *&constraints,
                           llvm::json::Path path) override;

  /// Encodes an unsigned to a string.
  static std::string encode(unsigned value) { return std::to_string(value); }

  /// Decodes a string into an optional unsigned if the string not represents a
  /// valid non-negative number. Otherwise, `std::nullopt` is returned.
  static std::optional<unsigned> decode(StringRef value) {
    if (llvm::any_of(value.str(), [](char c) { return !isdigit(c); }))
      return {};
    return std::stoi(value.str());
  }

private:
  /// Keywords
  static constexpr StringLiteral LB = StringLiteral("lb"),
                                 UB = StringLiteral("ub"),
                                 RANGE = StringLiteral("range"),
                                 EQ = StringLiteral("eq"),
                                 NE = StringLiteral("ne");

  /// Errors
  static constexpr StringLiteral
      ERR_ARRAY_FORMAT =
          StringLiteral("expected array to have [lb, ub] format"),
      ERR_LB = StringLiteral("lower bound already set"),
      ERR_UB = StringLiteral("upper bound already set"),
      ERR_UNSUPPORTED = StringLiteral(
          "unknown unsigned constraint: options are \"lb\", \"ub\", "
          "\"range\", \"eq\", or \"ne\"");
};

/// String RTL type, mappable to a `std::string` in C++.
class RTLStringType : public RTLType {
  using RTLType::RTLType;

public:
  /// String type constraints.
  struct StringConstraints : public Constraints {
    /// Equality constraint.
    std::optional<std::string> eq;
    /// Difference constraint.
    std::optional<std::string> ne;

    bool verify(StringRef paramValue) const override;
  };

  bool constraintsFromJSON(const llvm::json::Object &object,
                           Constraints *&constraints,
                           llvm::json::Path path) override;

  /// Returns the input string (method necessary to enable identical templated
  /// APIs for subtypes of `RTLType`).
  static std::string encode(StringRef value) { return value.str(); }

  /// Returns the input string (method necessary to enable identical templated
  /// APIs for subtypes of `RTLType`).
  static std::optional<std::string> decode(StringRef value) {
    return value.str();
  }

private:
  /// Keywords
  static constexpr StringLiteral EQ = StringLiteral("eq"),
                                 NE = StringLiteral("ne");

  /// Errors
  static constexpr StringLiteral ERR_UNSUPPORTED = StringLiteral(
      R"(unknown string constraint: options are "unsigned" or "string")");
};

/// ADL-findable LLVM-standard JSON deserializer for an RTL parameter pointer.
/// Allocates a concrete RTL type and stores its address in the type.
bool fromJSON(const llvm::json::Value &value, RTLType *&type,
              llvm::json::Path path);

/// Represents a named RTL parameter of a specific type, with optional
/// constraints on allowed values for it. This can be moved but not copied due
/// to underlying dynamatic memory allocation.
class RTLParameter {
public:
  /// Default constructor.
  RTLParameter() = default;

  /// Attempts to deserialize the parameter's state from a JSON value. Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  /// Verifies whether the parameter value satisifies all the parameter's
  /// constraints.
  bool verifyConstraints(StringRef paramValue) const {
    return type->constraints->verify(paramValue);
  }

  /// Returns the parameter's name.
  StringRef getName() const { return name; }

  /// Returns the parameter's type.
  const RTLType &getType() const { return *type; }

  /// Prohibit copy due to dynamic allocation of the parameter type.
  RTLParameter(const RTLParameter &) = delete;
  /// Prohibit copy due to dynamic allocation of the parameter type.
  RTLParameter &operator=(const RTLParameter &) = delete;

  RTLParameter(RTLParameter &&other) noexcept
      : name(other.name), type(std::exchange(other.type, nullptr)){};

  RTLParameter &operator=(RTLParameter &&other) noexcept {
    name = other.name;
    type = std::exchange(other.type, nullptr);
    return *this;
  }

  /// Deallocates the parameter if it is not `nullptr`.
  ~RTLParameter();

  /// RTL components need mutable access to their parameters to initialize them
  /// during JSON deserialization.
  friend RTLComponent;

private:
  /// The parameter's name.
  std::string name;
  /// The parameter's type.
  RTLType *type = nullptr;
  /// Whether the parameter should be used as a generic parameter during
  /// component instantiation. If none, the RTL parameter will be used as a
  /// generic parameter if and only if the associated RTL component is marked
  /// generic.
  std::optional<bool> useAsGeneric;
};

/// ADL-findable LLVM-standard JSON deserializer for an RTL parameter.
inline bool fromJSON(const llvm::json::Value &value, RTLParameter &parameter,
                     llvm::json::Path path) {
  return parameter.fromJSON(value, path);
}

/// Helper data-structure used to qeury for a component/model match between an
/// MLIR operation and an entity parsed from the RTL configuration file. Should
/// rarely be instantiated on its own; instead, its one-argument constructor
/// can implicitly perform the conversion when calling methods expecting an
/// instance of this struct.
struct RTLMatch {
  /// Attribute names under which the RTL component's name and parameters are
  /// stored on an MLIR operation, respectively.
  static constexpr StringLiteral NAME_ATTR = StringLiteral("hw.name"),
                                 PARAMETERS_ATTR =
                                     StringLiteral("hw.parameters");

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
  /// A timing model for the component, optionally constrained for specific RTL
  /// parameters.
  struct Model {
    using AddConstraints = std::pair<RTLParameter *, RTLType::Constraints *>;

    /// Path of the timing model on disk.
    std::string path;
    /// Additional parameter constraints under which the timing model is
    /// applicable.
    SmallVector<AddConstraints> addConstraints;

    /// Default constructor.
    Model() = default;

    Model(Model &&other) noexcept = default;
    Model &operator=(Model &&other) noexcept = default;

    /// Deallocates additional constraint objects.
    ~Model();
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
  std::string name;
  /// The component's RTL parameters.
  std::vector<RTLParameter> parameters;
  /// Maps each unique parameter name to its parameter object for easy access
  /// when concretizing components.
  llvm::StringMap<RTLParameter *> nameToParam;

  /// The component's timing models.
  SmallVector<Model> models;
  /// The component's dependencies (referenced by name).
  std::vector<std::string> dependencies;

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

/// Performs a series of regular expression match-and-replace in the input
/// string, and returns the resulting string. Each key-value pair in the
/// `replacements` map represent a regular expression to replace and the string
/// to replace it with, respectively. If a regex exists multiple times in the
/// input, it is replaced every time. Replacements happen one at a time in map
/// iteration order.
std::string
replaceRegexes(StringRef input,
               const std::map<std::string, std::string> &replacements);

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTL_H
