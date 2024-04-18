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
#include "llvm/Support/JSON.h"
#include <map>
#include <string>

namespace dynamatic {

/// Performs a series of regular expression match-and-replace in the input
/// string, and returns the resulting string. Each key-value pair in the
/// `replacements` map represent a regular expression to replace and the string
/// to replace it with, respectively. If a regex exists multiple times in the
/// input, it is replaced every time. Replacements happen one at a time in map
/// iteration order.
std::string
replaceRegexes(StringRef input,
               const std::map<std::string, std::string> &replacements);

/// Substitutes parameters in the input string. The map maps parameter names,
/// which need to be prefixed by a $ symbol to be replaced in the input, to
/// their respective value. Returns the input string after substitutions were
/// performed.
std::string substituteParams(StringRef input,
                             const llvm::StringMap<std::string> &parameters);

class RTLParameter;
class RTLComponent;

/// Abstract base class for RTL parameter types, with optional and customizable
/// type constraints.
class RTLType {
public:
  /// Abstract base class for RTL parameter type constraints. Each concrete RTL
  /// parameter type should subclass it if it wants to define custom
  /// constraints.
  struct Constraints {
    /// Determines whether constraints are satisfied with a specific parameter
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
  /// Reserved parameter names (used during RTL generation).
  static constexpr llvm::StringLiteral DYNAMATIC = StringLiteral("DYNAMATIC"),
                                       OUTPUT_DIR = StringLiteral("OUTPUT_DIR"),
                                       MODULE_NAME =
                                           StringLiteral("MODULE_NAME"),
                                       JSON_CONFIG =
                                           StringLiteral("JSON_CONFIG");

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

/// Helper data-structure used to query for a component/model match between an
/// MLIR operation and an entity parsed from the RTL configuration file. When an
/// RTL component matches, associate it to the match object using
/// `RTLMatch::setMatch` and then concretize it to RTL using
/// `RTLMatch::concretize`.
class RTLMatch {
  friend RTLComponent;

public:
  /// Attribute names under which the RTL component's name and parameters are
  /// stored on an MLIR operation, respectively.
  static constexpr StringLiteral NAME_ATTR = StringLiteral("hw.name"),
                                 PARAMETERS_ATTR =
                                     StringLiteral("hw.parameters");

  /// Fields below set before a match.

  /// The RTL component's name to look for.
  std::string name;
  /// Location at which to report errors.
  Location loc;
  /// If the match is associated to a HW module, references it.
  hw::HWModuleExternOp modOp = nullptr;
  /// If the match is associated to a HW module, maps RTL component's parameter
  /// names to their respective attribute.
  llvm::StringMap<Attribute> parameters;

  /// Fields below set after a match.

  /// Concrete entity name that the RTL component defines, derived from the
  /// entity name in the RTL component description with RTL parameter values
  /// substituted.
  std::string entityName;
  /// Concrete architecture name that the RTL component defines, derived from
  /// the entity name in the RTL component description with RTL parameter values
  /// substituted.
  std::string archName;
  /// Once a match is set, all RTL parameters of the matched components are
  /// encoded as strings in the map.
  llvm::StringMap<std::string> encodedParameters;

  /// Attempts to match external module operation from the HW dialect.
  RTLMatch(hw::HWModuleExternOp modOp);

  /// Attempts to match a component by name, without parameters.
  RTLMatch(StringRef name, Location loc);

  /// Returns the matched RTL component, or nullptr if no match was set.
  const RTLComponent *getMatchedComponent() { return component; }

  /// Matches the RTL component. Should only be called once per `RTLMatch`
  /// object, before calling `concretize`. Fails if at least one of the
  /// component's RTL parameters cannot be encoded to string; succeeds
  /// otherwise.
  LogicalResult setMatch(const RTLComponent &component);

  /// Attempts to concretize the matched RTL component. Generic components are
  /// copied to the output directory while generated components are produced by
  /// the user-provided generator.
  LogicalResult concretize(StringRef dynamaticPath, StringRef outputDir) const;

  /// Attempts to encode an attribute to a string and sets the second argument
  /// to the encoded string. Fails if the attribute type isn't supported for
  /// encoding.
  static LogicalResult encodeParameter(Attribute attr, std::string &value);

private:
  /// Matched RTL component.
  const RTLComponent *component = nullptr;
};

/// Represents an RTL component i.e., a top-level entry in the RTL configuration
/// file. A component maps to an MLIR operation identified by its canonical name
/// and has a set of named, typed, and  optionally constrained RTL parameters. A
/// component has optional timing models attached to it, whose applicability can
/// be restricted by parameter constraints too.
class RTLComponent {
  // Match objects can concretize RTL components and therefore need easy access
  // to their data.
  friend RTLMatch;

public:
  /// Hardware description languages.
  enum class HDL { VHDL, VERILOG };

  /// Denotes whether "arrays of ports" in the component's IO are expressed as
  /// multi-dimensional arrays (`HIERARCHICAL`) or as separate ports with
  /// indices encoded in their names (`FLAT`) as follows: <port-name>_<index>.
  enum class IOKind { HIERARCICAL, FLAT };

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

  /// Determines whether the RTL component has any timing model compatible
  /// with the match object. Returns the first compatible model (in model
  /// list order), if any exists.
  const RTLComponent::Model *getModel(const RTLMatch &match) const;

  /// Returns the list of RTL parameters, in order, which must be provided as
  /// generic parameters during instantiations of this component.
  SmallVector<const RTLParameter *> getGenericParameters() const;

  /// Returns whether the component is concretized using a "generic" RTL
  /// implementation.
  bool isGeneric() const { return !generic.empty(); }

  /// Returns the HDL in which the component is written.
  HDL getHDL() const { return hdl; }

  /// Returns the component's list of RTL dependencies.
  ArrayRef<std::string> getDependencies() const { return dependencies; }

  /// Returns the name of the component port matching the MLIR port name. If the
  /// component does not define any port name remapping this is simply the input
  /// MLIR port name.
  std::string getRTLPortName(StringRef mlirPortName) const;

  /// Returns the name of the component port matching the MLIR port name for the
  /// specific signal type. This is the remapped port name returned by the
  /// non-signal-specific version of that method suffixed by a string
  /// identifying the signal type (e.g., "_valid" for valid signals). Default
  /// suffixes may be overriden on a per-component basis.
  std::string getRTLPortName(StringRef mlirPortName, SignalType type) const;

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

  /// Path to the generic implementation of the component. Supports parameter
  /// substitution.
  std::string generic;
  /// Opaque command to issue when generating when concretizing the component
  /// for a specific set of parameter values. Supports parameter substitution.
  std::string generator;
  /// Name of the RTL entity. For generic components, by default this is the
  /// filename part (without extension) of the components's path. For generated
  /// component, it is the $MODULE_NAME parameter by default, which is provided
  /// at generation time. Supports parameter substitution.
  std::string entityName;
  /// Architecture's name, "arch" be default (only meaningful for VHDL
  /// components). Supports parameter substitution.
  std::string archName = "arch";
  /// HDL in which the component is written.
  HDL hdl = HDL::VHDL;
  /// If defined, instructs an RTL backend to serialize all RTL parameters to a
  /// JSON file at the provided filepath prior to component generation. It only
  /// makes sense for this to be defined for generated components. The filepath
  /// supports parameter substitution.
  std::optional<std::string> jsonConfig;

  /// IO kind used by the component, hierarchical by default.
  IOKind ioKind = IOKind::HIERARCICAL;
  /// Define port renamings for the component compared to the expected names.
  /// Supports wildcard matching with '*' chacacter.
  std::vector<std::pair<std::string, std::string>> ioMap;
  /// Maps each signal type of a dataflow channel to the suffix to use for port
  /// names of this type. For example, if the map contained the
  /// SignalType::VALID -> "valid" association, then it would be assumed that
  /// the valid wire of a channel-typed port with name <PORT-NAME> would be
  /// named <PORT-NAME>_valid in the RTL component.
  std::map<SignalType, std::string> ioChannels;

  /// Returns a pointer to the RTL parameter with a specific name, if it exists.
  RTLParameter *getParameter(StringRef name) const;

  /// Applies any component-specific remapping on a port name coming from MLIR
  /// and returns the remapped port name (which may be identical to the input).
  std::string portRemap(StringRef mlirPortName) const;

  /// If the component's IO type is hierarchical and the port name has indexed
  /// port name (portName == <baseName>_<arrayIdx>), returns true and stores the
  /// two port name components in the last two arguments; otherwise returns
  /// false.
  bool portNameIsIndexed(StringRef portName, StringRef &baseName,
                         size_t &arrayIdx) const;
};

/// ADL-findable LLVM-standard JSON deserializer for a signal type to string
/// mapping.
inline bool fromJSON(const llvm::json::Value &value,
                     std::map<SignalType, std::string> &ioChannels,
                     llvm::json::Path path);

/// ADL-findable LLVM-standard JSON deserializer for a HDL.
inline bool fromJSON(const llvm::json::Value &value, RTLComponent::HDL &hdl,
                     llvm::json::Path path);

/// ADL-findable LLVM-standard JSON deserializer for IO kinds.
inline bool fromJSON(const llvm::json::Value &value, RTLComponent::IOKind &io,
                     llvm::json::Path path);

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

  /// Tries to find an RTL component that is compatible with the match object.
  /// Returns true and records the match in the match object if at least one
  /// such component exist; returns false otherwise. If multiple components are
  /// compatible with the match object, the first one (in JSON-parsing-order) is
  /// matched.
  bool findCompatibleComponent(RTLMatch &match) const;

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

namespace llvm {
namespace json {
/// ADL-findable LLVM-standard JSON deserializer for a pair of strings (expect
/// an object with a single key-value pair).
inline bool fromJSON(const llvm::json::Value &value,
                     std::pair<std::string, std::string> &stringPair,
                     llvm::json::Path path);
} // namespace json
} // namespace llvm

#endif // DYNAMATIC_SUPPORT_RTL_H
