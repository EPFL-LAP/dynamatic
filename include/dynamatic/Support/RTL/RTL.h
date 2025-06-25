//===- RTL.h - RTL support --------------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL support, linking the IR to a concrete RTL implementation. In particular,
// this defines the parsing logic for the JSON-formatted RTL configuration file
// and data-structures to allow client code to request RTL components matching
// certain characteristics from an RTL configuration. See the formal
// specification for the RTL configuration file and details on the RTL matching
// logic in `docs/Specs/Backend.md`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_RTL_RTL_H
#define DYNAMATIC_SUPPORT_RTL_RTL_H

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTLTypes.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <string>

namespace dynamatic {

/// Hardware description languages.
enum class HDL { VHDL, VERILOG, SMV };

/// Returns the file extension (without a leading '.') for files of the HDL.
StringRef getHDLExtension(HDL hdl);

/// Mapping between parameter names and their respective values.
using ParameterMappings = llvm::StringMap<std::string>;

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
                             const ParameterMappings &parameters);

class RTLMatch;
class RTLParameter;
class RTLComponent;

/// Represents a named RTL parameter of a specific constrained type. This can be
/// moved but not copied due to underlying dynamatic memory allocation.
class RTLParameter {
  /// RTL components need mutable access to their parameters to initialize them
  /// during JSON deserialization.
  friend RTLComponent;

public:
  /// Reserved parameter names (used during RTL generation).
  static constexpr llvm::StringLiteral DYNAMATIC = StringLiteral("DYNAMATIC"),
                                       OUTPUT_DIR = StringLiteral("OUTPUT_DIR"),
                                       MODULE_NAME =
                                           StringLiteral("MODULE_NAME");

  /// Default constructor.
  RTLParameter() = default;

  /// Attempts to deserialize the parameter's state from a JSON value. Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  /// Returns the parameter's name.
  StringRef getName() const { return name; }

  /// Returns the parameter's type.
  const RTLType &getType() const { return type; }

  /// Prohibit copy due to dynamic allocation of the parameter type.
  RTLParameter(const RTLParameter &) = delete;
  /// Prohibit copy due to dynamic allocation of the parameter type.
  RTLParameter &operator=(const RTLParameter &) = delete;

  RTLParameter(RTLParameter &&other) noexcept
      : name(std::move(other.name)), type(std::move(other.type)){};

  RTLParameter &operator=(RTLParameter &&other) noexcept {
    name = std::move(other.name);
    type = std::move(other.type);
    return *this;
  }

private:
  /// The parameter's name.
  std::string name;
  /// The parameter's type.
  RTLType type;
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

/// Models the result of trying to match a constrained RTL parameter with an
/// explicit parameter value.
struct ParamMatch {
  /// Match state.
  enum State {
    DOES_NOT_EXIST,
    FAILED_VERIFICATION,
    FAILED_SERIALIZATION,
    SUCCESS
  } state;

  /// If the match is successful, holds the string-serialized version of the
  /// parameter value.
  std::string serialized = "";

  /// Denotes a parameter value that does not exist.
  static ParamMatch doesNotExist() { return ParamMatch(DOES_NOT_EXIST); }

  /// Denotes a parameter value that exists but failed to match the RTL
  /// parameter's constraints.
  static ParamMatch failedVerification() {
    return ParamMatch(FAILED_VERIFICATION);
  }

  /// Denotes a parameter value that exists and passed the RTL parameter's
  /// constraints but which failed to be serialized to a string.
  static ParamMatch failedSerialization() {
    return ParamMatch(FAILED_SERIALIZATION);
  }

  /// Denotes a successful parameter value match and serialization.
  static ParamMatch success(const llvm::Twine &serial) {
    return ParamMatch(SUCCESS, serial);
  }

protected:
  /// Construts a parameter match object from the state and an optional
  /// serialization for the parameter value.
  ParamMatch(State state, const llvm::Twine &serial = "")
      : state(state), serialized(serial.str()){};
};

/// A parameterized request for RTL components that match certain properties.
/// All of the class's methods are virtual to allow subclasses to define
/// context-specific matching logic depending on where the request originates
/// from.
class RTLRequest {
public:
  /// Location to report errors from.
  Location loc;

  /// Creates an RTL request reporting errors at the provided location.
  RTLRequest(Location loc) : loc(loc){};

  /// Returns the MLIR attribute holding the RTL parameter's value if it exists;
  /// otherwise returns nullptr.
  virtual Attribute getParameter(const RTLParameter &param) const {
    return nullptr;
  }

  /// Attemps to match the request to the component. On success, allocates a
  /// match on the heap and returns it; otherwise returns nullptr.
  virtual RTLMatch *tryToMatch(const RTLComponent &component) const = 0;

  /// Attempts to serialize the request's parameters to a JSON file at the
  /// provided filepath.
  virtual LogicalResult paramsToJSON(const llvm::Twine &filepath) const {
    return failure();
  };

  /// Default destructor.
  virtual ~RTLRequest() = default;
};

/// Request for RTL components matching an opaque MLIR operation.
class RTLRequestFromOp : public RTLRequest {
public:
  /// Creates the request from the opaque MLIR operation to match RTL components
  /// for, and the name that matching RTL components must have.
  RTLRequestFromOp(Operation *op, const llvm::Twine &name);

  /// Retuns the attribute value corresponding to the key named like the
  /// parameter under the `RTLRequest::PARAMETERS_ATTR` attribute dictionary, if
  /// it exists.
  Attribute getParameter(const RTLParameter &param) const override;

  LogicalResult paramsToJSON(const llvm::Twine &filepath) const override;

  /// Matches when the requests's name matched the component's and when the
  /// request's parameters are compatible with the component's parameters.
  RTLMatch *tryToMatch(const RTLComponent &component) const override;

protected:
  /// The name of the component to match.
  std::string name;
  /// The operation to find matches for.
  Operation *op;
  /// Parameter dictionary stored in the operation's attributes.
  DictionaryAttr parameters;

  /// Matches the RTL parameter with the value the request associates to it
  /// (according to `getParameter`). Returns a param match object denoting
  /// whether the match was successful and, if it was not, why it failed.
  ParamMatch matchParameter(const RTLParameter &param) const;

  /// Determines whether the request's parameters are compatible with the
  /// components's parameters, storing name-to-text-serialization mappings in
  /// the process. Succeeds when all component parameters are compatible with
  /// the request, fails othewise.
  LogicalResult areParametersCompatible(const RTLComponent &component,
                                        ParameterMappings &mappings) const;
};

/// Request for RTL components matchin an external hardware module.
class RTLRequestFromHWModule : public RTLRequestFromOp {
public:
  /// Creates the request from the external hardware module to match RTL
  /// components for.
  RTLRequestFromHWModule(hw::HWModuleExternOp modOp);

  RTLMatch *tryToMatch(const RTLComponent &component) const override;

private:
  /// Returns the canonical name of the MLIR operation that was converted into
  /// the external hardware module.
  std::string getName(hw::HWModuleExternOp modOp);
};

/// Request for dependencies of RTL components.
class RTLDependencyRequest : public RTLRequest {
public:
  /// Creates the request from the name of the RTL module to match.
  RTLDependencyRequest(const Twine &moduleName, Location loc);

  /// Matches when the component is generic and the request's module name
  /// matches the component's module name.
  RTLMatch *tryToMatch(const RTLComponent &component) const override;

private:
  /// The name of the RTL module to match.
  std::string moduleName;
};

/// A match between an RTL request and an RTL component, holding information
/// derived from the match and allowing to concretize an RTL implementation for
/// the matched component.
class RTLMatch {
public:
  /// Matched RTL component.
  const RTLComponent *component = nullptr;

  /// Default constructor so that RTL matches can be used as map values.
  RTLMatch() = default;

  /// Constructs a match for an RTL component, associating a mapping between
  /// parameter names and their respective serialized value that were used to
  /// determine the match. The component must outlive the match object.
  RTLMatch(const RTLComponent &component,
           const ParameterMappings &serializedParams);

  /// Returns the RTL component's concrete module name (i.e., with parameter
  /// values substituted).
  StringRef getConcreteModuleName() const { return moduleName; }

  /// Returns the RTL component's concrete architecture name (i.e., with
  /// parameter values substituted).
  StringRef getConcreteArchName() const { return archName; }

  /// Returns name-value mappings for all of the RTL component's generic
  /// parameters, in the order in which the component defines them,
  llvm::MapVector<StringRef, StringRef> getGenericParameterValues() const;

  /// Registers different parameters for each type of extern op.
  /// Temporary function. These parameters should be added to hw.parameters
  /// (generation_params in the future)
  void registerParameters(hw::HWModuleExternOp &modOp);

  void registerPortTypesParameter(hw::HWModuleExternOp &modOp,
                                  llvm::StringRef modName,
                                  hw::ModuleType &modType);
  void registerBitwidthParameter(hw::HWModuleExternOp &modOp,
                                 llvm::StringRef modName,
                                 hw::ModuleType &modType);
  void registerTransparentParameter(hw::HWModuleExternOp &modOp,
                                    llvm::StringRef modName,
                                    hw::ModuleType &modType);
  void registerExtraSignalParameters(hw::HWModuleExternOp &modOp,
                                     llvm::StringRef modName,
                                     hw::ModuleType &modType);
  void registerSelectedDelayParameter(hw::HWModuleExternOp &modOp,
                                      llvm::StringRef modName,
                                      hw::ModuleType &modType);

  /// Attempts to concretize the matched RTL component using the original RTL
  /// request that created the match. Generic components are copied to the
  /// output directory while generated components are produced by the
  /// user-provided generator.
  LogicalResult concretize(const RTLRequest &request, StringRef dynamaticPath,
                           StringRef outputDir) const;

private:
  /// Concrete module name that the RTL component defines, derived from the
  /// module name in the RTL component description with RTL parameter values
  /// substituted.
  std::string moduleName;
  /// Concrete architecture name that the RTL component defines, derived from
  /// the architecture name in the RTL component description with RTL parameter
  /// values substituted.
  std::string archName;
  /// Maps every RTL parameter in the matched RTL component to its value
  /// serialized to string obtained from the RTL request.
  ParameterMappings serializedParams;
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
  /// Denotes whether "arrays of ports" in the component's IO are expressed as
  /// multi-dimensional arrays (`HIERARCHICAL`) or as separate ports with
  /// indices encoded in their names (`FLAT`) as follows: <port-name>_<index>.
  enum class IOKind { HIERARCICAL, FLAT };

  /// A timing model for the component, optionally constrained for specific RTL
  /// parameters.
  struct Model {
    using AddConstraints = std::pair<RTLParameter *, RTLTypeConstraints *>;

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

  /// Determines whether the RTL component has any timing model compatible
  /// with the request. Returns the first compatible model (in model list
  /// order), if any exists.
  const RTLComponent::Model *getModel(const RTLRequest &request) const;

  /// Returns the list of RTL parameters, in order.
  SmallVector<const RTLParameter *> getParameters() const;

  /// Returns the list of RTL parameters, in order, which must be provided as
  /// generic parameters during instantiations of this component.
  SmallVector<const RTLParameter *> getGenericParameters() const;

  /// Returns the component's name.
  StringRef getName() const { return name; }

  /// Returns whether the component is concretized using a "generic" RTL
  /// implementation.
  bool isGeneric() const { return !generic.empty(); }

  /// Returns the component's module name.
  StringRef getModuleName() const { return moduleName; }

  /// Returns the HDL in which the component is written.
  HDL getHDL() const { return hdl; }

  /// Returns the component's list of RTL dependencies.
  ArrayRef<std::string> getDependencies() const { return dependencies; }

  /// Returns the name of the component port matching the MLIR port name and
  /// whether the port name indicates that is part of an array. If the component
  /// does not define any port name remapping this is simply the input MLIR port
  /// name.
  std::pair<std::string, bool> getRTLPortName(StringRef mlirPortName,
                                              HDL hdl) const;

  /// Returns the name of the component port matching the MLIR port name for the
  /// specific signal type and whether the port name indicates that it is part
  /// of an array. This is the remapped port name returned by the
  /// non-signal-specific version of that method suffixed by a string
  /// identifying the signal type (e.g., "_valid" for valid signals). Default
  /// suffixes may be overriden on a per-component basis.
  std::pair<std::string, bool>
  getRTLPortName(StringRef mlirPortName, SignalType signalType, HDL hdl) const;

  RTLComponent(RTLComponent &&) noexcept = default;
  RTLComponent &operator=(RTLComponent &&) noexcept = default;

private:
  /// Canonical name of the MLIR operation corresponding to this component.
  /// Empty if component does not directly correspond to an MLIR operation
  /// (e.g., for component dependencies).
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
  /// Name of the RTL module. For generic components, by default this is the
  /// filename part (without extension) of the components's path. For generated
  /// component, it is the $MODULE_NAME parameter by default, which is provided
  /// at generation time. Supports parameter substitution.
  std::string moduleName;
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
  std::map<SignalType, std::string> ioSignals;

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

  /// After all top-level JSON entries of the component have been deserialized,
  /// check that they are valid and set default values for some members. Returns
  /// false when the JSON description of the component is invalid.
  bool checkValidAndSetDefaults(llvm::json::Path path);
};

/// ADL-findable LLVM-standard JSON deserializer for a signal type to string
/// mapping.
inline bool fromJSON(const llvm::json::Value &value,
                     std::map<SignalType, std::string> &ioChannels,
                     llvm::json::Path path);

/// ADL-findable LLVM-standard JSON deserializer for a HDL.
inline bool fromJSON(const llvm::json::Value &value, HDL &hdl,
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

  /// Determines whether any RTL component is compatible with the request.
  bool hasMatchingComponent(const RTLRequest &request);

  /// Tries to find an RTL component that is compatible with the request.
  /// Returns a heap-allocated match to the first matching component (in
  /// JSON-parsing-order) if one exists; otherwise returs nullptr.
  RTLMatch *getMatchingComponent(const RTLRequest &request);

  /// Finds all RTL components compatible with the request and pushes
  /// corresponding heap-allocated matches to the vector (in
  /// JSON-parsing-order).
  void findMatchingComponents(const RTLRequest &request,
                              std::vector<RTLMatch *> &matches) const;

  /// Determines whether the RTL configuration has any component with a timing
  /// model compatible with the match object. Returns the first compatible model
  /// (in component and model list order), if any exists.
  const RTLComponent::Model *getModel(const RTLRequest &request) const;

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

#endif // DYNAMATIC_SUPPORT_RTL_RTL_H
