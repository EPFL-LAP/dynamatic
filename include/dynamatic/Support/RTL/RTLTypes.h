//===- RTLTypes.h - All supported RTL types ---------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares all supported RTL types for RTL parameters.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/JSON/JSON.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/JSON.h"

#ifndef DYNAMATIC_SUPPORT_RTL_RTLTYPES_H
#define DYNAMATIC_SUPPORT_RTL_RTLTYPES_H

namespace dynamatic {

//===----------------------------------------------------------------------===//
// RTLTypeConstraints and derived types
//===----------------------------------------------------------------------===//

/// Abstract base for RTL parameter type constraints. Each concrete RTL
/// parameter type should derive from this to define its own custom constraints.
struct RTLTypeConstraints {
  /// Determines whether constraints are satisfied with a specific parameter
  /// value stored in an MLIR attribute.
  virtual bool verify(mlir::Attribute attr) const = 0;

  /// Necessary because of virtual methods.
  virtual ~RTLTypeConstraints() = default;
};

/// Boolean type constraints.
struct BooleanConstraints : public RTLTypeConstraints {
  /// Equality constraint.
  std::optional<bool> eq;
  /// Difference constraint.
  std::optional<bool> ne;

  bool verify(mlir::Attribute attr) const override;
};

/// ADL-findable LLVM-standard JSON deserializer for boolean constraints.
bool fromJSON(const llvm::json::Value &value, BooleanConstraints &cons,
              llvm::json::Path path);

/// Unsigned type constraints.
struct UnsignedConstraints : public RTLTypeConstraints {
  /// Lower bound (inclusive).
  std::optional<unsigned> lb;
  /// Upper bound (inclusive).
  std::optional<unsigned> ub;
  /// Equality constraint.
  std::optional<unsigned> eq;
  /// Difference constraint.
  std::optional<unsigned> ne;

  bool verify(mlir::Attribute attr) const override;

  /// Attempts to deserialize the unsigned constraints using the provided
  /// deserializer. Exepcted key names are prefixed using the provided string.
  /// The method does not check for the deserializer's validity.
  json::ObjectDeserializer &deserialize(json::ObjectDeserializer &deserial,
                                        StringRef keyPrefix = {});

  /// Checks whether that the unsigned value honors the constraints.
  bool verify(unsigned value) const;

  /// Returns whether no constraint is set on the object.
  bool unconstrained() const;
};

/// ADL-findable LLVM-standard JSON deserializer for unsigned constraints.
bool fromJSON(const llvm::json::Value &value, UnsignedConstraints &cons,
              llvm::json::Path path);

/// String type constraints.
struct StringConstraints : public RTLTypeConstraints {
  /// Equality constraint.
  std::optional<std::string> eq;
  /// Difference constraint.
  std::optional<std::string> ne;

  bool verify(mlir::Attribute attr) const override;
};

/// ADL-findable LLVM-standard JSON deserializer for string constraints.
bool fromJSON(const llvm::json::Value &value, StringConstraints &cons,
              llvm::json::Path path);

/// Channel type constraints.
struct DataflowConstraints : public RTLTypeConstraints {
  /// Constraints on the data signal's width.
  UnsignedConstraints dataWidth;
  /// Constraints on the total number of extra signals.
  UnsignedConstraints numExtras;
  /// Constraints on the number of extra downstream signals.
  UnsignedConstraints numDownstreams;
  /// Constraints on the number of extra upstream signals.
  UnsignedConstraints numUpstreams;

  bool verify(mlir::Attribute attr) const override;
};

/// ADL-findable LLVM-standard JSON deserializer for channel constraints.
bool fromJSON(const llvm::json::Value &value, DataflowConstraints &cons,
              llvm::json::Path path);

/// Timing constraints.
struct TimingConstraints : public RTLTypeConstraints {
  /// Latency constraints between input/output ports of the same signal type.
  std::map<SignalType, UnsignedConstraints> latencies;

  TimingConstraints();

  bool verify(mlir::Attribute attr) const override;
};

/// ADL-findable LLVM-standard JSON deserializer for channel constraints.
bool fromJSON(const llvm::json::Value &value, TimingConstraints &cons,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// RTLType and derived types
//===----------------------------------------------------------------------===//

/// Base for constrained RTL parameter types. This implements
/// concept-based polymorphism to hide an instance of a templated concrete RTL
/// type class under a non-templated API.
struct RTLType {
  /// Abstract concept for concrete RTL types.
  struct Concept {
    virtual bool fromJSON(const llvm::json::Value &value,
                          llvm::json::Path path) = 0;
    virtual bool constraintsFromJSON(const llvm::json::Value &value,
                                     RTLTypeConstraints *&constraints,
                                     llvm::json::Path path) = 0;
    virtual bool verify(mlir::Attribute attr) const = 0;
    virtual std::string serializeImpl(mlir::Attribute attr) const = 0;
    virtual ~Concept() = default;
  };

  /// Base for concrete RTL parameter types. Each concrete RTL parameter type
  /// should derive from this. The first template parameter should be the
  /// derived type itself (in CRTP fashion) while the second template parameter
  /// should be the concrete type constraints type for the RTL type.
  template <typename DerivedT, typename ConstraintT>
  struct Model : public Concept {
    bool fromJSON(const llvm::json::Value &value,
                  llvm::json::Path path) override {
      return ::dynamatic::fromJSON(value, constraints, path);
    }
    bool constraintsFromJSON(const llvm::json::Value &value,
                             RTLTypeConstraints *&constraints,
                             llvm::json::Path path) override {
      ConstraintT *cons = new ConstraintT;
      constraints = cons;
      return ::dynamatic::fromJSON(value, *cons, path);
    }
    bool verify(mlir::Attribute attr) const override {
      return constraints.verify(attr);
    }
    std::string serializeImpl(mlir::Attribute attr) const override {
      return DerivedT::serialize(attr);
    }

    /// Constraints on the concrete RTL type.
    ConstraintT constraints;
  };

  /// Creates an empty type.
  RTLType() = default;

  /// Attempts to deserialize the RTL type (including its constraints). Returns
  /// true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Value &value, llvm::json::Path path);

  /// Attempts to deserialize a new set of RTL type constraints compatible with
  /// the RTL type. On success, the `constraints` object is
  /// allocated on the heap, initialized with the parsed constraints, and the
  /// method returns true. On failure, nothing is allocated and the method
  /// returns false.
  bool constraintsFromJSON(const llvm::json::Value &value,
                           RTLTypeConstraints *&constraints,
                           llvm::json::Path path) {
    return typeConcept->constraintsFromJSON(value, constraints, path);
  }

  /// Serializes the attribute to a string and returns it unless the attribute
  /// is of an incorrect type, in which case the returned string is empty.
  std::string serialize(mlir::Attribute attr) const {
    return typeConcept->serializeImpl(attr);
  };

  /// Determines whether the type's constraints are satisfied with a specific
  /// parameter value stored in the MLIR attribute.
  bool verify(mlir::Attribute attr) const { return typeConcept->verify(attr); };

  /// Prohibit copy due to dynamic allocation of the underlying concept.
  RTLType(const RTLType &) = delete;

  RTLType(RTLType &&other) noexcept
      : typeConcept(std::exchange(other.typeConcept, nullptr)) {}

  RTLType &operator=(RTLType &&other) noexcept {
    typeConcept = std::exchange(other.typeConcept, nullptr);
    return *this;
  }

  virtual ~RTLType() { delete typeConcept; }

private:
  Concept *typeConcept = nullptr;

  template <typename DerivedType>
  bool allocIf(StringRef typeStr) {
    if (typeStr != DerivedType::ID)
      return false;
    typeConcept = new DerivedType;
    return true;
  }

  template <typename First, typename Second, typename... Others>
  bool allocIf(StringRef typeStr) {
    if (allocIf<First>(typeStr))
      return true;
    return allocIf<Second, Others...>(typeStr);
  }
};

/// An RTL parameter representing a boolean, stored in the IR as a `BoolAttr`.
struct RTLBooleanType
    : public RTLType::Model<RTLBooleanType, BooleanConstraints> {
  static constexpr llvm::StringLiteral ID = "boolean";

  static std::string serialize(mlir::Attribute attr);
};

/// An RTL parameter representing a positive number, stored in the IR as a
/// `IntegerAttr` with unsigned semantics.
struct RTLUnsignedType
    : public RTLType::Model<RTLUnsignedType, UnsignedConstraints> {
  static constexpr llvm::StringLiteral ID = "unsigned";

  static std::string serialize(mlir::Attribute attr);
};

/// An RTL parameter representing a string, stored in the IR as a `StringAttr`.
struct RTLStringType : public RTLType::Model<RTLStringType, StringConstraints> {
  static constexpr llvm::StringLiteral ID = "string";

  static std::string serialize(mlir::Attribute attr);
};

/// An RTL parameter representing a dataflow type (`handshake::ControlType` or
/// `handshake::ChannelType`), stored in the IR as a `TypeAttr`.
struct RTLDataflowType
    : public RTLType::Model<RTLDataflowType, DataflowConstraints> {
  static constexpr llvm::StringLiteral ID = "dataflow";

  static std::string serialize(mlir::Attribute attr);
};

/// An RTL parameter representing timing information, stored in the IR as a
/// `handshake::TimingAttr`.
struct RTLTimingType : public RTLType::Model<RTLTimingType, TimingConstraints> {
  static constexpr llvm::StringLiteral ID = "timing", LATENCY = "-lat";

  /// Serializes timing information into a string.
  /// The output format is the TimingAttr assembly format with single quotes.
  /// E.g., '#handshake<timing {R: 1}>'
  static std::string serialize(mlir::Attribute attr);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTL_RTLTYPES_H
