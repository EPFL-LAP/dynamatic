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

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/JSON.h"

#ifndef DYNAMATIC_SUPPORT_RTL_RTLTYPES_H
#define DYNAMATIC_SUPPORT_RTL_RTLTYPES_H

namespace dynamatic {

class RTLParameter;

/// Abstract base class for RTL parameter types, with optional and customizable
/// type constraints.
class RTLType {
public:
  /// Abstract base class for RTL parameter type constraints. Each concrete RTL
  /// parameter type should subclass it if it wants to define custom
  /// constraints.
  struct Constraints {
    /// Determines whether constraints are satisfied with a specific parameter
    /// value stored in an MLIR attribute.
    virtual bool verify(mlir::Attribute attr) const { return true; };

    /// Necessary because of virtual method.
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

  /// Serializes the attribute to a string and returns it unless the attribute
  /// is of an incorrect type, in which case the returned string is empty.
  virtual std::string serialize(mlir::Attribute attr) const = 0;

  /// Determines whether the type's constraints are satisfied with a specific
  /// parameter value stored in the MLIR attribute.
  bool verify(mlir::Attribute attr) const { return constraints->verify(attr); };

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

/// Boolean RTL type, mappable to a `bol` in C++.
class RTLBooleanType : public RTLType {
  using RTLType::RTLType;

public:
  /// Unsigned type constraints.
  struct BoolConstraints : public Constraints {
    /// Equality constraint.
    std::optional<unsigned> eq;
    /// Difference constraint.
    std::optional<unsigned> ne;

    bool verify(mlir::Attribute attr) const override;
  };

  bool constraintsFromJSON(const llvm::json::Object &object,
                           Constraints *&constraints,
                           llvm::json::Path path) override;

  std::string serialize(mlir::Attribute attr) const override;

private:
  /// Keywords
  static constexpr llvm::StringLiteral EQ = llvm::StringLiteral("eq"),
                                       NE = llvm::StringLiteral("ne");

  /// Errors
  static constexpr llvm::StringLiteral ERR_UNSUPPORTED = llvm::StringLiteral(
      R"(unknown unsigned constraint: options are "eq" or "ne")");
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

    bool verify(mlir::Attribute attr) const override;
  };

  bool constraintsFromJSON(const llvm::json::Object &object,
                           Constraints *&constraints,
                           llvm::json::Path path) override;

  std::string serialize(mlir::Attribute attr) const override;

private:
  /// Keywords
  static constexpr llvm::StringLiteral LB = llvm::StringLiteral("lb"),
                                       UB = llvm::StringLiteral("ub"),
                                       RANGE = llvm::StringLiteral("range"),
                                       EQ = llvm::StringLiteral("eq"),
                                       NE = llvm::StringLiteral("ne");

  /// Errors
  static constexpr llvm::StringLiteral
      ERR_ARRAY_FORMAT =
          llvm::StringLiteral("expected array to have [lb, ub] format"),
      ERR_LB = llvm::StringLiteral("lower bound already set"),
      ERR_UB = llvm::StringLiteral("upper bound already set"),
      ERR_UNSUPPORTED = llvm::StringLiteral(
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

    bool verify(mlir::Attribute attr) const override;
  };

  bool constraintsFromJSON(const llvm::json::Object &object,
                           Constraints *&constraints,
                           llvm::json::Path path) override;

  std::string serialize(mlir::Attribute attr) const override;

private:
  /// Keywords
  static constexpr llvm::StringLiteral EQ = llvm::StringLiteral("eq"),
                                       NE = llvm::StringLiteral("ne");

  /// Errors
  static constexpr llvm::StringLiteral ERR_UNSUPPORTED = llvm::StringLiteral(
      R"(unknown string constraint: options are "unsigned" or "string")");
};

/// ADL-findable LLVM-standard JSON deserializer for an RTL parameter pointer.
/// Allocates a concrete RTL type and stores its address in the type.
bool fromJSON(const llvm::json::Value &value, RTLType *&type,
              llvm::json::Path path);

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTL_RTLTYPES_H
