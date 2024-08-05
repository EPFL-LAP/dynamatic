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

/// Abstract base for RTL parameter type constraints. Each concrete RTL
/// parameter type should derive from this to define its own custom constraints.
struct RTLTypeConstraints {
  /// Determines whether constraints are satisfied with a specific parameter
  /// value stored in an MLIR attribute.
  virtual bool verify(mlir::Attribute attr) const = 0;

  /// Attempts to deserialize the constraints from a JSON value. Returns true
  /// when parsing succeeded, false otherwise.
  virtual bool fromJSON(const llvm::json::Object &object,
                        llvm::json::Path path) = 0;

  /// Necessary because of virtual methods.
  virtual ~RTLTypeConstraints() = default;
};

/// Base for constrained RTL parameter types. This implements
/// concept-based polymorphism to hide an instance of a templated concrete RTL
/// type class under a non-templated API.
struct RTLType {
  /// Abstract concept for concrete RTL types.
  struct Concept {
    virtual bool fromJSON(const llvm::json::Object &object,
                          llvm::json::Path path) = 0;
    virtual bool constraintsFromJSON(const llvm::json::Object &object,
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
  template <typename DerivedType, typename ConstraintsType>
  struct Model : public Concept {
    bool fromJSON(const llvm::json::Object &object,
                  llvm::json::Path path) override {
      return constraints.fromJSON(object, path);
    }
    bool constraintsFromJSON(const llvm::json::Object &object,
                             RTLTypeConstraints *&constraints,
                             llvm::json::Path path) override {
      ConstraintsType *cons = new ConstraintsType;
      constraints = cons;
      return cons->fromJSON(object, path);
    }
    bool verify(mlir::Attribute attr) const override {
      return constraints.verify(attr);
    }
    std::string serializeImpl(mlir::Attribute attr) const override {
      return DerivedType::serialize(attr);
    }

    /// Constraints on the concrete RTL type.
    ConstraintsType constraints;
  };

  /// Creates an empty type.
  RTLType() = default;

  /// Attempts to deserialize the RTL type (including its constraints) from the
  /// JSON object. Returns true when parsing succeeded, false otherwise.
  bool fromJSON(const llvm::json::Object &object, llvm::json::Path path);

  /// Attempts to deserialize a new set of RTL type constraints compatible with
  /// the RTL type from the JSON object. On success, the `constraints` object is
  /// allocated on the heap, initialized with the parsed constraints, and the
  /// method returns true. On failure, nothing is allocated and the method
  /// returns false.
  bool constraintsFromJSON(const llvm::json::Object &object,
                           RTLTypeConstraints *&constraints,
                           llvm::json::Path path) {
    return typeConcept->constraintsFromJSON(object, constraints, path);
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

//===----------------------------------------------------------------------===//
// RTLBooleanType
//===----------------------------------------------------------------------===//

/// Boolean type constraints.
struct BooleanConstraints : public RTLTypeConstraints {
  /// Equality constraint.
  std::optional<bool> eq;
  /// Difference constraint.
  std::optional<bool> ne;

  bool verify(mlir::Attribute attr) const override;

  bool fromJSON(const llvm::json::Object &object,
                llvm::json::Path path) override;

private:
  /// Keywords.
  static constexpr llvm::StringLiteral EQ = "eq", NE = "ne";

  /// Errors.
  static constexpr llvm::StringLiteral ERR_UNSUPPORTED =
      R"(unknown unsigned constraint: options are "eq" or "ne")";
};

/// An RTL parameter representing a boolean, stored in the IR as a `BoolAttr`.
struct RTLBooleanType
    : public RTLType::Model<RTLBooleanType, BooleanConstraints> {
  static constexpr llvm::StringLiteral ID = "boolean";

  static std::string serialize(mlir::Attribute attr);
};

//===----------------------------------------------------------------------===//
// RTLUnsignedType
//===----------------------------------------------------------------------===//

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

  bool fromJSON(const llvm::json::Object &object,
                llvm::json::Path path) override;

private:
  /// Keywords.
  static constexpr llvm::StringLiteral LB = "lb", UB = "ub", RANGE = "range",
                                       EQ = "eq", NE = "ne";

  /// Errors.
  static constexpr llvm::StringLiteral
      ERR_ARRAY_FORMAT = "expected array to have [lb, ub] format",
      ERR_LB = "lower bound already set", ERR_UB = "upper bound already set",
      ERR_UNSUPPORTED =
          "unknown unsigned constraint: options are \"lb\", \"ub\", "
          "\"range\", \"eq\", or \"ne\"";
};

/// An RTL parameter representing a positive number, stored in the IR as a
/// `IntegerAttr` with unsigned semantics.
struct RTLUnsignedType
    : public RTLType::Model<RTLUnsignedType, UnsignedConstraints> {
  static constexpr llvm::StringLiteral ID = "unsigned";

  static std::string serialize(mlir::Attribute attr);
};

//===----------------------------------------------------------------------===//
// RTLStringType
//===----------------------------------------------------------------------===//

/// String type constraints.
struct StringConstraints : public RTLTypeConstraints {
  /// Equality constraint.
  std::optional<std::string> eq;
  /// Difference constraint.
  std::optional<std::string> ne;

  bool verify(mlir::Attribute attr) const override;

  bool fromJSON(const llvm::json::Object &object,
                llvm::json::Path path) override;

private:
  /// Keywords.
  static constexpr llvm::StringLiteral EQ = "eq", NE = "ne";

  /// Errors.
  static constexpr llvm::StringLiteral ERR_UNSUPPORTED =
      R"(unknown string constraint: options are "eq" or "ne")";
};

/// An RTL parameter representing a string, stored in the IR as a `StringAttr`.
struct RTLStringType : public RTLType::Model<RTLStringType, StringConstraints> {
  static constexpr llvm::StringLiteral ID = "string";

  static std::string serialize(mlir::Attribute attr);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTL_RTLTYPES_H
