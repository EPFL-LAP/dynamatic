//===- NumericAnalysis.h - Numeric analyis utilities ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Numeric analysis infrastructure. Its main purpose is to try to estimate the
// numeric range in which integer-like or floating-like SSA values belong in scf
// or cf-level IR.
//
// At the moment, this only really works well for simple cases. In more complex
// scenarios, the estimated ranges will be widely too conservative (e.g.,
// determining that a value is "unbounded"). There is a lot of room for
// improvements, both in the analysis method itself and in the "coverage" of
// MLIR operations that it can analyze. However, it already allows us to make
// simple queries about e.g., loop iterators, where we sometime care to prove
// that they are always positive.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace dynamatic {

/// A numeric range, representing a closed, half-open (i.e., unbounded on one
/// side), or open (i.e., unbounded on both sides) interval in which an integer
/// or float-like SSA value must belong. The class defines the logic for
/// creating these ranges, querying them, and combining them together in
/// different ways.
///
/// This class implements type erasure for the underlying `NumericRangeModel<T>`
/// class, which is templated by the type of the interval's bounds. This is
/// required to be able to use (mostly) the same untemplated API in client code
/// for representing integer and floating-point ranges.
///
/// NOTE: Right now the class uses static_cast everywhere, which is problematic
/// in case of a mistake in client-code. This should be replaced by dyn_cast
/// with runtime checks at some point.
class NumericRange {
public:
  struct NumericRangeConcept;
  template <typename T>
  struct NumericRangeModel;

  /// Underlying bound-type for ranges associated to SSA values with
  /// integer-like types.
  using IntBound = int64_t;
  /// Underlying bound-type for ranges associated to SSA values with
  /// floating-point-like types.
  using FloatBound = double;

  /// Default constructor, constructing an "uninitialized" range. Client-code
  /// outside the analysis should never need to construct such a range. Most of
  /// the querying methods will assert if they are called on an "uninitialized
  /// range".
  NumericRange() = default;

  /// Constructs an integer range.
  NumericRange(std::optional<IntBound> lb, std::optional<IntBound> ub)
      : concept(new NumericRangeModel<IntBound>(lb, ub)) {};

  /// Constructs an floating-point range.
  NumericRange(std::optional<FloatBound> lb, std::optional<FloatBound> ub)
      : concept(new NumericRangeModel<FloatBound>(lb, ub)) {};

  /// Determine whether the range is positive i.e., whether its lower bound is
  /// greater than or equal to 0.
  inline bool isPositive() const {
    assert(concept && "range is uninitialized");
    return concept->isPositive();
  }

  /// Returns an inverted range. If the receiving range is closed or fully open,
  /// then the returned one will be fully open too, otherwise it will be
  /// half-open.
  inline NumericRange invert() const {
    assert(concept && "range is uninitialized");
    return concept->invert();
  }

  /// Returns whether the `other` range is strictly greater than the receiving
  /// one i.e, whether its lower bound is strictly greater than the receiving
  /// range's upper bound. If any of these two bounds is std::nullopt, returns
  /// false.
  inline bool operator<(const NumericRange &other) {
    assert(concept && other.concept && "range is uninitialized");
    return concept->operator<(other);
  }

  /// Returns whether the `other` range is equal to the receiving one i.e.,
  /// whether their respective lower bounds and upper bounds are equal.
  inline bool operator==(const NumericRange &other) {
    assert(concept && other.concept && "range is uninitialized");
    return concept->operator==(other);
  }

  /// Returns whether the `other` range is greater than or equal to the
  /// receiving one i.e, whether its lower bound is greater than or equal to the
  /// receiving range's upper bound. If any of these two bounds is std::nullopt,
  /// returns false.
  inline bool operator<=(const NumericRange &other) {
    return *this < other || *this == other;
  }

  /// Returns the range's lower bound.
  template <typename T>
  std::optional<T> getLb() const;

  /// Returns the range's upper bound.
  template <typename T>
  std::optional<T> getUb() const;

  /// Returns a reference to the underlying range model.
  template <typename T>
  const NumericRangeModel<T> &getRange() const;

  /// Creates a copy of the range which will internally reference the same
  /// underlying range object.
  NumericRange(const NumericRange &other) = default;

  /// Copy-assigns a copy of the range which will internally reference the same
  /// underlying range object.
  NumericRange &operator=(const NumericRange &other) = default;

  /// Creates an unbounded range whose underlying bound type is determined by
  /// the provided MLIR type.
  static NumericRange unbounded(Type type);

  /// Creates a positive range [0, inf) whose underlying bound type is
  /// determined by the provided MLIR type.
  static NumericRange positive(Type type);

  /// Creates a range encompassing a single value [cst, cst] whose underlying
  /// bound type is determined by the provided MLIR type attribute.
  static NumericRange exactValue(mlir::TypedAttr attr);

  /// Adds two ranges together, producing a new range whose bounds result from
  /// the addition of the corresponding bounds in the two given ranges.
  static NumericRange add(const NumericRange &lhs, const NumericRange &rhs);

  /// Computes the intersection between two ranges.
  static NumericRange rangeIntersect(const NumericRange &lhs,
                                     const NumericRange &rhs);

  /// Computes the union between two ranges.
  static NumericRange rangeUnion(const NumericRange &lhs,
                                 const NumericRange &rhs);

  /// Concept (abstract) class for the underlying numeric range so that we can
  /// store an untemplated pointer to it in the type erased wrapper. Its methods
  /// correspond one to one with those defined above, see the latters for more
  /// comprehensive documentation.
  struct NumericRangeConcept {
    /// Returns whether the range is positive.
    virtual bool isPositive() const = 0;
    /// Returns an inverted range.
    virtual NumericRange invert() const = 0;
    /// Returns a new range that results from the addition of the receiving
    /// range and the `other` range.
    virtual NumericRange add(const NumericRange &other) const = 0;
    /// Returns a new range that results from the intersection of the receiving
    /// range and the `other` range.
    virtual NumericRange setIntersect(const NumericRange &other) const = 0;
    /// Returns a new range that results from the union of the receiving
    /// range and the `other` range.
    virtual NumericRange setUnion(const NumericRange &other) const = 0;
    /// Determines whether the receiving range is strictly less than the `other`
    /// range.
    virtual bool operator<(const NumericRange &other) const = 0;
    /// Determines whether the receiving range is equal to the `other` range.
    virtual bool operator==(const NumericRange &other) const = 0;
    /// Default virtual destructor.
    virtual ~NumericRangeConcept() = default;
  };

  /// Underlying range model, which is templated by the type used to store its
  /// lower and upper bound.
  template <typename T>
  struct NumericRangeModel : NumericRangeConcept {
    /// Range's lower bound. A none value signifies that the represented
    /// interval is open to the left.
    std::optional<T> lb;
    /// Range's lower bound. A none value signifies that the represented
    /// interval is open to the right.
    std::optional<T> ub;

    /// Simple constructor that constructs a range from a given lower bound and
    /// upper bound.
    NumericRangeModel(std::optional<T> lb, std::optional<T> ub)
        : lb(lb), ub(ub) {};

    bool isPositive() const override {
      return lb.has_value() && *lb >= static_cast<T>(0);
    }

    NumericRange invert() const override {
      std::optional<T> none = std::nullopt;
      if (lb != std::nullopt)
        return NumericRange(none, lb);
      if (ub != std::nullopt)
        return NumericRange(ub, none);
      return NumericRange(none, none);
    };

    NumericRange add(const NumericRange &other) const override {
      const NumericRangeModel<T> &rhs =
          static_cast<const NumericRangeModel<T> &>(*other.concept);

      // Add corresponding bounds together to get the new ranges (unless one
      // of them is unbounded, in that case the side remains unbounded)
      std::optional<T> newLb =
          lb.has_value() && rhs.lb.has_value()
              ? static_cast<std::optional<T>>(*lb + *rhs.lb)
              : std::nullopt;
      std::optional<T> newUb =
          ub.has_value() && rhs.ub.has_value()
              ? static_cast<std::optional<T>>(*ub + *rhs.ub)
              : std::nullopt;
      return NumericRange(newLb, newUb);
    }

    NumericRange setIntersect(const NumericRange &other) const override {
      const NumericRangeModel<T> &rhs =
          static_cast<const NumericRangeModel<T> &>(*other.concept);

      // Take the most restrictive lower bound
      std::optional<T> newLb;
      if (!lb.has_value())
        newLb = rhs.lb;
      else if (!rhs.lb.has_value())
        newLb = lb;
      else
        newLb = std::max(*lb, *rhs.lb);
      // Take the most restrictive upper bound
      std::optional<T> newUb;
      if (!ub.has_value())
        newUb = rhs.ub;
      else if (!rhs.ub.has_value())
        newUb = ub;
      else
        newUb = std::min(*ub, *rhs.ub);

      return NumericRange(newLb, newUb);
    }

    NumericRange setUnion(const NumericRange &other) const override {
      const NumericRangeModel<T> &rhs =
          static_cast<const NumericRangeModel<T> &>(*other.concept);
      // Take the least restrictive lower bound
      std::optional<T> newLb = lb.has_value() && rhs.lb.has_value()
                                   ? std::optional<T>(std::min(*lb, *rhs.lb))
                                   : std::nullopt;
      // Take the least restrictive upper bound
      std::optional<T> newUb = ub.has_value() && rhs.ub.has_value()
                                   ? std::optional<T>(std::max(*ub, *rhs.ub))
                                   : std::nullopt;
      return NumericRange(newLb, newUb);
    }

    bool operator<(const NumericRange &other) const override {
      const NumericRangeModel<T> &rhs =
          static_cast<const NumericRangeModel<T> &>(*other.concept);
      return ub.has_value() && rhs.lb.has_value() && *ub < *rhs.lb;
    };

    bool operator==(const NumericRange &other) const override {
      const NumericRangeModel<T> &rhs =
          static_cast<const NumericRangeModel<T> &>(*other.concept);
      return lb == rhs.lb && ub == rhs.ub;
    };
  };

private:
  /// Shared pointer to the underlying concept object holding the range.
  std::shared_ptr<NumericRangeConcept> concept;
};

/// Provides access to the numerical analysis infrastructure, allowing to query
/// the numeric in which any value in the IR belongs. The analysis is
/// fundamentally conservative in that it will never return an incorrect range,
/// though it will sometimes be overly conservative.
class NumericAnalysis {
public:
  /// Default constructor.
  NumericAnalysis() = default;

  /// Determines whether a numeric value is always positive. Asserts if
  /// the provided value isn't an integer or float-like.
  bool isPositive(Value val);

  /// Returns the numeric range in which the value belongs, estimating the range
  /// first if it is not already available. The SSA value is guaranteed to be in
  /// the returned range, and may be unbounded on any or both side if no precise
  /// estimation could be achieved. Asserts if the provided value isn't an
  /// integer or float-like.
  NumericRange getRange(Value val);

private:
  /// Represents a set of values whose range was already queried as part of a
  /// call to `getRange`.
  using VisitSet = llvm::SmallDenseSet<Value>;

  /// Maps each value whose numeric range was queried in the past to its
  /// estimated range. No range is ever updated by the class's public methods
  /// once it's been set, so modifying the IR after querying a range may make
  /// the results inconsistent.
  DenseMap<Value, NumericRange> rangeMap;

  /// Recursive version of the identically-named public method, which
  /// additionally maintains the set of values it has already queried along the
  /// way. Before returning, the `rangeMap` is updated with the estimated range
  /// for the value. If the provided value is already in the visited set,
  /// returns an unbounded range without updating the `rangeMap`.
  NumericRange getRange(Value val, VisitSet &visited);

  /// Returns the estimated range of a block argument. At the moment, it
  /// implements logic for blocks within structured for and while loops
  /// (scf::ForOp and scf::WhileOp, respectively), and for blocks with cf-style
  /// predecessors. If any other structure is encountered it will just return an
  /// unbounded range.
  NumericRange getRangeOfBlockArg(BlockArgument blockArg, VisitSet &visited);

  /// If the provided value is an operand of the comparison operation, returns
  /// the range it belongs to when the condition evaluates to true; otherwise
  /// returns an unbounded range. To get the same result but for when the
  /// condition evaluates to false, simply invert the returned range.
  NumericRange fromCmp(Value val, mlir::arith::CmpIOp cmpOp, VisitSet &visited);

  /// Copies all the values from one set of visisted values to another.
  void setDeepCopy(VisitSet &from, VisitSet &to);
};

} // namespace dynamatic