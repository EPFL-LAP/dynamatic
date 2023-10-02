//===- NumericAnalysis.cpp - Numeric analyis utilities ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of numeric analysis infrastructure. Right now, ranges are
// estimated by "backtracking" values through their def-use chains, which has
// strong limitations. In the future, a "forward analysis" may yield better
// results.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NumericAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace dynamatic;

template <typename T>
std::optional<T> NumericRange::getLb() const {
  assert(concept && "range is uninitialized");
  const NumericRangeModel<T> &range =
      static_cast<const NumericRangeModel<T> &>(*concept);
  return range.lb;
}

template <typename T>
std::optional<T> NumericRange::getUb() const {
  assert(concept && "range is uninitialized");
  const NumericRangeModel<T> &range =
      static_cast<const NumericRangeModel<T> &>(*concept);
  return range.ub;
}

template <typename T>
const NumericRange::NumericRangeModel<T> &
NumericRange::NumericRange::getRange() const {
  assert(concept && "range is uninitialized");
  const NumericRangeModel<T> &range =
      static_cast<const NumericRangeModel<T> &>(*concept);
  return range;
}

NumericRange NumericRange::unbounded(Type type) {
  if (isa<IntegerType, IndexType>(type))
    return NumericRange(static_cast<std::optional<int64_t>>(std::nullopt),
                        static_cast<std::optional<int64_t>>(std::nullopt));
  assert(isa<mlir::FloatType>(type) && "range only support int or float type");
  return NumericRange(static_cast<std::optional<double>>(std::nullopt),
                      static_cast<std::optional<double>>(std::nullopt));
}

NumericRange NumericRange::positive(Type type) {
  if (isa<IntegerType, IndexType>(type))
    return NumericRange(static_cast<std::optional<int64_t>>(0),
                        static_cast<std::optional<int64_t>>(std::nullopt));
  assert(isa<mlir::FloatType>(type) && "range only support int or float type");
  return NumericRange(static_cast<std::optional<double>>(0.0),
                      static_cast<std::optional<double>>(std::nullopt));
}

NumericRange NumericRange::exactValue(mlir::TypedAttr attr) {
  if (mlir::IntegerAttr intAttr = dyn_cast<mlir::IntegerAttr>(attr)) {
    APInt ap = intAttr.getValue();
    int64_t val =
        ap.isNegative() ? ap.getSExtValue() : (int64_t)ap.getZExtValue();
    return NumericRange(static_cast<std::optional<int64_t>>(val),
                        static_cast<std::optional<int64_t>>(val));
  }
  mlir::FloatAttr floatAttr = dyn_cast<mlir::FloatAttr>(attr);
  assert(floatAttr && "range only support int or float type");
  APFloat ap = floatAttr.getValue();
  return NumericRange(static_cast<std::optional<double>>(ap.convertToDouble()),
                      static_cast<std::optional<double>>(ap.convertToDouble()));
}

NumericRange NumericRange::add(const NumericRange &lhs,
                               const NumericRange &rhs) {
  if (!lhs.concept)
    return rhs;
  if (!rhs.concept)
    return lhs;
  return lhs.concept->add(rhs);
}

NumericRange NumericRange::rangeIntersect(const NumericRange &lhs,
                                          const NumericRange &rhs) {
  if (!lhs.concept)
    return rhs;
  if (!rhs.concept)
    return lhs;
  return lhs.concept->setIntersect(rhs);
}

NumericRange NumericRange::rangeUnion(const NumericRange &lhs,
                                      const NumericRange &rhs) {
  if (!lhs.concept)
    return rhs;
  if (!rhs.concept)
    return lhs;
  return lhs.concept->setUnion(rhs);
}

bool NumericAnalysis::isPositive(Value val) {
  return getRange(val).isPositive();
}

NumericRange NumericAnalysis::getRange(Value val) {
  if (rangeMap.contains(val))
    return rangeMap[val];
  llvm::SmallDenseSet<Value> visited;
  return getRange(val, visited);
}

// NOLINTBEGIN(misc-no-recursion)
NumericRange NumericAnalysis::getRange(Value val, VisitSet &visited) {
  Type valType = val.getType();
  assert((isa<IntegerType, IndexType>(valType) || isa<FloatType>(valType)) &&
         "value must be integer or float type");

  // Do not loop forever if encountering the same value again
  if (auto [_, newVal] = visited.insert(val); !newVal) {
    return NumericRange::unbounded(valType);
  }

  // Check if the range for that value is already know; if yes there is no need
  // to recompute anything.
  if (auto rangeIt = rangeMap.find(val); rangeIt != rangeMap.end())
    return rangeIt->second;

  Operation *defOp = val.getDefiningOp();
  // Handle the case where the value is a block argument
  if (!defOp)
    return rangeMap[val] =
               getRangeOfBlockArg(cast<BlockArgument>(val), visited);

  // If the defining operation is a constant, just lookup its value attribute
  if (arith::ConstantOp cstOp = dyn_cast<arith::ConstantOp>(defOp))
    return rangeMap[val] = NumericRange::exactValue(cstOp.getValue());

  // If the defining operation is an addition, add the operands' ranges
  // together
  if (isa<arith::AddIOp, arith::AddFOp>(defOp)) {
    VisitSet visitedRhs;
    setDeepCopy(visited, visitedRhs);
    return rangeMap[val] =
               NumericRange::add(getRange(defOp->getOperand(0), visited),
                                 getRange(defOp->getOperand(1), visitedRhs));
  }

  // Unsupported operation, make the value's range unbounded
  return rangeMap[val] = NumericRange::unbounded(valType);
}

NumericRange NumericAnalysis::getRangeOfBlockArg(BlockArgument arg,
                                                 VisitSet &visited) {
  Type argType = arg.getType();
  Block *block = arg.getParentBlock();
  size_t argIdx = arg.getArgNumber();

  Operation *parentOp = block->getParentOp();
  if (!parentOp)
    return NumericRange();

  // Check whether the block is in a while loop
  if (scf::WhileOp whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
    if (block == &whileOp.getBefore().front()) {
      // Argument to the before block come either from the while operation's
      // operands or from the after block's yield terminator
      VisitSet visitedYield;
      setDeepCopy(visited, visitedYield);

      return NumericRange::rangeUnion(
          getRange(whileOp.getOperand(argIdx), visited),
          getRange(whileOp.getYieldOp().getOperand(argIdx), visitedYield));
    }

    // The argument to the after block always comes from the before block
    scf::ConditionOp condOp = whileOp.getConditionOp();
    Value yieldArg = condOp.getArgs()[argIdx];
    Operation *condValDefOp = condOp.getCondition().getDefiningOp();

    // We may get some tighter range on the value by looking at how the
    // condition operand for the scf.condition is computed.
    if (arith::CmpIOp cmpOp = dyn_cast_or_null<arith::CmpIOp>(condValDefOp)) {
      VisitSet visitedCmp;
      setDeepCopy(visited, visitedCmp);
      NumericRange cmpRange = fromCmp(yieldArg, cmpOp, visitedCmp);
      return NumericRange::rangeIntersect(getRange(yieldArg, visited),
                                          cmpRange);
    }
    /// TODO: do the same for arith::CmpFOp

    return getRange(yieldArg, visited);
  }

  // Check whether the block is in for while loop
  if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
    // The value could be the induction variable, in which case it is bounded
    // by the for loop's lower (inclusive) and upper (exclusive) bounds
    if (arg == forOp.getInductionVar()) {
      VisitSet visitedUb;
      setDeepCopy(visited, visitedUb);
      NumericRange upRange = getRange(forOp.getUpperBound(), visited);
      std::optional<int64_t> ub = upRange.getUb<int64_t>();
      if (ub.has_value())
        *ub -= 1;
      return NumericRange::rangeUnion(
          getRange(forOp.getLowerBound(), visitedUb),
          NumericRange(upRange.getLb<int64_t>(), ub));
    }

    // Loop-carried values come either from the for operation's operands or
    // from the loop's yield terminator
    scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(block->getTerminator());
    assert(yieldOp && "for loop terminator must be yield");
    return NumericRange::rangeUnion(
        getRange(forOp.getInitArgs()[argIdx], visited),
        getRange(yieldOp.getOperand(argIdx), visited));
  }

  // Check whether the block simply has predecessors
  if (!block->hasNoPredecessors()) {
    NumericRange range;
    auto updateRangeCond = [&](Block *successor, ValueRange operands,
                               Value cond, bool brDest) {
      if (successor != block)
        return;

      VisitSet visitedBr;
      setDeepCopy(visited, visitedBr);
      Value brArg = operands[argIdx];
      NumericRange brArgRange = getRange(brArg, visitedBr);

      Operation *condDefOp = cond.getDefiningOp();
      if (arith::CmpIOp cmpOp = dyn_cast_or_null<arith::CmpIOp>(condDefOp)) {
        VisitSet visitedCmp;
        setDeepCopy(visited, visitedCmp);

        NumericRange cmpRange = fromCmp(brArg, cmpOp, visitedCmp);
        if (!brDest)
          cmpRange = cmpRange.invert();
        brArgRange = NumericRange::rangeIntersect(brArgRange, cmpRange);
      }
      /// TODO: do the same for arith::CmpFOp

      range = NumericRange::rangeUnion(range, brArgRange);
    };

    // Compute the union of the ranges from all predecessors
    for (Block *pred : block->getPredecessors()) {
      Operation *termOp = pred->getTerminator();
      if (cf::BranchOp brOp = dyn_cast<cf::BranchOp>(termOp)) {
        VisitSet visitedBr;
        setDeepCopy(visited, visitedBr);
        range = NumericRange::rangeUnion(
            range, getRange(brOp.getOperand(argIdx), visitedBr));
      } else if (cf::CondBranchOp condOp = dyn_cast<cf::CondBranchOp>(termOp)) {
        Value cond = condOp.getCondition();
        updateRangeCond(condOp.getTrueDest(), condOp.getTrueDestOperands(),
                        cond, true);
        updateRangeCond(condOp.getFalseDest(), condOp.getFalseDestOperands(),
                        cond, false);
      } else
        // Unsupported terminator
        return NumericRange::unbounded(argType);
    }

    return range;
  }

  // Unsupported operation, return an unbounded range
  return NumericRange::unbounded(argType);
}

NumericRange NumericAnalysis::fromCmp(Value val, mlir::arith::CmpIOp cmpOp,
                                      VisitSet &visited) {
  // The value must be an operand of the comparison for us to derive any insight
  // from it
  bool isLhs;
  NumericRange otherRange;
  if (val == cmpOp.getLhs()) {
    isLhs = true;
    otherRange = getRange(cmpOp.getRhs(), visited);
  } else if (val == cmpOp.getRhs()) {
    isLhs = false;
    otherRange = getRange(cmpOp.getLhs(), visited);
  } else
    return NumericRange::unbounded(cmpOp.getLhs().getType());

  std::optional<int64_t> lb = otherRange.getLb<int64_t>();
  std::optional<int64_t> ub = otherRange.getUb<int64_t>();

  // Wrapper around the NumericRange constructor to avoid having static casts
  // everywhere
  auto newRange = [&](std::optional<int64_t> newLb,
                      std::optional<int64_t> mewUb) -> NumericRange {
    return NumericRange(newLb, mewUb);
  };

  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    // val in [lb, ub]
    return otherRange;
  case arith::CmpIPredicate::ne:
    // val outside [lb, ub]
    return otherRange.invert();
  case arith::CmpIPredicate::ult:
    if (isLhs)
      // val in [0, ub - 1]
      return newRange(0, *ub - 1);
    // val in [lb + 1, inf)
    return newRange(*lb + 1, std::nullopt);
  case arith::CmpIPredicate::ule:
    if (isLhs)
      // val in [0, ub]
      return newRange(0, *ub);
    // val in [lb, inf)
    return newRange(*lb, std::nullopt);
  case arith::CmpIPredicate::slt:
    if (isLhs)
      // val in (-inf, ub - 1]
      return newRange(std::nullopt, *ub - 1);
    // val in [lb + 1, inf)
    return newRange(*lb + 1, std::nullopt);
  case arith::CmpIPredicate::sle:
    if (isLhs)
      // val in (-inf, ub]
      return newRange(std::nullopt, *ub);
    // val in [lb, inf)
    return newRange(*lb, std::nullopt);
  case arith::CmpIPredicate::ugt:
    if (isLhs)
      // val in [lb + 1, inf)
      return newRange(*lb + 1, std::nullopt);
    // val in [0, ub - 1]
    return newRange(0, *ub - 1);
  case arith::CmpIPredicate::uge:
    if (isLhs)
      // val in [lb, inf)
      return newRange(*lb, std::nullopt);
    // val in [0, ub]
    return newRange(0, *ub);
  case arith::CmpIPredicate::sgt:
    if (isLhs)
      // val in [lb + 1, inf)
      return newRange(*lb + 1, std::nullopt);
    // val in (-inf, ub - 1]
    return newRange(std::nullopt, *ub - 1);
  case arith::CmpIPredicate::sge:
    if (isLhs)
      // val in [lb, inf)
      return newRange(*lb, std::nullopt);
    // val in (-inf, ub]
    return newRange(std::nullopt, *ub);
  }
}
// NOLINTEND(misc-no-recursion)

void NumericAnalysis::setDeepCopy(VisitSet &from, VisitSet &to) {
  for (Value val : from)
    to.insert(val);
}
