//===- Backedge.h - Support for building backedges --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// Backedges are operations/values which have to exist as operands before
// they are produced in a result. Since it isn't clear how to build backedges
// in MLIR, these helper classes set up a canonical way to do so.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BACKEDGEBUILDER_H
#define DYNAMATIC_SUPPORT_BACKEDGEBUILDER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace dynamatic {

class Backedge;

/// Instantiates one of these and use it to build typed backedges. Backedges
/// which get used as operands must be assigned to with the actual value before
/// this class is destructed, usually at the end of a scope. It will check that
/// invariant then erase all the backedge ops during destruction.
///
/// Example use:
/// ```
///   dynamatic::BackedgeBuilder back(rewriter, loc);
///   dynamatic::Backedge ready = back.get(rewriter.getI1Type());
///   // Use `ready` as a `Value`.
///   auto addOp = rewriter.create<addOp>(loc, ready);
///   // When the actual value is available,
///   ready.set(anotherOp.getResult(0));
/// ```
class BackedgeBuilder {
  friend class Backedge;

public:
  /// Creates a bacedge builder that will internally use an operation builder.
  BackedgeBuilder(OpBuilder &builder, Location loc);
  /// Creates a bacedge builder that will internally use a pattern rewriter.
  BackedgeBuilder(PatternRewriter &rewriter, Location loc);

  /// Create a typed backedge. If no location is provided, the one passed to the
  /// constructor will be used.
  Backedge get(mlir::Type resultType, mlir::LocationAttr optionalLoc = {});

  /// Clear the backedges, erasing any remaining cursor ops. Fails and and emits
  /// diagnostic messages if a backedge is still active.
  LogicalResult clearOrEmitError();

  /// Abandon the backedges, suppressing any diagnostics if they are still
  /// active upon destruction of the backedge builder.
  void abandon();

  /// Simply calls `BackedgeBuilder::clearOrEmitError` while suppressing its
  /// result.
  ~BackedgeBuilder();

private:
  /// Builder to create backedges with.
  mlir::OpBuilder *builder = nullptr;
  /// Rewriter to create backedges and erase operations with.
  mlir::PatternRewriter *rewriter = nullptr;
  mlir::Location loc;
  llvm::SmallVector<Operation *, 16> edges;
};

/// `Backedge` is a wrapper class around a `Value`. When assigned another
/// `Value`, it replaces all uses of itself with the new `Value` then become a
/// wrapper around the new `Value`.
class Backedge {
  friend class BackedgeBuilder;

public:
  /// A backedge cannot be by client-code.
  Backedge() = delete;

  /// Returns true if the backedge was resolved.
  explicit operator bool() const { return !!value; }
  /// Converts the backedge to the value it is current holding.
  operator Value() const { return value; }
  /// Assigns a "real" value to the backedge. This should be called excatly
  /// once per backedge.
  void setValue(Value value);

private:
  /// The current value held by the backedge.
  Value value = nullptr;
  /// Whether the backedge was already assigned a real value.
  bool set = false;

  /// `Backedge` is constructed exclusively by `BackedgeBuilder`.
  Backedge(mlir::Operation *op);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BACKEDGEBUILDER_H
