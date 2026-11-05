//===- TempOps.h - Temporary producers and consumers ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Temporary operations standing in for a producer or a consumer while the IR
// is being converted.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_TEMPOPS_H
#define EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_TEMPOPS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace dynamatic {
namespace experimental {
namespace cf_to_handshake {

using mlir::UnrealizedConversionCastOp;

/// A temporary value used for the block inputs into a block
/// We route down from them to the bottom of each block
/// allowing us to process each block independently.
/// We then replace these temporary values with the real live-outs
/// to correctly connect all the values through the CFG
class TempProd {
public:
  // An empty temporary, so that temporaries can be held in maps.
  TempProd() = default;

  /// Creates a temporary of the type, at the builder's position.
  TempProd(OpBuilder &builder, Type type) {
    castOp = UnrealizedConversionCastOp::create(
        builder, builder.getUnknownLoc(), TypeRange{type}, ValueRange{});
  }

  Value getValue() const { return castOp->getResult(0); }

  /// Replaces every use of the temporary with the real channel,
  /// and erases the cast.
  void replace(Value real) {
    castOp.getResult(0).replaceAllUsesWith(real);
    castOp.erase();
  }

  /// Erases the cast, once something else has replaced every use of it.
  /// The memory interface builder plugs the interface's data into each load
  /// itself, rather than giving it to us to replace the temporary with.
  void erase() {
    assert(castOp.getResult(0).use_empty() && "temporary is still in use");
    castOp.erase();
  }

private:
  UnrealizedConversionCastOp castOp;
};

/// A temporary consumer of a value.
/// As it is an operation in the IR,
/// any changes to the producer of the value will be redirected:
/// it remains a valid way to identify the value
/// no matter what stage of the conversion we are in.
/// Needs to be removed from the IR once we are done with it
class TempCons {
public:
  // An empty temporary, so that temporaries can be held in maps.
  TempCons() = default;

  /// Creates a temporary consumer of the value, at the builder's position.
  TempCons(OpBuilder &builder, Value value) {
    castOp = UnrealizedConversionCastOp::create(
        builder, builder.getUnknownLoc(), TypeRange{}, ValueRange{value});
  }

  Value getValue() const { return castOp->getOperand(0); }

  /// Erases the cast.
  void erase() { castOp.erase(); }

private:
  UnrealizedConversionCastOp castOp;
};

} // namespace cf_to_handshake
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_TEMPOPS_H
