//===- UtilsBitsUpdate.h - Utils support bits optimization ------*- C++ -*-===//
//
// This file declares supports for --optimize-bits pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H
#define DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

const unsigned CPP_MAX_WIDTH = 64;
const unsigned ADDRESS_WIDTH = 32;

IntegerType getNewType(Value opVal, unsigned bitswidth, bool signless = false);

IntegerType getNewType(Value opVal, unsigned bitswidth,
                       IntegerType::SignednessSemantics ifSign);

std::optional<Operation *> insertWidthMatchOp(Operation *newOp, int opInd,
                                              Type newType, MLIRContext *ctx);

namespace dynamatic {
namespace bitwidth {

/// Construct the functions w.r.t. the operation name in the forward process.
void constructForwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::operand_range vecOperands)>>
        &mapOpNameWidth);

/// Construct the functions w.r.t. the operation name in the backward process.
void constructBackwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::result_range vecResults)>>
        &mapOpNameWidth);

/// Construct the functions w.r.t. the operation name in the validation process.
void constructUpdateFuncMap(
    DenseMap<mlir::StringRef,
             std::function<unsigned(Operation::operand_range vecOperands)>>
        &mapOpNameWidth);

/// For branch and conditional branch operations, propagate the bits width of
/// the operands to the result.
bool propType(Operation *Op);

/// Insert width match operations (extension or truncation) for the operands and
/// the results.
void matchOpResWidth(Operation *op, MLIRContext *ctx,
                     SmallVector<Operation *> &newMatchedOps);

/// Replace the operation's operand with the its successor.
void replaceWithPredecessor(Operation *op);

/// Replace the operation's operand with the its successor, and set the operation's
/// resultOp according to its successor's resultOp type.
void replaceWithPredecessor(Operation *op, Type resType);

// Validate the truncation and extension operation in case its operand and
// result operand width are not consistent by reverting or deleting the
// operations.
void revertTruncOrExt(Operation *op, MLIRContext *ctx);

/// Set the pass, match, and revert flags to choose the methods that validate 
/// the operations.
static bool setPassFlag(Operation *op);

static bool setMatchFlag(Operation *op);

static bool setRevertFlag(Operation *op);

/// Validate the operations after bits optimization to generate .mlir file.
void validateOp(Operation *op, MLIRContext *ctx,
                SmallVector<Operation *> &newMatchedOps);
} // namespace bitwidth
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H