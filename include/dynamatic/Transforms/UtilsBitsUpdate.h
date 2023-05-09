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

IntegerType getNewType(Value opType, unsigned bitswidth, bool signless = false);

IntegerType getNewType(Value opType, unsigned bitswidth,
                       IntegerType::SignednessSemantics ifSign);

std::optional<Operation *> insertWidthMatchOp(Operation *newOp, int opInd,
                                              Type newType, MLIRContext *ctx);

namespace dynamatic::update {

void constructUpdateFuncMap(
    DenseMap<mlir::StringRef,
             std::function<unsigned(Operation::operand_range vecOperands)>>
        &mapOpNameWidth);

void constructBackwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::result_range vecResults)>>
        &mapOpNameWidth);

void constructForwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::operand_range vecOperands)>>
        &mapOpNameWidth);

void validateOp(Operation *Op, MLIRContext *ctx,
                SmallVector<Operation *> &newMatchedOps);

bool propType(Operation *Op);

void matchOpResWidth(Operation *Op, MLIRContext *ctx,
                     SmallVector<Operation *> &newMatchedOps);

void replaceWithSuccessor(Operation *Op);

void replaceWithSuccessor(Operation *Op, Type resType);

void revertTruncOrExt(Operation *Op, MLIRContext *ctx);

void setValidateType(Operation *Op, bool &passtype, bool &match, bool &revert);

} // namespace dynamatic::update

#endif // DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H