//===- elastic-miter.cpp - The elastic-miter driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/JSON.h"

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"

using namespace mlir;
using namespace dynamatic::handshake;

namespace dynamatic::experimental {
// Instead of keeping the original BB for the LHS/RHS circuit, the BB is used to
// organize the different part of the miter circuit. The BBs can therefore not
// be used for any control-flow analysis.
constexpr unsigned int BB_IN = 0;  // Input auxillary logic
constexpr unsigned int BB_LHS = 1; // Operations of the LHS circuit
constexpr unsigned int BB_RHS = 2; // Operations of the RHS circuit
constexpr unsigned int BB_OUT = 3; // Output auxillary logic

void setHandshakeName(OpBuilder &builder, Operation *op);

void setHandshakeAttributes(OpBuilder &builder, Operation *op, int bb,
                            const std::string &name);

// Add a prefix to the handshake.name attribute of an operation.
// This avoids naming conflicts when merging two functions together.
// Returns failure() when the operation doesn't already have a handshake.name
LogicalResult prefixOperation(Operation &op, const std::string &prefix);

FailureOr<std::pair<FuncOp, Block *>>
buildNewFuncWithBlock(OpBuilder builder, const std::string &name,
                      ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
                      NamedAttribute argNamedAttr, NamedAttribute resNamedAttr);

// Build a elastic-miter template function with the interface of the LHS FuncOp.
// The result names have EQ_ prefixed.
FailureOr<std::pair<FuncOp, Block *>>
buildEmptyMiterFuncOp(OpBuilder builder, FuncOp &lhsFuncOp, FuncOp &rhsFuncOp);

// Get the first and only non-external FuncOp from the module.
// Additionally check some properites:
// 1. The FuncOp is materialized (each Value is only used once).
// 2. There are no memory interfaces
// 3. Arguments  and results are all handshake.channel or handshake.control type
FailureOr<FuncOp> getModuleFuncOpAndCheck(ModuleOp module);

LogicalResult createFiles(StringRef outputDir, StringRef mlirFilename,
                          ModuleOp mod, llvm::json::Object jsonObject);

// This creates an elastic-miter module given the path to two MLIR files. The
// files need to contain exactely one module each. Each module needs to contain
// exactely one handshake.func.
FailureOr<std::pair<ModuleOp, llvm::json::Object>>
createElasticMiter(MLIRContext &context, StringRef lhsFilename,
                   StringRef rhsFilename, size_t bufferSlots);

} // namespace dynamatic::experimental