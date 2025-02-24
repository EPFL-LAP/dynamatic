//===- FabricGeneration.h - Generate Elastic Miter Circuit ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a generator for an elastic-miter circuit.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_FABRIC_GENERATION_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_FABRIC_GENERATION_H

#include <filesystem>

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

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

struct ElasticMiterConfig {
  SmallVector<std::pair<std::string, std::string>> inputBuffers;
  SmallVector<std::pair<std::string, std::string>> inputNDWires;
  SmallVector<std::pair<std::string, std::string>> outputBuffers;
  SmallVector<std::pair<std::string, std::string>> outputNDWires;
  SmallVector<std::pair<std::string, Type>> arguments;
  SmallVector<std::pair<std::string, Type>> results;
  SmallVector<std::string> eq;
  std::string funcName;
  std::string lhsFuncName;
  std::string rhsFuncName;
};

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

LogicalResult createMlirFile(const std::filesystem::path &mlirPath,
                             ModuleOp mod);

// This creates an elastic-miter module given the path to two MLIR files.
// The files need to contain exactely one module each. Each module needs to
// contain exactely one handshake.func.
FailureOr<std::pair<ModuleOp, struct ElasticMiterConfig>>
createElasticMiter(MLIRContext &context, ModuleOp lhsModule, ModuleOp rhsModule,
                   size_t bufferSlots);

// Creates a reachability circuit. Essentially ND wires are put at all in- and
// outputs of the circuit. Additionally creates a json config file with the name
// of the funcOp, and its argument and results.
FailureOr<std::pair<ModuleOp, struct ElasticMiterConfig>>
createReachabilityCircuit(MLIRContext &context,
                          const std::filesystem::path &filename);

// This creates an elastic-miter MLIR module and a JSON config file given the
// path to two MLIR files. The input files need to contain exactly one module
// each. Each module needs to contain exactely one handshake.func.
FailureOr<std::pair<std::filesystem::path, struct ElasticMiterConfig>>
createMiterFabric(MLIRContext &context, const std::filesystem::path &lhsPath,
                  const std::filesystem::path &rhsPath,
                  const std::filesystem::path &outputDir, size_t nrOfTokens);

} // namespace dynamatic::experimental

#endif