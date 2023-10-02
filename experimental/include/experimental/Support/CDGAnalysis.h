//===- CDGAnalysis.h - Exp. support for CDG analysis -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the function for CDG analysis.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H
#define EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace dynamatic {
namespace experimental {

/// Structure to represent the neighbors of a Block in the
/// Control-Dependence Graph (CDG).
struct BlockNeighbors {
  SmallVector<Block *, 4> predecessors;
  SmallVector<Block *, 4> successors;
};

/// Function to perform control-dependence graph (CDG) analysis on the given
/// function and return a map where each Block is mapped to its predecessors
/// and successors in the CDG.
DenseMap<Block *, BlockNeighbors> cdgAnalysis(func::FuncOp &funcOp,
                                              MLIRContext &ctx);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H
