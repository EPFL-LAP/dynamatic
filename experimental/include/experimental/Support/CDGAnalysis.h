//===- CDGAnalysis.h - Exp. support for CDG analysis -------*- C++ -*-===//
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

/// @brief Structure to represent the neighbors of a Block in the
/// Control-Dependence Graph (CDG).
///
/// Each Block in the CDG is mapped to a BlockNeighbors structure, which
/// contains information about its predecessors and successors in the CDG.
struct BlockNeighbors {
  SmallVector<Block *, 4> predecessors;
  SmallVector<Block *, 4> successors;
};

/// @brief CDG analysis function.
///
/// Function to perform control-dependence graph (CDG) analysis on the given
/// function and return a map where each Block is mapped to its predecessors
/// and successors in the CDG.
///
/// @param funcOp The FuncOp representing the function to analyze.
/// @param ctx The MLIRContext used for the analysis.
/// @return A DenseMap<Block*, BlockNeighbors*> containing the CDG information.
DenseMap<Block *, BlockNeighbors *> *cdgAnalysis(func::FuncOp &funcOp,
                                                 MLIRContext &ctx);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H
