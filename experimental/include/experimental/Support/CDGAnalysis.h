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
#include <set>

using namespace mlir;

namespace dynamatic {
namespace experimental {

struct Edge {
  DominanceInfoNode *from; // A
  DominanceInfoNode *to;   // B

  Edge(DominanceInfoNode *from, DominanceInfoNode *to);

  DominanceInfoNode *findLowestCommonAncestor();
};

void hello();

// CDG analysis function
LogicalResult CDGAnalysis(func::FuncOp funcOp, MLIRContext *ctx);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H