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

// Helper struct for CDG analysis that represents the edge (A,B) in CFG.
struct CFGEdge {
  DominanceInfoNode *from; // A
  DominanceInfoNode *to;   // B

  CFGEdge(DominanceInfoNode *from, DominanceInfoNode *to);

  // Finds the least common ancesstor (LCA) in post-dominator tree 
  // for nodes A and B of a CFG edge (A,B).
  DominanceInfoNode *findLCAInPostDomTree();
};

// CDG analysis function
LogicalResult CDGAnalysis(func::FuncOp funcOp, MLIRContext *ctx);

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CDG_ANALYSIS_H