//===- GsaAnalysis.h - Gated Single Assignment analyis utilities
//----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assingment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_GSAANALYSIS_H
#define DYNAMATIC_ANALYSIS_GSAANALYSIS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace dynamatic {

/// Analysis to return information about static single assignment and gated
/// single assignment.
///
struct SSAPhi {
  Block *owner_block;
  SmallVector<Operation *, 4> producer_operations;
};

class GsaAnalysis {
public:
  /// Constructor called automatically by
  /// `getAnalysis<GsaAnalysis>()` if the analysis is not already
  /// cached. It expects to be passed a FuncOp operation where it can loop over
  /// its blocks (mimicking the BBs of the CFG)
  GsaAnalysis(Operation *operation) {
    // type-cast it into mlir::func::FuncOp
    mlir::func::FuncOp funcOp = dyn_cast<mlir::func::FuncOp>(operation);
    if (funcOp) {
      identifySsaPhis(funcOp);
    } else {
      // type-cast it into mlir::ModuleOp
      ModuleOp modOp = dyn_cast<ModuleOp>(operation);
      if (modOp) {
        auto funcOps = modOp.getOps<mlir::func::FuncOp>();
        // call those in a loop and create a big structure
        for (mlir::func::FuncOp funcOp : llvm::make_early_inc_range(funcOps)) {
          identifySsaPhis(funcOp);
        }
      } else
        // report an error indicating that the anaylsis is instantiated over
        // an inappropriate operation
        llvm::errs() << "GsaAnalysis is instantiated over an "
                        "operation that is not FuncOp or ModuleOp!\n";
    }
  };

  /// Invalidation hook to keep the analysis cached across passes. Returns true
  /// if the analysis should be invalidated and fully reconstructed the next
  /// time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<GsaAnalysis>();
  }

private:
  // Contains the ssa phis of every func::FuncOp,
  SmallVector<SmallVector<SSAPhi *, 4>, 4> all_ssa_phis;
  // Contains the gsa gates of every func::FuncOp, which is a map from every ssa
  // phi to a string representing the predicate
  SmallVector<DenseMap<SSAPhi *, std::string>, 4> all_gsa_gates;

  void identifySsaPhis(mlir::func::FuncOp &funcOp);
};

} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_GSAANALYSIS_H