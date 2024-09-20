//===- GsaAnalysis.h - GSA analyis utilities --------------------*- C++ -*-===//
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
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace dynamatic {
namespace experimental {
namespace gsa {

/// Define the three possible kinds of phi inputs
enum PhiInputType { OpInputType, PhiInputType, ArgInputType };

struct Phi;

/// Single structure to collect a possible phi input. Either `v` or `phi` are
/// used to maintain information about the input. The value of `type` discerns
/// what kind of input it is.
struct PhiInput {

  /// type of the input
  enum PhiInputType type;

  /// Value in case of result operation or block argument
  Value v;

  /// Pointer to the phi result in case of a phi
  Phi *phi;

  /// Constructor for the result of an operation
  PhiInput(Value v) : type(OpInputType), v(v), phi(nullptr){};
  /// Constructor for the result of a phi
  PhiInput(Phi *p) : type(PhiInputType), v(nullptr), phi(p){};
  /// Constructor for the result of a block argument
  PhiInput(BlockArgument ba) : type(ArgInputType), v(Value(ba)), phi(nullptr){};
};

/// The structure collects all the information related to a phi. Each block
/// argument is associated to a phi, and has a set of inputs
struct Phi {

  /// Reference to the value produced by the phi (block argument)
  Value result;
  /// Index of the block argument
  unsigned argNumber;
  /// List of operands of the phi
  DenseSet<PhiInput *> operands;
  /// Pointer to the block argument
  Block *blockOwner;

  /// Initialize the values of the phi
  Phi(Value v, unsigned n, DenseSet<PhiInput *> &pi, Block *b)
      : result(v), argNumber(n), operands(pi), blockOwner(b) {}
};

/// Class in charge of performing the GSA analysis prior to the cf to handshake
/// conversion. For each block arguments, it takes note of the possible
/// predecessors (input of a phi function). The input of a
/// phi can be:
/// - the value resulting from an operation;
/// - the value resulting from another phi;
/// - an argument of the parent function.
template <typename FunctionType>
class GsaAnalysis {

public:
  /// Constructor for the GSA analysis
  GsaAnalysis(Operation *operation) {

    // Accepts only modules as input
    ModuleOp modOp = dyn_cast<ModuleOp>(operation);

    // Only one function should be present in the module, excluding external
    // functions
    int functionsCovered = 0;
    if (modOp) {
      auto funcOps = modOp.getOps<FunctionType>();
      for (FunctionType funcOp : funcOps) {

        // Skip if external
        if (funcOp.isExternal())
          continue;

        // Analyze the function
        if (!functionsCovered) {
          identifyAllPhi(funcOp);
          functionsCovered++;
        } else {
          llvm::errs() << "[GSA] Too many functions to handle in the module";
        }
      }
    } else {
      // report an error indicating that the analysis is instantiated over
      // an inappropriate operation
      llvm::errs()
          << "[GSA] GsaAnalysis is instantiated over an operation that is "
             "not ModuleOp!\n";
    }
  };

  /// Invalidation hook to keep the analysis cached across passes. Returns
  /// true if the analysis should be invalidated and fully reconstructed the
  /// next time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<GsaAnalysis>();
  }

  /// Get a pointer to the vector containing the phi functions of a block
  SmallVector<Phi *> *getPhis(Block *bb);

private:
  /// For each block in the function, keep a list of phi functions with all
  /// their information. The size of the list associate to each block is equal
  /// to the number of block arguments.
  DenseMap<Block *, SmallVector<Phi *>> phiList;

  /// Identify the phi necessary in the function, referencing all of their
  /// inputs
  void identifyAllPhi(FunctionType &funcOp);

  /// Print the list of the phi functions
  void printPhiList();
};

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_GSAANALYSIS_H
