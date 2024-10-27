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

#include <utility>

#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Shannon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace dynamatic {
namespace experimental {
namespace gsa {

/// Define the three possible kinds of phi inputs
enum PhiInputType { OpInputType, PhiInputType, ArgInputType, EmptyInputType };
enum GsaGateFunction { GammaGate, MuGate, PhiGate };

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
  struct Phi *phi;

  /// Pointer to the block
  Block *blockOwner;

  /// Constructor for the result of an operation
  PhiInput(Value v, Block *bb)
      : type(OpInputType), v(v), phi(nullptr), blockOwner(bb){};
  /// Constructor for the result of a phi
  PhiInput(struct Phi *p, Block *bb)
      : type(PhiInputType), v(nullptr), phi(p), blockOwner(bb){};
  /// Constructor for the result of a block argument
  PhiInput(BlockArgument ba, Block *bb)
      : type(ArgInputType), v(Value(ba)), phi(nullptr), blockOwner(bb){};
  /// Constructor for an empty input
  PhiInput()
      : type(EmptyInputType), v(nullptr), phi(nullptr), blockOwner(nullptr){};

  Block *getBlock();
};

/// The structure collects all the information related to a phi. Each block
/// argument is associated to a phi, and has a set of inputs
struct Phi {

  /// Reference to the value produced by the phi (block argument)
  Value result;
  /// Index of the block argument
  unsigned argNumber;
  /// List of operands of the phi
  SmallVector<PhiInput *> operands;
  /// Pointer to the block argument
  Block *blockOwner;
  /// Type of GSA gate function
  GsaGateFunction gsaGateFunction;
  /// Condition used to determine the outcome of the choice
  std::string condition;
  /// Index of the current phi
  unsigned index;
  /// Determintes whether it is a root or not (all MUs are roots, only the base
  /// of a tree of GAMMAs is the root)
  bool isRoot = false;

  /// Initialize the values of the phi
  Phi(Value v, unsigned n, SmallVector<PhiInput *> &pi, Block *b,
      GsaGateFunction ggf, std::string c = "")
      : result(v), argNumber(n), operands(pi), blockOwner(b),
        gsaGateFunction(ggf), condition(std::move(c)) {}

  void print();
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
  // Associate an index to each phi
  unsigned uniquePhiIndex;

  /// For each block in the function, keep a list of phi functions with all
  /// their information. The size of the list associate to each block is equal
  /// to the number of block arguments.
  DenseMap<Block *, SmallVector<Phi *>> phiList;

  /// Identify the phi necessary in the function, referencing all of their
  /// inputs
  void identifyAllPhi(FunctionType &funcOp);

  /// Print the list of the phi functions
  void printPhiList();

  /// Mark as mu all the phi functions which correspond to loop variables
  void convertPhiToMu(FunctionType &funcOp);

  /// Convert each phi function to a gamma function
  void convertPhiToGamma(FunctionType &funcOp);

  /// Given a boolean expression for each of a phi's inputs, expand it in a tree
  /// of gamma functions
  Phi *expandExpressions(
      std::vector<std::pair<boolean::BoolExpression *, PhiInput *>>
          &expressions,
      std::vector<std::string> &cofactors, Phi *originalPhi);
};

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_GSAANALYSIS_H
