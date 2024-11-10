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
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include <utility>

namespace dynamatic {
namespace experimental {
namespace gsa {

/// In this library, the word `gate` is used both for a `PHI` in SSA
/// representation and for `GAMMA`/`MU` in GSA representation

/// Define the three possible kinds of gate:
/// - A GAMMA gate, having two inputs and a predicate;
/// - A MU gate, having one init input and a `next` input;
/// - a PHI gate, having N possible inputs chosen in a *merge* fashon.
enum GateType { GammaGate, MuGate, PhiGate };

struct Gate;

/// Single class to collect a possible gate input, among these three
/// alternatives:
/// - a value, either produced by an operation in the IR or a block argument;
/// - another gate;
/// - an empty input (this happens for gammas having onlyo one input).
struct GateInput {

  /// Depending on the type of the input, it might be a reference to a value on
  /// the IR, another gate or empty
  std::variant<Value, struct Gate *> input;

  /// Constructor a gate input being the result of an operation
  GateInput(Value v) : input(v) {};

  /// Constructor for a gate input being the output of another gate
  GateInput(struct Gate *p) : input(p) {};

  /// Constructor for a gate input being empty
  GateInput() : input(nullptr) {};

  /// Returns the block owner of the input
  Block *getBlock();

  /// Returns true if the input is of type `Value`
  bool isTypeValue() { return std::holds_alternative<Value>(input); }

  /// Returns true if the input is empty
  bool isTypeEmpty() {
    return std::holds_alternative<Gate *>(input) && !std::get<Gate *>(input);
  }

  /// Returns true if the input is a gate
  bool isTypeGate() {
    return std::holds_alternative<Gate *>(input) && std::get<Gate *>(input);
  }

  /// Returns the input gate (raise an error if input is not a  gate)
  Gate *getGate() { return std::get<Gate *>(input); }
  /// Returns the input value (raise an error if input is not a value)
  Value getValue() { return std::get<Value>(input); }
};

using ListExpressionsPerGate =
    std::vector<std::pair<boolean::BoolExpression *, GateInput *>>;

/// The structure collects all the information related to a gate. Each gate has
/// a set of inputs, a type, a condition, an index and it might be a root.
struct Gate {

  /// Reference to the value produced by the gate (block argument)
  BlockArgument result;

  /// List of operands of the gate
  SmallVector<GateInput *> operands;

  /// Type of gate function
  GateType gsaGateFunction;

  /// Condition used to determine the outcome of the choice. The format is `cX`
  /// where `X` is the number of the block argument where
  std::string condition;

  /// Index of the current gate, which uniquely identifies it
  unsigned index;

  /// Determines whether it is a root or not (all MUs are roots, only the base
  /// of a tree of GAMMAs is the root)
  bool isRoot = false;

  /// Initialize the values of the gate
  Gate(BlockArgument v, SmallVector<GateInput *> &pi, GateType gt, int i,
       std::string c = "")
      : result(v), operands(pi), gsaGateFunction(gt), condition(std::move(c)),
        index(i) {}

  void print();

  Block *getBlock() { return result.getParentBlock(); }
  unsigned getArgumentNumber() { return result.getArgNumber(); }
};

/// Class in charge of performing the GSA analysis prior to the cf to handshake
/// conversion. For each block arguments, it provides the information necessary
/// for a conversion into a `Gate` structure.
template <typename FunctionType>
class GsaAnalysis {

public:
  /// Constructor for the GSA analysis. It requires an operation consisting of a
  /// functino to get the SSA information from.
  GsaAnalysis(Operation *operation) {

    // Only one function should be present in the module, excluding external
    // functions
    int functionsCovered = 0;

    // The analysis can be instantiated either over a module containing one
    // function only or over a function
    if (ModuleOp modOp = dyn_cast<ModuleOp>(operation); modOp) {
      for (FunctionType funcOp : modOp.getOps<FunctionType>()) {

        // Skip if external
        if (funcOp.isExternal())
          continue;

        // Analyze the function
        if (!functionsCovered) {
          identifyAllGates(funcOp);
          functionsCovered++;
        } else {
          llvm::errs() << "[GSA] Too many functions to handle in the module";
        }
      }
    } else if (FunctionType fOp = dyn_cast<FunctionType>(operation); fOp) {
      identifyAllGates(fOp);
      functionsCovered = 1;
    }

    // report an error indicating that the analysis is instantiated over
    // an inappropriate operation
    if (functionsCovered != 1)
      llvm::errs() << "[GSA] GsaAnalysis failed due to a wrong input type\n";
  };

  /// Invalidation hook to keep the analysis cached across passes. Returns
  /// true if the analysis should be invalidated and fully reconstructed the
  /// next time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<GsaAnalysis>();
  }

  /// Get a pointer to the vector containing the gate functions of a block
  SmallVector<Gate *> *getGates(Block *bb);

private:
  // Associate an index to each gate
  unsigned uniqueGateIndex;

  /// For each block in the function, keep a list of gate functions with all
  /// their information.
  DenseMap<Block *, SmallVector<Gate *>> gateList;

  /// Identify the gates necessary in the function, referencing all of their
  /// inputs
  void identifyAllGates(FunctionType &funcOp);

  /// Print the list of the gate functions
  void printGateList();

  /// Mark as mu all the phi gates which correspond to loop variables
  void convertPhiToMu(FunctionType &funcOp);

  /// Convert some phi gates to trees of gamma gates
  void convertPhiToGamma(FunctionType &funcOp);

  /// Given a boolean expression for each phi's inputs, expand it in a tree
  /// of gamma functions
  Gate *expandGammaTree(ListExpressionsPerGate &expressions,
                        std::vector<std::string> &conditions,
                        Gate *originalPhi);
};

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_GSAANALYSIS_H
