//===- GSAAnalysis.h - GSA analyis utilities --------------------*- C++ -*-===//
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
#include "llvm/Support/Debug.h"
#include <queue>
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
  /// the IR, another gate or empty.
  std::variant<Value, Gate *> input;

  /// Constructor a gate input being the result of an operation.
  GateInput(Value v) : input(v) {};

  /// Constructor for a gate input being the output of another gate.
  GateInput(Gate *p) : input(p) {};

  /// Constructor for a gate input being empty.
  GateInput() : input(nullptr) {};

  /// Returns the block owner of the input.
  Block *getBlock();

  /// Returns true if the input is of type `Value`.
  bool isTypeValue() { return std::holds_alternative<Value>(input); }

  /// Returns true if the input is empty.
  bool isTypeEmpty() {
    return std::holds_alternative<Gate *>(input) && !std::get<Gate *>(input);
  }

  /// Returns true if the input is a gate.
  bool isTypeGate() {
    return std::holds_alternative<Gate *>(input) && std::get<Gate *>(input);
  }

  /// Returns the input gate (raise an error if input is not a  gate).
  Gate *getGate() { return std::get<Gate *>(input); }

  /// Returns the input value (raise an error if input is not a value).
  Value getValue() { return std::get<Value>(input); }

  ~GateInput() = default;
};

using ListExpressionsPerGate =
    std::vector<std::pair<boolean::BoolExpression *, GateInput *>>;

/// The structure collects all the information related to a gate. Each gate has
/// a set of inputs, a type, a condition, an index and it might be a root.
struct Gate {

  /// Reference to the value produced by the gate (block argument).
  BlockArgument result;

  /// List of operands of the gate.
  SmallVector<GateInput *> operands;

  /// Type of gate function.
  GateType gsaGateFunction;

  /// Block whose terminator is used to drive the condition of the gate.
  Block *conditionBlock;

  /// Index of the current gate, which uniquely identifies it.
  unsigned index;

  /// Determines whether it is a root or not (all MUs are roots, only the base
  /// of a tree of GAMMAs is the root).
  bool isRoot = false;

  /// Initialize the values of the gate.
  Gate(BlockArgument v, ArrayRef<GateInput *> pi, GateType gt, unsigned i,
       Block *c = nullptr)
      : result(v), operands(pi), gsaGateFunction(gt), conditionBlock(c),
        index(i) {}

  /// Print the gate.
  void print();

  /// Get the block the gate refers to.
  inline Block *getBlock() { return result.getParentBlock(); }

  /// Get the argument numebr the gate refers to.
  inline unsigned getArgumentNumber() { return result.getArgNumber(); }

  ~Gate() = default;
};

/// Class in charge of performing the GSA analysis prior to the cf to handshake
/// conversion. For each block arguments, it provides the information necessary
/// for a conversion into a `Gate` structure.
class GSAAnalysis {

public:
  /// Constructor for the GSA analysis. It requires an operation consisting of a
  /// functino to get the SSA information from.
  GSAAnalysis(Operation *operation);

  /// Copy constructor for the anlaysis pass, rerunning the anlaysis so that new
  /// pointers are created
  GSAAnalysis(GSAAnalysis &gsa) {
    this->inputOp = gsa.inputOp;
    this->convertSSAToGSA(this->inputOp);
  }

  /// Invalidation hook to keep the analysis cached across passes. Returns
  /// true if the analysis should be invalidated and fully reconstructed the
  /// next time it is queried.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<GSAAnalysis>();
  }

  /// Get a pointer to the vector containing the gate functions of a block.
  ArrayRef<Gate *> getGates(Block *bb) const;

  /// Deallocates all the gates created.
  ~GSAAnalysis();

private:
  /// Keep track of the original operation the analysis was run on
  mlir::func::FuncOp inputOp;

  /// Associate an index to each gate.
  unsigned uniqueGateIndex;

  /// Associate a index to each block.
  DenseMap<Block *, unsigned> indexPerBlock;

  /// For each block in the function, keep a list of gate functions with all
  /// their information.
  DenseMap<Block *, SmallVector<Gate *>> gatesPerBlock;

  /// List of all gate inputs which have been created (and must be thus
  /// deallocated).
  SmallVector<GateInput *> gateInputList;

  /// Identify the gates necessary in the function, referencing all of their
  /// inputs.
  void convertSSAToGSA(mlir::func::FuncOp &funcOp);

  /// Print the list of the gate functions.
  void printAllGates();

  /// Mark as mu all the phi gates which correspond to loop variables.
  void convertPhiToMu(mlir::func::FuncOp &funcOp);

  /// Convert some phi gates to trees of gamma gates.
  void convertPhiToGamma(mlir::func::FuncOp &funcOp);

  /// Given a boolean expression for each phi's inputs, expand it in a tree
  /// of gamma functions.
  Gate *expandGammaTree(ListExpressionsPerGate &expressions,
                        std::queue<unsigned> &conditions, Gate *originalPhi);

  /// Map each block to an unsigned index, so that the following relationship
  /// holds: if Bi dominates Bj than i < j.
  void mapBlocksToIndex(mlir::func::FuncOp &funcOp);

  /// Return the index of a block
  unsigned getIndexFromBlock(Block *bb) { return indexPerBlock[bb]; }

  Block *getBlockFromIndex(unsigned index);
};

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_GSAANALYSIS_H
