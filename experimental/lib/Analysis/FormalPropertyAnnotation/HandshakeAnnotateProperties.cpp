//===- HandshakeAnnotateProperties.cpp - Property annotation ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-annotate-properties pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/FormalPropertyAnnotation/HandshakeAnnotateProperties.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>
#include <unordered_set>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::formalprop;

namespace {

struct HandshakeAnnotatePropertiesPass
    : public dynamatic::experimental::formalprop::impl::
          HandshakeAnnotatePropertiesBase<HandshakeAnnotatePropertiesPass> {

  HandshakeAnnotatePropertiesPass(const std::string &jsonPath = "") {
    this->jsonPath = jsonPath;
    this->uid = 0;
  }

  void runDynamaticPass() override;

private:
  unsigned int uid;
  json::Array propertyTable;

  LogicalResult annotateAbsenceOfBackpressure(ModuleOp modOp);
  LogicalResult annotateValidEquivalence(ModuleOp modOp);
  LogicalResult annotateValidEquivalenceBetweenOps(Operation &op1,
                                                   Operation &op2);
  LogicalResult annotateEagerForkNotAllOutputSent(ModuleOp modOp);
  LogicalResult
  annotateCopiedSlotsRec(std::unordered_set<std::string> &visitedSet,
                         handshake::EagerForkLikeOpInterface &originFork,
                         Operation &curOp);
  LogicalResult annotateCopiedSlots(Operation &op);
  LogicalResult annotateCopiedSlotsOfAllForks(ModuleOp modOp);
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
  bool isChannelToBeChecked(OpResult res);
};
} // namespace

bool HandshakeAnnotatePropertiesPass::isChannelToBeChecked(OpResult res) {
  // The channel connected to EndOp, MemoryControllerOp, and LSQOp don't appear
  // in the properties database for the following reasons:
  // - EndOp: the operation doesn't exist in the output model; the property
  //   creation is still possible but requires to get the names of the model's
  //   I/O signals (not implemented yet)
  // - MemeoryControllerOp and LSQOp: only load and stores can be connected to
  //   these Ops, therefore we cannot rigidify their channels with the
  //   ReadyRemoverOp and ValidMergerOp
  if (isa<handshake::EndOp, handshake::MemoryControllerOp, handshake::LSQOp>(
          res.getOwner()))
    return false;

  return std::all_of(
      res.getUsers().begin(), res.getUsers().end(), [](auto *user) {
        return !isa<handshake::EndOp, handshake::MemoryControllerOp,
                    handshake::LSQOp>(*user);
      });
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalenceBetweenOps(
    Operation &op1, Operation &op2) {
  for (auto [i, res1] : llvm::enumerate(op1.getResults()))
    for (auto [j, res2] : llvm::enumerate(op2.getResults())) {
      // equivalence is symmetrical so it needs to be checked only once for
      // each pair of signals when the Ops are the same
      if ((&op1 != &op2 || i < j) && isChannelToBeChecked(res1) &&
          isChannelToBeChecked(res2)) {
        ValidEquivalence p(uid, FormalProperty::TAG::OPT, res1, res2);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalence(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto &op : funcOp.getOps()) {
      if (failed(annotateValidEquivalenceBetweenOps(op, op))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateAbsenceOfBackpressure(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      for (auto [resIndex, res] : llvm::enumerate(op.getResults()))
        if (isChannelToBeChecked(res)) {

          AbsenceOfBackpressure p(uid, FormalProperty::TAG::OPT, res);

          propertyTable.push_back(p.toJSON());
          uid++;
        }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateEagerForkNotAllOutputSent(
    ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        EagerForkNotAllOutputSent p(uid, FormalProperty::TAG::INVAR, forkOp);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  }
  return success();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateCopiedSlotsRec(
    std::unordered_set<std::string> &visitedSet,
    handshake::EagerForkLikeOpInterface &originFork, Operation &curOp) {

  // If this operation has been visited, there is nothing to do
  std::string id = getUniqueName(&curOp).str();
  if (auto iter = visitedSet.find(id); iter != visitedSet.end()) {
    return success();
  }
  visitedSet.insert(id);

  // If this operation contains a slot, the copied slot has been found and can
  // be annotated
  if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(curOp)) {
    CopiedSlotsOfActiveForkAreFull p(uid, FormalProperty::TAG::INVAR, bufferOp,
                                     originFork);
    propertyTable.push_back(p.toJSON());
    uid++;
    return success();
  }

  if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(curOp)) {
    // TODO: Which of the previous paths should be followed?
    return success();
  }

  // Only JoinLikeOps or single-operand ops are remaining, but ideally a
  // dyn_cast would happen for either case
  for (auto value : curOp.getOperands()) {
    Operation *prevOpPtr = value.getDefiningOp();
    if (prevOpPtr == nullptr)
      // if there is no defining op, the value must be a constant, and does not
      // need to be annotated
      continue;
    Operation &prevOp = *prevOpPtr;
    if (failed(annotateCopiedSlotsRec(visitedSet, originFork, prevOp))) {
      return failure();
    }
  }

  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlots(Operation &op) {
  std::unordered_set<std::string> visitedSet = {};
  if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
    return annotateCopiedSlotsRec(visitedSet, forkOp, op);
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlotsOfAllForks(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (failed(annotateCopiedSlots(op)))
        return failure();
    }
  }
  return success();
}

#include "dynamatic/Support/LinearAlgebra/Gaussian.h"
// The structs FlowVariable and FlowExpression together form a DSL that help
// with writing flow equations. A similar DSL exists for constraint programming
// in `ConstraintProgramming.h`, but it is not reused for the following reasons:
// 1. FlowExpression uses integer coefficients, whereas CPVars have doubles as
// coefficients
// 2. Metadata that is necessary for FlowExpressions can easily be added (type,
// operation, index)
// 3. No name is necessary, as a variable is uniquely defined by the metadata
// 4. Dedicated conversion function to a matrix, as this is necessary anyway
struct FlowVariable {
  enum TYPE { full, sent, inputLambda, outputLambda, internalLambda };
  TYPE type;
  // `typeIndex` has a different meaning depending on `type`:
  // `full` => index of slot that is full
  // `sent` => index of output that is sent
  // `inputLambda` => index of operand channel
  // `outputLambda` => index of result channel
  // `internalLambda` => index of internal channel
  unsigned typeIndex;
  Operation *op;

  // utility functions for initializing variables
  static FlowVariable internalChannel(Operation *op, unsigned index) {
    return (FlowVariable){FlowVariable::TYPE::internalLambda, index, op};
  }
  static FlowVariable inputChannel(Operation *op, unsigned index) {
    return (FlowVariable){FlowVariable::TYPE::inputLambda, index, op};
  }
  static FlowVariable outputChannel(Operation *op, unsigned index) {
    return (FlowVariable){FlowVariable::TYPE::outputLambda, index, op};
  }
  static FlowVariable slot(Operation *op, unsigned index) {
    return (FlowVariable){FlowVariable::TYPE::full, index, op};
  }
  static FlowVariable sentOutput(Operation *op, unsigned index) {
    return (FlowVariable){FlowVariable::TYPE::sent, index, op};
  }

  bool operator==(const FlowVariable &other) const {
    return type == other.type && typeIndex == other.typeIndex && op == other.op;
  }

  bool isLambda() const {
    return type == FlowVariable::TYPE::inputLambda ||
           type == FlowVariable::TYPE::outputLambda ||
           type == FlowVariable::TYPE::internalLambda;
  }

  std::string getName() const {
    switch (type) {
    case full:
      return llvm::formatv("{0}.full_{1}", getUniqueName(op), typeIndex).str();
    case sent:
      return llvm::formatv("{0}.sent_{1}", getUniqueName(op), typeIndex).str();
    case inputLambda:
    case outputLambda:
    case internalLambda:
      assert(false && "lambda channels are not named");
    };
  }
};

// Hash implementation required so that FlowVariable can be used in an
// unordered_map
template <>
struct std::hash<FlowVariable> {
  size_t operator()(const FlowVariable &var) const {
    using std::hash;
    return (hash<FlowVariable::TYPE>()(var.type) ^
            hash<unsigned>()(var.typeIndex) ^ hash<Operation *>()(var.op));
  }
};

namespace {
// Only the operators that are used have been implemented...
struct FlowExpression {
  std::unordered_map<FlowVariable, int> terms;
  FlowExpression() = default;
  FlowExpression(const FlowVariable &v) { terms[v] = 1; };
};

FlowExpression operator-(FlowVariable v) {
  FlowExpression f{};
  f.terms[v] = -1;
  return f;
}
FlowExpression operator-(FlowVariable left, FlowVariable right) {
  FlowExpression f{};
  f.terms[left] = 1;
  f.terms[right] -= 1;
  return f;
}
FlowExpression operator+(FlowVariable left, FlowVariable right) {
  FlowExpression f{};
  f.terms[left] = 1;
  f.terms[right] += 1;
  return f;
}
FlowExpression operator-(FlowExpression left, FlowVariable right) {
  left.terms[right] -= 1;
  return left;
}
void operator+=(FlowExpression &left, const FlowVariable &right) {
  left.terms[right] += 1;
}

// Used to assign dense indices to FlowVariables based on a list of
// FlowExpression, i.e. indices 0 to n-1 are used for n variables, while keeping
// lambda variables with low indices to ensure they are eliminated first within
// the row-echelon form
class IndexMap {
  std::unordered_map<FlowVariable, size_t> map;
  std::vector<FlowVariable> variables;
  size_t nLambdas;

public:
  size_t getNLambdas() { return nLambdas; }
  size_t size() { return variables.size(); }
  size_t getIndex(FlowVariable v) { return map[v]; }
  FlowVariable getVariable(size_t index) { return variables[index]; }
  void verify() {
    assert(map.size() == variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      FlowVariable a = variables[i];
      size_t j = map[a];
      assert(i == j);
    }
    for (auto [key, value] : map) {
      assert(variables[value] == key);
    }
  }

  IndexMap() = default;
  IndexMap(const std::vector<FlowExpression> &exprs) {
    size_t index = 0;
    // annotate lambdas first
    for (auto &expr : exprs) {
      for (auto [key, value] : expr.terms) {
        if (!key.isLambda())
          continue;
        if (map.count(key) == 0) {
          map[key] = index;
          ++index;
          variables.push_back(key);
        }
      }
    }
    nLambdas = index;
    // annotate remaining variables
    for (auto &expr : exprs) {
      for (auto [key, value] : expr.terms) {
        if (map.count(key) == 0) {
          map[key] = index;
          ++index;
          variables.push_back(key);
        }
      }
    }
  }
};

std::vector<FlowExpression> extractLocalEquations(ModuleOp modOp) {
  std::vector<FlowExpression> equations{};
  // annotate equations derived from operations
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      FlowVariable i1 = FlowVariable::internalChannel(&op, 1);
      FlowVariable i2 = FlowVariable::internalChannel(&op, 2);
      assert(!(i1 == i2) && "expected internal channels to be different");

      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        for (auto &use : res.getUses()) {
          unsigned j = use.getOperandNumber();
          Operation &nextOp = *use.getOwner();
          assert(nextOp.getOperands().size() > j);
          FlowVariable forward = FlowVariable::outputChannel(&op, i);
          FlowVariable back = FlowVariable::inputChannel(&nextOp, j);
          // forward and back represent the same channel but from different
          // sides, so their lambdas have to be equal
          equations.push_back(forward - back);
        }
      }

      // A general structure for an operation is assumed:
      // in1, in2, ... -> Join/Merge/Mux -> internal channel 1
      // internal channel 1 -> slots? -> internal channel 2
      // internal channel 2 -> fork -> out1, out2, ...
      //
      // Some operations do not follow this structure, and should be handled
      // separately to avoid making false assumptions.
      if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
        // From inspecting .td declaration... probably a better way of doing
        // this
        int addressInputIndex = 0;
        int dataOutputIndex = 1;
        FlowVariable in = FlowVariable::inputChannel(&op, addressInputIndex);
        FlowVariable out = FlowVariable::outputChannel(&op, dataOutputIndex);
        FlowVariable addrSlot = FlowVariable::slot(&op, 0);
        FlowVariable dataSlot = FlowVariable::slot(&op, 1);

        equations.push_back(in - i1 - addrSlot);
        equations.push_back(i2 - out - dataSlot);
        continue;
      }
      if (auto controllerOp = dyn_cast<handshake::MemoryControllerOp>(op)) {
        continue;
      }

      // Join operation, merge operation, or mux
      if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
        if (auto muxOp = dyn_cast<handshake::MuxOp>(op)) {
          // mux : select input has same as output lambda, data inputs act like
          // merge
          Value a = muxOp.getSelectOperand();
          unsigned selectIndex = -1;
          for (auto &use : a.getUses()) {
            selectIndex = use.getOperandNumber();
          }
          assert(selectIndex != (unsigned)-1);
          FlowExpression dataEq = -i1;
          for (auto [i, channel] : llvm::enumerate(op.getOperands())) {
            FlowVariable chVar = FlowVariable::inputChannel(&op, i);
            if (i == selectIndex) {
              // select channel
              equations.push_back(chVar - i1);
            } else {
              // dataEq : sum(dataChannelLambda) = outputChannelLambda
              dataEq += chVar;
            }
          }
          equations.push_back(dataEq);
        } else {
          // merge : the sum of input lambdas is the output lambda
          FlowExpression mergeEq = -i1;
          for (auto [i, channel] : llvm::enumerate(op.getOperands())) {
            mergeEq += FlowVariable::inputChannel(&op, i);
          }
          equations.push_back(mergeEq);
        }
      } else {
        // join : for every input, lambda_in = lambda_out
        for (auto [i, channel] : llvm::enumerate(op.getOperands())) {
          equations.push_back(FlowVariable::inputChannel(&op, i) - i1);
        }
      }

      if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(op)) {
        // TODO: handle multiple slots
        if (bufferOp.getNumSlots() != 1) {
          llvm::errs() << getUniqueName(&op) << " has multiple slots\n";
        }
        assert(bufferOp.getNumSlots() == 1);
        FlowVariable full = FlowVariable::slot(&op, 0);
        equations.push_back(i1 - full - i2);
      } else {
        equations.push_back(i1 - i2);
      }

      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        // eagerfork: for every channel, either same tokens in as out, or in
        // `sent` state and in = out - 1
        for (auto [i, channel] : llvm::enumerate(op.getResults())) {
          FlowVariable sent = FlowVariable::sentOutput(&op, i);
          FlowVariable result = FlowVariable::outputChannel(&op, i);
          equations.push_back(i2 + sent - result);
        }
      } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(op)) {
        continue;
      } else {
        // lazy fork: all outputs have same tokens in as out
        for (auto [i, channel] : llvm::enumerate(op.getResults())) {
          FlowVariable result = FlowVariable::outputChannel(&op, i);
          equations.push_back(i2 - result);
        }
      }
    }
  }
  return equations;
}
} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  // The equations are represented by a FlowExpression that is equal to zero
  std::vector<FlowExpression> equations = extractLocalEquations(modOp);

  // Map all variables used in `equations` to an index in the matrix
  IndexMap indices(equations);
  indices.verify();

  // matrix with one row per equation, and column per variable
  MatIntType matrix(MatIntZero(equations.size(), indices.size()));

  // insert equations into the matrix
  for (auto [row, expr] : llvm::enumerate(equations)) {
    for (auto [key, value] : expr.terms) {
      unsigned index = indices.getIndex(key);
      matrix(row, index) = (int)value;
    }
  }

  // bring to row-echelon form
  gaussianElimination(matrix);

  size_t rows = matrix.size1();
  size_t cols = matrix.size2();
  // ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR);
  // uid++;

  for (size_t row = 0; row < rows; ++row) {
    bool canAnnotate = true;
    for (size_t col = 0; col < indices.getNLambdas(); ++col) {
      if (matrix(row, col) != 0) {
        canAnnotate = false;
        break;
      }
    }

    if (!canAnnotate) {
      continue;
    }

    std::vector<int> coefs{};
    std::vector<std::string> names{};

    for (size_t col = indices.getNLambdas() + 1; col < cols; ++col) {
      if (matrix(row, col) != 0) {
        coefs.push_back(matrix(row, col));
        names.push_back(indices.getVariable(col).getName());
      }
    }
    if (coefs.size() > 0) {
      ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR);
      uid++;
      p.addEquation(coefs, names);
      if (p.getEquations().size() > 0) {
        propertyTable.push_back(p.toJSON());
      }
    }
  }
  /*
  if (p.getEquations().size() > 0) {
    propertyTable.push_back(p.toJSON());
  }
  for (auto &expr : equations) {
    std::vector<int> coefs{};
    std::vector<std::string> names{};
    for (auto [key, value] : expr.terms) {
      assert(metaData.count(key) == 1);
      coefs.push_back((int)value);
      names.push_back(key.getName());
    }
    ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR, coefs, names);
    propertyTable.push_back(p.toJSON());
    uid++;
  }
  */

  return success();
}

void HandshakeAnnotatePropertiesPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  if (failed(annotateAbsenceOfBackpressure(modOp)))
    return signalPassFailure();
  if (failed(annotateValidEquivalence(modOp)))
    return signalPassFailure();
  if (annotateInvariants) {
    if (failed(annotateEagerForkNotAllOutputSent(modOp)))
      return signalPassFailure();
    if (failed(annotateCopiedSlotsOfAllForks(modOp)))
      return signalPassFailure();
    if (failed(annotateReconvergentPathFlow(modOp)))
      return signalPassFailure();
  }

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::formalprop::createAnnotateProperties(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeAnnotatePropertiesPass>(jsonPath);
}
