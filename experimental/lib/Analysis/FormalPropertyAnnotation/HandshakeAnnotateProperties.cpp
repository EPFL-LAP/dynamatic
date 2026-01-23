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

#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include <boost/numeric/ublas/matrix.hpp>
CPVar resultVar(Operation &op, unsigned index) {
  std::string name =
      llvm::formatv("#out{0}{1}", index, getUniqueName(&op)).str();
  return CPVar(name, VarType::INTEGER);
}

CPVar inputVar(Operation &op, unsigned index) {
  std::string name =
      llvm::formatv("#in{0}{1}", index, getUniqueName(&op)).str();
  return CPVar(name, VarType::INTEGER);
}

CPVar sentVar(Operation &forkOp, unsigned index) {
  std::string name =
      llvm::formatv("{1}.sent_{0}", index, getUniqueName(&forkOp));
  return CPVar(name, VarType::INTEGER);
}

CPVar slotVar(Operation &slotOp, unsigned index) {
  std::string name =
      llvm::formatv("{1}.full_{0}", index, getUniqueName(&slotOp));
  return CPVar(name, VarType::INTEGER);
}

struct FlowVariableInfo {
  enum TYPE { full, sent, lambda };
  TYPE type;
  Value channel;
  Operation *op;
  unsigned id;
};

void swapRows(boost::numeric::ublas::matrix<int> &m, size_t row1, size_t row2) {
  size_t cols = m.size2();
  for (size_t i = 0; i < cols; ++i) {
    int t = m(row1, i);
    m(row1, i) = m(row2, i);
    m(row2, i) = t;
  }
}

void gaussianElimination(boost::numeric::ublas::matrix<int> &m) {
  size_t rows = m.size1();
  size_t cols = m.size2();

  // h: row index
  // k: leading non-zero position
  size_t h = 0, k = 0;
  while (h < rows && k < cols) {
    int pivotRow = -1;

    int pivotValue = std::numeric_limits<int>::max();

    // Find the row to pivot around
    for (size_t i = h; i < rows; ++i) {
      if (m(i, k) != 0) {
        if (std::abs(m(i, k)) < std::abs(pivotValue)) {
          pivotValue = m(i, k);
          pivotRow = i;
        }
      }
    }

    // no row with non-zero index at k -> look at next column
    if (pivotRow == -1) {
      ++k;
      continue;
    }

    swapRows(m, h, pivotRow);
    if (pivotValue < 0) {
      for (size_t i = k; i < cols; ++i) {
        m(h, i) *= -1;
      }
    }

    // eliminate other rows
    for (size_t i = h + 1; i < rows; ++i) {
      int factorPivot = m(h, k);
      int factorRow = m(i, k);
      for (size_t j = k; j < cols; ++j) {
        m(i, j) *= factorPivot;
        m(i, j) -= factorRow * m(h, j);
      }
    }
    ++h;
    ++k;
  }
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  // all equations are equal to zero
  std::vector<LinExpr> equations{};
  std::map<CPVar, FlowVariableInfo> metaData{};
  // annotate equations from channels
  unsigned id = 0;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      // channel metadata
      CPVar i1 = CPVar(llvm::formatv("#x1{0}", getUniqueName(&op)).str(),
                       VarType::INTEGER);
      CPVar i2 = CPVar(llvm::formatv("#x2{0}", getUniqueName(&op)).str(),
                       VarType::INTEGER);

      metaData[i1] = (FlowVariableInfo){
          FlowVariableInfo::TYPE::lambda,
          nullptr,
          &op,
          id++,
      };
      metaData[i2] = (FlowVariableInfo){
          FlowVariableInfo::TYPE::lambda,
          nullptr,
          &op,
          id++,
      };
      for (auto [i, res] : llvm::enumerate(op.getOperands())) {
        metaData[inputVar(op, i)] = (FlowVariableInfo){
            FlowVariableInfo::TYPE::lambda,
            res,
            &op,
            id++,
        };
      }
      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        CPVar out = resultVar(op, i);
        metaData[out] = (FlowVariableInfo){
            FlowVariableInfo::TYPE::lambda,
            res,
            &op,
            id++,
        };
        for (auto &use : res.getUses()) {
          unsigned j = use.getOperandNumber();
          Operation &nextOp = *use.getOwner();
          CPVar in = inputVar(nextOp, j);
          // lambda_out = lambda_in   as they represent the same channel
          equations.push_back(out - in);
        }
      }
    }
  }
  unsigned nLambda = id;
  // annotate equations derived from operations
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      unsigned numIn = op.getOperands().size();
      unsigned numOut = op.getResults().size();
      CPVar i1 = CPVar(llvm::formatv("#x1{0}", getUniqueName(&op)).str(),
                       VarType::INTEGER);
      CPVar i2 = CPVar(llvm::formatv("#x2{0}", getUniqueName(&op)).str(),
                       VarType::INTEGER);
      if (numIn == 1) {
        equations.push_back(inputVar(op, 0) - i1);
      } else {
        // Join operation, merge operation, or mux
        if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
          if (isa<handshake::MuxOp>(op)) {
            // mux
            // TODO: check if select is always input 0
            equations.push_back(inputVar(op, 0) - i1);
            LinExpr dataEq = 0 - i1;
            for (unsigned i = 1; i < numIn; ++i) {
              dataEq += inputVar(op, i);
            }
            equations.push_back(dataEq);
          } else {
            // merge
            LinExpr mergeEq = 0 - i1;
            for (unsigned i = 0; i < numIn; ++i) {
              mergeEq += inputVar(op, i);
            }
            equations.push_back(mergeEq);
          }
        } else {
          // join
          for (unsigned i = 0; i < numIn; ++i) {
            equations.push_back(inputVar(op, i) - i1);
          }
        }
      }

      if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(op)) {
        // TODO: handle multiple slots
        CPVar full = slotVar(op, 0);
        equations.push_back(i1 - full - i2);
        metaData[full] = (FlowVariableInfo){
            FlowVariableInfo::TYPE::full,
            nullptr,
            &op,
            id++,
        };
      } else {
        equations.push_back(i1 - i2);
      }

      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        for (unsigned i = 0; i < numOut; ++i) {
          CPVar sent = sentVar(op, i);
          equations.push_back(i2 + sent - resultVar(op, i));

          metaData[sent] = (FlowVariableInfo){
              FlowVariableInfo::TYPE::sent,
              nullptr,
              &op,
              id++,
          };
        }
      } else {
        for (unsigned i = 0; i < numOut; ++i) {
          equations.push_back(i2 - resultVar(op, i));
        }
      }
    }
  }

  boost::numeric::ublas::matrix<int> matrix(equations.size(), id);
  std::map<unsigned, CPVar> idInfo;
  for (auto [row, expr] : llvm::enumerate(equations)) {
    for (auto [key, value] : expr.terms) {
      unsigned index = metaData[key].id;
      idInfo[index] = key;
      matrix(row, index) = (int)value;
    }
  }
  for (unsigned i = 0; i < matrix.size2(); ++i) {
    assert(idInfo.count(i) == 1);
  }
  gaussianElimination(matrix);
  size_t rows = matrix.size1();
  size_t cols = matrix.size2();
  for (size_t row = 0; row < rows; ++row) {
    bool canAnnotate = true;
    for (size_t col = 0; col < nLambda; ++col) {
      if (matrix(row, col) != 0) {
        canAnnotate = false;
        break;
      }
    }

    if (!canAnnotate)
      continue;

    std::vector<int> coefs{};
    std::vector<std::string> names{};

    for (size_t col = nLambda + 1; col < cols; ++col) {
      if (matrix(row, col) != 0) {
        coefs.push_back(matrix(row, col));
        names.push_back(idInfo[col].getName());
      }
    }
    ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR, coefs, names);
    propertyTable.push_back(p.toJSON());
    uid++;
  }
  /*
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
