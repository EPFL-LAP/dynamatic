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
#include "dynamatic/Support/LinearAlgebra/Gaussian.h"
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

// [START Boilerplate code for the MLIR pass]
#include "experimental/Analysis/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKEANNOTATEPROPERTIES
#include "experimental/Analysis/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct HandshakeAnnotatePropertiesPass
    : public dynamatic::experimental::impl::HandshakeAnnotatePropertiesBase<
          HandshakeAnnotatePropertiesPass> {

  using HandshakeAnnotatePropertiesBase::HandshakeAnnotatePropertiesBase;

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
};

bool isChannelToBeChecked(OpResult res) {
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
} // namespace

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

namespace dynamatic {
// Used to assign dense indices to FlowVariables based on a list of
// FlowExpression, i.e. indices 0 to n-1 are used for n variables, while keeping
// lambda variables with low indices to ensure they are eliminated first within
// the row-echelon form
struct FlowEquationsMatrix {
  std::unordered_map<FlowVariable, size_t> varToIndex;
  std::vector<FlowVariable> indexToVar;
  size_t nLambdas;
  MatIntType matrix;

  size_t size() const { return indexToVar.size(); }
  void verify() {
    assert(varToIndex.size() == indexToVar.size());
    for (size_t i = 0; i < indexToVar.size(); ++i) {
      FlowVariable &a = indexToVar[i];
      size_t j = varToIndex[a];
      assert(i == j);
    }
    for (auto &[key, value] : varToIndex) {
      assert(indexToVar[value] == key);
    }
  }

  FlowExpression getRowAsExpression(size_t row) const {
    FlowExpression ret;
    for (size_t col = 0; col < indexToVar.size(); ++col) {
      int coef = matrix(row, col);
      if (coef != 0) {
        ret += coef * indexToVar[col];
      }
    }
    return ret;
  }

  FlowEquationsMatrix() = default;
  FlowEquationsMatrix(const std::vector<FlowExpression> &exprs) {
    size_t index = 0;
    // give lower indices to variables that cannot be annotated
    for (auto &expr : exprs) {
      for (auto &[key, value] : expr.terms) {
        // skip variables that can be annotated in SMV
        if (key.getAnnotater() != nullptr) {
          continue;
        }
        // PlusAndMinus variables should never be inserted, as the DSL will
        // insert them as two separate variables
        assert(!key.isPlusMinus());
        if (varToIndex.count(key) == 0) {
          varToIndex[key] = index;
          ++index;
          indexToVar.push_back(key);
        }
      }
    }
    nLambdas = index;
    // annotate remaining variables
    for (auto &expr : exprs) {
      for (auto &[key, value] : expr.terms) {
        if (varToIndex.count(key) == 0) {
          varToIndex[key] = index;
          ++index;
          indexToVar.push_back(key);
        }
      }
    }

    // matrix with one row per equation, and column per variable
    matrix = MatIntZero(exprs.size(), size());

    // insert equations into the matrix
    for (auto [row, expr] : llvm::enumerate(exprs)) {
      for (auto &[key, value] : expr.terms) {
        unsigned index = varToIndex[key];
        matrix(row, index) = (int)value;
      }
    }
  }
};

std::vector<FlowExpression> extractLocalEquations(ModuleOp modOp) {
  std::vector<FlowExpression> equations{};
  // annotate equations derived from operations
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      // A general structure for an operation is assumed:
      // in1, in2, ... -> Join/Merge/Mux -> entry channel
      // entry channel -> slots? -> exit channel
      // exit channel 2 -> Fork/Branch -> out1, out2, ...
      //
      // Some operations do not follow this structure, and should be handled
      // separately to avoid making false assumptions.
      if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
        continue;
      }
      if (auto storeOp = dyn_cast<handshake::StoreOp>(op)) {
        continue;
      }
      if (auto controllerOp = dyn_cast<handshake::MemoryControllerOp>(op)) {
        continue;
      }

      FlowVariable entry = FlowVariable::internalChannel(&op, 0);
      // Join operation, merge operation, or mux
      if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
        if (auto muxOp = dyn_cast<handshake::MuxOp>(op)) {
          // mux : select input has same as output lambda, data inputs act like
          Value a = muxOp.getSelectOperand();
          unsigned selectIndex = -1;
          for (auto &use : a.getUses()) {
            selectIndex = use.getOperandNumber();
          }
          assert(selectIndex == 0);

          OpOperand &selectChannel = op.getOpOperands()[0];
          FlowVariable selectVar(selectChannel, op);
          if (selectVar.isPlusMinus()) {
            // two inputs! can do +- analysis
            assert(muxOp.getDataOperands().size() == 2);
            FlowVariable falseVar = FlowVariable(op.getOpOperands()[1], op);
            FlowVariable trueVar = FlowVariable(op.getOpOperands()[2], op);
            equations.push_back(selectVar.getPlus() - trueVar);
            equations.push_back(selectVar.getMinus() - falseVar);
            equations.push_back(selectVar - entry);
          } else {
            FlowExpression dataEq = -entry;
            for (OpOperand &channel : op.getOpOperands()) {
              FlowVariable chVar(channel, op);
              // FlowVariable chVar = FlowVariable::inputChannel(&op, i);
              if (channel.getOperandNumber() == selectIndex) {
                // select channel
                equations.push_back(chVar - entry);
              } else {
                // dataEq : sum(dataChannelLambda) = outputChannelLambda
                dataEq += chVar;
              }
            }
            equations.push_back(dataEq);
          }
        } else {
          // merge : the sum of input lambdas is the output lambda
          FlowExpression mergeEq = -entry;
          FlowExpression plusEq;
          FlowExpression minusEq;
          bool allPM = true;
          bool nonePM = true;
          auto channels = op.getOpOperands();
          for (auto &channel : channels) {
            FlowVariable ch(channel, op);
            if (ch.isPlusMinus()) {
              nonePM = false;
              plusEq += ch.getPlus();
              minusEq += ch.getMinus();
            } else {
              allPM = false;
              mergeEq += ch;
            }
          }
          assert((allPM || nonePM) && "why are merge inputs not all the same?");
          if (allPM) {
            entry.pm = FlowVariable::plusAndMinus;
            plusEq -= entry.getPlus();
            minusEq -= entry.getMinus();
            equations.push_back(plusEq);
            equations.push_back(minusEq);
          } else {
            equations.push_back(mergeEq);
          }
        }
      } else {
        // join : for every input, lambda_in = lambda_out
        auto channels = op.getOpOperands();
        if (channels.size() == 1) {
          // Only 1 input channel
          auto &channel = channels[0];
          FlowVariable chVar = FlowVariable(channel, op);
          // If input is +-, then intermediate channel is as well
          entry.pm = chVar.pm;
          if (chVar.isPlusMinus()) {
            equations.push_back(chVar.getPlus() - entry.getPlus());
            equations.push_back(chVar.getMinus() - entry.getMinus());
          } else {
            equations.push_back(chVar - entry);
          }
        } else {
          for (auto &channel : channels) {
            equations.push_back(FlowVariable(channel, op) - entry);
          }
        }
      }

      FlowVariable exit = entry;
      if (auto arithOp = dyn_cast<handshake::ArithOpInterface>(op)) {
        // Arithmetic operations modify the channel - unless further analysis is
        // done, information about the bit is lost
        exit = entry.nextInternal();
        exit.pm = FlowVariable::PLUSMINUS::notApplicable;
        equations.push_back(entry - exit);
      }

      // Annotate latency-induced slots
      if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(op)) {
        for (auto &latencySlot : latencyOp.getLatencyInducedSlots()) {
          FlowVariable full = FlowVariable(
              std::make_shared<LatencyInducedSlotNamer>(latencySlot));

          FlowVariable before = exit;
          FlowVariable after = before.nextInternal();
          if (before.isPlusMinus()) {
            assert(after.isPlusMinus());
            FlowVariable fullPM = full;
            fullPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            equations.push_back(full - fullPM);
            equations.push_back(before.getPlus() - fullPM.getPlus() -
                                after.getPlus());
            equations.push_back(before.getMinus() - fullPM.getMinus() -
                                after.getMinus());
          } else {
            equations.push_back(before - full - after);
          }
          exit = after;
        }
      }

      // Annotate buffer slots
      if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(op)) {
        for (auto &slotFull : bufferOp.getInternalSlotStateNamers()) {
          FlowVariable full =
              FlowVariable(std::make_shared<BufferSlotFullNamer>(slotFull));

          FlowVariable before = exit;
          FlowVariable after = before.nextInternal();
          if (before.isPlusMinus()) {
            assert(after.isPlusMinus());
            FlowVariable fullPM = full;
            fullPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            equations.push_back(before.getPlus() - fullPM.getPlus() -
                                after.getPlus());
            equations.push_back(before.getMinus() - fullPM.getMinus() -
                                after.getMinus());
          } else {
            equations.push_back(before - full - after);
          }
          exit = after;
        }
      }

      auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op);
      if (cmergeOp && op.getOpOperands().size() == 2) {
        auto sentStates = cmergeOp.getInternalSentStateNamers();

        FlowVariable dataChannel(cmergeOp.getDataResult());
        FlowVariable dataSent(
            std::make_shared<EagerForkSentNamer>(sentStates[0]));
        // Handle case where the data is a bit
        if (exit.isPlusMinus()) {
          assert(dataChannel.isPlusMinus());
          FlowVariable sentPM = dataSent;
          sentPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
          equations.push_back(dataSent - sentPM);
          equations.push_back(exit.getPlus() + sentPM.getPlus() -
                              dataChannel.getPlus());
          equations.push_back(exit.getMinus() + sentPM.getMinus() -
                              dataChannel.getMinus());
        } else {
          equations.push_back(exit + dataSent - dataChannel);
        }

        FlowVariable indexChannel(cmergeOp.getResults()[1]);
        FlowVariable indexSent(
            std::make_shared<EagerForkSentNamer>(sentStates[1]));
        assert(indexChannel.isPlusMinus() &&
               "cmerge with 2 inputs should have 1 bit to determine index");

        auto opops = op.getOpOperands();
        FlowVariable inM(opops[0], op);
        FlowVariable inP(opops[1], op);
        auto slots = cmergeOp.getInternalSlotStateNamers();
        FlowVariable slot(std::make_shared<BufferSlotFullNamer>(slots[0]));
        FlowVariable slotPM = slot;
        slotPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
        FlowVariable sentPM = indexSent;
        sentPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;

        equations.push_back(indexChannel.getMinus() + slotPM.getMinus() - inM -
                            sentPM.getMinus());
        equations.push_back(indexChannel.getPlus() + slotPM.getPlus() - inP -
                            sentPM.getPlus());
      } else if (auto forkOp =
                     dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        // eagerfork: for every channel, either same tokens in as out, or in
        // `sent` state and in = out - 1
        for (auto [i, sentVariable] :
             llvm::enumerate(forkOp.getInternalSentStateNamers())) {
          FlowVariable sent =
              FlowVariable(std::make_shared<EagerForkSentNamer>(sentVariable));
          FlowVariable result = FlowVariable(op.getResults()[i]);
          if (exit.isPlusMinus()) {
            assert(result.isPlusMinus());
            // FlowVariable sentPM = sent;
            sent.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            // equations.push_back(sent - sentPM);
            equations.push_back(exit.getPlus() + sent.getPlus() -
                                result.getPlus());
            equations.push_back(exit.getMinus() + sent.getMinus() -
                                result.getMinus());
          } else {
            equations.push_back(exit + sent - result);
          }
        }
      } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(op)) {
        continue;
      } else {
        // lazy fork: all outputs have same tokens in as out
        for (auto [i, channel] : llvm::enumerate(op.getResults())) {
          FlowVariable result = FlowVariable(channel);
          if (exit.isPlusMinus() && result.isPlusMinus()) {
            equations.push_back(exit.getPlus() - result.getPlus());
            equations.push_back(exit.getMinus() - result.getMinus());
          } else {
            equations.push_back(exit - result);
          }
        }
      }
    }
  }
  return equations;
}
} // namespace dynamatic

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  // The equations are represented by a FlowExpression that is equal to zero
  std::vector<FlowExpression> equations = extractLocalEquations(modOp);

  // Map all variables used in `equations` to an index in the matrix
  FlowEquationsMatrix indices(equations);
  MatIntType &matrix = indices.matrix;
  indices.verify();

  // bring to row-echelon form
  gaussianElimination(matrix);

  size_t rows = matrix.size1();
  for (size_t row = 0; row < rows; ++row) {
    bool canAnnotate = true;
    for (size_t col = 0; col < indices.nLambdas; ++col) {
      if (matrix(row, col) != 0) {
        canAnnotate = false;
        break;
      }
    }

    if (!canAnnotate) {
      continue;
    }

    FlowExpression expr = indices.getRowAsExpression(row);
    ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR);
    p.addEquation(expr);
    if (p.getEquations().size() > 0) {
      uid++;
      propertyTable.push_back(p.toJSON());
    }
  }
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
