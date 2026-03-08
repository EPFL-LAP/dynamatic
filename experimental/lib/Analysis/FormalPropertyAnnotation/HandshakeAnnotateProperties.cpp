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
        assert(!key.constraint || key.constraint->singleValue);
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

std::vector<FlowExpression>
extractLocalEquations(ModuleOp modOp,
                      const DenseMap<mlir::Value, IndexInfo> &map) {
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

      if (auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op)) {
        size_t numInputs = cmergeOp.getDataOperands().size();

        FlowVariable x0(InternalLambda(&op, 0));
        x0.constraint = IndexConstraint(numInputs);

        for (auto [i, channel] : llvm::enumerate(cmergeOp.getDataOperands())) {
          FlowVariable channelVar(ChannelLambda(channel), map);
          equations.push_back(channelVar - x0.getConstrained(i));
        }

        auto slots = cmergeOp.getInternalSlotStateNamers();
        std::shared_ptr<InternalStateNamer> slotNamer =
            std::make_shared<BufferSlotFullNamer>(slots[0]);
        FlowVariable slot(slotNamer);
        slot.constraint = IndexConstraint(numInputs);

        FlowVariable indexChannel = x0.nextInternal();
        for (size_t i = 0; i < numInputs; ++i) {
          equations.push_back(x0.getConstrained(i) -
                              indexChannel.getConstrained(i) -
                              slot.getConstrained(i));
        }
        FlowVariable dataChannel = indexChannel.nextInternal();
        dataChannel.constraint.reset();
        equations.push_back(x0 - dataChannel - slot);

        auto sentNamers = cmergeOp.getInternalSentStateNamers();

        std::shared_ptr<InternalStateNamer> dataNamer =
            std::make_shared<EagerForkSentNamer>(sentNamers[0]);
        std::shared_ptr<InternalStateNamer> indexNamer =
            std::make_shared<EagerForkSentNamer>(sentNamers[1]);
        FlowVariable dataSent(dataNamer);
        FlowVariable indexSent(indexNamer);
        indexSent.constraint = IndexConstraint(numInputs);

        auto outputs = cmergeOp.getResults();
        FlowVariable data(ChannelLambda(outputs[0]), map);
        FlowVariable index(ChannelLambda(outputs[1]), map);

        for (size_t i = 0; i < numInputs; ++i) {
          equations.push_back(index.getConstrained(i) -
                              indexSent.getConstrained(i) -
                              indexChannel.getConstrained(i));
        }
        equations.push_back(data - dataSent - dataChannel);
        continue;
      }

      FlowVariable entry = InternalLambda(&op, 0);
      // Join operation, merge operation, or mux
      if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
        if (auto muxOp = dyn_cast<handshake::MuxOp>(op)) {
          // mux : select input has same as output lambda, data inputs act like
          Value select = muxOp.getSelectOperand();
          FlowVariable selectVar(ChannelLambda(select), map);
          equations.push_back(selectVar - entry);
          if (selectVar.isIndex()) {
            auto dataOperands = muxOp.getDataOperands();
            assert(selectVar.constraint->info.numValues == dataOperands.size());
            for (auto [i, operand] : llvm::enumerate(dataOperands)) {
              FlowVariable var(ChannelLambda(operand), map);
              equations.push_back(selectVar.getConstrained(i) - var);
            }
          } else {
            assert(false && "muxOp select var should always be index");
            FlowExpression dataEq = -entry;
            for (auto operand : muxOp.getDataOperands()) {
              FlowVariable chVar(ChannelLambda(operand), map);
              dataEq += chVar;
            }
            equations.push_back(dataEq);
          }
        } else {
          // merge : the sum of input lambdas is the output lambda
          FlowExpression mergeEq = -entry;
          std::vector<FlowExpression> constrainedEqs;
          size_t indexValue;
          bool foundIndexed = false;
          bool foundUnindexed = false;
          auto channels = op.getOperands();
          for (auto channel : channels) {
            FlowVariable ch(ChannelLambda(channel), map);
            if (ch.isIndex()) {
              if (!foundIndexed) {
                indexValue = ch.constraint->info.numValues;
                constrainedEqs.resize(indexValue);
              }
              foundIndexed = true;
              assert(indexValue == ch.constraint->info.numValues &&
                     "found differing index");
              for (size_t i = 0; i < indexValue; ++i) {
                constrainedEqs[i] += ch.getConstrained(i);
              }
            } else {
              foundUnindexed = true;
              assert(foundIndexed == false);
              mergeEq += ch;
            }
          }
          assert(!(foundIndexed && foundUnindexed) &&
                 "some index channels and some normal channels");
          if (foundIndexed) {
            entry.constraint = IndexConstraint(IndexInfo(indexValue));
            for (size_t i = 0; i < indexValue; ++i) {
              constrainedEqs[i] -= entry.getConstrained(i);
              equations.push_back(constrainedEqs[i]);
            }
          } else {
            equations.push_back(mergeEq);
          }
        }
      } else {
        // join : for every input, lambda_in = lambda_out
        auto channels = op.getOperands();
        if (channels.size() == 1) {
          // Only 1 input channel
          auto channel = channels[0];
          FlowVariable chVar(ChannelLambda(channel), map);
          // If input is +-, then intermediate channel is as well
          entry.constraint = chVar.constraint;
          if (chVar.isIndex()) {
            for (size_t i = 0; i < chVar.constraint->info.numValues; ++i) {
              equations.push_back(chVar.getConstrained(i) -
                                  entry.getConstrained(i));
            }
          } else {
            equations.push_back(chVar - entry);
          }
        } else {
          for (auto channel : channels) {
            FlowVariable chVar(ChannelLambda(channel), map);
            equations.push_back(chVar - entry);
          }
        }
      }

      FlowVariable exit = entry;
      if (auto arithOp = dyn_cast<handshake::ArithOpInterface>(op)) {
        // Arithmetic operations modify the channel - unless further analysis is
        // done, information about the bit is lost
        if (entry.isIndex()) {
          exit = entry.nextInternal();
          exit.constraint.reset();
          equations.push_back(entry - exit);
        }
      }

      // Annotate latency-induced slots
      if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(op)) {
        for (auto &latencySlot : latencyOp.getLatencyInducedSlots()) {
          std::shared_ptr<InternalStateNamer> namer =
              std::make_shared<LatencyInducedSlotNamer>(latencySlot);
          FlowVariable full(namer);

          FlowVariable before = exit;
          FlowVariable after = before.nextInternal();
          if (before.isIndex()) {
            assert(after.isIndex());
            FlowVariable fullPM = full;
            size_t numValues = before.constraint->info.numValues;
            fullPM.constraint = IndexConstraint(numValues);
            equations.push_back(full - fullPM);
            for (size_t i = 0; i < numValues; ++i) {
              equations.push_back(before.getConstrained(i) -
                                  fullPM.getConstrained(i) -
                                  after.getConstrained(i));
            }
          } else {
            equations.push_back(before - full - after);
          }
          exit = after;
        }
      }

      // Annotate buffer slots
      if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(op)) {
        for (auto &slotFull : bufferOp.getInternalSlotStateNamers()) {
          std::shared_ptr<InternalStateNamer> namer =
              std::make_shared<BufferSlotFullNamer>(slotFull);
          FlowVariable full(namer);

          FlowVariable before = exit;
          FlowVariable after = before.nextInternal();
          if (before.isIndex()) {
            assert(after.isIndex());
            FlowVariable fullPM = full;
            size_t numValues = before.constraint->info.numValues;
            fullPM.constraint = IndexConstraint(numValues);
            equations.push_back(full - fullPM);
            for (size_t i = 0; i < numValues; ++i) {
              equations.push_back(before.getConstrained(i) -
                                  fullPM.getConstrained(i) -
                                  after.getConstrained(i));
            }
          } else {
            equations.push_back(before - full - after);
          }
          exit = after;
        }
      }

      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        // eagerfork: for every channel, either same tokens in as out, or in
        // `sent` state and in = out - 1
        for (auto [i, sentVariable] :
             llvm::enumerate(forkOp.getInternalSentStateNamers())) {
          std::shared_ptr<InternalStateNamer> namer =
              std::make_shared<EagerForkSentNamer>(sentVariable);
          FlowVariable sent(namer);
          FlowVariable result(ChannelLambda(op.getResults()[i]), map);
          if (exit.isIndex()) {
            assert(result.isIndex());
            sent.constraint = IndexConstraint(exit.constraint->info.numValues);
            // equations.push_back(sent - sentPM);
            for (size_t i = 0; i < exit.constraint->info.numValues; ++i) {
              equations.push_back(exit.getConstrained(i) +
                                  sent.getConstrained(i) -
                                  result.getConstrained(i));
            }
          } else {
            equations.push_back(exit + sent - result);
          }
        }
      } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(op)) {
        /*
        FlowVariable trueVar(ChannelLambda(branchOp.getTrueResult()), map);
        FlowVariable falseVar(ChannelLambda(branchOp.getFalseResult()), map);
        assert(exit.isIndex() && exit.constraint->info.numValues == 2);
        equations.push_back(falseVar - exit.getConstrained(0));
        equations.push_back(falseVar - exit.getConstrained(1));
        */
      } else {
        // lazy fork: all outputs have same tokens in as out
        for (auto [i, channel] : llvm::enumerate(op.getResults())) {
          FlowVariable result(ChannelLambda(channel), map);
          if (exit.isIndex() && result.isIndex()) {
            for (size_t i = 0; i < exit.constraint->info.numValues; ++i) {
              equations.push_back(exit.getConstrained(i) -
                                  result.getConstrained(i));
            }
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

void annotateIndexChannels(llvm::DenseMap<mlir::Value, IndexInfo> &map,
                           mlir::Value index, IndexInfo info) {
  map.insert({index, info});

  Operation *op = index.getDefiningOp();
  if (!op)
    return;

  // Arithmetic ops can turn non-index into index, so stop following
  if (isa<ArithOpInterface>(op))
    return;

  if (auto bufOp = dyn_cast<BufferOp>(op)) {
    annotateIndexChannels(map, bufOp.getOperand(), info);
    return;
  }

  if (auto forkOp = dyn_cast<ForkOp>(op)) {
    annotateIndexChannels(map, forkOp.getOperand(), info);
    return;
  }

  if (auto muxOp = dyn_cast<MuxOp>(op)) {
    for (auto prevOp : muxOp.getDataOperands()) {
      annotateIndexChannels(map, prevOp, info);
    }
    return;
  }

  if (auto mergeOp = dyn_cast<MergeOp>(op)) {
    for (auto prevOp : mergeOp.getOperands()) {
      annotateIndexChannels(map, prevOp, info);
    }
    return;
  }
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  llvm::DenseMap<mlir::Value, IndexInfo> map;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto &op : funcOp.getOps()) {
      mlir::Value index = nullptr;
      std::optional<IndexInfo> info;
      if (auto muxOp = dyn_cast<MuxOp>(op)) {
        index = muxOp.getSelectOperand();
        info = IndexInfo(muxOp.getDataOperands().size());
      } else if (auto branchOp = dyn_cast<ConditionalBranchOp>(op)) {
        index = branchOp.getConditionOperand();
        info = IndexInfo(2);
      }

      if (index == nullptr) {
        continue;
      }
      annotateIndexChannels(map, index, *info);
    }
  }

  // The equations are represented by a FlowExpression that is equal to zero
  std::vector<FlowExpression> equations = extractLocalEquations(modOp, map);

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
    if (expr.terms.size() == 0) {
      continue;
    }
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
