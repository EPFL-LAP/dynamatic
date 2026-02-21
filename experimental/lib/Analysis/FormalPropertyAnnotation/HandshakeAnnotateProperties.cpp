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
  enum TYPE { internalState, inputLambda, outputLambda, internalLambda };
  enum PLUSMINUS { notApplicable, plusAndMinus, plus, minus };
  // A Lambda variable is defined by type, lambdaIndex, and op.
  // An internal state is defined by type, state
  TYPE type;
  unsigned lambdaIndex;
  Operation *op;
  PLUSMINUS pm;
  std::shared_ptr<InternalStateNamer> state;

  FlowVariable(std::shared_ptr<InternalStateNamer> state)
      : type(TYPE::internalState), pm(PLUSMINUS::notApplicable),
        state(std::move(state)) {}
  FlowVariable(TYPE t, Operation *op, unsigned lambdaIndex)
      : type(t), lambdaIndex(lambdaIndex), op(op),
        pm(PLUSMINUS::notApplicable) {
    assert(
        t != TYPE::internalState &&
        "internal states should be initialized with the according constructor");
  }

  FlowVariable(const OpResult &channel) {
    op = channel.getDefiningOp();
    assert(op && "cannot get FlowVariable from channel without owner");
    type = outputLambda;
    lambdaIndex = channel.getResultNumber();

    pm = PLUSMINUS::notApplicable;
    if (auto ct = dyn_cast<handshake::ChannelType>(channel.getType())) {
      if (ct.getDataBitWidth() == 1) {
        pm = PLUSMINUS::plusAndMinus;
      }
    }
  }

  FlowVariable(OpOperand &back, Operation &resOp) {
    Value channel = back.get();
    op = channel.getDefiningOp();
    pm = PLUSMINUS::notApplicable;
    if (auto ct = dyn_cast<handshake::ChannelType>(channel.getType())) {
      if (ct.getDataBitWidth() == 1) {
        pm = PLUSMINUS::plusAndMinus;
      }
    }
    if (op == nullptr) {
      type = inputLambda;
      op = &resOp;
      lambdaIndex = back.getOperandNumber();
    } else {
      type = outputLambda;
      bool found = false;
      for (auto res : op->getResults()) {
        if (res == channel) {
          assert(!found && "found multiple matches");
          found = true;
          lambdaIndex = res.getResultNumber();
        }
      }
      assert(found && "did not find matching OpResult");
    }
  }

  // utility functions for initializing variables
  static FlowVariable internalChannel(Operation *op, unsigned index) {
    return FlowVariable(TYPE::internalLambda, op, index);
  }

  FlowVariable nextInternal() const {
    assert(type == TYPE::internalLambda);
    FlowVariable next = *this;
    next.lambdaIndex = lambdaIndex + 1;
    return next;
  }

  bool operator==(const FlowVariable &other) const {
    if (type == TYPE::internalState && other.type == TYPE::internalState) {
      return pm == other.pm && state.get() == other.state.get();
    }

    return type == other.type && lambdaIndex == other.lambdaIndex &&
           op == other.op && pm == other.pm;
  }

  bool sameChannel(const FlowVariable &other) const {
    assert(isLambda());
    assert(other.isLambda());
    return type == other.type && lambdaIndex == other.lambdaIndex &&
           op == other.op;
  }

  inline bool isPlusMinus() const { return pm == plusAndMinus; }
  inline FlowVariable getPlus() const {
    assert(isPlusMinus());
    FlowVariable p = *this;
    p.pm = plus;
    return p;
  }

  inline FlowVariable getMinus() const {
    assert(isPlusMinus());
    FlowVariable p = *this;
    p.pm = minus;
    return p;
  }

  bool isLambda() const {
    return type == FlowVariable::TYPE::inputLambda ||
           type == FlowVariable::TYPE::outputLambda ||
           type == FlowVariable::TYPE::internalLambda;
  }

  std::string getName() const {
    std::string sign;
    switch (pm) {
    case notApplicable:
      sign = "";
      break;
    case plus:
      sign = "+";
      break;
    case minus:
      sign = "-";
      break;
    case plusAndMinus:
      sign = "+-";
      break;
    }
    switch (type) {
    case internalState:
      return llvm::formatv("{0}{1}", state->getSMVName(), sign);
    case inputLambda:
      return llvm::formatv("{0}.in_{1}{2}", getUniqueName(op), lambdaIndex,
                           sign)
          .str();
    case outputLambda:
      return llvm::formatv("{0}.out_{1}{2}", getUniqueName(op), lambdaIndex,
                           sign)
          .str();
    case internalLambda:
      return llvm::formatv("{0}.#{1}{2}", getUniqueName(op), lambdaIndex, sign)
          .str();
    };
  }
};

// Hash implementation required so that FlowVariable can be used in an
// unordered_map
template <>
struct std::hash<FlowVariable> {
  size_t operator()(const FlowVariable &var) const {
    using std::hash;
    if (var.type == FlowVariable::TYPE::internalState) {
      return hash<FlowVariable::TYPE>()(var.type) ^
             hash<std::string>()(var.state->getSMVName());
    }
    return (hash<FlowVariable::TYPE>()(var.type) ^
            hash<unsigned>()(var.lambdaIndex) ^ hash<Operation *>()(var.op));
  }
};

namespace {
// Only the operators that are used have been implemented...
struct FlowExpression {
  std::unordered_map<FlowVariable, int> terms;
  FlowExpression() = default;
  FlowExpression(const FlowVariable &v) {
    if (v.isPlusMinus()) {
      // If plusAndMinus, separate into plus and minus parts
      terms[v.getPlus()] = 1;
      terms[v.getMinus()] = 1;
    } else {
      terms[v] = 1;
    }
  };
  void debug() const {
    std::string txt = "0 = ";
    bool first = true;
    for (auto &[key, value] : terms) {
      if (!first) {
        if (value > 0) {
          txt += " + ";
        } else if (value < 0) {
          txt += " - ";
        }
      } else {
        if (value < 0)
          txt += "-";
        first = false;
      }
      if (abs(value) == 1) {
        txt += key.getName();

      } else {
        txt += llvm::formatv("{0} * {1}", value, key.getName()).str();
      }
    }
    llvm::errs() << txt << "\n";
  }
};

FlowExpression operator-(FlowExpression expr) {
  for (auto &[key, value] : expr.terms) {
    expr.terms[key] = -value;
  }
  return expr;
}

FlowExpression operator+(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
  return left;
}

FlowExpression operator-(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
  return left;
}

void operator+=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
}

void operator-=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
}

// Used to assign dense indices to FlowVariables based on a list of
// FlowExpression, i.e. indices 0 to n-1 are used for n variables, while keeping
// lambda variables with low indices to ensure they are eliminated first within
// the row-echelon form
class FlowEquationsMatrix {
  std::unordered_map<FlowVariable, size_t> map;
  std::vector<FlowVariable> variables;
  size_t nLambdas;

public:
  MatIntType matrix;
  size_t getNLambdas() { return nLambdas; }
  size_t size() { return variables.size(); }
  size_t getIndex(const FlowVariable &v) { return map[v]; }
  FlowVariable &getVariable(size_t index) { return variables[index]; }
  void verify() {
    assert(map.size() == variables.size());
    for (size_t i = 0; i < variables.size(); ++i) {
      FlowVariable &a = variables[i];
      size_t j = map[a];
      assert(i == j);
    }
    for (auto &[key, value] : map) {
      assert(variables[value] == key);
    }
  }

  FlowEquationsMatrix() = default;
  FlowEquationsMatrix(const std::vector<FlowExpression> &exprs) {
    size_t index = 0;
    // annotate lambdas/+- first
    for (auto &expr : exprs) {
      for (auto &[key, value] : expr.terms) {
        // skip non-lambda variables that are not +-
        if (!key.isLambda() && key.pm == FlowVariable::PLUSMINUS::notApplicable)
          continue;
        // PlusAndMinus variables should never be inserted, as the DSL will
        // insert them as two separate variables
        assert(!key.isPlusMinus());
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
      for (auto &[key, value] : expr.terms) {
        if (map.count(key) == 0) {
          map[key] = index;
          ++index;
          variables.push_back(key);
        }
      }
    }

    // matrix with one row per equation, and column per variable
    matrix = MatIntZero(exprs.size(), size());

    // insert equations into the matrix
    for (auto [row, expr] : llvm::enumerate(exprs)) {
      llvm::errs() << "row " << row << ": ";
      expr.debug();
      for (auto &[key, value] : expr.terms) {
        unsigned index = getIndex(key);
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
      // Annotate channel equations
#if false
      for (auto [i, res] : llvm::enumerate(op.getResults())) {
        bool bitChannel = false;
        if (auto ct = dyn_cast<handshake::ChannelType>(res.getType())) {
          //llvm::errs() << llvm::formatv("{0}.{1} is {2} bits wide\n", getUniqueName(&op), i, ct.getDataBitWidth());
          if (ct.getDataBitWidth() == 1) {
            bitChannel = true;
          }
        }
        if (!isChannelToBeChecked(res))
          continue;

        for (auto &use : res.getUses()) {
          unsigned j = use.getOperandNumber();
          Operation &nextOp = *use.getOwner();
          assert(nextOp.getOperands().size() > j);
          if (bitChannel) {
            // + and - are both forwarded without change
            // This equation will be expanded into 2 equations later
            // lambda_forward+ = lambda_backward+
            // lambda_forward- = lambda_backward-
            FlowVariable forward(res);
            FlowVariable back(use, nextOp);
            llvm::errs() << llvm::formatv("from {0} to {1}\n", forward.getName(), back.getName());
            /*
            FlowVariable forward = FlowVariable::outputChannel(&op, i);
            forward.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            FlowVariable back = FlowVariable::inputChannel(&nextOp, j);
            back.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            */
            equations.push_back(forward - back);
          } else {
            FlowVariable forward = FlowVariable::outputChannel(&op, i);
            FlowVariable back = FlowVariable::inputChannel(&nextOp, j);
            // forward and back represent the same channel but from different
            // sides, so their lambdas have to be equal
            equations.push_back(forward - back);
          }
        }
      }
#endif

      // A general structure for an operation is assumed:
      // in1, in2, ... -> Join/Merge/Mux -> entry channel
      // entry channel -> slots? -> exit channel
      // exit channel 2 -> Fork/Branch -> out1, out2, ...
      //
      // Some operations do not follow this structure, and should be handled
      // separately to avoid making false assumptions.
      if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
        /*
        // From inspecting .td declaration... probably a better way of doing
        // this
        int addressInputIndex = 0;
        int dataOutputIndex = 1;
        FlowVariable in = FlowVariable::inputChannel(&op, addressInputIndex);
        FlowVariable out = FlowVariable::outputChannel(&op, dataOutputIndex);
        FlowVariable addrSlot = FlowVariable::slot(&op, 0);
        FlowVariable dataSlot = FlowVariable::slot(&op, 1);

        FlowVariable i1 = FlowVariable::internalChannel(&op, 1);
        FlowVariable i2 = FlowVariable::internalChannel(&op, 2);

        equations.push_back(in - i1 - addrSlot);
        equations.push_back(i2 - out - dataSlot);
        */
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

      // Annotate latency-induced slots
      FlowVariable exit = entry;
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
        equations.push_back(slot - slotPM);
        FlowVariable sentPM = indexSent;
        sentPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
        equations.push_back(indexSent - sentPM);

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
            FlowVariable sentPM = sent;
            sentPM.pm = FlowVariable::PLUSMINUS::plusAndMinus;
            equations.push_back(sent - sentPM);
            equations.push_back(exit.getPlus() + sentPM.getPlus() -
                                result.getPlus());
            equations.push_back(exit.getMinus() + sentPM.getMinus() -
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
} // namespace

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

    for (size_t col = indices.getNLambdas(); col < cols; ++col) {
      if (matrix(row, col) != 0) {
        coefs.push_back(matrix(row, col));
        names.push_back(indices.getVariable(col).getName());
      }
    }
    if (coefs.size() > 0) {
      ReconvergentPathFlow p(uid, FormalProperty::TAG::OPT);
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

  if (false) {
    if (failed(annotateAbsenceOfBackpressure(modOp)))
      return signalPassFailure();
    if (failed(annotateValidEquivalence(modOp)))
      return signalPassFailure();
  }
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
