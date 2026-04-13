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

#include "dynamatic/Analysis/IndexChannelAnalysis.h"
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
#include "mlir/Pass/Pass.h"
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

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  auto &indexChannelAnalysis = getAnalysis<dynamatic::IndexChannelAnalysis>();

  // Local equations extracted in constructor
  FlowEquationExtractor extractor(indexChannelAnalysis);
  // This fails when some operations in the module are not yet handled
  if (failed(extractor.extractAll(modOp))) {
    return failure();
  }

  // Create a matrix, and map all variables to an column index
  FlowSystem indices(extractor.equations);
  MatIntType &matrix = indices.matrix;

  // Verify that the registry data structure is correct
  assert(indices.registry.verify());

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

namespace {
struct IOG {
  IOG() = default;
  std::unordered_set<Operation *> units;
  llvm::DenseSet<mlir::Value> channels;
  mlir::Value entry;

  bool contains(Operation *op) {
    auto iter = units.find(op);
    return iter != units.end();
  }

  bool contains(mlir::Value channel) {
    auto iter = channels.find(channel);
    return iter != channels.end();
  }

  void debug() {
    std::vector<mlir::Value> stack;
    llvm::DenseSet<mlir::Value> visited;
    stack.push_back(entry);
    visited.insert(entry);
    while (!stack.empty()) {
      mlir::Value channel = stack.back();
      stack.pop_back();
      Operation *prev = channel.getDefiningOp();
      if (prev == nullptr) {
        llvm::errs() << "entry";
      } else {
        llvm::errs() << getUniqueName(prev);
      }
      Operation *next = channel.getUses().begin()->getOwner();
      assert(next);
      assert(contains(next));
      llvm::errs() << " -> " << getUniqueName(next) << "\n";
      for (mlir::Value out : next->getResults()) {
        if (auto iter = visited.find(out); iter != visited.end())
          continue;
        if (contains(out)) {
          visited.insert(out);
          stack.push_back(out);
        }
      }
    }
  }
};

struct IOGFinderCheckpoint {
  IOG partial;
  // Stack of channels at the edge that are not yet inserted into the IOG
  std::vector<mlir::Value> unhandled;
  // Set of channels that have been declared illegal by local rules
  llvm::DenseSet<mlir::Value> illegals;
  // Is the partial IOG still following the rules? Note that DONE does not
  // actually mean done, as the `unhandled` stack should also be empty.
  enum PROGRESS { ILLEGAL, PARTIAL, DONE };
  PROGRESS progress = PARTIAL;
  Operation *endOp;
  std::vector<mlir::Value> entries;

  PROGRESS getProgress() {
    if (progress == ILLEGAL)
      return ILLEGAL;
    if (!unhandled.empty())
      return PARTIAL;
    return progress;
  }

  bool isLegal(mlir::Value channel) {
    auto iter = illegals.find(channel);
    return iter == illegals.end();
  }

  void follow(mlir::Value channel) {
    if (!isLegal(channel)) {
      // A non-legal channel has been followed, so mark this IOG as non-valid
      progress = ILLEGAL;
      return;
    }
    if (partial.contains(channel)) {
      return;
    }
    unhandled.push_back(channel);
  }

  void makeIllegal(mlir::Value channel) {
    if (partial.contains(channel))
      progress = ILLEGAL;
    // The legality of unhandled channels is checked in getNextOp
    illegals.insert(channel);
  }

  Operation *getNextOp() {
    // Stop making progress if an illegal edge has been taken
    if (progress == ILLEGAL) {
      return nullptr;
    }

    // Handle the `endOp` if it has not yet been inserted into the IOG
    if (!partial.contains(endOp)) {
      return endOp;
    }

    if (unhandled.empty()) {
      // All handling done!
      return nullptr;
    }

    // Look at channel at the top of the `unhandled` stack
    auto channel = unhandled.back();

    // If this channel has been marked as illegal, abort this branch
    if (!isLegal(channel)) {
      progress = ILLEGAL;
      return nullptr;
    }

    // If this channel is one of the target entry channels, mark the appropriate
    // `entry` of the IOG and set the state accordingly
    if (std::find(entries.begin(), entries.end(), channel) != entries.end()) {
      partial.entry = channel;
      progress = DONE;
    }

    // To fully handle a channel, both the operation before and after have to be
    // handled
    Operation *before = channel.getDefiningOp();
    if (before && !partial.contains(before)) {
      partial.units.insert(before);
      return before;
    }

    Operation *after = channel.getUses().begin()->getOwner();
    if (after && !partial.contains(after)) {
      partial.units.insert(after);
      return after;
    }

    // Both sides of this channel have been handled, so this channel is done and
    // can be removed/inserted into the appropriate locations
    unhandled.pop_back();
    partial.channels.insert(channel);

    // Return next channel on unhandled stack
    return getNextOp();
  }
};

struct IOGFinder {
  IOGFinder() = default;
  ~IOGFinder() = default;
  std::vector<IOGFinderCheckpoint> partials;
  std::vector<IOG> finals;

  // This function takes a checkpoint and a list of channels. For each channel
  // c, it pushes the IOG where *only* channel c is taken, and all the rest are
  // illegal. Used for forks (which only allow exactly one of its outputs) and
  // join operations (which only allow exactly one of its inputs)
  template <typename T>
  void followSingle(const IOGFinderCheckpoint &checkpoint, T options) {
    for (mlir::Value channel : options) {
      IOGFinderCheckpoint goHere = checkpoint;
      goHere.follow(channel);
      for (mlir::Value other : options) {
        if (channel == other)
          continue;
        goHere.makeIllegal(other);
      }
      partials.push_back(goHere);
    }
  }

  // Take a step towards computing all IOGs. Does nothing if there are no
  // partial IOGs.
  // Example usage:
  // IOGFinder finder = ...
  // while (!finder.partials.empty()) {
  //   finder.step();
  // }
  void step() {
    if (partials.empty())
      return;
    auto checkpoint = partials.back();
    partials.pop_back();
    if (checkpoint.progress == IOGFinderCheckpoint::PROGRESS::ILLEGAL)
      return;

    Operation *op = checkpoint.getNextOp();
    if (op == nullptr) {
      if (checkpoint.progress == IOGFinderCheckpoint::PROGRESS::DONE) {
        finals.push_back(checkpoint.partial);
      }
      return;
    }
    checkpoint.partial.units.insert(op);

    if (auto endOp = dyn_cast<EndOp>(op)) {
      followSingle(checkpoint, endOp.getOperands());
    } else if (auto bufOp = dyn_cast<BufferOp>(op)) {
      checkpoint.follow(bufOp.getOperand());
      checkpoint.follow(bufOp.getResult());
      partials.push_back(checkpoint);
    } else if (auto condBr = dyn_cast<ConditionalBranchOp>(op)) {
      checkpoint.follow(condBr.getTrueResult());
      checkpoint.follow(condBr.getFalseResult());
      followSingle(checkpoint, condBr.getOperands());
    } else if (auto forkOp = dyn_cast<ForkOp>(op)) {
      checkpoint.follow(forkOp.getOperand());
      followSingle(checkpoint, forkOp.getResults());
    } else if (auto muxOp = dyn_cast<MuxOp>(op)) {
      // Either all data inputs, or the select input. Output always
      checkpoint.follow(muxOp.getResult());
      auto dataFollower = checkpoint;
      auto selectFollower = checkpoint;
      for (auto data : muxOp.getDataOperands()) {
        dataFollower.follow(data);
        selectFollower.makeIllegal(data);
      }

      dataFollower.makeIllegal(muxOp.getSelectOperand());
      selectFollower.follow(muxOp.getSelectOperand());

      partials.push_back(dataFollower);
      partials.push_back(selectFollower);
    } else if (auto cmerge = dyn_cast<ControlMergeOp>(op)) {
      for (auto data : cmerge.getDataOperands()) {
        checkpoint.follow(data);
      }
      followSingle(checkpoint, cmerge.getResults());
    } else if (auto arithOp = dyn_cast<ArithOpInterface>(op)) {
      checkpoint.follow(arithOp->getResults()[0]);
      followSingle(checkpoint, arithOp->getOperands());
    } else if (isa<SourceOp, SinkOp, StoreOp>(op)) {
      // These operations can not be part of an IOG, as they generate/consume
      // tokens. This means that the partial IOGs looking at these operations
      // can be discarded, so nothing needs to be done
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      checkpoint.follow(loadOp.getAddress());
      checkpoint.follow(loadOp.getAddressResult());
      checkpoint.follow(loadOp.getData());
      checkpoint.follow(loadOp.getDataResult());
      partials.push_back(checkpoint);
    } else if (auto memCon = dyn_cast<MemoryControllerOp>(op)) {
      // Memory controllers will be handled at the annotation part. To make sure
      // this IOG is not discarded, this checkpoint is added back to the stack
      // of partial IOGs
      partials.push_back(checkpoint);
    } else {
      op->emitError("unhandled case in IOG finding");
    }
  }
};

std::vector<IOG> findAllIOGsWithEndOp(std::vector<mlir::Value> entries,
                                      EndOp endOp) {
  IOGFinder finder{};
  auto x = IOGFinderCheckpoint{
      .partial = IOG(),
      .unhandled = std::vector<mlir::Value>(),
      .illegals = llvm::DenseSet<mlir::Value>(),
      .endOp = endOp,
      .entries = std::move(entries),
  };
  finder.partials.push_back(x);

  // Compute all IOGs
  while (!finder.partials.empty()) {
    finder.step();
  }
  return finder.finals;
}

std::vector<IOG> findAllIOGs(ModuleOp modOp) {
  std::vector<IOG> ret{};
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    // Each argument corresponds to an entry node.
    std::vector<mlir::Value> arguments;
    for (mlir::Value arg : funcOp.getRegion().getArguments()) {
      arguments.push_back(arg);
    }

    for (auto &op : funcOp.getOps()) {
      if (auto endOp = dyn_cast<EndOp>(op)) {
        for (auto &x : findAllIOGsWithEndOp(arguments, endOp)) {
          ret.push_back(x);
        }
      }
    }
  }
  for (auto &x : ret) {
    llvm::errs() << "---------DEBUG----------\n";
    x.debug();
    llvm::errs() << "------------------------\n";
  }
  return ret;
}
} // namespace

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

    findAllIOGs(modOp);
  }

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}
