#include "HandshakeSpecPostBuffer.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

namespace dynamatic {
namespace experimental {
namespace speculation {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_HANDSHAKESPECPOSTBUFFER
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

struct HandshakeSpecPostBufferPass
    : public dynamatic::experimental::speculation::impl::
          HandshakeSpecPostBufferBase<HandshakeSpecPostBufferPass> {
  using HandshakeSpecPostBufferBase<
      HandshakeSpecPostBufferPass>::HandshakeSpecPostBufferBase;
  void runDynamaticPass() override;
};

Operation *getUserSkippingBuffers(Value val) {
  Operation *uniqueUser = *val.getUsers().begin();
  if (auto bufOp = dyn_cast<BufferOp>(uniqueUser)) {
    return getUserSkippingBuffers(bufOp.getResult());
  }
  return uniqueUser;
}

Operation *getDefiningOpSkippingBuffersAndFork(Value val) {
  Operation *definingOp = val.getDefiningOp();
  if (auto bufOp = dyn_cast<BufferOp>(definingOp)) {
    return getDefiningOpSkippingBuffersAndFork(bufOp.getOperand());
  }
  if (auto forkOp = dyn_cast<ForkOp>(definingOp)) {
    return getDefiningOpSkippingBuffersAndFork(forkOp.getOperand());
  }
  return definingOp;
}

void HandshakeSpecPostBufferPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  SpecPreBufferOp1 specOp1 = *funcOp.getOps<SpecPreBufferOp1>().begin();
  SpecPreBufferOp2 specOp2 = *funcOp.getOps<SpecPreBufferOp2>().begin();

  OpBuilder builder(&getContext());
  builder.setInsertionPoint(specOp1);

  SpeculatorOp speculator = builder.create<SpeculatorOp>(
      specOp1.getLoc(), specOp1.getDataOut().getType(), specOp2.getDataIn(),
      specOp1.getTrigger(), specOp1.getFifoDepth());
  inheritBB(specOp1, speculator);

  specOp1.getDataOut().replaceAllUsesWith(speculator.getDataOut());
  specOp2.getSaveCtrl().replaceAllUsesWith(speculator.getSaveCtrl());
  specOp2.getCommitCtrl().replaceAllUsesWith(speculator.getCommitCtrl());
  specOp2.getSCIsMisspec().replaceAllUsesWith(speculator.getSCIsMisspec());

  // Construct Save Commit Control
  auto branchDiscardCondNonMisspec = cast<ConditionalBranchOp>(
      getUserSkippingBuffers(speculator.getSCIsMisspec()));

  // This branch will propagate the signal SCCommitControl according to
  // the control branch condition, which comes from branchDiscardCondNonMisSpec
  auto branchReplicated = builder.create<ConditionalBranchOp>(
      branchDiscardCondNonMisspec.getLoc(),
      branchDiscardCondNonMisspec.getTrueResult(),
      speculator.getSCCommitCtrl());
  inheritBB(speculator, branchReplicated);

  // We create a Merge operation to join SCCSaveCtrl and SCCommitCtrl signals
  SmallVector<Value, 2> mergeOperands;
  mergeOperands.push_back(speculator.getSCSaveCtrl());

  // Helper function to check if a value leads to a Backedge
  auto isBranchBackedge = [&](Value result) {
    return llvm::any_of(result.getUsers(), [&](Operation *user) {
      return isBackedge(result, user);
    });
  };

  // We need to send the control token to the same path that the speculative
  // token followed. Hence, if any branch output leads to a backedge, replicate
  // the branch in the SaveCommit control path.

  // Check if trueResult of controlBranch leads to a backedge (loop)
  auto specBranch =
      cast<SpeculatingBranchOp>(getDefiningOpSkippingBuffersAndFork(
          branchDiscardCondNonMisspec.getDataOperand()));
  bool branchFound = false;
  for (Operation *user :
       iterateOverPossiblyIndirectUsers(specBranch.getDataOperand())) {
    if (auto branch = dyn_cast<ConditionalBranchOp>(user)) {
      if (isBranchBackedge(branch.getTrueResult())) {
        mergeOperands.push_back(branchReplicated.getTrueResult());
        branchFound = true;
        break;
      }
      // Check if falseResult of controlBranch leads to a backedge (loop)
      if (isBranchBackedge(branch.getFalseResult())) {
        mergeOperands.push_back(branchReplicated.getFalseResult());
        branchFound = true;
        break;
      }
    }
  }
  if (!branchFound) {
    llvm::report_fatal_error("No valid branch found");
    return signalPassFailure();
  }

  // All the inputs to the merge operation are ready
  auto mergeOp = builder.create<handshake::MergeOp>(branchReplicated.getLoc(),
                                                    mergeOperands);
  inheritBB(speculator, mergeOp);

  specOp1.getSCSaveCtrl().replaceAllUsesWith(mergeOp.getResult());

  specOp1->erase();
  specOp2->erase();

  for (auto commitOp : funcOp.getOps<SpecCommitOp>()) {
    builder.setInsertionPoint(commitOp);
    // To maintain high throughput: commit op *sometimes* joins a control with
    // data from the iteration i-1. We need a 1-slot buffer to hold the control
    // signal
    auto bufOp1 =
        builder.create<BufferOp>(builder.getUnknownLoc(), commitOp.getCtrl(), 1,
                                 BufferType::FIFO_BREAK_NONE);
    inheritBB(commitOp, bufOp1);
    bufOp1.getOperand().replaceAllUsesExcept(bufOp1.getResult(), bufOp1);

    // To avoid deadlock: On misspeculation, kill follows resend. When "resend"
    // is sent, misspeculated data doesn't join with "kill" signal. The stall of
    // this data possibly leads to a deadlock ("resend" is never accepted by a
    // save-commit). Therefore, we need a 1-slot buffer to store the token
    auto bufOp2 =
        builder.create<BufferOp>(builder.getUnknownLoc(), commitOp.getDataIn(),
                                 1, BufferType::FIFO_BREAK_NONE);
    inheritBB(commitOp, bufOp2);
    bufOp2.getOperand().replaceAllUsesExcept(bufOp2.getResult(), bufOp2);
  }

  for (auto specBranchOp : funcOp.getOps<SpeculatingBranchOp>()) {
    Value specResult = specBranchOp.getTrueResult();
    for (Operation *user : iterateOverPossiblyIndirectUsers(specResult)) {
      if (isa<SinkOp>(user))
        continue;
      if (auto branch = dyn_cast<ConditionalBranchOp>(user)) {
        Value delayedOperand;
        if (equalsIndirectly(branch.getDataOperand(), specResult))
          delayedOperand = branch.getDataOperand();
        else if (equalsIndirectly(branch.getConditionOperand(), specResult))
          delayedOperand = branch.getConditionOperand();
        else
          llvm::report_fatal_error(
              "Could not find the correct operand to delay");

        builder.setInsertionPoint(branch);
        // Create a buffer to hold the delayed operand
        auto bufOp =
            builder.create<BufferOp>(builder.getUnknownLoc(), delayedOperand, 1,
                                     BufferType::FIFO_BREAK_NONE);
        inheritBB(branch, bufOp);
        bufOp.getOperand().replaceAllUsesExcept(bufOp.getResult(), bufOp);
      } else {
        user->emitError() << "Unexpected user of speculative branch: " << *user;
      }
    }
  }
}
