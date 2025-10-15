#include "HandshakeSpecPostBuffer.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
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

static Operation *getUserSkippingBuffers(Value val) {
  Operation *uniqueUser = *val.getUsers().begin();
  if (auto bufOp = dyn_cast<BufferOp>(uniqueUser)) {
    return getUserSkippingBuffers(bufOp.getResult());
  }
  return uniqueUser;
}

static handshake::ConditionalBranchOp findControlBranch(FuncOp funcOp,
                                                        unsigned bb) {
  for (auto condBrOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    if (auto brBB = getLogicBB(condBrOp); !brBB || brBB != bb)
      continue;

    llvm::errs() << "Candidate: ";
    condBrOp->dump();

    for (Value result : condBrOp->getResults()) {
      for (Operation *user : result.getUsers()) {

        if (isBackedge(result, user))
          return condBrOp;
        llvm::errs() << "Rejecting user: ";
        user->dump();
      }
    }
  }

  return nullptr;
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

  unsigned specBB = getLogicBB(specOp1).value();

  OpBuilder builder(&getContext());
  builder.setInsertionPoint(specOp1);

  SpeculatorOp speculator = builder.create<SpeculatorOp>(
      specOp1.getLoc(), specOp1.getDataOut().getType(), specOp2.getDataIn(),
      specOp1.getTrigger(), specOp1.getFifoDepth());
  setBB(speculator, specBB);

  // speculator.setConstant(constant);
  // speculator.setDefaultValue(defaultValue);

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
  setBB(branchReplicated, specBB);

  // We create a Merge operation to join SCCSaveCtrl and SCCommitCtrl signals
  SmallVector<Value, 2> mergeOperands;
  mergeOperands.push_back(speculator.getSCSaveCtrl());

  ConditionalBranchOp controlBranch = findControlBranch(funcOp, specBB);
  if (controlBranch == nullptr) {
    specOp1->emitError() << "Could not find backedge within speculation bb: "
                         << specBB << ".\n";
    return signalPassFailure();
  }

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
  if (isBranchBackedge(controlBranch.getTrueResult())) {
    mergeOperands.push_back(branchReplicated.getTrueResult());
  }
  // Check if falseResult of controlBranch leads to a backedge (loop)
  else if (isBranchBackedge(controlBranch.getFalseResult())) {
    mergeOperands.push_back(branchReplicated.getFalseResult());
  }
  // If neither trueResult nor falseResult leads to a backedge, handle the error
  else {
    controlBranch->emitError()
        << "Could not find the backedge in the Control Branch " << specBB
        << "\n";
    return signalPassFailure();
  }

  // All the inputs to the merge operation are ready
  auto mergeOp = builder.create<handshake::MergeOp>(branchReplicated.getLoc(),
                                                    mergeOperands);
  setBB(mergeOp, specBB);

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
    builder.setInsertionPointAfterValue(specResult);
    // Create a buffer to hold the delayed operand
    auto bufOp = builder.create<BufferOp>(builder.getUnknownLoc(), specResult,
                                          1, BufferType::FIFO_BREAK_NONE);
    inheritBBFromValue(specResult, bufOp);
    specResult.replaceAllUsesExcept(bufOp.getResult(), bufOp);
  }
}
