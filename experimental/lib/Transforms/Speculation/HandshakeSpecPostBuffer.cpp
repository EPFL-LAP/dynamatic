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
#include "mlir/Support/LogicalResult.h"
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

    for (Value result : condBrOp->getResults()) {
      for (Operation *user : result.getUsers()) {

        if (isBackedge(result, user))
          return condBrOp;
      }
    }
  }

  return nullptr;
}

static FailureOr<Value> constructSaveCommitControl(SpeculatorOp speculator) {
  OpBuilder builder(speculator.getContext());
  builder.setInsertionPoint(speculator);
  unsigned specBB = getLogicBB(speculator).value();

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

  ConditionalBranchOp controlBranch =
      findControlBranch(speculator->getParentOfType<FuncOp>(), specBB);
  if (controlBranch == nullptr) {
    speculator->emitError()
        << "Could not find backedge within speculation bb: " << specBB << ".\n";
    return failure();
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
    return failure();
  }

  // All the inputs to the merge operation are ready
  auto mergeOp = builder.create<handshake::MergeOp>(branchReplicated.getLoc(),
                                                    mergeOperands);
  mergeOp->setAttr("specv1_sc_merge", builder.getUnitAttr());
  setBB(mergeOp, specBB);

  return mergeOp.getResult();
}

static LogicalResult placeAdditionalBuffers(SpeculatorOp speculator) {
  FuncOp funcOp = speculator->getParentOfType<FuncOp>();
  OpBuilder builder(funcOp.getContext());

  for (auto commitOp : funcOp.getOps<SpecCommitOp>()) {
    builder.setInsertionPoint(commitOp);
    // To maintain high throughput: commit op *sometimes* joins a control from
    // the iteration `i` with data from the iteration `i-1`. We need a 1-slot
    // buffer to hold the control signal
    auto bufOp1 =
        builder.create<BufferOp>(builder.getUnknownLoc(), commitOp.getCtrl(), 1,
                                 BufferType::FIFO_BREAK_NONE);
    inheritBB(commitOp, bufOp1);
    bufOp1.getOperand().replaceAllUsesExcept(bufOp1.getResult(), bufOp1);
  }

  // To avoid deadlock: on misspeculation, `kill` is only sent after `resend` is
  // accepted. Buffering algorithm ignores `resend`, and insufficient buffering
  // may cause deadlock. We buffer dataOut and commitCtrl of speculator and the
  // merged save-commit control for a `resend` iteration.

  auto bufDataOut =
      builder.create<BufferOp>(builder.getUnknownLoc(), speculator.getDataOut(),
                               1, BufferType::FIFO_BREAK_NONE);
  inheritBB(speculator, bufDataOut);
  speculator.getDataOut().replaceAllUsesExcept(bufDataOut.getResult(),
                                               bufDataOut);

  auto bufCommitCtrl = builder.create<BufferOp>(builder.getUnknownLoc(),
                                                speculator.getCommitCtrl(), 1,
                                                BufferType::FIFO_BREAK_NONE);
  inheritBB(speculator, bufCommitCtrl);
  speculator.getCommitCtrl().replaceAllUsesExcept(bufCommitCtrl.getResult(),
                                                  bufCommitCtrl);

  MergeOp merge;
  bool mergeFound = false;
  for (auto candidate : funcOp.getOps<MergeOp>()) {
    if (candidate->hasAttr("specv1_sc_merge")) {
      merge = candidate;
      mergeFound = true;
      break;
    }
  }
  if (!mergeFound) {
    funcOp.emitError("specv1_sc_merge not found");
    return failure();
  }
  auto bufSCControl =
      builder.create<BufferOp>(builder.getUnknownLoc(), merge.getResult(), 1,
                               BufferType::FIFO_BREAK_NONE);
  inheritBB(speculator, bufSCControl);
  merge.getResult().replaceAllUsesExcept(bufSCControl.getResult(),
                                         bufSCControl);

  return success();
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

  // Build the (post-buffer) SpeculatorOp
  SpeculatorOp speculator = builder.create<SpeculatorOp>(
      specOp1.getLoc(), specOp1.getDataOut().getType(), specOp2.getDataIn(),
      specOp1.getTrigger(), specOp1.getFifoDepth());
  setBB(speculator, specBB);

  specOp1.getDataOut().replaceAllUsesWith(speculator.getDataOut());
  specOp2.getSaveCtrl().replaceAllUsesWith(speculator.getSaveCtrl());
  specOp2.getCommitCtrl().replaceAllUsesWith(speculator.getCommitCtrl());
  specOp2.getSCIsMisspec().replaceAllUsesWith(speculator.getSCIsMisspec());

  auto scControl = constructSaveCommitControl(speculator);
  if (failed(scControl))
    return signalPassFailure();

  specOp1.getSCSaveCtrl().replaceAllUsesWith(scControl.value());

  specOp1->erase();
  specOp2->erase();

  if (failed(placeAdditionalBuffers(speculator)))
    return signalPassFailure();
}
