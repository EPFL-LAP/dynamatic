// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKESPECPOSTBUFFER
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

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

namespace {

struct HandshakeSpecPostBufferPass
    : public dynamatic::experimental::impl::HandshakeSpecPostBufferBase<
          HandshakeSpecPostBufferPass> {
  using HandshakeSpecPostBufferBase::HandshakeSpecPostBufferBase;
  void runDynamaticPass() override;
};

/// Struct to allow packaged return of the funcOp
/// and pre-buffering spec ops
struct SpecOps {
  FuncOp funcOp;
  SpecPreBufferOp1 specOp1;
  SpecPreBufferOp2 specOp2;
};

/// Safely get the FuncOp, SpecPreBufferOp1, and SpecPreBufferOp2
FailureOr<SpecOps> getSpecOps(ModuleOp modOp) {

  // There should be exactly one function
  auto funcOps = modOp.getOps<FuncOp>();
  if (std::distance(funcOps.begin(), funcOps.end()) != 1) {
    modOp.emitError() << "Expected a single FuncOp";
    return failure();
  }
  FuncOp funcOp = *funcOps.begin();

  // The speculation pass splits the speculator into two placeholder
  // ops (SpecPreBuffer1 and SpecPreBuffer2) so the MILP can treat
  // them as standard join-semantic units. There should be exactly
  // one of each — they get coalesced into the real speculator
  // by this pass.
  auto op1Range = funcOp.getOps<SpecPreBufferOp1>();
  if (std::distance(op1Range.begin(), op1Range.end()) != 1) {
    funcOp.emitError() << "Expected exactly one SpecPreBufferOp1";
    return failure();
  }

  auto op2Range = funcOp.getOps<SpecPreBufferOp2>();
  if (std::distance(op2Range.begin(), op2Range.end()) != 1) {
    funcOp.emitError() << "Expected exactly one SpecPreBufferOp2";
    return failure();
  }

  return SpecOps{funcOp, *op1Range.begin(), *op2Range.begin()};
}

Operation *getUserSkippingBuffers(Value val) {
  Operation *uniqueUser = *val.getUsers().begin();
  if (auto bufOp = dyn_cast<BufferOp>(uniqueUser)) {
    return getUserSkippingBuffers(bufOp.getResult());
  }
  return uniqueUser;
}

FailureOr<handshake::ConditionalBranchOp>
findControlBranch(FuncOp funcOp, unsigned bb) {
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

  return failure();
}



/// Stores in a do-while loop have commit units before them,
/// but the values in iteration 0 are non-speculative,
/// and so pass through the commit without any control signal.
///
/// The commit control generated in iteration 0 is then used
/// to kill/discard the value going to the store in iteration 1.
/// In general, at that commit unit, the commit control joins
/// with data from the next iteration.
///
/// Since the MILP buffering algorithm cannot see this,
/// the commit control path is sometimes underbuffered,
/// which causes backpressure into the speculator
/// until the data from the next iteration arrives.
///
/// We therefore add 1 extra buffer on the control path
/// to the commit, if the commit is inside the same BB
/// as the speculator
///
/// Maybe more are actually needed on some kernels, we do not
/// know exactly what the buffering requirements here are.
void bufferCommitCtrl(FuncOp funcOp, SpeculatorOp speculator) {
  OpBuilder builder(funcOp.getContext());
  unsigned specBB = getLogicBB(speculator).value();

  // any commit could be in the do-while loop
  for (auto commitOp : funcOp.getOps<SpecCommitOp>()) {
    builder.setInsertionPoint(commitOp);

    // only buffer ctrl for commits in the do-while loop
    if (getLogicBB(commitOp) == specBB) {
      // get the control signal
      Value ctrlInput = commitOp.getCtrl();

      // place a new 1 slot break none buffer
      // which consumes the control signal
      auto bufCtrl = builder.create<BufferOp>(
          /*error message origin=*/commitOp.getLoc(),
          /*input=*/ctrlInput,
          /*numSlots=*/1,
          /*type=*/BufferType::FIFO_BREAK_NONE);

      // give it the right bb
      inheritBB(commitOp, bufCtrl);

      // Rewire IR to use the buffered control signal
      ctrlInput.replaceAllUsesExcept(
          /*newValue=*/bufCtrl.getResult(),
          /*exceptedUser=*/bufCtrl);
    }
  }
}


FailureOr<SpeculatorOp> coalesceSpecOps(FuncOp funcOp,
                                        SpecPreBufferOp1 specOp1,
                                        SpecPreBufferOp2 specOp2,
                                        unsigned specBB,
                                        OpBuilder &builder) {
  // Build the SpeculatorOp
  // (specify inputs at construction)
  SpeculatorOp speculator = builder.create<SpeculatorOp>(
      /*error message origin=*/specOp1.getLoc(),
      /*type we are speculating=*/specOp1.getDataOut().getType(),
      /*the actual value=*/specOp2.getDataIn(),
      /*trigger=*/specOp1.getTrigger(),
      /*prediction fifo depth=*/specOp1.getFifoDepth());

  // inherit bb
  setBB(speculator, specBB);

  // rewire the rest of the IR to use the speculator outputs
  specOp1.getDataOut().replaceAllUsesWith(speculator.getDataOut());
  specOp2.getCommitCtrl().replaceAllUsesWith(speculator.getCommitCtrl());

  specOp1.getIssueCtrl().replaceAllUsesWith(speculator.getIssueCtrl());
  specOp1.getResolveCtrl().replaceAllUsesWith(speculator.getResolveCtrl());

  specOp1->erase();
  specOp2->erase();

  return speculator;
}

LogicalResult placeAdditionalBuffers(FuncOp funcOp,
                                            SpeculatorOp speculator) {
  // solve underbuffering of cross iteration joining
  // at commits on stores in do-while loops
  bufferCommitCtrl(funcOp, speculator);

  return success();
}

void HandshakeSpecPostBufferPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // safely get the funcOp, specOp1 and specOp2
  auto ops = getSpecOps(modOp);
  if (failed(ops))
    return signalPassFailure();

  // getSpecOps returns a struct, unpacking here
  auto [funcOp, specOp1, specOp2] = *ops;

  // get the spec bb
  unsigned specBB = getLogicBB(specOp1).value();

  // setup builder, set insertion point in serialized IR
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(specOp1);

  auto speculator = coalesceSpecOps(funcOp, specOp1, specOp2, specBB, builder);
  if (failed(speculator))
    return signalPassFailure();

  if (failed(placeAdditionalBuffers(funcOp, *speculator)))
    return signalPassFailure();
}

} // namespace
