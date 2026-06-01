// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKESPECPOSTBUFFER
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

#include "dynamatic/Analysis/NameAnalysis.h"
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
  void runOnOperation() override;
};

void HandshakeSpecPostBufferPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  // There should be exactly one function.
  auto funcOps = modOp.getOps<FuncOp>();
  if (std::distance(funcOps.begin(), funcOps.end()) != 1) {
    modOp.emitError() << "Expected a single FuncOp";
    return signalPassFailure();
  }
  FuncOp funcOp = *funcOps.begin();

  // No speculator: nothing for this pass to do.
  auto specOps = funcOp.getOps<SpeculatorOp>();
  if (specOps.empty())
    return;
  if (std::distance(specOps.begin(), specOps.end()) != 1) {
    funcOp.emitError() << "Expected at most one SpeculatorOp";
    return signalPassFailure();
  }
  SpeculatorOp specOp = *specOps.begin();

  // Stores in a do-while loop have commit units before them,
  // but the values in iteration 0 are non-speculative,
  // and so pass through the commit without any control signal.
  //
  // The commit control generated in iteration 0 is then used
  // to kill/discard the value going to the store in iteration 1.
  // In general, at that commit unit, the commit control joins
  // with data from the next iteration.
  //
  // Since the MILP buffering algorithm cannot see this,
  // the commit control path is sometimes underbuffered,
  // which causes backpressure into the speculator
  // until the data from the next iteration arrives.
  //
  // We therefore add 1 extra buffer on the control path
  // to the commit, if the commit is inside the same BB
  // as the speculator
  //
  // Maybe more are actually needed on some kernels, we do not
  // know exactly what the buffering requirements here are.
  OpBuilder builder(funcOp.getContext());
  unsigned specBB = getLogicBB(specOp).value();

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

  // Name any ops created by this pass.
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  if (failed(nameAnalysis.walk(NameAnalysis::UnnamedBehavior::NAME)))
    return signalPassFailure();
  markAnalysesPreserved<NameAnalysis>();
}

} // namespace
