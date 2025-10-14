#include "SpecV1PostAdaptor.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_SPECV1POSTADAPTOR
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV1PostAdaptorPass
    : public dynamatic::experimental::speculationv2::impl::
          SpecV1PostAdaptorBase<SpecV1PostAdaptorPass> {
  using SpecV1PostAdaptorBase<SpecV1PostAdaptorPass>::SpecV1PostAdaptorBase;
  void runDynamaticPass() override;
};

void SpecV1PostAdaptorPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp.getContext());

  auto speculator = *funcOp.getOps<SpecPreBufferOp1>().begin();
  auto specBB = getLogicBB(speculator);
  assert(specBB.has_value() && "Speculator must be in a basic block");
  // Value loopCondition = nullptr;

  SmallVector<unsigned> loopBBs = {specBB.value()};
  for (auto branch :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branch) != *specBB)
      continue;
    if (branch->hasAttr("specv1_cond_br"))
      continue;
    if (disablePasserAtExits &&
        (!isInsideLoop(branch.getTrueResult(), loopBBs) ||
         !isInsideLoop(branch.getFalseResult(), loopBBs))) {
      // Do not insert passers at exits
      continue;
    }

    builder.setInsertionPoint(branch);
    Value inverted = nullptr;
    for (auto *user :
         iterateOverPossiblyIndirectUsers(branch.getConditionOperand())) {
      if (auto notOp = dyn_cast<NotOp>(user)) {
        inverted = notOp.getResult();
        break;
      }
    }
    if (inverted == nullptr) {
      NotOp notOp =
          builder.create<NotOp>(branch.getLoc(), branch.getConditionOperand());
      setBB(notOp, *specBB);
      inverted = notOp.getResult();
    }

    // Build a passer for the trueResult
    PasserOp trueResultPasser = builder.create<PasserOp>(
        branch.getLoc(), branch.getDataOperand(), branch.getConditionOperand());
    setBB(trueResultPasser, *specBB);
    branch.getTrueResult().replaceAllUsesWith(trueResultPasser.getResult());

    // Build a passer for the falseResult
    // The passer ctrl is inverted condition.
    PasserOp falseResultPasser = builder.create<PasserOp>(
        branch.getLoc(), branch.getDataOperand(), inverted);
    setBB(falseResultPasser, *specBB);
    branch.getFalseResult().replaceAllUsesWith(falseResultPasser.getResult());

    if (branch->hasAttr("specv1_adaptor_inner_loop")) {
      trueResultPasser->setAttr("specv1_adaptor_inner_loop",
                                branch->getAttr("specv1_adaptor_inner_loop"));
      falseResultPasser->setAttr("specv1_adaptor_inner_loop",
                                 branch->getAttr("specv1_adaptor_inner_loop"));
    } else {
      // if (isBackedge(trueResultPasser.getResult())) {
      //   loopCondition = trueResultPasser.getCtrl();
      // } else if (isBackedge(falseResultPasser.getResult())) {
      //   loopCondition = falseResultPasser.getCtrl();
      // }
    }

    branch->erase();
  }
  // assert(
  //     loopCondition != nullptr &&
  //     "Could not find the loop condition from branches in the speculator
  //     BB");
  // // Update the loop header (CMerge -> Init)
  // if (failed(updateLoopHeader(funcOp, loopBBs, loopCondition)))
  //   return signalPassFailure();
}
