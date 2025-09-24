#include "PreSpecV2Gamma.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <fstream>

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
#define GEN_PASS_DEF_PRESPECV2GAMMA
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct PreSpecV2GammaPass
    : public dynamatic::experimental::speculationv2::impl::PreSpecV2GammaBase<
          PreSpecV2GammaPass> {
  using PreSpecV2GammaBase<PreSpecV2GammaPass>::PreSpecV2GammaBase;
  void runDynamaticPass() override;
};

static void introduceGSA(FuncOp &funcOp, unsigned branchBB) {
  OpBuilder builder(funcOp->getContext());
  for (auto branchOp : funcOp.getOps<ConditionalBranchOp>()) {
    if (getLogicBB(branchOp) != branchBB)
      continue;
    if (!branchOp.getDataOperand().getType().isa<ControlType>())
      continue;

    ControlMergeOp trueCMerge =
        cast<ControlMergeOp>(getUniqueUser(branchOp.getTrueResult()));
    ControlMergeOp falseCMerge =
        cast<ControlMergeOp>(getUniqueUser(branchOp.getFalseResult()));

    assert(isa<SinkOp>(getUniqueUser(trueCMerge.getIndex())));
    assert(isa<SinkOp>(getUniqueUser(falseCMerge.getIndex())));

    BranchOp trueBranch;
    BranchOp falseBranch;
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(trueCMerge.getResult())) {
      if (auto br = dyn_cast<BranchOp>(user)) {
        trueBranch = br;
        break;
      }
    }
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(falseCMerge.getResult())) {
      if (auto br = dyn_cast<BranchOp>(user)) {
        falseBranch = br;
        break;
      }
    }
    assert(trueBranch && falseBranch);

    ControlMergeOp confluenceCMerge =
        cast<ControlMergeOp>(getUniqueUser(trueBranch.getResult()));
    assert(confluenceCMerge.getDataOperands()[0] == falseBranch.getResult());
    assert(confluenceCMerge.getDataOperands()[1] == trueBranch.getResult());

    Value condition = branchOp.getConditionOperand();

    builder.setInsertionPoint(confluenceCMerge);
    MuxOp newMux = builder.create<MuxOp>(
        builder.getUnknownLoc(), confluenceCMerge.getResult().getType(),
        condition,
        ArrayRef<Value>{confluenceCMerge.getDataOperands()[0],
                        confluenceCMerge.getDataOperands()[1]});
    inheritBB(confluenceCMerge, newMux);

    confluenceCMerge.getResult().replaceAllUsesWith(newMux.getResult());
    confluenceCMerge.getIndex().replaceAllUsesWith(condition);
    confluenceCMerge->erase();
  }
}

void PreSpecV2GammaPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  introduceGSA(funcOp, branchBB);

  if (failed(replaceBranchesWithPassers(funcOp, branchBB))) {
    funcOp.emitError("Failed to replace branches in BB 1 with passers");
    return signalPassFailure();
  }
}
