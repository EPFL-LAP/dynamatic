#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace dynamatic;
using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PREEAGERLYELASTIC
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

struct PreEagerlyElasticPass
    : public dynamatic::experimental::impl::PreEagerlyElasticBase<
          PreEagerlyElasticPass> {
  using PreEagerlyElasticBase::PreEagerlyElasticBase;

  void runOnOperation() override;

  // private:
};

void PreEagerlyElasticPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  if (!funcOp) {
    llvm::errs() << "No funcOp found!\n";
    return signalPassFailure();
  }

  SmallVector<Operation *> branchesToErase;

  // iterate over all basic blocks
  for (Block &block : funcOp.getBody()) {
    // iterate over all operations inside the current bb
    for (Operation &op : block) {
      // check if the operation is a conditional branch
      auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(&op);
      if (!branchOp || !branchOp->hasAttr("ftd.skip"))
        continue;

      // skip suppressors with a sink
      bool trueRes = branchOp.getTrueResult().use_empty();
      bool falseRes = branchOp.getFalseResult().use_empty();
      if (trueRes || falseRes)
        continue;

      // convert suppressors without sinks into two suppressors
      // set up the insertion point right before the original branch
      OpBuilder builder(branchOp);

      Value condition = branchOp.getConditionOperand();
      Value data = branchOp.getDataOperand();
      Location loc = branchOp.getLoc();

      // create suppressor for true path
      auto suppressorA =
          builder.create<handshake::ConditionalBranchOp>(loc, condition, data);
      suppressorA->setAttr("ftd.skip", branchOp->getAttr("ftd.skip"));
      if (auto bbAttr = branchOp->getAttr("handshake.bb"))
        suppressorA->setAttr("handshake.bb", bbAttr);

      // create suppressor for false path
      auto suppressorB =
          builder.create<handshake::ConditionalBranchOp>(loc, condition, data);
      suppressorB->setAttr("ftd.skip", branchOp->getAttr("ftd.skip"));
      if (auto bbAttr = branchOp->getAttr("handshake.bb"))
        suppressorB->setAttr("handshake.bb", bbAttr);

      // rewire the downstream consumers to point to our new split branches
      branchOp.getTrueResult().replaceAllUsesWith(suppressorA.getTrueResult());
      branchOp.getFalseResult().replaceAllUsesWith(
          suppressorB.getFalseResult());

      // erase the original combined branch
      branchesToErase.push_back(branchOp);

      /*
      // print the name of the branch as well as to where it goes
      std::string branchName = "unnamed_branch";
      if (auto nameAttr = branchOp->getAttrOfType<StringAttr>("handshake.name"))
        branchName = nameAttr.getValue().str();
      llvm::errs() << "Suppressor Branch: " << branchName << '\n';

      if (branchOp.getTrueResult().use_empty()) {
        llvm::errs() << "     True Result: SINK\n";
      } else {
        for (Operation *user : branchOp.getTrueResult().getUsers()) {
          std::string userName =
              user->getAttrOfType<StringAttr>("handshake.name")
                  .getValue()
                  .str();
          llvm::errs() << "    True Result: " << user->getName().getStringRef()
                       << " (" << userName << ")\n";
        }
      }

      if (branchOp.getFalseResult().use_empty()) {
        llvm::errs() << "     False Result: SINK\n";
      } else {
        for (Operation *user : branchOp.getFalseResult().getUsers()) {
          std::string userName =
              user->getAttrOfType<StringAttr>("handshake.name")
                  .getValue()
                  .str();
          llvm::errs() << "    False Result: " << user->getName().getStringRef()
                       << " (" << userName << ")\n";
        }
      }
      */
    }
  }
  // safely erase after loop is done
  for (Operation *deadOp : branchesToErase) {
    deadOp->erase();
  }
}
