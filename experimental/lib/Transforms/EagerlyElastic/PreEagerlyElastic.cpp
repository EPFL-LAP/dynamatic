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
#define GEN_PASS_DEF_HANDSHAKESIZELSQS
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
  handshake::FuncOp funcOp = getOperation();

  // iterate over all basic blocks
  for (Block &block : funcOp.getBody()) {
    // iterate over all operations inside the current bb
    for (Operation &op : block) {
      // check if the operation is a conditional branch
      auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(&op);
      if (!branchOp || !branchOp->hasAttr("ftd.skip"))
        continue;

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
          llvm::outs() << "    True Result: " << user->getName().getStringRef()
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
          llvm::outs() << "    False Result: " << user->getName().getStringRef()
                       << " (" << userName << ")\n";
        }
      }
    }
  }
}
