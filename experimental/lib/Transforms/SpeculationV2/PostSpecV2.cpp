#include "PostSpecV2.h"
#include "JSONImporter.h"
#include "MaterializationUtil.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
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
#define GEN_PASS_DEF_POSTSPECV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct PostSpecV2Pass
    : public dynamatic::experimental::speculationv2::impl::PostSpecV2Base<
          PostSpecV2Pass> {
  using PostSpecV2Base<PostSpecV2Pass>::PostSpecV2Base;
  void runDynamaticPass() override;
};

static InitOp moveInitsUp(Value fedValue) {
  fedValue = getForkTop(fedValue);
  materializeValue(fedValue);

  Operation *uniqueUser = getUniqueUser(fedValue);
  if (auto init = dyn_cast<InitOp>(uniqueUser)) {
    // Single use. No need to move repeating inits.
    return init;
  }

  OpBuilder builder(fedValue.getContext());
  builder.setInsertionPointAfterValue(fedValue);

  ForkOp fork = cast<ForkOp>(uniqueUser);

  auto newInit = builder.create<InitOp>(fedValue.getLoc(), fedValue, 0);
  inheritBB(fedValue.getDefiningOp(), newInit);
  fedValue.replaceAllUsesExcept(newInit.getResult(), newInit);

  for (auto res : fork->getResults()) {
    if (auto init = dyn_cast<InitOp>(getUniqueUser(res))) {
      init.getResult().replaceAllUsesWith(newInit.getResult());
      init->erase();
    } else {
      res.replaceAllUsesWith(fedValue);
    }
  }

  fork->erase();

  materializeValue(fedValue);
  materializeValue(newInit.getResult());

  return newInit;
}

bool tryErasePasser(PasserOp passer) {
  Value result = passer.getResult();

  if (result.use_empty()) {
    passer->erase();
    return true;
  }
  assert(result.hasOneUse());
  Operation *user = getUniqueUser(result);
  if (isa<SinkOp>(user)) {
    user->erase();
    passer->erase();
    return true;
  }
  if (auto childPasser = dyn_cast<PasserOp>(user)) {
    if (tryErasePasser(childPasser)) {
      passer->erase();
      return true;
    }
  }
  return false;
}

void PostSpecV2Pass::runDynamaticPass() {
  // Parse json (jsonPath is a member variable handled by tablegen)
  auto bbOrFailure = readFromJSON(jsonPath);
  if (failed(bbOrFailure))
    return signalPassFailure();

  auto [loopBBs] = bbOrFailure.value();

  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  unsigned headBB = loopBBs[0];

  // Move the init op up
  for (auto init : funcOp.getOps<InitOp>()) {
    if (getLogicBB(init) != headBB)
      continue;
    moveInitsUp(init.getOperand());
    break;
  }

  // Erase unused PasserOps
  for (auto passerOp : llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
    tryErasePasser(passerOp);
  }
}
