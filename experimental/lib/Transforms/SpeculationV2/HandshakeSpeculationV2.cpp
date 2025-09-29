#include "HandshakeSpeculationV2.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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
#define GEN_PASS_DEF_HANDSHAKESPECULATIONV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct HandshakeSpeculationV2Pass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeSpeculationV2Base<HandshakeSpeculationV2Pass> {
  using HandshakeSpeculationV2Base<
      HandshakeSpeculationV2Pass>::HandshakeSpeculationV2Base;
  void runDynamaticPass() override;
};

/// Returns if the circuit is eligible for MuxPasserSwap.
static bool isEligibleForMuxPasserSwap(MuxOp muxOp, bool reason = false) {
  // Ensure the rewritten subcircuit structure.
  Operation *backedgeDefiningOp = muxOp.getDataOperands()[1].getDefiningOp();
  if (!isa<PasserOp>(backedgeDefiningOp)) {
    if (reason)
      llvm::errs() << "data1 is not passer\n";
    return false;
  }
  auto passerOp = cast<PasserOp>(backedgeDefiningOp);

  Operation *selectorDefiningOp = muxOp.getSelectOperand().getDefiningOp();
  if (!isa<InitOp>(selectorDefiningOp)) {
    if (reason)
      llvm::errs() << "mux select is not init\n";
    return false;
  }
  auto initOp = cast<InitOp>(selectorDefiningOp);

  if (!equalsIndirectly(passerOp.getCtrl(), initOp.getOperand())) {
    if (reason)
      llvm::errs() << "mux select and passer ctrl not equal\n";
    return false;
  }

  // TODO: Verify token counts

  return true;
}

static bool isMuxPasserToAndEligible(MuxOp muxOp, bool reason = false) {
  auto *data0 = muxOp.getDataOperands()[0].getDefiningOp();
  auto *data1 = muxOp.getDataOperands()[1].getDefiningOp();

  if (!data0 || !data1) {
    if (reason)
      llvm::errs() << "data0 or data1 has no defining op\n";
    return false;
  }

  if (!isa<ConstantOp>(data0)) {
    if (reason)
      llvm::errs() << "data0 is not constant\n";
    return false;
  }
  auto constOp = cast<ConstantOp>(data0);
  if (isa<IntegerAttr>(constOp.getValue()) &&
      constOp.getValue().cast<IntegerAttr>().getValue() != 0) {
    if (reason)
      llvm::errs() << "data0 is not constant 0\n";
    return false;
  }
  auto *constDefiningOp = constOp.getOperand().getDefiningOp();
  if (!isa<SourceOp>(constDefiningOp)) {
    if (reason)
      llvm::errs() << "data0 is not from source\n";
    return false;
  }

  if (!isa<PasserOp>(data1)) {
    if (reason)
      llvm::errs() << "data1 is not passer\n";
    return false;
  }
  auto passerOp = cast<PasserOp>(data1);

  if (!equalsIndirectly(passerOp.getCtrl(), muxOp.getSelectOperand())) {
    if (reason)
      llvm::errs() << "mux select and passer ctrl not equal\n";
    return false;
  }

  return true;
}
static AndIOp muxPasserToAnd(MuxOp muxOp) {
  OpBuilder builder(muxOp.getContext());
  builder.setInsertionPoint(muxOp);

  auto passerOp = cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());
  auto constOp = cast<ConstantOp>(muxOp.getDataOperands()[0].getDefiningOp());
  auto sourceOp = cast<SourceOp>(constOp.getOperand().getDefiningOp());

  AndIOp andIOp = builder.create<AndIOp>(
      muxOp->getLoc(), muxOp.getResult().getType(),
      ArrayRef<Value>{passerOp.getCtrl(), passerOp.getData()});
  inheritBB(muxOp, andIOp);

  muxOp.getResult().replaceAllUsesWith(andIOp.getResult());

  // Erase the old ops
  eraseMaterializedOperation(muxOp);
  eraseMaterializedOperation(passerOp);
  eraseMaterializedOperation(constOp);
  eraseMaterializedOperation(sourceOp);

  materializeValue(andIOp.getLhs());
  materializeValue(andIOp.getRhs());
  materializeValue(andIOp.getResult());

  return andIOp;
}

/// Performs the MuxPasserSwap, swapping the MuxOp and PasserOp, and updating
/// the select operand and control of the MuxOp and PasserOp.
/// Returns the PasserOp that was swapped.
static PasserOp performMuxPasserSwap(MuxOp muxOp) {
  OpBuilder builder(muxOp.getContext());
  builder.setInsertionPoint(muxOp);

  auto passerOp =
      dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());

  // Materialization is required for swapping
  assertMaterialization(passerOp.getResult());

  // Swap mux and passer
  muxOp.getDataOperandsMutable()[1].set(passerOp.getData());
  passerOp.getDataMutable()[0].set(muxOp.getResult());
  muxOp.getResult().replaceAllUsesExcept(passerOp.getResult(), passerOp);

  InitOp init = cast<InitOp>(muxOp.getSelectOperand().getDefiningOp());
  assertMaterialization(init.getOperand());
  // Add repeating inits
  SpecV2RepeatingInitOp ri1 = builder.create<SpecV2RepeatingInitOp>(
      muxOp->getLoc(), init.getOperand(), 1);
  inheritBB(init, ri1);
  init.getOperandMutable()[0].set(ri1.getResult());

  assertMaterialization(passerOp.getCtrl());
  SpecV2RepeatingInitOp ri2 = builder.create<SpecV2RepeatingInitOp>(
      muxOp->getLoc(), passerOp.getCtrl(), 1);
  inheritBB(passerOp, ri2);
  passerOp.getCtrlMutable()[0].set(ri2.getResult());

  // Materialize passer's result for further rewriting.
  materializeValue(passerOp.getResult());

  return passerOp;
}

static SpecV2RepeatingInitOp moveRepeatingInitsUp(Value fedValue) {
  fedValue = getForkTop(fedValue);
  materializeValue(fedValue);

  Operation *uniqueUser = getUniqueUser(fedValue);
  if (auto ri = dyn_cast<SpecV2RepeatingInitOp>(uniqueUser)) {
    // Single use. No need to move repeating inits.
    return ri;
  }

  OpBuilder builder(fedValue.getContext());
  builder.setInsertionPointAfterValue(fedValue);

  ForkOp fork = cast<ForkOp>(uniqueUser);

  auto newRI =
      builder.create<SpecV2RepeatingInitOp>(fedValue.getLoc(), fedValue, 1);
  inheritBB(fedValue.getDefiningOp(), newRI);
  fedValue.replaceAllUsesExcept(newRI.getResult(), newRI);

  for (auto res : fork->getResults()) {
    if (auto ri = dyn_cast<SpecV2RepeatingInitOp>(getUniqueUser(res))) {
      ri.getResult().replaceAllUsesWith(newRI.getResult());
      ri->erase();
    } else {
      res.replaceAllUsesWith(fedValue);
    }
  }

  fork->erase();

  materializeValue(fedValue);
  materializeValue(newRI.getResult());

  return newRI;
}

/// Returns if the simplification of 3 passers is possible.
static bool isPasserSimplifiable(PasserOp ctrlDefiningPasser,
                                 bool reason = false) {
  // Ensure the structure
  Operation *bottomOp = getUniqueUser(ctrlDefiningPasser.getResult());
  if (!isa<PasserOp>(bottomOp)) {
    if (reason)
      llvm::errs() << "Bottom passer not found\n";
    return false;
  }
  auto bottomPasser = cast<PasserOp>(bottomOp);
  if (ctrlDefiningPasser.getResult() != bottomPasser.getCtrl()) {
    if (reason)
      llvm::errs() << "Ctrl mismatch\n";
    return false;
  }

  Operation *topOp = bottomPasser.getData().getDefiningOp();
  if (!isa<PasserOp>(topOp)) {
    if (reason)
      llvm::errs() << "Top passer not found\n";
    return false;
  }
  auto topPasser = cast<PasserOp>(topOp);

  // Confirm the context
  if (!equalsIndirectly(ctrlDefiningPasser.getCtrl(), topPasser.getCtrl())) {
    if (reason)
      llvm::errs() << "Ctrl mismatch\n";
    return false;
  }

  return true;
}

/// Simplify 3 passers into a single one.
static AndIOp simplifyPasser(PasserOp ctrlDefiningPasser) {
  OpBuilder builder(ctrlDefiningPasser.getContext());
  builder.setInsertionPoint(ctrlDefiningPasser);

  AndIOp andOp = builder.create<AndIOp>(builder.getUnknownLoc(),
                                        ctrlDefiningPasser.getData(),
                                        ctrlDefiningPasser.getCtrl());
  inheritBB(ctrlDefiningPasser, andOp);

  auto bottomPasser =
      cast<PasserOp>(getUniqueUser(ctrlDefiningPasser.getResult()));
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());

  bottomPasser.getCtrlMutable()[0].set(andOp.getResult());
  bottomPasser.getDataMutable()[0].set(topPasser.getData());

  topPasser->erase();
  ctrlDefiningPasser->erase();

  materializeValue(andOp.getLhs());
  materializeValue(andOp.getRhs());
  materializeValue(andOp.getResult());

  return andOp;
}

static bool isAndIUpEligible(AndIOp candidateAndI) {
  Value lhs = getForkTop(candidateAndI.getLhs());
  Value rhs = getForkTop(candidateAndI.getRhs());

  materializeValue(lhs);

  Operation *lhsUniqueUser = getUniqueUser(lhs);

  if (!isa<ForkOp>(lhsUniqueUser)) {
    return false;
  }

  ForkOp lhsFork = cast<ForkOp>(lhsUniqueUser);

  for (auto res : lhsFork->getResults()) {
    if (auto andI = dyn_cast<AndIOp>(getUniqueUser(res))) {
      if (andI == candidateAndI)
        continue;
      if ((andI.getLhs() == res && equalsIndirectly(andI.getRhs(), rhs)) ||
          (andI.getRhs() == res && equalsIndirectly(andI.getLhs(), rhs))) {
        return true;
      }
    }
  }

  return false;
}

static AndIOp moveAndIUp(AndIOp candidateAndI) {
  Value lhs = getForkTop(candidateAndI.getLhs());
  Value rhs = getForkTop(candidateAndI.getRhs());

  materializeValue(lhs);

  Operation *lhsUniqueUser = getUniqueUser(lhs);

  if (!isa<ForkOp>(lhsUniqueUser)) {
    return cast<AndIOp>(lhsUniqueUser);
  }

  ForkOp lhsFork = cast<ForkOp>(lhsUniqueUser);

  OpBuilder builder(lhs.getContext());
  builder.setInsertionPointAfterValue(rhs);

  AndIOp newAndI = builder.create<AndIOp>(lhs.getLoc(), lhs, rhs);
  inheritBB(lhsFork, newAndI);

  newAndI->setAttrs(candidateAndI->getAttrs());

  for (auto res : lhsFork->getResults()) {
    bool rewritten = false;
    if (auto andI = dyn_cast<AndIOp>(getUniqueUser(res))) {
      if ((andI.getLhs() == res && equalsIndirectly(andI.getRhs(), rhs)) ||
          (andI.getRhs() == res && equalsIndirectly(andI.getLhs(), rhs))) {
        andI.getResult().replaceAllUsesWith(newAndI.getResult());
        andI->erase();
        rewritten = true;
      }
    }
    if (!rewritten) {
      res.replaceAllUsesWith(lhs);
    }
  }

  materializeValue(lhs);
  materializeValue(rhs);
  materializeValue(newAndI.getResult());

  return newAndI;
}

static bool isMotionPastGammaEligible(MuxOp muxOp, bool reason = false) {
  Operation *falseDefiningOp = muxOp.getDataOperands()[0].getDefiningOp();
  Operation *trueDefiningOp = muxOp.getDataOperands()[1].getDefiningOp();

  if (!falseDefiningOp || !trueDefiningOp) {
    if (reason)
      llvm::errs() << "data0 or data1 has no defining op\n";
    return false;
  }

  if (!isa<PasserOp>(falseDefiningOp)) {
    if (reason)
      llvm::errs() << "data0 is not passer\n";
    return false;
  }
  auto falsePasser = cast<PasserOp>(falseDefiningOp);

  if (!isa<PasserOp>(trueDefiningOp)) {
    if (reason)
      llvm::errs() << "data1 is not passer\n";
    return false;
  }
  auto truePasser = cast<PasserOp>(trueDefiningOp);

  Operation *falsePasserCtrlDefOp = falsePasser.getCtrl().getDefiningOp();
  Operation *truePasserCtrlDefOp = truePasser.getCtrl().getDefiningOp();

  if (!isa<PasserOp>(falsePasserCtrlDefOp)) {
    if (reason)
      llvm::errs() << "passer ctrl not defined by passer\n";
    return false;
  }
  auto falsePasserCtrlDef = cast<PasserOp>(falsePasserCtrlDefOp);

  if (!isa<PasserOp>(truePasserCtrlDefOp)) {
    if (reason)
      llvm::errs() << "passer ctrl not defined by passer\n";
    return false;
  }
  auto truePasserCtrlDef = cast<PasserOp>(truePasserCtrlDefOp);

  Operation *falsePasserDataDefOp = falsePasser.getData().getDefiningOp();
  Operation *truePasserDataDefOp = truePasser.getData().getDefiningOp();

  if (!isa<PasserOp>(falsePasserDataDefOp)) {
    if (reason)
      llvm::errs() << "passer data not defined by passer\n";
    return false;
  }
  auto falsePasserDataDef = cast<PasserOp>(falsePasserDataDefOp);

  if (!isa<PasserOp>(truePasserDataDefOp)) {
    if (reason)
      llvm::errs() << "passer data not defined by passer\n";
    return false;
  }
  auto truePasserDataDef = cast<PasserOp>(truePasserDataDefOp);

  Operation *selDefiningOp = muxOp.getSelectOperand().getDefiningOp();
  if (!isa<PasserOp>(selDefiningOp)) {
    if (reason)
      llvm::errs() << "mux select not defined by passer\n";
    return false;
  }
  auto selPasser = cast<PasserOp>(selDefiningOp);

  if (!equalsIndirectly(falsePasserCtrlDef.getCtrl(),
                        truePasserCtrlDef.getCtrl()) ||
      !equalsIndirectly(falsePasserCtrlDef.getCtrl(), selPasser.getCtrl()) ||
      !equalsIndirectly(falsePasserCtrlDef.getCtrl(),
                        falsePasserDataDef.getCtrl()) ||
      !equalsIndirectly(falsePasserCtrlDef.getCtrl(),
                        truePasserDataDef.getCtrl())) {
    if (reason)
      llvm::errs() << "ctrl1 not equal\n";
    return false;
  }

  if (!equalsIndirectly(selPasser.getData(), truePasserCtrlDef.getData())) {
    if (reason)
      llvm::errs() << "ctrl2 not equal\n";
    return false;
  }
  Operation *defOp = getIndirectDefiningOp(selPasser.getData());
  if (auto notOp = dyn_cast<NotOp>(defOp)) {
    if (!equalsIndirectly(notOp.getOperand(), falsePasserCtrlDef.getData())) {
      if (reason)
        llvm::errs() << "ctrl3 not equal\n";
      return false;
    }
  } else {
    Operation *defOp2 = getIndirectDefiningOp(falsePasserCtrlDef.getData());
    if (auto notOp2 = dyn_cast<NotOp>(defOp2)) {
      if (!equalsIndirectly(notOp2.getOperand(), selPasser.getData())) {
        if (reason)
          llvm::errs() << "ctrl4 not equal\n";
        return false;
      }
    } else {
      if (reason)
        llvm::errs() << "mux select not defined by NotOp\n";
      return false;
    }
  }

  return true;
}
static void erasePassersBeforeMotionPastGamma(MuxOp muxOp,
                                              DenseSet<PasserOp> &frontiers) {
  auto falsePasser = cast<PasserOp>(muxOp.getDataOperands()[0].getDefiningOp());
  auto truePasser = cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());

  auto falsePasserCtrlDef =
      cast<PasserOp>(falsePasser.getCtrl().getDefiningOp());
  auto truePasserCtrlDef = cast<PasserOp>(truePasser.getCtrl().getDefiningOp());

  auto falsePasserDataDef =
      cast<PasserOp>(falsePasser.getData().getDefiningOp());
  auto truePasserDataDef = cast<PasserOp>(truePasser.getData().getDefiningOp());

  auto selPasser = cast<PasserOp>(muxOp.getSelectOperand().getDefiningOp());

  frontiers.erase(falsePasserCtrlDef);
  frontiers.erase(truePasserCtrlDef);
  frontiers.erase(falsePasserDataDef);
  frontiers.erase(truePasserDataDef);
  frontiers.erase(selPasser);
}

static PasserOp performMotionPastGamma(MuxOp muxOp) {
  PasserOp selPasser = cast<PasserOp>(muxOp.getSelectOperand().getDefiningOp());
  PasserOp falsePasser =
      cast<PasserOp>(muxOp.getDataOperands()[0].getDefiningOp());
  PasserOp truePasser =
      cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());
  PasserOp falsePasserCtrlDef =
      cast<PasserOp>(falsePasser.getCtrl().getDefiningOp());
  PasserOp truePasserCtrlDef =
      cast<PasserOp>(truePasser.getCtrl().getDefiningOp());
  PasserOp falsePasserDataDef =
      cast<PasserOp>(falsePasser.getData().getDefiningOp());
  PasserOp truePasserDataDef =
      cast<PasserOp>(truePasser.getData().getDefiningOp());

  Value ctrl1 = selPasser.getCtrl();
  Value ctrl2 = truePasserCtrlDef.getData();
  Value ctrl2inv = falsePasserCtrlDef.getData();

  OpBuilder builder(muxOp.getContext());
  builder.setInsertionPoint(muxOp);
  PasserOp newPasser =
      builder.create<PasserOp>(muxOp->getLoc(), muxOp.getResult(), ctrl1);
  inheritBB(muxOp, newPasser);
  muxOp.getResult().replaceAllUsesExcept(newPasser.getResult(), newPasser);

  selPasser.getResult().replaceAllUsesWith(ctrl2);
  selPasser->erase();

  falsePasserCtrlDef.getResult().replaceAllUsesWith(
      falsePasserCtrlDef.getData());
  falsePasserCtrlDef->erase();

  truePasserCtrlDef.getResult().replaceAllUsesWith(truePasserCtrlDef.getData());
  truePasserCtrlDef->erase();

  falsePasserDataDef.getResult().replaceAllUsesWith(
      falsePasserDataDef.getData());
  falsePasserDataDef->erase();

  truePasserDataDef.getResult().replaceAllUsesWith(truePasserDataDef.getData());
  truePasserDataDef->erase();

  materializeValue(ctrl1);
  materializeValue(ctrl2);
  materializeValue(ctrl2inv);

  return newPasser;
}

static SpecV2InterpolatorOp introduceIdentInterpolator(Value value) {
  assertMaterialization(value);

  OpBuilder builder(value.getContext());
  builder.setInsertionPointAfterValue(value);

  SpecV2InterpolatorOp interpolatorOp =
      builder.create<SpecV2InterpolatorOp>(value.getLoc(), value, value);
  inheritBB(value.getDefiningOp(), interpolatorOp);

  value.replaceAllUsesExcept(interpolatorOp.getResult(), interpolatorOp);

  materializeValue(value);

  return interpolatorOp;
}

static bool isInterpolatorInductionEligible(SpecV2InterpolatorOp interpOp) {
  Operation *user = getUniqueUser(interpOp.getResult());
  if (!user) {
    llvm::errs() << "no unique user\n";
    return false;
  }

  if (!isa<AndIOp>(user)) {
    llvm::errs() << "user is not AndI\n";
    return false;
  }

  auto andI = cast<AndIOp>(user);
  if (!andI->hasAttr("specv2_tmp_and")) {
    llvm::errs() << "AndI is not a tmp_and\n";
    return false;
  }

  if (andI.getLhs() != interpOp.getResult()) {
    llvm::errs() << "interpOp result is not lhs of AndI\n";
    return false;
  }

  Operation *defOp = getIndirectDefiningOp(andI.getRhs());
  if (!defOp || !isa<SpecV2RepeatingInitOp>(defOp)) {
    llvm::errs() << "rhs of AndI not defined by repeating init\n";
    return false;
  }

  auto ri = cast<SpecV2RepeatingInitOp>(defOp);
  if (!equalsIndirectly(ri.getOperand(), interpOp.getLongOperand())) {
    llvm::errs()
        << "repeating init operand does not match interpOp long operand\n";
    return false;
  }

  bool equal = false;
  Value longOperandUpstream = interpOp.getLongOperand();
  while (true) {
    if (equalsIndirectly(longOperandUpstream, interpOp.getShortOperand())) {
      equal = true;
      break;
    }
    if (auto ri = dyn_cast<SpecV2RepeatingInitOp>(
            getIndirectDefiningOp(longOperandUpstream))) {
      longOperandUpstream = ri.getOperand();
      continue;
    }
    break;
  }

  if (!equal) {
    llvm::errs() << "long operand does not reach short operand upstream\n";
    return false;
  }

  return true;
}

static SpecV2InterpolatorOp
performInterpolatorInduction(SpecV2InterpolatorOp interpOp) {
  OpBuilder builder(interpOp.getContext());
  builder.setInsertionPointAfterValue(interpOp);

  auto andI = cast<AndIOp>(getUniqueUser(interpOp.getResult()));

  // Value oldLongOperand = getForkTop(interpOp.getLongOperand());

  SpecV2InterpolatorOp newInterpOp = builder.create<SpecV2InterpolatorOp>(
      interpOp->getLoc(), interpOp.getShortOperand(), andI.getRhs());
  inheritBB(andI, newInterpOp);

  andI.getResult().replaceAllUsesWith(newInterpOp.getResult());

  andI->erase();
  interpOp->erase();

  // materializeValue(oldLongOperand);

  return newInterpOp;
}

// static Value moveRepeatingInitDown(SpecV2RepeatingInitOp ri) {
//   OpBuilder builder(ri->getContext());

//   materializeValue(ri);
//   Operation *user = getUniqueUser(ri.getResult());
//   ForkOp fork = cast<ForkOp>(user);
//   builder.setInsertionPoint(fork);
//   for (auto res : fork->getResults()) {
//     SpecV2RepeatingInitOp newRI =
//         builder.create<SpecV2RepeatingInitOp>(ri.getLoc(), res, 1);
//     inheritBB(ri, newRI);
//     res.replaceAllUsesExcept(newRI.getResult(), newRI);
//   }

//   Value riOperand = ri.getOperand();

//   ri.getResult().replaceAllUsesWith(riOperand);
//   ri->erase();

//   // Possibly two forks are nested, so flatten them
//   materializeValue(riOperand);

//   return riOperand;
// }

// /// Move the top (least recently added) repeating init and passer below the
// fork
// /// as the preparation for the resolver insertion.
// static void moveTopRIAndPasser(SpecV2InterpolatorOp interpolator) {
//   auto topRI = cast<SpecV2RepeatingInitOp>(
//       getIndirectDefiningOp(interpolator.getShortOperand()));
//   auto oldPasserOp = cast<PasserOp>(topRI.getOperand().getDefiningOp());

//   materializeValue(interpolator.getShortOperand());
//   materializeValue(interpolator.getLongOperand());

//   OpBuilder builder(topRI->getContext());

//   moveRepeatingInitDown(topRI);

//   Operation *fork = getUniqueUser(oldPasserOp.getResult());
//   builder.setInsertionPoint(fork);
//   // getUniqueUser(oldPasserOp.getResult())->dump();

//   if (movePassersDownPM(fork).failed()) {
//     llvm::errs() << "Failed to perform motion for PasserOp";
//     llvm_unreachable("SpeculationV2 algorithm failed");
//   }
// }

// /// Returns if the circuit is eligible for the introduction of the resolver.
// static bool
// isEligibleForResolverIntroduction(SpecV2InterpolatorOp interpolator) {
//   // Ensure the structure
//   Operation *shortOperandDefiningOp =
//       interpolator.getShortOperand().getDefiningOp();
//   if (!isa<SpecV2RepeatingInitOp>(shortOperandDefiningOp)) {
//     llvm::errs() << "No repeating init\n";
//     return false;
//   }
//   auto riOp = cast<SpecV2RepeatingInitOp>(shortOperandDefiningOp);

//   Operation *riOpUpstream = riOp.getOperand().getDefiningOp();
//   if (!isa<PasserOp>(riOpUpstream)) {
//     llvm::errs() << "No passer\n";
//     return false;
//   }
//   auto passerOp = cast<PasserOp>(riOpUpstream);

//   if (!equalsIndirectly(passerOp.getCtrl(), interpolator.getResult())) {
//     llvm::errs() << "Passer ctrl and interpolator result doesn't match\n";
//     return false;
//   }

//   // TODO: Confirm the longOperand
//   return true;
// }

// /// Introduces a spec resolver.
// /// Returns the resolver result value.
// static SpecV2ResolverOp
// introduceSpecResolver(SpecV2InterpolatorOp interpolator) {
//   auto riOp = cast<SpecV2RepeatingInitOp>(
//       (interpolator.getShortOperand().getDefiningOp()));
//   auto passerOp = cast<PasserOp>(riOp.getOperand().getDefiningOp());

//   OpBuilder builder(interpolator->getContext());
//   builder.setInsertionPoint(interpolator);

//   auto resolverOp = builder.create<SpecV2ResolverOp>(
//       interpolator.getLoc(), passerOp.getData(),
//       interpolator.getLongOperand());
//   inheritBB(interpolator, resolverOp);

//   interpolator.getResult().replaceAllUsesWith(resolverOp.getResult());
//   interpolator->erase();
//   riOp->erase();
//   passerOp->erase();
//   return resolverOp;
// }

static bool isAndAssociativityEligible(AndIOp andI) {
  Operation *user = getUniqueUser(andI.getResult());
  if (!user)
    return false;
  if (!isa<AndIOp>(user))
    return false;
  auto userAndI = cast<AndIOp>(user);
  auto userLhs = userAndI.getLhs();
  auto userRhs = userAndI.getRhs();
  auto theOtherValue = (userLhs == andI.getResult()) ? userRhs : userLhs;
  Operation *theOtherValueDefOp = theOtherValue.getDefiningOp();
  if (!theOtherValueDefOp) {
    llvm::errs() << "not supported: andI user other value has no defOp\n";
    return false;
  }
  Operation *andILhsDefOp = andI.getLhs().getDefiningOp();
  Operation *andIRhsDefOp = andI.getRhs().getDefiningOp();
  if (!andILhsDefOp || !andIRhsDefOp) {
    llvm::errs() << "not supported: andI lhs or rhs has no defOp\n";
    return false;
  }
  return (theOtherValueDefOp->isBeforeInBlock(andILhsDefOp) ||
          theOtherValueDefOp->isBeforeInBlock(andIRhsDefOp));
}

static void performAndAssociativity(AndIOp andI) {
  auto userAndI = cast<AndIOp>(getUniqueUser(andI.getResult()));
  auto userLhs = userAndI.getLhs();
  auto userRhs = userAndI.getRhs();
  auto theOtherValue = (userLhs == andI.getResult()) ? userRhs : userLhs;
  Operation *andILhsDefOp = andI.getLhs().getDefiningOp();
  Operation *andIRhsDefOp = andI.getRhs().getDefiningOp();
  Value keep;
  Value goDown;
  if (andILhsDefOp->isBeforeInBlock(andIRhsDefOp)) {
    keep = andI.getLhs();
    goDown = andI.getRhs();
  } else {
    keep = andI.getRhs();
    goDown = andI.getLhs();
  }

  OpBuilder builder(andI.getContext());
  builder.setInsertionPoint(andI);

  auto newAndI1 = builder.create<AndIOp>(andI.getLoc(), keep, theOtherValue);
  inheritBB(andI, newAndI1);

  auto newAndI2 =
      builder.create<AndIOp>(andI.getLoc(), newAndI1.getResult(), goDown);
  inheritBB(andI, newAndI2);

  userAndI.getResult().replaceAllUsesWith(newAndI2.getResult());
  userAndI->erase();
  andI->erase();
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
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
  OpBuilder builder(funcOp->getContext());

  DenseSet<PasserOp> frontiers;
  // n is a member variable handled by tablegen.
  // Storing repeating inits for post-processing.
  SmallVector<SpecV2RepeatingInitOp> repeatingInits(n);

  if (!disableInitialMotion) {
    frontiers.clear();

    for (auto passer : funcOp.getOps<PasserOp>()) {
      if (llvm::find(loopBBs, getLogicBB(passer)) == loopBBs.end())
        continue;
      passer->setAttr("specv2_frontier", builder.getBoolAttr(false));
      frontiers.insert(passer);
    }

    bool frontiersUpdated;
    do {
      do {
        frontiersUpdated = false;
        for (auto passerOp : frontiers) {
          Operation *uniqueUser = getUniqueUser(passerOp.getResult());
          if (llvm::find(loopBBs, getLogicBB(uniqueUser)) == loopBBs.end()) {
            // The passer is exiting the loop. Do not move it.
            // (Exit passer motion is performed later if enabled)
            continue;
          }

          if (isEligibleForPasserMotionOverPM(passerOp)) {
            performPasserMotionPastPM(passerOp, frontiers);
            frontiersUpdated = true;
            // If frontiers are updated, the iterator is outdated.
            // Break and restart the loop.
            break;
          }

          if (auto mux = dyn_cast<MuxOp>(getUniqueUser(passerOp.getResult()))) {
            if (isMotionPastGammaEligible(mux)) {
              erasePassersBeforeMotionPastGamma(mux, frontiers);
              auto passer = performMotionPastGamma(mux);
              passer->setAttr("specv2_frontier", builder.getBoolAttr(false));
              frontiers.insert(passer);
              frontiersUpdated = true;
              // If frontiers are updated, the iterator is outdated.
              // Break and restart the loop.
              break;
            }
          }
        }
        // If no frontiers were updated, we can stop.
      } while (frontiersUpdated);

      bool passerSimplified;
      do {
        passerSimplified = false;
        for (auto passer : frontiers) {
          if (isPasserSimplifiable(passer)) {
            auto bottomPasser =
                cast<PasserOp>(getUniqueUser(passer.getResult()));
            auto topPasser =
                cast<PasserOp>(bottomPasser.getData().getDefiningOp());
            frontiers.erase(bottomPasser);
            frontiers.erase(topPasser);
            frontiers.erase(passer);
            AndIOp andIOp = simplifyPasser(passer);
            auto newFrontier =
                cast<PasserOp>(getUniqueUser(andIOp.getResult()));
            newFrontier->setAttr("specv2_frontier", builder.getBoolAttr(false));
            frontiers.insert(newFrontier);
            // moveAndIUp(andIOp);
            frontiersUpdated = true;
            passerSimplified = true;
            break;
          }
        }
      } while (passerSimplified);

      bool andMovedUp;
      do {
        andMovedUp = false;
        for (auto andi : funcOp.getOps<AndIOp>()) {
          if (llvm::find(loopBBs, getLogicBB(andi)) == loopBBs.end())
            continue;

          if (andi->hasAttr("specv2_tmp_and"))
            continue;

          if (isAndIUpEligible(andi)) {
            moveAndIUp(andi);
            andMovedUp = true;
            frontiersUpdated = true;
            break;
          }

          if (isAndAssociativityEligible(andi)) {
            performAndAssociativity(andi);
            andMovedUp = true;
            frontiersUpdated = true;
            break;
          }
        }
      } while (andMovedUp);
      // Passer simplification may have some passers eligible for motion
    } while (frontiersUpdated);

    bool muxUpdated;
    do {
      muxUpdated = false;
      for (auto muxOp : funcOp.getOps<MuxOp>()) {
        if (llvm::find(loopBBs, getLogicBB(muxOp)) == loopBBs.end())
          continue;

        if (isMuxPasserToAndEligible(muxOp)) {
          AndIOp andIOp = muxPasserToAnd(muxOp);
          moveAndIUp(andIOp);
          muxUpdated = true;
          break;
        }
      }
    } while (muxUpdated);
  }

  // Repeatedly move passers past Muxes and PMSC.
  for (unsigned i = 0; i < n; i++) {
    frontiers.clear();

    // Perform MuxPasserSwap for each Mux
    for (auto muxOp :
         llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
      if (getLogicBB(muxOp) != loopBBs.front())
        continue;

      if (!isEligibleForMuxPasserSwap(muxOp)) {
        muxOp.emitWarning(
            "MuxOp is not eligible for Passer swap, exiting the pass.");
        return;
      }

      auto passerOp = performMuxPasserSwap(muxOp);

      // Add the moved passer to the frontiers
      passerOp->setAttr("specv2_frontier", builder.getBoolAttr(false));
      frontiers.insert(passerOp);
    }

    Value riResult = frontiers.begin()->getCtrl();
    auto newRI = moveRepeatingInitsUp(riResult.getDefiningOp()->getOperand(0));
    if (i == 0) {
      newRI->setAttr("specv2_top_ri", builder.getBoolAttr(true));
    }

    repeatingInits[i] = newRI;

    // Repeatedly move passers inside PMSC.
    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (isEligibleForPasserMotionOverPM(passerOp)) {
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }

        if (auto mux = dyn_cast<MuxOp>(getUniqueUser(passerOp.getResult()))) {
          if (isMotionPastGammaEligible(mux)) {
            erasePassersBeforeMotionPastGamma(mux, frontiers);
            auto passer = performMotionPastGamma(mux);
            passer->setAttr("specv2_frontier", builder.getBoolAttr(false));
            frontiers.insert(passer);
            frontiersUpdated = true;
            // If frontiers are updated, the iterator is outdated.
            // Break and restart the loop.
            break;
          }
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);
  }

  if (n >= 2) {
    // Reduce the passer chain by introducing interpolator op and performing
    // induction.

    SmallVector<PasserOp> bottomPassers;
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(repeatingInits[0].getResult())) {
      if (auto passer = dyn_cast<PasserOp>(user)) {
        bottomPassers.push_back(passer);
      }
    }

    SmallVector<AndIOp> andChain;
    for (unsigned i = 1; i < n; i++) {
      bool firstPasser = true;
      for (auto bottomPasser : bottomPassers) {
        auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());
        builder.setInsertionPoint(topPasser);

        // Place AND temporarily for buffering
        auto tmpAnd =
            builder.create<AndIOp>(builder.getUnknownLoc(),
                                   bottomPasser.getCtrl(), topPasser.getCtrl());
        inheritBB(topPasser, tmpAnd);
        tmpAnd->setAttr("specv2_tmp_and", builder.getBoolAttr(true));

        if (firstPasser) {
          andChain.push_back(tmpAnd);
          firstPasser = false;
        }

        bottomPasser.getCtrlMutable()[0].set(tmpAnd.getResult());

        topPasser.getResult().replaceAllUsesWith(topPasser.getData());
        topPasser->erase();
      }
    }

    for (auto andOp : andChain) {
      moveAndIUp(andOp);
    }
  }

  // Simplify the exit passers.
  SmallVector<PasserOp> ctrlDefiningPassers;
  for (auto passer : funcOp.getOps<PasserOp>()) {
    if (isPasserSimplifiable(passer)) {
      ctrlDefiningPassers.push_back(passer);
    }
  }
  frontiers.clear();
  for (auto passer : ctrlDefiningPassers) {
    AndIOp andIOp = simplifyPasser(passer);
    if (exitEagerEval) {
      PasserOp exitPasser = cast<PasserOp>(getUniqueUser(andIOp.getResult()));
      exitPasser->setAttr("specv2_frontier", builder.getBoolAttr(false));
      frontiers.insert(exitPasser);
    }
    moveAndIUp(andIOp);
  }
  if (exitEagerEval) {
    llvm::errs() << "Performing exit eager evaluation\n";
    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (isEligibleForPasserMotionOverPM(passerOp)) {
          Operation *uniqueUser = getUniqueUser(passerOp.getResult());
          auto bbOrNull = getLogicBB(uniqueUser);
          if (bbOrNull.has_value() &&
              llvm::find(loopBBs, bbOrNull) == loopBBs.end()) {
            loopBBs.push_back(bbOrNull.value());
          }
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);
  }

  if (resolver && n >= 2) {
    std::optional<SpecV2InterpolatorOp> interpolator = std::nullopt;
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(repeatingInits[0].getResult())) {
      if (auto andI = dyn_cast<AndIOp>(user)) {
        if (andI->hasAttr("specv2_tmp_and")) {
          interpolator = introduceIdentInterpolator(andI.getLhs());
          break;
        }
      }
    }
    if (!interpolator) {
      llvm::errs() << "No AndI found for introducing interpolator\n";
      return signalPassFailure();
    }
    while (isInterpolatorInductionEligible(interpolator.value())) {
      interpolator = performInterpolatorInduction(interpolator.value());
    }

    // // Preparation for the resolver insertion
    // moveTopRIAndPasser(interpolator.value());

    // // Introduce the resolver
    // if (!isEligibleForResolverIntroduction(interpolator.value())) {
    //   interpolator.value().emitError(
    //       "The circuit is not eligible for the resolver introduction");
    //   return signalPassFailure();
    // }
    // introduceSpecResolver(interpolator.value());
  }

  DenseMap<unsigned, unsigned> bbMap = unifyBBs(loopBBs, funcOp);
  if (bbMapping != "") {
    // Convert bbMap to a json file
    std::ofstream csvFile(bbMapping);
    csvFile << "before,after\n";
    for (const auto &entry : bbMap) {
      csvFile << entry.first << "," << entry.second << "\n";
    }
    csvFile.close();
  }

  recalculateMCBlocks(funcOp);
}
