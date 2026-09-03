#include "experimental/Transforms/EagerlyElastic/EagerlyElasticLib.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

/// Helper function to add the bbAttr and name to new operations.
void setHandshakeAttrs(Attribute bbAttr, NameAnalysis &namer,
                       ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    assert(op);
    if (bbAttr)
      op->setAttr("handshake.bb", bbAttr);
    namer.setName(op);
  }
}

/// Helper to trace a condition value back through NotIOps AND RepeatingInits
static Value traceConditionRoot(Value val, bool &inverted) {
  while (Operation *defOp = val.getDefiningOp()) {
    if (auto notOp = dyn_cast<handshake::NotIOp>(defOp)) {
      inverted = !inverted;
      val = notOp.getOperand();
    } else if (auto repInit = dyn_cast<handshake::RepeatingInitOp>(defOp)) {
      val = repInit.getOperand();
    } else if (auto mux = dyn_cast<handshake::MuxOp>(defOp)) {
      if (mux->hasAttr("cmerge_mux")) {
        val = mux.getDataOperands()[1];
      } else break;
    } else {
      break;
    }
  }
  return val;
}

/// Checks whether two condition values originate from the same root source.
bool checkConditionsMatch(Value valA, Value valB, bool expectSamePolarity) {
  bool invA = false, invB = false;

  /* // Trace valA back through NotIOps
  while (auto notOp =
             dyn_cast_or_null<handshake::NotIOp>(valA.getDefiningOp())) {
    invA = !invA;
    valA = notOp.getOperand();
  }

  // Trace valB back through NotIOps
  while (auto notOp =
             dyn_cast_or_null<handshake::NotIOp>(valB.getDefiningOp())) {
    invB = !invB;
    valB = notOp.getOperand();
  }  */
  valA = traceConditionRoot(valA, invA);
  valB = traceConditionRoot(valB, invB);

  // They must originate from the exact same root wire
  if (valA != valB)
    return false;

  return expectSamePolarity ? (invA == invB) : (invA != invB);
}

/// Recursive function to determine whether a value originates from a constant
/// source.
bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // No constant source possible here
  if (isa<handshake::MuxOp>(definingOp) or
      isa<handshake::RepeatingInitOp>(definingOp))
    return false;

  if (isa<handshake::SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

/// Checks if a suppressor (BranchOp) can be pushed past its downstream
/// operation (for Rewrite A). Eligible operations are pure and matched and need
/// to have the same condition on all their input operands or a source.
BypassResult isEligibleForBypass(handshake::ConditionalBranchOp branchOp,
                                 Operation *targetOp) {

  // verify the targetOp is a PM unit
  if (!isa<handshake::ArithOpInterface, handshake::ForkOp, handshake::NotIOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return BypassResult::Ineligible; // reject anything that isn't a PM unit or
                                     // a 1-input merge
  }

  // loadOps receive address control independently, the can always be bypassed
  if (isa<handshake::LoadOp>(targetOp)) {
    return BypassResult::Eligible;
  }

  // only move past NotOps if they aren't part of a suppressor condition
  if (targetOp->hasAttr("is_suppressor_not")) {
    return BypassResult::Ineligible;
  }

  // do not move past ors that guard the stores TODO: make sure other ones can be bypassed
  if (isa<handshake::OrIOp>(targetOp)) {
    return BypassResult::Ineligible;
  }

  // ensure all other inputs to the target op match this suppressor's condition
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    // if we have a branch it must have the same condition
    if (auto siblingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
            operand.getDefiningOp())) {
      if (siblingBranch == branchOp)
        continue;

      // check whether condition matches indirectly
      if (!checkConditionsMatch(currentCond,
                                siblingBranch.getConditionOperand(), true)) {
        return BypassResult::Ineligible;
      }
    } else if (!isSourced(operand)) {
      return BypassResult::Ineligible;
    }
  }
  return BypassResult::Eligible;
}

/// Move the suppressors past the following operation, targetOp. Erase all the
/// suppressors going into targetOp and create new suppressors on every output
/// of targetOp.
void moveSuppressorPastOp(handshake::ConditionalBranchOp branchOp,
                          Operation *targetOp,
                          DenseSet<handshake::ConditionalBranchOp> &frontier,
                          NameAnalysis &namer, int DRewrite) {

  llvm::errs() << "Moving past: " << targetOp->getAttr("handshake.name")
               << '\n';

  Location loc = targetOp->getLoc();
  Value condition = branchOp.getConditionOperand();
  mlir::Attribute headerBBAttr = nullptr;

  // rewire the suppressor
  for (OpOperand &use : targetOp->getOpOperands()) {
    auto incomingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
        use.get().getDefiningOp());
    if (!incomingBranch)
      continue;

    // for Rewrite D, move only the suppressor connected to A of the Mux
    if (incomingBranch != branchOp && DRewrite) {
      continue;
    }

    // rewire the target operand directly to the suppressor's input data
    use.set(incomingBranch.getDataOperand());
    
    if (!headerBBAttr)
      headerBBAttr = branchOp->getAttr("subloop_header_bb");

    // check if the suppressor has any remaining downstream uses
    if (incomingBranch.getFalseResult().use_empty()) {
      frontier.erase(incomingBranch);
      incomingBranch->erase();
    }
  }

  OpBuilder builder(targetOp);
  builder.setInsertionPointAfter(targetOp);

  // place the new suppressors on every result of the targetOp
  for (Value result : targetOp->getResults()) {
    // If this is a LoadOp, skip index 0 (the address output going to the MC)
    if (isa<handshake::LoadOp>(targetOp) &&
        llvm::cast<mlir::OpResult>(result).getResultNumber() == 0) {
      continue;
    }

    auto newBranch =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, result);
    setHandshakeAttrs(targetOp->getAttr("handshake.bb"), namer, {newBranch});
    if (headerBBAttr) {
      newBranch->setAttr("subloop_header_bb", headerBBAttr);
    }

    // reroute downstream consumers to look at the new branch's FalseResult
    result.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
    frontier.insert(newBranch);
  }
}

void applyRewriteB(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   handshake::ConditionalBranchOp falseBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {
  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite B for: "
               << dataMux->getAttr("handshake.name") << '\n';

  Value conditionC = dataMux.getSelectOperand();

  // build the additional control structure
  OpBuilder builder(dataMux);

  // create the feedback loop: not -> initOp -> mux, not
  // initialize initOp with conditionC temporarily
  auto initOp = builder.create<handshake::InitOp>(loc, conditionC);
  auto redNot = builder.create<handshake::NotIOp>(loc, initOp.getResult());
  setHandshakeAttrs(bbAttr, namer, {initOp, redNot});
  // Close the loop: connect the output of redNot back into the InitOp's input
  initOp.getOperandMutable().assign(redNot.getResult());
  // connect the new structure back to the old one
  dataMux.getSelectOperandMutable().assign(initOp.getResult());

  // create the green loop
  auto greenNot = builder.create<handshake::NotIOp>(loc, conditionC);
  llvm::SmallVector<Value, 2> greenMuxInputs = {greenNot, conditionC};
  auto greenMux = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), initOp.getResult(), greenMuxInputs);
  setHandshakeAttrs(bbAttr, namer, {greenNot, greenMux});

  // update the suppressor's condition (TODO: put this into function for B,C,D?)
  auto notOp = dyn_cast_or_null<handshake::NotIOp>(
      trueBranch.getConditionOperand().getDefiningOp());
  if (notOp && notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(greenMux.getResult());
  else { // create a new isolated NotIOp
    builder.setInsertionPoint(notOp ? notOp : trueBranch);
    auto newNotOp = builder.create<handshake::NotIOp>(trueBranch.getLoc(),
                                                      greenMux.getResult());
    setHandshakeAttrs(bbAttr, namer, {newNotOp});
    newNotOp->setAttr("is_suppressor_not", builder.getUnitAttr());
    trueBranch.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  moveSuppressorPastOp(trueBranch, dataMux, frontier, namer);
}

void applyRewriteC(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite C for: "
               << dataMux->getAttr("handshake.name") << '\n';

  Value conditionC = dataMux.getSelectOperand();

  // build the additional control structure for the rewrite
  OpBuilder builder(dataMux);

  auto repeatingInit =
      builder.create<handshake::RepeatingInitOp>(loc, conditionC, 1);
  setHandshakeAttrs(bbAttr, namer, {repeatingInit});

  // create a mux with a true on false and on true the condition
  auto sourceOp = builder.create<handshake::SourceOp>(loc);
  auto cnstTrue = builder.create<handshake::ConstantOp>(
      loc, builder.getBoolAttr(true), sourceOp.getResult());
  llvm::SmallVector<Value, 2> muxGreenInputs = {cnstTrue.getResult(),
                                                conditionC};
  auto muxGreen = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), repeatingInit.getResult(), muxGreenInputs);
  setHandshakeAttrs(bbAttr, namer, {sourceOp, cnstTrue, muxGreen});

  // connect the repeatingInit to the loop mux
  dataMux.getSelectOperandMutable().assign(repeatingInit.getResult());

  /// update the suppressor's condition
  auto notOp = dyn_cast_or_null<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (notOp && notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(muxGreen.getResult());
  else { // create a new isolated NotIOp
    builder.setInsertionPoint(notOp ? notOp : branchOp);
    auto newNotOp = builder.create<handshake::NotIOp>(branchOp.getLoc(),
                                                      muxGreen.getResult());
    setHandshakeAttrs(bbAttr, namer, {newNotOp});
    newNotOp->setAttr("is_suppressor_not", builder.getUnitAttr());
    branchOp.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  // move suppressor past the mux
  moveSuppressorPastOp(branchOp, dataMux, frontier, namer, 1);
}

/// Apply Rewrite D once by connecting a repeatingInitOp to the circuit and then
/// moving the suppressor past the mux.
void applyRewriteD(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   handshake::InitOp initOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite D for: "
               << dataMux->getAttr("handshake.name") << '\n';
  Value conditionC = initOp.getOperand();

  // build the additional control structure for the rewrite
  OpBuilder builder(dataMux);

  auto repeatingInit =
      builder.create<handshake::RepeatingInitOp>(loc, conditionC, 1);
  setHandshakeAttrs(bbAttr, namer, {repeatingInit});
  Value specOutput = repeatingInit.getResult();

  // connect the repeatingInit to the loop init
  initOp.getOperandMutable().assign(specOutput);

  // update the suppressor's condition - must be inverted relative to the init
  auto notOp = dyn_cast_or_null<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (notOp && notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(specOutput);
  else { // create a new isolated NotIOp
    builder.setInsertionPoint(notOp ? notOp : branchOp);
    auto newNotOp =
        builder.create<handshake::NotIOp>(branchOp.getLoc(), specOutput);
    setHandshakeAttrs(bbAttr, namer, {newNotOp});
    newNotOp->setAttr("is_suppressor_not", builder.getUnitAttr());
    branchOp.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  // move suppressor past the mux
  moveSuppressorPastOp(branchOp, dataMux, frontier, namer, 1);
}

void applyRewriteDMerged(handshake::ControlMergeOp mergeOp,
                   handshake::ConditionalBranchOp branchOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {

  Location loc = mergeOp->getLoc();
  auto bbAttr = mergeOp->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite D for: "
               << mergeOp->getAttr("handshake.name") << '\n';

  Value branchOut = branchOp.getFalseResult();
  int suppressedIdx = -1;
  for (auto [idx, operand] : llvm::enumerate(mergeOp.getDataOperands())) {
    if (operand == branchOut) {
      suppressedIdx = static_cast<int>(idx);
      break;
    }
  }

  // Extract the original input condition driving the suppressor
  Value suppCond = branchOp.getConditionOperand();
  auto notOp = dyn_cast_or_null<handshake::NotIOp>(suppCond.getDefiningOp());
  Value conditionC = notOp ? notOp.getOperand() : suppCond;

  Value rawData = branchOp.getDataOperand();
  mergeOp->getOpOperand(suppressedIdx).set(rawData);

  // 4. Create a constant false (<i1>) in the same basic block
  OpBuilder builder(mergeOp);
  builder.setInsertionPointAfter(mergeOp);

  auto sourceOp = builder.create<handshake::SourceOp>(loc);
  auto constFalse = builder.create<handshake::ConstantOp>(
      loc, builder.getBoolAttr(true), sourceOp.getResult());
  setHandshakeAttrs(bbAttr, namer, {sourceOp, constFalse});

  // 5. Construct Mux data inputs: conditionC at suppressedIdx, false elsewhere
  unsigned numOperands = mergeOp.getDataOperands().size();
  SmallVector<Value> muxInputs(numOperands, constFalse.getResult());
  muxInputs[suppressedIdx] = conditionC;

  // 6. Create the Mux driven by the cmerge index output
  Value cmergeIndex = mergeOp.getIndex();
  auto condMux = builder.create<handshake::MuxOp>(loc, conditionC.getType(), cmergeIndex, muxInputs);
  setHandshakeAttrs(bbAttr, namer, {condMux});
  condMux->setAttr("cmerge_mux", builder.getUnitAttr());

  // 7. Update and move the NOT operation after the Mux
  notOp->moveAfter(condMux);
  notOp.getOperandMutable().assign(condMux.getResult());
  notOp->setAttr("handshake.bb", bbAttr);

  // 8. Move the suppressor branchOp after the NOT operation
  branchOp->moveAfter(notOp);
  branchOp.getConditionOperandMutable().assign(notOp.getResult());
  branchOp->setAttr("handshake.bb", bbAttr);

  // 9. Attach the cmerge result to branchOp and rewire downstream consumers
  Value rawCmergeRes = mergeOp.getResult();
  branchOp.getDataOperandMutable().assign(rawCmergeRes);
  rawCmergeRes.replaceAllUsesExcept(branchOp.getFalseResult(), branchOp);

  frontier.insert(branchOp);
}

void applyRewriteE(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer, int inverted) {
  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite E for: "
               << dataMux->getAttr("handshake.name") << '\n';

  // identify the inputs
  Value inputA = dataMux.getSelectOperand();
  Value inputB = trueBranch.getDataOperand();

  // build the new AND gate
  OpBuilder builder(dataMux);
  Value newInputA = inputA;
  if (inverted) {
    auto notOp = builder.create<handshake::NotIOp>(loc, inputA);
    setHandshakeAttrs(bbAttr, namer, {notOp});
    newInputA = notOp.getResult();
  }
  auto andOp = builder.create<handshake::AndIOp>(loc, newInputA, inputB);
  setHandshakeAttrs(bbAttr, namer, {andOp});

  // reroute all downstream operations from the dataMux to the new AND
  dataMux.getResult().replaceAllUsesWith(andOp.getResult());

  // clean up
  dataMux.erase();
  if (trueBranch.getFalseResult().use_empty()) {
    frontier.erase(trueBranch);
    trueBranch.erase();
  }
}

void applyRewriteF(handshake::ConditionalBranchOp branchOp,
                   handshake::ConditionalBranchOp topSuppLeft,
                   handshake::ConditionalBranchOp topSuppRight,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {

  Location loc = branchOp->getLoc();
  OpBuilder builder(branchOp);
  llvm::errs() << "Applying Rewrite F!\n";

  Value conditionB = topSuppLeft.getConditionOperand();
  auto notOpB = dyn_cast<handshake::NotIOp>(conditionB.getDefiningOp());
  if (notOpB) {
    conditionB = notOpB.getOperand();
  } else {
    // input to the and gate for B must be inverted - add a NotIOp
    auto newNotB = builder.create<handshake::NotIOp>(loc, conditionB);
    setHandshakeAttrs(branchOp->getAttr("handshake.bb"), namer, {newNotB});
    conditionB = newNotB.getResult();
  }
  Value conditionC = topSuppRight.getDataOperand();

  // create the new AND gate
  auto andOp = builder.create<handshake::AndIOp>(loc, conditionB, conditionC);
  setHandshakeAttrs(branchOp->getAttr("handshake.bb"), namer, {andOp});

  // safely bypass or remove topSuppLeft
  Value wireLeft = topSuppLeft.getFalseResult();
  branchOp->replaceUsesOfWith(wireLeft, topSuppLeft.getDataOperand());
  if (wireLeft.use_empty()) {
    frontier.erase(topSuppLeft);
    topSuppLeft->erase();
  }

  // safely bypass or remove topSuppRight
  Value wireRight = topSuppRight.getFalseResult();

  // check whether we have to reconnect the notOp or the branchOp directly
  if (auto notOp = dyn_cast_or_null<handshake::NotIOp>(
          branchOp.getConditionOperand().getDefiningOp())) {
    notOp->replaceUsesOfWith(wireRight, andOp.getResult());
  } else {
    branchOp.getConditionOperandMutable().assign(andOp.getResult());
  }
  if (wireRight.use_empty()) {
    frontier.erase(topSuppRight);
    topSuppRight->erase();
  }
}

void applyRewriteG(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp blueBranchA,
                   handshake::ConditionalBranchOp blueBranchB,
                   handshake::ConditionalBranchOp topSuppressorA,
                   handshake::ConditionalBranchOp topSuppressorB,
                   handshake::ConditionalBranchOp topSuppressorC,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {
  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite G for: "
               << dataMux->getAttr("handshake.name") << '\n';

  // capture the shared control signal D from one of the top suppressors
  Value conditionD = topSuppressorA.getConditionOperand();

  // safely bypass topSuppA
  Value wireA = topSuppressorA.getFalseResult();
  blueBranchA->replaceUsesOfWith(wireA, topSuppressorA.getDataOperand());
  if (wireA.use_empty()) {
    frontier.erase(topSuppressorA);
    topSuppressorA->erase();
  }

  // safely bypass topSuppB
  Value wireB = topSuppressorB.getFalseResult();
  blueBranchB->replaceUsesOfWith(wireB, topSuppressorB.getDataOperand());
  if (wireB.use_empty()) {
    frontier.erase(topSuppressorB);
    topSuppressorB->erase();
  }

  // bypass topSuppC
  SmallPtrSet<Operation *, 3> expectedRewriteOps;
  if (auto notA = dyn_cast_or_null<handshake::NotIOp>(
          blueBranchA.getConditionOperand().getDefiningOp()))
    expectedRewriteOps.insert(notA);
  else
    expectedRewriteOps.insert(blueBranchA);
  if (auto notB = dyn_cast_or_null<handshake::NotIOp>(
          blueBranchB.getConditionOperand().getDefiningOp()))
    expectedRewriteOps.insert(notB);
  else
    expectedRewriteOps.insert(blueBranchB);
  expectedRewriteOps.insert(dataMux);

  // safely rewire only the operands belonging to these specific operations
  Value wireC = topSuppressorC.getFalseResult();
  Value inputC = topSuppressorC.getDataOperand();

  wireC.replaceUsesWithIf(inputC, [&](OpOperand &use) {
    return expectedRewriteOps.contains(use.getOwner());
  });

  if (wireC.use_empty()) {
    frontier.erase(topSuppressorC);
    topSuppressorC->erase();
  }

  // build the new suppressor
  OpBuilder builder(dataMux);
  builder.setInsertionPointAfter(dataMux);

  auto movedSuppressor = builder.create<handshake::ConditionalBranchOp>(
      loc, conditionD, dataMux.getResult());
  setHandshakeAttrs(bbAttr, namer, {movedSuppressor});
  frontier.insert(movedSuppressor);

  // reroute all downstream operations to the new moved suppressor's result
  dataMux.getResult().replaceAllUsesExcept(movedSuppressor.getFalseResult(),
                                           movedSuppressor);
}

void applyRewriteH(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   handshake::InitOp initOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {
  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite H for: "
               << dataMux->getAttr("handshake.name") << '\n';

  // find the condition signal C
  Value conditionC = trueBranch.getConditionOperand();
  auto notOp = dyn_cast<handshake::NotIOp>(
      trueBranch.getConditionOperand().getDefiningOp());
  if (notOp) {
    conditionC = notOp.getOperand();
  }

  // build the new Mux
  OpBuilder builder(dataMux);
  llvm::SmallVector<Value, 2> newMuxInputs = {conditionC, conditionC};
  auto newMux = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), initOp.getResult(), newMuxInputs);
  setHandshakeAttrs(bbAttr, namer, {newMux});

  // rewire the input to the suppressor to the new mux
  if (notOp.getResult().hasOneUse()) {
    notOp.getOperandMutable().assign(newMux.getResult());
  } else {
    // if other operations share this NotIOp, create a new isolated one
    builder.setInsertionPoint(notOp);
    auto newNotOp =
        builder.create<handshake::NotIOp>(notOp->getLoc(), newMux.getResult());
    setHandshakeAttrs(notOp->getAttr("handshake.bb"), namer, {newNotOp});

    trueBranch.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  moveSuppressorPastOp(trueBranch, dataMux, frontier, namer);
}

void markMultiSuccessorHeaderBranches(
    const llvm::DenseSet<handshake::ConditionalBranchOp> &frontier,
    ModuleOp modOp) {
  auto subloopInfoAttr = modOp->getAttrOfType<ArrayAttr>("handshake.subloop_info");
  if (!subloopInfoAttr)
    return;
  OpBuilder builder(modOp.getContext());
  
  for (handshake::ConditionalBranchOp branchOp : frontier) {
    // get the bb of the current branch
    auto bbAttr = branchOp->getAttrOfType<IntegerAttr>("handshake.bb");
    int64_t currentBB = bbAttr.getInt();

    for (Attribute loopAttr : subloopInfoAttr) {
      auto dict = cast<DictionaryAttr>(loopAttr);
      auto headerBBAttr = dict.getAs<IntegerAttr>("header_bb");
      auto successorBBsArray = dict.getAs<ArrayAttr>("successor_bbs");

      if (!headerBBAttr || !successorBBsArray)
        continue;

      int64_t headerBB = headerBBAttr.getInt();

      // Condition 1: Must belong to this header block
      if (currentBB != headerBB)
        continue;

      // Condition 2: Header must have MULTIPLE successors
      if (successorBBsArray.size() <= 1)
        continue;

      // Condition 3: Verify the branch targets one of the successor blocks
      bool targetsSuccessor = false;
      for (Operation *user : branchOp.getFalseResult().getUsers()) {
        if (isa<handshake::ControlMergeOp>(user))
          continue;
          
        auto userBBAttr = user->getAttrOfType<IntegerAttr>("handshake.bb");
        int64_t userBB = userBBAttr.getInt();

        // Check if userBB is in successor_bbs
        for (Attribute succAttr : successorBBsArray) {
          if (dyn_cast<IntegerAttr>(succAttr).getInt() == userBB) {
            targetsSuccessor = true;
            break;
          }
        }
        if (targetsSuccessor)
          break;
      }

      // mark the cond_branch with the header bb
      if (targetsSuccessor) {
        branchOp->setAttr("subloop_header_bb",
                          builder.getI64IntegerAttr(headerBB));
        // branchOp.dump();
        break; // match found for this branch
      }
    }
  }
}
