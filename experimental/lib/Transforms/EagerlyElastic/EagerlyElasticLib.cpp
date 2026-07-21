#include "experimental/Transforms/EagerlyElastic/EagerlyElasticLib.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

/// Recursively trace a value back to its source, traversing through
/// non-blocking structural ops such as forks, buffers, and logical inversions.
Value getForkTop(Value value, bool &isInverted) {
  Operation *defOp = value.getDefiningOp();
  // look through logical inversions
  if (auto notOp = dyn_cast<handshake::NotIOp>(defOp)) {
    isInverted = !isInverted;
    return getForkTop(notOp.getOperand(), isInverted);
  }
  // look through standard handshake forks
  if (auto fork = dyn_cast<handshake::ForkOp>(defOp)) {
    return getForkTop(fork.getOperand(), isInverted);
  }
  // look through handshake buffers
  if (auto buf = dyn_cast<handshake::BufferOp>(defOp)) {
    return getForkTop(buf.getOperand(), isInverted);
  }
  return value;
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
bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                                   Operation *targetOp) {

  // can be pushed past a Mux but not during rewrite A - there is an additional
  // check in the rewriteA() function
  if (isa<handshake::MuxOp>(targetOp))
    return true;

  // verify the targetOp is a PM unit
  if (!isa<handshake::ArithOpInterface, handshake::ForkOp, handshake::NotIOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return false; // reject anything that isn't a PM unit or a 1-input merge
  }

  // the other input to the loadOp is coming from the memory controller meaning
  // we can always move past a loadOp
  if (isa<handshake::LoadOp>(targetOp)) {
    return true;
  }

  // only move past NotOps if it's not part of a suppressor condition
  if (isa<handshake::NotIOp>(targetOp)) {
    auto users = targetOp->getResult(0).getUsers();
    // If there are no users, return false
    if (users.empty()) {
      return false;
    }

    // check if any of the users match the specific operations
    for (Operation *user : users) {
      if (llvm::isa<handshake::ConditionalBranchOp, handshake::RepeatingInitOp>(
              user)) {
        return false;
      }
    }
  }

  // ensure all other inputs to the target op match this suppressor's condition
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    // if we have a branch it must have the same condition
    if (auto siblingBranch =
            dyn_cast<handshake::ConditionalBranchOp>(operand.getDefiningOp())) {
      if (siblingBranch == branchOp) continue;
      // if the branch has more than one use, a fork needs to be created first
      if (!siblingBranch.getFalseResult().hasOneUse()) {
        return false;
      }
      // check whether condition matches indirectly
      bool currentInverted = false, siblingInverted = false;
      Value currentRoot = getForkTop(currentCond, currentInverted);
      Value siblingRoot =
          getForkTop(siblingBranch.getConditionOperand(), siblingInverted);

      // They must originate from the same wire AND have the exact same polarity
      if (currentRoot != siblingRoot || currentInverted != siblingInverted) {
        return false;
      }
    } else if (!isSourced(operand)) {
      return false;
    }
  }
  return true;
}

/// Move the suppressors past the following operation, targetOp. Erase all the
/// suppressors going into targetOp and create new suppressors on every output
/// of targetOp.
void performSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                             DenseSet<handshake::ConditionalBranchOp> &frontier,
                             NameAnalysis &namer, int DRewrite) {

  // identify the operation we want to move past
  Value dataPath = branchOp.getFalseResult();
  Operation *targetOp = *dataPath.user_begin();
  llvm::errs() << "Moving past: "
                          << targetOp->getAttr("handshake.name") << '\n';

  // ensure that targetOp and the new branch are in the same bb as the old
  // branch. skip for load operations due to the memory controller
  auto bbAttr = targetOp->getAttr("handshake.bb");
  if (!isa<handshake::LoadOp>(targetOp) &&
      targetOp->getAttr("handshake.bb") != bbAttr) {
    targetOp->moveAfter(branchOp);
    bbAttr = branchOp->getAttr("handshake.bb");
    targetOp->setAttr("handshake.bb", bbAttr);
  }

  Location loc = targetOp->getLoc();
  Value condition = branchOp.getConditionOperand();

  // erase all old suppressors feeding into the target operation
  for (Value operand : targetOp->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (auto incomingBranch =
              dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
        // for Rewrite D, move only the suppressor connected to A of the Mux
        if (incomingBranch != branchOp && DRewrite) {
          continue;
        }
        // erase the branch
        frontier.erase(incomingBranch);
        targetOp->replaceUsesOfWith(operand, incomingBranch.getDataOperand());
        incomingBranch->erase();
      }
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
    setupMetadata(bbAttr, namer, newBranch);

    // reroute downstream consumers to look at the new branch's FalseResult
    result.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
    frontier.insert(newBranch);
  }
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

  // find the condition signal C
  auto notOp = dyn_cast<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (!notOp) {
    // get the original condition signal and create two not operations to get
    // the correct circuit necessary for the Rewrite
    Value originalCondition = branchOp.getConditionOperand();
    OpBuilder builder(branchOp);
    auto firstNot =
        builder.create<handshake::NotIOp>(branchOp.getLoc(), originalCondition);
    // Create the second NotIOp consuming the result of the first
    auto secondNot = builder.create<handshake::NotIOp>(branchOp.getLoc(),
                                                       firstNot.getResult());
    setupMetadata(bbAttr, namer, firstNot, secondNot);

    // Rewire the branchOp to consume the second NotIOp's result
    branchOp.getConditionOperandMutable().assign(secondNot.getResult());
    notOp = secondNot;
  }
  Value conditionC = notOp.getOperand();

  // if Rewrite D is applied multiple times, connect the new repeatinginitOp to
  // the old one
  Value oldInitInput = initOp.getOperand();
  auto prevRepeatingInit = dyn_cast_or_null<handshake::RepeatingInitOp>(
      oldInitInput.getDefiningOp());
  Value inputOperand = prevRepeatingInit ? oldInitInput : conditionC;

  // build the additional control structure for the rewrite
  OpBuilder builder(dataMux);

  auto repeatingInit =
      builder.create<handshake::RepeatingInitOp>(loc, inputOperand, 1);
  setupMetadata(bbAttr, namer, repeatingInit);
  Value specOutput = repeatingInit.getResult();

  // connect the repeatingInit to the loop init and the suppressor
  initOp.getOperandMutable().assign(specOutput);

  // rewire the NotIOp's input to the new repeating init result
  if (notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(specOutput);
  else {
    // if the notop has other uses, create a new one with only one use
    builder.setInsertionPoint(notOp);
    auto newNotOp =
        builder.create<handshake::NotIOp>(notOp->getLoc(), specOutput);
    setupMetadata(notOp->getAttr("handshake.bb"), namer, newNotOp);
    // rewire the current branchOp to use the new NotIOp's result
    branchOp.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  // move suppressor past the mux
  performSuppressorMotion(branchOp, frontier, namer, 1);
}

void applyRewriteC(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite C for: "
                          << dataMux->getAttr("handshake.name") << '\n';

  // find the condition signal C
  auto notOp = dyn_cast<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (!notOp) {
    // get the original condition signal and create two not operations to get
    // the correct circuit necessary for the Rewrite
    Value originalCondition = branchOp.getConditionOperand();
    OpBuilder builder(branchOp);
    auto firstNot =
        builder.create<handshake::NotIOp>(branchOp.getLoc(), originalCondition);
    // Create the second NotIOp consuming the result of the first
    auto secondNot = builder.create<handshake::NotIOp>(branchOp.getLoc(),
                                                       firstNot.getResult());
    setupMetadata(bbAttr, namer, firstNot, secondNot);

    // Rewire the branchOp to consume the second NotIOp's result
    branchOp.getConditionOperandMutable().assign(secondNot.getResult());
    notOp = secondNot;
  }
  Value conditionC = notOp.getOperand();

  // build the additional control structure for the rewrite
  OpBuilder builder(dataMux);

  auto repeatingInit =
      builder.create<handshake::RepeatingInitOp>(loc, conditionC, 1);
  setupMetadata(bbAttr, namer, repeatingInit);

  // create a mux with a true on false and on true the condition
  auto sourceOp = builder.create<handshake::SourceOp>(loc);
  auto cnstTrue = builder.create<handshake::ConstantOp>(
      loc, builder.getBoolAttr(true), sourceOp.getResult());
  llvm::SmallVector<Value, 2> muxGreenInputs = {cnstTrue.getResult(),
                                                conditionC};
  auto muxGreen = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), repeatingInit.getResult(), muxGreenInputs);
  setupMetadata(bbAttr, namer, sourceOp, cnstTrue, muxGreen);

  // connect the repeatingInit to the loop mux and the new mux
  dataMux.getSelectOperandMutable().assign(repeatingInit.getResult());

  // rewire the NotIOp's input to the new muxes result
  if (notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(muxGreen.getResult());
  else {
    // if the notop has other uses, create a new one with only one use
    builder.setInsertionPoint(notOp);
    auto newNotOp = builder.create<handshake::NotIOp>(notOp->getLoc(),
                                                      muxGreen.getResult());
    setupMetadata(notOp->getAttr("handshake.bb"), namer, newNotOp);
    // rewire the current branchOp to use the new NotIOp's result
    branchOp.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  // move suppressor past the mux
  performSuppressorMotion(branchOp, frontier, namer);
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

  auto firstNot = dyn_cast_or_null<handshake::NotIOp>(
      falseBranch.getConditionOperand().getDefiningOp());
  handshake::NotIOp secondNot = nullptr;
  if (!firstNot) {
    // if the false branch does not have a notOp, construct double-inverse
    Value originalCondition = falseBranch.getConditionOperand();
    OpBuilder builder(falseBranch);

    firstNot = builder.create<handshake::NotIOp>(falseBranch.getLoc(),
                                                      originalCondition);
    secondNot = builder.create<handshake::NotIOp>(falseBranch.getLoc(),
                                                  firstNot.getResult());
    setupMetadata(bbAttr, namer, firstNot, secondNot);

    falseBranch.getConditionOperandMutable().assign(secondNot.getResult());
  } else {
    auto outerNot = firstNot;
    firstNot = dyn_cast<handshake::NotIOp>(outerNot.getOperand().getDefiningOp());
    if (!firstNot)
      // TODO: what if the nots are inverted, maybe something else should be
      return;
    
    secondNot = outerNot;
  }
  Value conditionC = secondNot.getOperand();

  // build the additional control structure
  OpBuilder builder(dataMux);

  // create the feedback loop: not -> initOp -> mux, not
  // initialize initOp with conditionC temporarily
  auto initOp = builder.create<handshake::InitOp>(loc, conditionC);
  auto redNot = builder.create<handshake::NotIOp>(loc, initOp.getResult());
  setupMetadata(bbAttr, namer, initOp, redNot);
  // Close the loop: connect the output of redNot back into the InitOp's input
  initOp.getOperandMutable().assign(redNot.getResult());

  // create the green loop
  // secondNot is the green Not already
  // llvm::SmallVector<Value, 2> greenMuxInputs = {secondNot.getResult(),
                                                // conditionC};
  llvm::SmallVector<Value, 2> greenMuxInputs = {conditionC,
                                                secondNot.getResult()};
  auto greenMux = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), initOp.getResult(), greenMuxInputs);
  setupMetadata(bbAttr, namer, greenMux);

  // connect the new structure back to the old one
  dataMux.getSelectOperandMutable().assign(initOp.getResult());

  // 5. Rewire the first not's input to the green Mux's result
  if (firstNot.getResult().hasOneUse()) {
    firstNot.getOperandMutable().assign(greenMux.getResult());
  } else {
    // If the NotIOp has other downstream uses, isolate it
    builder.setInsertionPoint(firstNot);
    auto newNotOp = builder.create<handshake::NotIOp>(firstNot->getLoc(),
                                                      greenMux.getResult());
    setupMetadata(firstNot->getAttr("handshake.bb"), namer, newNotOp);

    // Rewire the false suppressor to use this isolated NotIOp
    falseBranch.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  performSuppressorMotion(falseBranch, frontier, namer);
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
  auto notOp = dyn_cast<handshake::NotIOp>(
      trueBranch.getConditionOperand().getDefiningOp());
  if (!notOp) {
    // get the original condition signal and create two not operations to get
    // the correct circuit necessary for the Rewrite
    // condition->firstNot->secondNot->trueBranch
    Value originalCondition = trueBranch.getConditionOperand();
    OpBuilder builder(trueBranch);
    auto firstNot = builder.create<handshake::NotIOp>(trueBranch.getLoc(),
                                                      originalCondition);
    auto secondNot = builder.create<handshake::NotIOp>(trueBranch.getLoc(),
                                                       firstNot.getResult());
    setupMetadata(bbAttr, namer, firstNot, secondNot);

    // Rewire the trueBranch to consume the second NotIOp's result
    trueBranch.getConditionOperandMutable().assign(secondNot.getResult());
    notOp = secondNot;
  }
  Value conditionC = notOp.getOperand();

  // build the new Mux
  OpBuilder builder(dataMux);
  llvm::SmallVector<Value, 2> newMuxInputs = {conditionC, conditionC};
  auto newMux = builder.create<handshake::MuxOp>(
      loc, conditionC.getType(), initOp.getResult(), newMuxInputs);
  setupMetadata(bbAttr, namer, newMux);

  // rewire the input to the suppressor to the new mux
  if (notOp.getResult().hasOneUse()) {
    notOp.getOperandMutable().assign(newMux.getResult());
  } else {
    // if other operations share this NotIOp, create a new isolated one
    builder.setInsertionPoint(notOp);
    auto newNotOp =
        builder.create<handshake::NotIOp>(notOp->getLoc(), newMux.getResult());
    setupMetadata(notOp->getAttr("handshake.bb"), namer, newNotOp);

    trueBranch.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  performSuppressorMotion(trueBranch, frontier, namer);
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
    setupMetadata(bbAttr, namer, notOp);
    newInputA = notOp.getResult();
  }
  auto andOp = builder.create<handshake::AndIOp>(loc, newInputA, inputB);
  setupMetadata(bbAttr, namer, andOp);

  // reroute all downstream operations from the dataMux to the new AND
  dataMux.getResult().replaceAllUsesWith(andOp.getResult());

  // clean up
  frontier.erase(trueBranch);
  trueBranch.erase();
  dataMux.erase();
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

  // bypass topSuppC TODO: make this safe if there are other uses
  Value wireC = topSuppressorC.getFalseResult();
  wireC.replaceAllUsesWith(topSuppressorC.getDataOperand());
  if (wireC.use_empty()) {
    frontier.erase(topSuppressorC);
    topSuppressorC->erase();
  }

  // build the new suppressor
  OpBuilder builder(dataMux);
  builder.setInsertionPointAfter(dataMux);

  auto movedSuppressor = builder.create<handshake::ConditionalBranchOp>(
      loc, conditionD, dataMux.getResult());
  setupMetadata(bbAttr, namer, movedSuppressor);
  frontier.insert(movedSuppressor);

  // reroute all downstream operations to the new moved suppressor's result
  dataMux.getResult().replaceAllUsesExcept(movedSuppressor.getFalseResult(),
                                           movedSuppressor);
}

void applyRewriteF(handshake::ConditionalBranchOp branchOp,
                   handshake::ConditionalBranchOp topSuppLeft,
                   handshake::ConditionalBranchOp topSuppRight,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer) {
  Location loc = branchOp->getLoc();
  OpBuilder builder(branchOp);

  llvm::errs() << "Applying Rewrite F!\n";
  branchOp.dump();
  topSuppLeft.dump();
  topSuppRight.dump();
  Value conditionB = topSuppLeft.getConditionOperand();
  auto notOpB = dyn_cast<handshake::NotIOp>(conditionB.getDefiningOp());
  if (notOpB) {
    conditionB = notOpB.getOperand();
    llvm::errs() << "there's a not\n";
  } else {
    // input to the and gate for B must be inverted - add a NotIOp
    auto newNotB = builder.create<handshake::NotIOp>(loc, conditionB);
    setupMetadata(branchOp->getAttr("handshake.bb"), namer, newNotB);
    conditionB = newNotB.getResult();
    llvm::errs() << "Inserted NOT op for non-inverted condition B\n";
  }
  Value conditionC = topSuppRight.getDataOperand();

  conditionB.dump();
  topSuppRight.getConditionOperand().dump();
  
  // create the new AND gate
  auto andOp = builder.create<handshake::AndIOp>(loc, conditionB,
                                                 conditionC);
  setupMetadata(branchOp->getAttr("handshake.bb"), namer, andOp);

  // safely bypass or remove topSuppLeft
  Value wireLeft = topSuppLeft.getFalseResult();
  branchOp->replaceUsesOfWith(wireLeft, topSuppLeft.getDataOperand());
  if (wireLeft.use_empty()) {
    frontier.erase(topSuppLeft);
    topSuppLeft->erase();
    llvm::errs() << "erased topSuppLeft\n";
  }

  // safely bypass or remove topSuppRight
  Value wireRight = topSuppRight.getFalseResult();
  
  // check whether we have to reconnect the notOp or the branchOp directly
  if (auto notOp = dyn_cast_or_null<handshake::NotIOp>(
          branchOp.getConditionOperand().getDefiningOp())) {
    notOp->replaceUsesOfWith(wireRight, andOp.getResult());
    llvm::errs() << "reconnect if\n";
  } else {
    branchOp.getConditionOperandMutable().assign(andOp.getResult());
    llvm::errs() << "reconnect else\n";
  }
  if (wireRight.use_empty()) {
    frontier.erase(topSuppRight);
    topSuppRight->erase();
    llvm::errs() << "erased topSuppRight\n";
  } 
}
