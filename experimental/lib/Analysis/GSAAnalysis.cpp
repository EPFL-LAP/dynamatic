//===- GSAAnalysis.cpp - GSA analyis utilities ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assignment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vector"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "gsa"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;
using namespace dynamatic::experimental::ftd;
using namespace dynamatic::experimental::boolean;

experimental::gsa::GSAAnalysis::GSAAnalysis(handshake::MergeOp &merge,
                                            Region &region) {
  inputOp = &region;
  convertSSAToGSAMerges(merge, region);
}

void experimental::gsa::GSAAnalysis::convertSSAToGSAMerges(
    handshake::MergeOp &mergeOp, Region &region) {

  // Associate an index to each basic block in "funcOp" so that if Bi
  // dominates Bj than i < j
  BlockIndexing bi(region);

  // Initialize the index of the class
  uniqueGateIndex = 0;

  Block *block = mergeOp.getResult().getParentBlock();

  // Create an empty list for the phi functions corresponding to the block
  gatesPerBlock.insert({block, llvm::SmallVector<Gate *>()});

  // Create a set for the operands of the corresponding phi function
  SmallVector<GateInput *> operands;

  auto isAlreadyPresent = [&](Value c) -> bool {
    return std::any_of(operands.begin(), operands.end(), [c](GateInput *in) {
      return in->isTypeValue() && in->getValue() == c;
    });
  };

  // Add to the list of operands of the new gate all the values which were not
  // already used
  for (Value v : mergeOp.getOperands()) {
    if (!isAlreadyPresent(v)) {
      GateInput *gateInput = new GateInput(v);
      gateInputList.push_back(gateInput);
      operands.push_back(gateInput);
    }
  }

  // If the list of operands is not empty (i.e. the phi has at least
  // one input), add it to the phis associated to that block
  if (!operands.empty()) {
    Gate *newPhi = new Gate(mergeOp.getResult(), operands, GateType::PhiGate,
                            ++uniqueGateIndex);
    gatesPerBlock[block].push_back(newPhi);
  }

  convertPhiToMu(region);
  convertPhiToGamma(region, bi);
  printAllGates();
}

experimental::gsa::GSAAnalysis::GSAAnalysis(Operation *operation) {

  // Only one function should be present in the module, excluding external
  // functions
  unsigned functionsCovered = 0;

  // TODO: Extend to support multiple functions

  // The analysis can be instantiated either over a module containing one
  // function only or over a function
  if (ModuleOp modOp = dyn_cast<ModuleOp>(operation); modOp) {
    for (func::FuncOp funcOp : modOp.getOps<func::FuncOp>()) {

      // Skip if external
      if (funcOp.isExternal())
        continue;

      // Analyze the function
      if (!functionsCovered) {
        inputOp = &funcOp.getRegion();
        convertSSAToGSA(*inputOp);
        functionsCovered++;
      } else {
        llvm::errs() << "[GSA] Too many functions to handle in the module";
      }
    }
  } else if (func::FuncOp fOp = dyn_cast<func::FuncOp>(operation); fOp) {
    convertSSAToGSA(fOp.getRegion());
    functionsCovered = 1;
  }

  // report an error indicating that the analysis is instantiated over
  // an inappropriate operation
  if (functionsCovered != 1)
    llvm::errs() << "[GSA] GSAAnalysis failed due to a wrong input type\n";
};

experimental::gsa::Gate *experimental::gsa::GSAAnalysis::expandGammaTree(
    ListExpressionsPerGate &expressions, std::queue<unsigned> conditions,
    Gate *originalPhi, const BlockIndexing &bi) {

  // At each iteration, we want to use a cofactor that is present in all the
  // expressions in `expressions`. Since the cofactors are ordered according to
  // the basic block number, we can say that if a cofactor is present in one
  // expression, then it must be present in all the others, since they all have
  // the blocks associated to that cofactor as common dominator.
  unsigned indexToUse;
  std::string conditionToUse;
  while (true) {
    indexToUse = conditions.front();
    conditionToUse = "c" + std::to_string(indexToUse);
    conditions.pop();
    unsigned cofactorUsage =
        std::count_if(expressions.begin(), expressions.end(),
                      [&](std::pair<BoolExpression *, GateInput *> g) {
                        return g.first->containsMintern(conditionToUse);
                      });

    // None is using that cofactor, go to the next one
    if (cofactorUsage == 0)
      continue;

    // All are using the cofactor, expand the expressions over it
    if (cofactorUsage == expressions.size())
      break;

    llvm_unreachable("A cofactor is used only by some expressions");
  }

  ListExpressionsPerGate conditionsFalseExpressions;
  ListExpressionsPerGate conditionsTrueExpressions;

  // For each expression
  for (const auto &expression : expressions) {

    // Restrict an expression according to a condition ans an input boolean
    // value and return it
    auto getRestrictedExpression = [&](bool value) -> BoolExpression * {
      BoolExpression *expr = expression.first->deepCopy();
      restrict(expr, conditionToUse, value);
      return expr->boolMinimize();
    };

    // Substitute a condition in the expression both with value true and false
    BoolExpression *exprTrue = getRestrictedExpression(true);
    BoolExpression *exprFalse = getRestrictedExpression(false);

    // One of the two expressions above will be zero: add the input to the
    // corresponding list
    if (exprTrue->type != ExpressionType::Zero)
      conditionsTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->type != ExpressionType::Zero)
      conditionsFalseExpressions.emplace_back(exprFalse, expression.second);
  }

  SmallVector<GateInput *> operandsGamma(2);

  // If the number of non-null expressions obtained with cofactor = X is greater
  // than 1, then the expansions must be done again over those inputs with a new
  // cofactor. The resulting gamma is going to be used as X input of this
  // gamma.
  //
  // If only one element is present, that same element is used as input of the
  // gamma.
  //
  // If no elements are present, then that input will never be used, and it is
  // considered as empty.
  auto setGammaOperand = [&](int input, ListExpressionsPerGate &list) -> void {
    if (list.size() > 1) {
      Gate *gamma = expandGammaTree(list, conditions, originalPhi, bi);
      operandsGamma[input] = new GateInput(gamma);
      gateInputList.push_back(operandsGamma[input]);
    } else if (list.size() == 1) {
      operandsGamma[input] = list[0].second;
    } else {
      operandsGamma[input] = new GateInput();
      gateInputList.push_back(operandsGamma[input]);
    }
  };

  setGammaOperand(1, conditionsTrueExpressions);
  setGammaOperand(0, conditionsFalseExpressions);

  // Create a new gamma and add it to the list of phis for the original basic
  // block.

  // Get the index of the condition (it is associated to a basic block in
  // "indexPerBlock" mapping)
  Gate *newGate =
      new Gate(originalPhi->result, operandsGamma, GateType::GammaGate,
               ++uniqueGateIndex, bi.getBlockFromIndex(indexToUse));
  gatesPerBlock[originalPhi->getBlock()].push_back(newGate);

  return newGate;
}

void experimental::gsa::GSAAnalysis::convertSSAToGSA(Region &region) {

  if (region.getBlocks().size() == 1)
    return;

  // Associate an index to each basic block in "funcOp" so that if Bi
  // dominates Bj than i < j
  BlockIndexing bi(region);

  // This function works in two steps. First, all the block arguments in the
  // IR are converted into PHIs, taking care or properly extracting the
  // information about the producers of the operands. Then, the phis are
  // converted either to GAMMAs or MUs.

  // The input of a phi might be another gate. This is the case when the
  // input of a phi is a block argument from a block with index different
  // from 0. In this situation, we mark the phi input as "missing", and we
  // store the necessary information (block argument) to later reconstruct
  // the relationship.
  struct MissingPhi {

    // Which input is missing
    GateInput *pi;

    // Related block argument
    BlockArgument blockArg;

    MissingPhi(GateInput *pi, BlockArgument blockArg)
        : pi(pi), blockArg(blockArg) {}
  };

  // Initialize the index of the class
  uniqueGateIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phisToConnect;

  // For each block in the function
  for (Block &block : region.getBlocks()) {

    // Create an empty list for the phi functions corresponding to the block
    gatesPerBlock.insert({&block, llvm::SmallVector<Gate *>()});

    // For each block argument
    for (BlockArgument &arg : block.getArguments()) {
      unsigned argNumber = arg.getArgNumber();
      // Create a set for the operands of the corresponding phi function
      SmallVector<GateInput *> operands;
      DenseSet<Block *> coveredPredecessors;
      // For each predecessor of the block, which is in charge of
      // providing the inputs of the phi functions
      for (Block *pred : block.getPredecessors()) {

        // Make sure that a predecessor is covered only once
        if (coveredPredecessors.contains(pred))
          continue;
        coveredPredecessors.insert(pred);

        // Get the branch terminator
        auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
        assert(branchOp && "Expected terminator operation in a predecessor "
                           "block feeding a block argument!");

        // Check if the input value "c" of type Value is already present among
        // the operands of the phi function
        auto isAlreadyPresent = [&](Value c) -> bool {
          return std::any_of(operands.begin(), operands.end(),
                             [c](GateInput *in) {
                               return in->isTypeValue() && in->getValue() == c;
                             });
        };

        // For each alternative in the branch terminator
        for (auto [successorId, successorBlock] :
             llvm::enumerate(branchOp->getSuccessors())) {

          // Skip the successor if it is not the block under analysis
          if (successorBlock != &block)
            continue;

          // Get the values used for that branch
          auto successorOperands = branchOp.getSuccessorOperands(successorId);
          // Get the value used as input of the gate
          Value producer = successorOperands[argNumber];
          GateInput *gateInput = nullptr;

          /// If it is a block argument whose parent block has some predecessor,
          /// then the value is the output of a phi, and we add it to the list
          /// of missing phis. Otherwise, the input is a value, and it can be
          /// safely added directly.
          if (BlockArgument blockArg = dyn_cast<BlockArgument>(producer);
              blockArg && !producer.getParentBlock()->hasNoPredecessors()) {
            gateInput = new GateInput((Gate *)nullptr);
            phisToConnect.push_back(MissingPhi(gateInput, blockArg));
            gateInputList.push_back(gateInput);
          } else {
            if (!isAlreadyPresent(dyn_cast<Value>(producer))) {
              gateInput = new GateInput(producer);
              gateInputList.push_back(gateInput);
            }
          }

          // Insert the value among the inputs of the phi
          if (gateInput)
            operands.push_back(gateInput);

          break;
        }
      }

      // If the list of operands is not empty (i.e. the phi has at least
      // one input), add it to the phis associated to that block
      if (!operands.empty()) {
        Gate *newPhi =
            new Gate(arg, operands, GateType::PhiGate, ++uniqueGateIndex);
        gatesPerBlock[&block].push_back(newPhi);
      }
    }
  }

  // Find the missing phi and correct the pointers
  for (MissingPhi &missing : phisToConnect) {
    auto list = gatesPerBlock[missing.blockArg.getParentBlock()];
    Gate **foundGate = std::find_if(list.begin(), list.end(),
                                    [&](Gate *&t) {
                                      return t->getArgumentNumber() ==
                                             missing.blockArg.getArgNumber();
                                    }

    );
    assert(foundGate != list.end() && "[GSA] Not found phi to reconnect");
    missing.pi->input = *foundGate;
  }

  convertPhiToMu(region);
  convertPhiToGamma(region, bi);
  printAllGates();
}

void experimental::gsa::GSAAnalysis::convertPhiToGamma(
    Region &region, const BlockIndexing &bi) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  // For each block
  for (auto const &[phiBlock, phis] : gatesPerBlock) {

    // For each phi
    for (Gate *phi : phis) {

      // Skip if the phi is not of type `Phi`
      if (phi->gsaGateFunction != PhiGate)
        continue;

      // Sort the operands of the phi so that Bi comes before Bj if Bi dominates
      // Bj
      SmallVector<GateInput *> phiOperands = phi->operands;
      llvm::sort(phiOperands.begin(), phiOperands.end(),
                 [&](GateInput *a, GateInput *b) {
                   return domInfo.dominates(a->getBlock(), b->getBlock());
                 });

      // Find the nearest common dominator among all the blocks involved
      // in the phi inputs. Also add all the blocks to the list of 'blocks to
      // avoid'
      std::vector<Block *> blocksToAvoid = {phiOperands[0]->getBlock()};
      Block *commonDominator = phiOperands[0]->getBlock();
      for (GateInput *operand : llvm::drop_begin(phiOperands)) {
        Block *bb = operand->getBlock();
        commonDominator =
            domInfo.findNearestCommonDominator(bb, commonDominator);
        blocksToAvoid.push_back(bb);
      }

      // When traversing a path, some blocks have a condition which steers the
      // control flow execution. This set stores the list of the blocks
      // accordingly. They will become boolean conditions afterwards
      DenseSet<unsigned> blocksWithConditionInPath;

      // Vector associating each input of the Phi to a boolean expression
      std::vector<std::pair<BoolExpression *, GateInput *>> expressionsList;

      // For each input of the phi, compute the boolean expression which defines
      // its usage
      for (GateInput *operand : phiOperands) {

        // Remove the current operand from the list of blocks to avoid
        blocksToAvoid.erase(blocksToAvoid.begin());

        // through operand's block but not through any of the `blocksToAvoid`
        auto paths = findAllPaths(commonDominator, phiBlock, bi,
                                  operand->getBlock(), blocksToAvoid);

        BoolExpression *phiInputCondition = BoolExpression::boolZero();

        // Sum all the conditions for each path
        for (std::vector<Block *> &path : paths) {
          boolean::BoolExpression *condition =
              getPathExpression(path, blocksWithConditionInPath, bi);
          phiInputCondition =
              BoolExpression::boolOr(condition, phiInputCondition);
          phiInputCondition = phiInputCondition->boolMinimize();
        }

        // Associate the expression to the phi operand
        expressionsList.emplace_back(phiInputCondition, operand);
      }

      // Get a queue with all the necessary conditions, ordered according to
      // their basic block index
      std::vector<unsigned> conditionsToOrder;
      for (unsigned &index : blocksWithConditionInPath)
        conditionsToOrder.push_back(index);
      std::sort(conditionsToOrder.begin(), conditionsToOrder.end());
      std::queue<unsigned> conditionsOrdered;
      for (unsigned &index : conditionsToOrder)
        conditionsOrdered.push(index);

      // Expand the expressions to get the tree of gammas
      Gate *gammaRoot =
          expandGammaTree(expressionsList, conditionsOrdered, phi, bi);
      gammaRoot->isRoot = true;

      // Once that a phi has been converted into a tree of gammas, all the
      // gates which used the original phi as input must be connected to the
      // root of the gamma tree
      for (auto &[bb, phis] : gatesPerBlock) {
        for (Gate *phii : phis) {
          for (GateInput *op : phii->operands) {
            if (op->isTypeGate() && op->getGate() == phi)
              op->input = gammaRoot;
          }
        }
      }
    }
  }
}

void experimental::gsa::GSAAnalysis::convertPhiToMu(Region &region) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  // For each phi
  for (const std::pair<Block *, SmallVector<Gate *>> &entry : gatesPerBlock) {
    Block *phiBlock = entry.first;
    SmallVector<Gate *> phis = entry.second;
    for (Gate *phi : phis) {

      // A phi might be a MU iff it is inside a for loop and has exactly
      // two operands
      if (!loopInfo.getLoopFor(phiBlock) || phi->operands.size() != 2)
        continue;

      Block *op0Block = phi->operands[0]->getBlock(),
            *op1Block = phi->operands[1]->getBlock();

      // Checks whether the block of the merge is a loop header
      bool isBlockHeader =
          loopInfo.getLoopFor(phiBlock)->getHeader() == phiBlock;

      // Checks whether the two operands come from different loops (in
      // this case, one of the values is the initial definition)
      bool operandFromOutsideLoop =
          loopInfo.getLoopFor(op0Block) != loopInfo.getLoopFor(op1Block);

      // If both the conditions hold, then we have a MU gate
      if (!(isBlockHeader && operandFromOutsideLoop))
        continue;

      phi->gsaGateFunction = GateType::MuGate;

      // Use the initial value of MU as first input of the gate
      if (domInfo.dominates(op1Block, phiBlock))
        std::swap(phi->operands[0], phi->operands[1]);

      // The block determining the MU condition is the exiting block of the
      // innermost loop the MU is in
      phi->conditionBlock =
          loopInfo.getLoopFor(phi->getBlock())->getExitingBlock();
      phi->isRoot = true;
    }
  }
}

ArrayRef<experimental::gsa::Gate *>
experimental::gsa::GSAAnalysis::getGatesPerBlock(Block *bb) const {
  auto it = gatesPerBlock.find(bb);
  return it == gatesPerBlock.end() ? ArrayRef<Gate *>() : it->getSecond();
}

void experimental::gsa::GSAAnalysis::printAllGates() {
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      g->print();
  }
}

experimental::gsa::GSAAnalysis::~GSAAnalysis() {
  for (GateInput *gi : gateInputList)
    delete gi;
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      delete g;
  }
}

void experimental::gsa::Gate::print() {

  LLVM_DEBUG(
      auto getPhiName = [](Gate *p) -> std::string {
        switch (p->gsaGateFunction) {
        case GammaGate:
          return "GAMMA";
        case MuGate:
          return "MU";
        default:
          return "PHI";
        }
      };

      llvm::dbgs() << "[GSA] Block "; getBlock()->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " arg " << getArgumentNumber() << " type "
                   << getPhiName(this) << "_" << index;

      if (gsaGateFunction == GammaGate || gsaGateFunction == MuGate) {
        llvm::dbgs() << " condition ";
        conditionBlock->printAsOperand(llvm::dbgs());
      }

      llvm::dbgs()
      << "\n";

      for (GateInput *&op : operands) {
        if (op->isTypeValue()) {
          llvm::dbgs() << "[GSA]\t VALUE\t: ";
          op->getValue().print(llvm::dbgs());
        } else if (op->isTypeEmpty()) {
          llvm::dbgs() << "[GSA]\t EMPTY";
        } else {
          llvm::dbgs() << "[GSA]\t GATE\t: " << getPhiName(op->getGate()) << "_"
                       << op->getGate()->index;
        }

        if (!op->isTypeEmpty()) {
          llvm::dbgs() << "\t(";
          op->getBlock()->printAsOperand(llvm::dbgs());
          llvm::dbgs() << ")";
        }
        llvm::dbgs() << "\n";
      });
}

Block *experimental::gsa::GateInput::getBlock() {
  if (isTypeEmpty())
    return nullptr;
  return isTypeGate() ? getGate()->getBlock() : getValue().getParentBlock();
}

LogicalResult experimental::gsa::GSAAnalysis::addGsaGates(
    Region &region, ConversionPatternRewriter &rewriter, const GSAAnalysis &gsa,
    Backedge startValue, bool removeTerminators) {

  using namespace experimental::gsa;

  // The function instantiates the GAMMA and MU gates as provided by the GSA
  // analysis pass. A GAMMA function is translated into a multiplxer driven by
  // single control signal and fed by two operands; a MU function is
  // translated into a multiplxer driven by an init (it is currently
  // implemented as a Merge fed by a constant triggered from Start once and
  // from the loop condition thereafter). The input of one of these functions
  // might be another GSA function, and it's possible that the function was
  // not instantiated yet. For this reason, we keep track of the missing
  // operands, and reconnect them later on.
  //
  // To simplify the way GSA functions are handled, each of them has an unique
  // index.

  struct MissingGsa {
    // Index of the GSA function to modify
    unsigned phiIndex;
    // Index of the GSA function providing the result
    unsigned edgeIndex;
    // Index of the operand to modify
    unsigned operandInput;

    MissingGsa(unsigned pi, unsigned ei, unsigned oi)
        : phiIndex(pi), edgeIndex(ei), operandInput(oi) {}
  };

  if (region.getBlocks().size() == 1)
    return success();

  // List of missing GSA functions
  SmallVector<MissingGsa> missingGsaList;
  // List of gammas with only one input
  DenseSet<Operation *> oneInputGammaList;
  // Maps the index of each GSA function to each real operation
  DenseMap<unsigned, Operation *> gsaList;

  // For each block excluding the first one, which has no gsa
  for (Block &block : llvm::drop_begin(region)) {

    // For each GSA function
    ArrayRef<Gate *> phis = gsa.getGatesPerBlock(&block);
    for (Gate *phi : phis) {

      // TODO: No point of this skipping if we have a guarantee that the
      // phi->gsaGateFunction is exclusively either MuGate or GammaGate...
      // Skip if it's an SSA phi that has more than 2 inputs (not yet broken
      // down to multiple GSA gates)
      if (phi->gsaGateFunction == PhiGate)
        continue;

      Location loc = block.front().getLoc();
      rewriter.setInsertionPointToStart(&block);
      SmallVector<Value> operands;

      // Maintain the index of the current operand
      unsigned operandIndex = 0;
      // Checks whether one index is empty
      int nullOperand = -1;

      // For each of its operand
      for (auto *operand : phi->operands) {
        // If the input is another GSA function, then a dummy value is used as
        // operand and the operations will be reconnected later on.
        // If the input is empty, we keep track of its index.
        // In the other cases, we already have the operand of the function.
        if (operand->isTypeGate()) {
          Gate *g = std::get<Gate *>(operand->input);
          operands.emplace_back(g->result);
          missingGsaList.emplace_back(
              MissingGsa(phi->index, g->index, operandIndex));
        } else if (operand->isTypeEmpty()) {
          nullOperand = operandIndex;
          operands.emplace_back(nullptr);
        } else {
          auto val = std::get<Value>(operand->input);
          operands.emplace_back(val);
        }
        operandIndex++;
      }

      // The condition value is provided by the `condition` field of the phi
      rewriter.setInsertionPointAfterValue(phi->result);
      Value conditionValue =
          phi->conditionBlock->getTerminator()->getOperand(0);

      // If the function is MU, then we create a merge
      // and use its result as condition
      if (phi->gsaGateFunction == MuGate) {
        mlir::DominanceInfo domInfo;
        mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

        // The inputs of the merge are the condition value and a `false`
        // constant driven by the start value of the function. This will
        // created later on, so we use a dummy value.
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(conditionValue);
        mergeOperands.push_back(conditionValue);

        auto initMergeOp =
            rewriter.create<handshake::MergeOp>(loc, mergeOperands);

        initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

        // Replace the new condition value
        conditionValue = initMergeOp->getResult(0);
        conditionValue.setType(channelifyType(conditionValue.getType()));

        // Add the activation constant driven by the backedge value, which will
        // be then updated with the real start value, once available
        auto cstType = rewriter.getIntegerType(1);
        auto cstAttr = IntegerAttr::get(cstType, 0);
        rewriter.setInsertionPointToStart(initMergeOp->getBlock());
        auto constOp = rewriter.create<handshake::ConstantOp>(
            initMergeOp->getLoc(), cstAttr, startValue);
        constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
        initMergeOp->setOperand(0, constOp.getResult());
      }

      // When a single input gamma is encountered, a mux is inserted as a
      // placeholder to perform the gamma/mu allocation flow. In the end,
      // these muxes are erased from the IR
      if (nullOperand >= 0) {
        operands[0] = operands[1 - nullOperand];
        operands[1] = operands[1 - nullOperand];
      }

      // Create the multiplexer
      auto mux = rewriter.create<handshake::MuxOp>(loc, phi->result.getType(),
                                                   conditionValue, operands);

      // The one input gamma is marked at an operation to skip in the IR and
      // later removed
      if (nullOperand >= 0)
        oneInputGammaList.insert(mux);

      if (phi->isRoot)
        rewriter.replaceAllUsesWith(phi->result, mux.getResult());

      gsaList.insert({phi->index, mux});
      mux->setAttr(FTD_EXPLICIT_PHI, rewriter.getUnitAttr());
    }
  }

  // For each of the GSA missing inputs, perform a replacement
  for (auto &missingMerge : missingGsaList) {

    auto *operandMerge = gsaList[missingMerge.phiIndex];
    auto *resultMerge = gsaList[missingMerge.edgeIndex];

    operandMerge->setOperand(missingMerge.operandInput + 1,
                             resultMerge->getResult(0));

    // In case of a one-input gamma, the other input must be replaced as well,
    // to avoid errors when the block arguments are erased later on
    if (oneInputGammaList.contains(operandMerge))
      operandMerge->setOperand(2 - missingMerge.operandInput,
                               resultMerge->getResult(0));
  }

  // Get rid of the multiplexers adopted as place-holders of one input gamma
  for (auto &op : llvm::make_early_inc_range(oneInputGammaList)) {
    int operandToUse = llvm::isa_and_nonnull<handshake::MuxOp>(
                           op->getOperand(1).getDefiningOp())
                           ? 1
                           : 2;
    op->getResult(0).replaceAllUsesWith(op->getOperand(operandToUse));
    rewriter.eraseOp(op);
  }

  if (!removeTerminators)
    return success();

  // Remove all the block arguments for all the non starting blocks
  for (Block &block : llvm::drop_begin(region))
    block.eraseArguments(0, block.getArguments().size());

  // Each terminator must be replaced so that it does not provide any block
  // arguments (possibly only the final control argument)
  for (Block &block : region) {
    if (Operation *terminator = block.getTerminator(); terminator) {
      rewriter.setInsertionPointAfter(terminator);
      if (auto cbr = dyn_cast<cf::CondBranchOp>(terminator); cbr) {
        while (!cbr.getTrueOperands().empty())
          cbr.eraseTrueOperand(0);
        while (!cbr.getFalseOperands().empty())
          cbr.eraseFalseOperand(0);
      } else if (auto br = dyn_cast<cf::BranchOp>(terminator); br) {
        while (!br.getOperands().empty())
          br.eraseOperand(0);
      }
    }
  }

  return success();
}
