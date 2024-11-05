//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the fast token delivery methodology
// https://ieeexplore.ieee.org/abstract/document/10035134, together with the
// straight LSQ allocation https://dl.acm.org/doi/abs/10.1145/3543622.3573050.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Analysis/GsaAnalysis.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::boolean;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake, ConvertCalls,
                 ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
                 ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
                 OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
                 OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
                 OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
                 OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
                 OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
                 OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
                 OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
                 OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
                 OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
                 OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
                 OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
                 OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
                 OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
                 OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
                 OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
                 OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
                 OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
                 OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
                 OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
                 OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
                 OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
                 OneToOneConversion<arith::TruncFOp, handshake::TruncFOp>,
                 OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>,
                 OneToOneConversion<arith::SIToFPOp, handshake::SIToFPOp>,
                 OneToOneConversion<arith::FPToSIOp, handshake::FPToSIOp>,
                 OneToOneConversion<arith::ExtFOp, handshake::ExtFOp>,
                 OneToOneConversion<math::AbsFOp, handshake::AbsFOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace ftd {

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

// ------------------------ Forwarded declarations ------------------------

static void mapConditionsToValues(Region &region, FtdStoredOperations &ftdOps);

static void connectInitMerges(ConversionPatternRewriter &rewriter,
                              handshake::FuncOp funcOp,
                              FtdStoredOperations &ftdOps);

// ------------------------ End forwarded declarations ------------------------

void FtdLowerFuncToHandshake::analyzeLoop(handshake::FuncOp funcOp,
                                          FtdStoredOperations &ftdOps) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  std::ofstream ofs;

  ofs.open("ftdscripting/loopinfo.txt", std::ofstream::out);
  std::string loopDescription;
  llvm::raw_string_ostream loopDescriptionStream(loopDescription);

  auto muxes = funcOp.getBody().getOps<handshake::MuxOp>();
  for (auto phi : muxes) {
    if (!loopInfo.getLoopFor(phi->getBlock()))
      continue;
    ofs << namer.getName(phi).str();
    if (llvm::isa<handshake::MergeOp>(phi->getOperand(0).getDefiningOp()))
      ofs << " (MU)\n";
    else
      ofs << " (GAMMA)\n";
    loopInfo.getLoopFor(phi->getBlock())
        ->print(loopDescriptionStream, false, false, 0);
    ofs << loopDescription << "\n";
    loopDescription = "";
  }

  ofs.close();
}

LogicalResult FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  FtdStoredOperations ftdOps;

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // Map for each block its exit condition (if exists). This allows to build
  // boolean expressions as circuits
  mapConditionsToValues(lowerFuncOp.getRegion(), ftdOps);

  // Add the muxes as obtained by the GSA analysis pass
  if (failed(addExplicitPhi(lowerFuncOp, rewriter, ftdOps)))
    return failure();

  // First lower the parent function itself, without modifying its body
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // When GSA-MU functions are translated into multiplexers, an `init merge`
  // is created to feed them. This merge requires the start value of the
  // function as one of its data inputs. However, the start value was not
  // present yet when `addExplicitPhi` is called, thus we need to reconnect
  // it.
  connectInitMerges(rewriter, funcOp, ftdOps);

  // Stores mapping from each value that passes through a merge-like
  // operation to the data result of that merge operation
  ArgReplacements argReplacements;

  // Currently, the following 2 functions do nothing but construct the network
  // of CMerges in complete isolation from the rest of the components
  // implementing the operations
  // In particular, the addMergeOps relies on adding Merges for every block
  // argument but because we removed all "real" arguments, we are only left
  // with the Start value as an argument for every block
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports
  // get tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);

  // Create the memory interface according to the algorithm from FPGA'23. This
  // functions introduce new data dependencies that are then passed to FTD for
  // correctly delivering data between them like any real data dependencies
  if (failed(
          ftdVerifyAndCreateMemInterfaces(funcOp, rewriter, memInfo, ftdOps)))
    return failure();

  // Convert the constants and undefined values from the `arith` dialect to
  // the `handshake` dialect, while also using the start value as their
  // control value
  if (failed(convertConstants(rewriter, funcOp)) ||
      failed(convertUndefinedValues(rewriter, funcOp)))
    return failure();

  if (funcOp.getBlocks().size() != 1) {
    // Add muxes for regeneration of values in loop
    if (failed(addRegen(rewriter, funcOp, ftdOps)))
      return failure();

    analyzeLoop(funcOp, ftdOps);

    // Add suppression blocks between each pair of producer and consumer
    if (failed(addSupp(rewriter, funcOp, ftdOps)))
      return failure();
  }

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}

// --- Helper functions ---

/// When init merges for MU functions are instantiated, the function does not
/// have a start signal yet. Once that the start signal is created, it needs
/// to be connected to all the init merges.
static void connectInitMerges(ConversionPatternRewriter &rewriter,
                              handshake::FuncOp funcOp,
                              FtdStoredOperations &ftdOps) {
  auto startValue = (Value)funcOp.getArguments().back();
  auto cstType = rewriter.getIntegerType(1);
  auto cstAttr = IntegerAttr::get(cstType, 0);
  for (auto initMerge : funcOp.getBody().getOps<handshake::MergeOp>()) {
    rewriter.setInsertionPointToStart(initMerge->getBlock());
    auto constOp = rewriter.create<handshake::ConstantOp>(initMerge->getLoc(),
                                                          cstAttr, startValue);
    ftdOps.initMergesOperations.insert(constOp);
    initMerge->setOperand(0, constOp.getResult());
  }
}

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, crate a group graph.
static void constructGroupsGraph(SmallVector<Operation *> &operations,
                                 SmallVector<ProdConsMemDep> &allMemDeps,
                                 DenseSet<Group *> &groups) {

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : operations) {
    if (isHandhsakeLSQOperation(op)) {
      Block *b = op->getBlock();
      auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
      if (it == groups.end()) {
        Group *g = new Group(b);
        groups.insert(g);
      }
    }
  }

  // If there exist a relationship O_i -> O_j, with O_i being in basic BB_i
  // (thus group i) and O_j being in BB_j (thus in group j), add G_i to the
  // predecessors of G_j, G_j to the successors of G_i
  for (ProdConsMemDep memDep : allMemDeps) {
    // Find the group related to the producer
    Group *producerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.prodBb;
        });

    // Find the group related to the consumer
    Group *consumerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.consBb;
        });

    // create edges to link the groups
    producerGroup->succs.insert(consumerGroup);
    consumerGroup->preds.insert(producerGroup);
  }
}

/// Minimizes the connections between groups based on dominance info. Let's
/// consider the graph
///
/// B -> C -> D
/// |         ^
/// |---------|
///
/// having B, C and D as groups, B being predecessor of both C and D, C of D.
/// Since C has to wait for B to be done, and D has to wait for C to be done,
/// there is no point in D waiting for C to be done. For this reason, the
/// graph can be simplified, saving and edge:
///
/// B -> C -> D
static void minimizeGroupsConnections(DenseSet<Group *> &groupsGraph) {

  // Get the dominance info for the region
  DominanceInfo domInfo;

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the
  // whole group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<Group *> predsToRemove;
    for (auto &bp : group->preds) {

      for (auto &sp : group->preds) {

        // if we are considering the same elements, ignore them
        if (sp->bb == bp->bb || greaterThanBlocks(sp->bb, bp->bb))
          continue;

        // Add the small predecessors to the list of elements to remove in
        // case the big predecessor has the small one among its
        // predecessors, and the big precessor's BB properly dominates the
        // BB of the group currently under analysis
        if ((bp->preds.find(sp) != bp->preds.end()) &&
            domInfo.properlyDominates(bp->bb, group->bb)) {
          predsToRemove.insert(sp);
        }
      }
    }

    for (auto *pred : predsToRemove) {
      group->preds.erase(pred);
      pred->succs.erase(group);
    }
  }
}

/// Allocate some joins in front of each lazy fork, so that the number of
/// inputs for each of them is exactly one. The current inputs of the lazy
/// forks become inputs for the joins.
static LogicalResult joinInsertion(OpBuilder &builder,
                                   DenseSet<Group *> &groups,
                                   DenseMap<Block *, Operation *> &forksGraph) {
  // For each group
  for (Group *group : groups) {
    // Get the corresponding fork and operands
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    // If the number of inputs is higher than one
    if (operands.size() > 1) {

      // Join all the inputs, and set the output of this new element as input
      // of the lazy fork
      builder.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          builder.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
  return success();
}

/// For each block extract the terminator condition, i.e. the value driving
/// the final conditional branch (in case it exists)
static void mapConditionsToValues(Region &region, FtdStoredOperations &ftdOps) {
  for (Block &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (isa_and_nonnull<cf::CondBranchOp>(terminator)) {
      auto condBranch = dyn_cast<cf::CondBranchOp>(terminator);
      ftdOps.conditionToValue[getBlockCondition(&block)] =
          condBranch.getCondition();
    }
  }
}

/// Get a value out of the input boolean expression
static Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                                   experimental::boolean::BoolExpression *expr,
                                   Block *block, FtdStoredOperations &ftdOps) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  auto condition = ftdOps.conditionToValue[singleCond->id];
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        channelifyType(condition.getType()), condition);
    ftdOps.opsToSkip.insert(notOp);
    return notOp->getResult(0);
  }
  condition.setType(channelifyType(condition.getType()));
  return condition;
}

/// Get a circuit out a boolean expression, depending on the different kinds
/// of expressions you might have
static Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                     BoolExpression *expr, Block *block,
                                     FtdStoredOperations &ftdOps) {

  // Variable case
  if (expr->type == ExpressionType::Variable)
    return boolVariableToCircuit(rewriter, expr, block, ftdOps);

  // Constant case (either 0 or 1)
  rewriter.setInsertionPointToStart(block);
  auto sourceOp = rewriter.create<handshake::SourceOp>(
      block->getOperations().front().getLoc());
  Value cnstTrigger = sourceOp.getResult();

  auto intType = rewriter.getIntegerType(1);
  auto cstAttr = rewriter.getIntegerAttr(
      intType, (expr->type == ExpressionType::One ? 1 : 0));

  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->getOperations().front().getLoc(), cstAttr, cnstTrigger);
  ftdOps.opsToSkip.insert(constOp);

  return constOp.getResult();
}

/// Convert a `BDD` object as obtained from the bdd expansion to a
/// circuit
static Value bddToCircuit(ConversionPatternRewriter &rewriter, BDD *bdd,
                          Block *block, FtdStoredOperations &ftdOps) {
  if (!bdd->inputs.has_value())
    return boolExpressionToCircuit(rewriter, bdd->boolVariable, block, ftdOps);

  rewriter.setInsertionPointToStart(block);

  // Get the two operands by recursively calling `bddToCircuit` (it possibly
  // creates other muxes in a hierarchical way)
  SmallVector<Value> muxOperands;
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().first, block, ftdOps));
  muxOperands.push_back(
      bddToCircuit(rewriter, bdd->inputs.value().second, block, ftdOps));
  Value muxCond =
      boolExpressionToCircuit(rewriter, bdd->boolVariable, block, ftdOps);

  // Create the multiplxer and add it to the rest of the circuit
  auto muxOp = rewriter.create<handshake::MuxOp>(
      block->getOperations().front().getLoc(), muxCond, muxOperands);
  ftdOps.opsToSkip.insert(muxOp);

  return muxOp.getResult();
}

/// Insert a branch to the correct position, taking into account whether it
/// should work to suppress the over-production of tokens or self-regeneration
static Value addSuppressionInLoop(ConversionPatternRewriter &rewriter,
                                  CFGLoop *loop, Operation *consumer,
                                  Value connection, BranchToLoopType btlt,
                                  FtdStoredOperations &ftdOps, CFGLoopInfo &li,
                                  std::vector<Operation *> &producersToCover) {

  handshake::ConditionalBranchOp branchOp;

  // Case in which there is only one termination block
  if (Block *loopExit = loop->getExitingBlock(); loopExit) {

    // Do not add the branch in case of a while loop with backward edge
    if (btlt == BranchToLoopType::BackwardRelationship &&
        greaterThanBlocks(connection.getParentBlock(), loopExit))
      return connection;

    // Get the termination operation, which is supposed to be conditional
    // branch.
    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    // A conditional branch is now to be added next to the loop terminator, so
    // that the token can be suppressed
    auto *exitCondition = getBlockLoopExitCondition(loopExit, loop, li);
    auto conditionValue =
        boolVariableToCircuit(rewriter, exitCondition, loopExit, ftdOps);

    rewriter.setInsertionPointToStart(loopExit);

    // Since only one output is used, the other one will be connected to sink
    // in the materialization pass, as we expect from a suppress branch
    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().back().getLoc(),
        getBranchResultTypes(connection.getType()), conditionValue, connection);

  } else {

    std::vector<std::string> cofactorList;
    SmallVector<Block *> exitBlocks;
    loop->getExitingBlocks(exitBlocks);
    loopExit = exitBlocks.front();

    BoolExpression *fLoopExit = BoolExpression::boolZero();

    // Get the list of all the cofactors related to possible exit conditions
    for (Block *exitBlock : exitBlocks) {
      BoolExpression *blockCond =
          getBlockLoopExitCondition(exitBlock, loop, li);
      fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
      cofactorList.push_back(getBlockCondition(exitBlock));
    }

    // Sort the cofactors alphabetically
    std::sort(cofactorList.begin(), cofactorList.end());

    // Apply a BDD expansion to the loop exit expression and the list of
    // cofactors
    BDD *bdd = buildBDD(fLoopExit, cofactorList);

    // Convert the boolean expression obtained through bdd to a circuit
    Value branchCond = bddToCircuit(rewriter, bdd, loopExit, ftdOps);

    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    rewriter.setInsertionPointToStart(loopExit);

    branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().front().getLoc(),
        getBranchResultTypes(connection.getType()), branchCond, connection);
  }

  // If we are handling a case with more producers than consumers, the new
  // branch must undergo the `addSupp` function so we add it to our structure
  // to be able to loop over it
  if (btlt == BranchToLoopType::MoreProducerThanConsumers) {
    ftdOps.suppBranches.insert(branchOp);
    producersToCover.push_back(branchOp);
  }

  Value newConnection = btlt == BranchToLoopType::MoreProducerThanConsumers
                            ? branchOp.getTrueResult()
                            : branchOp.getFalseResult();

  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
static LogicalResult
insertDirectSuppression(ConversionPatternRewriter &rewriter,
                        handshake::FuncOp &funcOp, Operation *consumer,
                        Value connection, FtdStoredOperations &ftdOps) {
  Block *entryBlock = &funcOp.getBody().front();
  ControlDependenceAnalysis<dynamatic::handshake::FuncOp> cdgAnalysis(funcOp);
  Block *producerBlock = connection.getParentBlock();

  // Get the control dependencies from the producer
  auto res = cdgAnalysis.getBlockForwardControlDeps(producerBlock);
  DenseSet<Block *> prodControlDeps = res.value_or(DenseSet<Block *>());

  // Get the control dependencies from the consumer
  res = cdgAnalysis.getBlockForwardControlDeps(consumer->getBlock());
  DenseSet<Block *> consControlDeps = res.value_or(DenseSet<Block *>());

  // Get rid of common entries in the two sets
  eliminateCommonBlocks(prodControlDeps, consControlDeps);

  // Compute the activation function of producer and consumer
  BoolExpression *fProd =
      enumeratePaths(entryBlock, producerBlock, prodControlDeps);
  BoolExpression *fCons =
      enumeratePaths(entryBlock, consumer->getBlock(), consControlDeps);

  // The condition related to the select signal of the consumer mux must be
  // added if the following conditions hold: The consumer is a mux; The
  // mux was a GAMMA from GSA analysis; The input of the mux (i.e., coming
  // from the producer) is a data input.
  if (llvm::isa_and_nonnull<handshake::MuxOp>(consumer) &&
      ftdOps.explicitPhiMerges.contains(consumer) &&
      consumer->getOperand(0) != connection &&
      consumer->getOperand(0).getParentBlock() != consumer->getBlock() &&
      consumer->getBlock() != producerBlock) {

    auto selectOperand = consumer->getOperand(0);
    BoolExpression *selectOperandCondition = BoolExpression::parseSop(
        getBlockCondition(selectOperand.getDefiningOp()->getBlock()));

    // The condition must be taken into account for `fCons` only if the
    // producer is not control dependent from the block which produces the
    // condition of the mux
    if (!prodControlDeps.contains(selectOperand.getParentBlock())) {
      if (consumer->getOperand(1) == connection)
        fCons = BoolExpression::boolAnd(fCons,
                                        selectOperandCondition->boolNegate());
      else
        fCons = BoolExpression::boolAnd(fCons, selectOperandCondition);
    }
  }

  /// f_supp = f_prod and not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fSup, cofactorList);
    Value branchCond =
        bddToCircuit(rewriter, bdd, consumer->getBlock(), ftdOps);

    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), getBranchResultTypes(connection.getType()),
        branchCond, connection);
    consumer->replaceUsesOfWith(connection, branchOp.getFalseResult());
  }

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::addSupp(ConversionPatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp,
                                 FtdStoredOperations &ftdOps) const {
  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  // A set of relationships between producer and consumer needs to be covered.
  // To do that, we consider each possible operation in the circuit as
  // producer. However, some operations are added throughout the execution of
  // the function, and those are possibly to be analyzed as well. This vector
  // maintains the list of operations to be analyzed.
  // [TBD] We could consider slightly changing the analysis to avoid needing
  // to rerun the analysis on the operations inserted throughout the
  // execution, but this is future work...
  std::vector<Operation *> producersToCover;

  // Add all the operations in the IR to the above vector
  for (Block &producerBlock : region.getBlocks()) {
    for (Operation &producerOp : producerBlock.getOperations())
      producersToCover.push_back(&producerOp);
  }

  // Loop through the vector until all the elements have been analyzed
  unsigned producerIndex = 0;
  while (producerIndex < producersToCover.size()) {

    Operation *producerOp = producersToCover.at(producerIndex++);
    Block *producerBlock = producerOp->getBlock();

    // Skip the prod-cons if the producer is part of the operations related to
    // the BDD expansion or INIT merges
    if (ftdOps.opsToSkip.contains(producerOp) ||
        ftdOps.initMergesOperations.contains(producerOp))
      continue;

    // Consider all the consumers of each value of the producer
    for (Value result : producerOp->getResults()) {

      std::vector<Operation *> users(result.getUsers().begin(),
                                     result.getUsers().end());
      users.erase(unique(users.begin(), users.end()), users.end());

      for (Operation *consumerOp : users) {
        Block *consumerBlock = consumerOp->getBlock();

        // If the consumer and the producer are in the same block without the
        // consumer being a multiplxer skip because no delivery is needed
        if (consumerBlock == producerBlock &&
            !isa<handshake::MuxOp>(consumerOp))
          continue;

        // Skip the prod-cons if the consumer is part of the operations
        // related to the BDD expansion or INIT merges
        if (ftdOps.opsToSkip.contains(consumerOp) ||
            ftdOps.initMergesOperations.contains(consumerOp))
          continue;

        // TODO: Group the conditions of memory and the conditions of Branches
        // in 1 function?
        // Skip if either the producer of the consumer are
        // related to memory operations, or if the consumer is a conditional
        // branch
        if (llvm::isa_and_nonnull<handshake::MemoryControllerOp>(consumerOp) ||
            llvm::isa_and_nonnull<handshake::MemoryControllerOp>(producerOp) ||
            llvm::isa_and_nonnull<handshake::LSQOp>(producerOp) ||
            llvm::isa_and_nonnull<handshake::LSQOp>(consumerOp) ||
            llvm::isa_and_nonnull<handshake::ControlMergeOp>(producerOp) ||
            llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
            llvm::isa<handshake::ConditionalBranchOp>(consumerOp) ||
            llvm::isa<cf::CondBranchOp>(consumerOp) ||
            llvm::isa<cf::BranchOp>(consumerOp) ||
            (llvm::isa<memref::LoadOp>(consumerOp) &&
             !llvm::isa<handshake::LSQLoadOp>(consumerOp)) ||
            (llvm::isa<memref::StoreOp>(consumerOp) &&
             !llvm::isa<handshake::LSQStoreOp>(consumerOp)) ||
            (llvm::isa<memref::LoadOp>(consumerOp) &&
             !llvm::isa<handshake::MCLoadOp>(consumerOp)) ||
            (llvm::isa<memref::StoreOp>(consumerOp) &&
             !llvm::isa<handshake::MCStoreOp>(consumerOp)) ||
            llvm::isa<mlir::MemRefType>(result.getType()))
          continue;

        // The next step is to identify the relationship between the producer
        // and consumer in hand: Are they in the same loop or at different
        // loop levels? Are they connected through a bwd edge?

        // Set true if the producer is in a loop which does not contains
        // the consumer
        bool producingGtUsing =
            loopInfo.getLoopFor(producerBlock) &&
            !loopInfo.getLoopFor(producerBlock)->contains(consumerBlock);

        auto *consumerLoop = loopInfo.getLoopFor(consumerBlock);

        // Set to true if the consumer uses its own result
        bool selfRegeneration =
            llvm::any_of(consumerOp->getResults(),
                         [&result](const Value &v) { return v == result; });

        // We need to suppress all the tokens produced within a loop and
        // used outside each time the loop is not terminated. This should be
        // done for as many loops there are
        if (producingGtUsing && !isBranchLoopExit(producerOp, loopInfo)) {
          Value con = result;
          for (CFGLoop *loop = loopInfo.getLoopFor(producerBlock); loop;
               loop = loop->getParentLoop()) {

            // For each loop containing the producer but not the consumer, add
            // the branch
            if (!loop->contains(consumerBlock))
              con = addSuppressionInLoop(
                  rewriter, loop, consumerOp, con,
                  BranchToLoopType::MoreProducerThanConsumers, ftdOps, loopInfo,
                  producersToCover);
          }
        }

        // We need to suppress a token if the consumer is the producer itself
        // within a loop
        else if (selfRegeneration && consumerLoop &&
                 !ftdOps.suppBranches.contains(producerOp)) {
          addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                               BranchToLoopType::SelfRegeneration, ftdOps,
                               loopInfo, producersToCover);
        }

        // We need to suppress a token if the consumer comes before the
        // producer (backward edge)
        else if ((greaterThanBlocks(producerBlock, consumerBlock) ||
                  (isa<handshake::MuxOp>(consumerOp) &&
                   producerBlock == consumerBlock &&
                   isaMergeLoop(consumerOp, loopInfo))) &&
                 consumerLoop) {
          addSuppressionInLoop(rewriter, consumerLoop, consumerOp, result,
                               BranchToLoopType::BackwardRelationship, ftdOps,
                               loopInfo, producersToCover);
        }

        // If no loop is involved, then there is a direct relationship between
        // consumer and producer
        else if (failed(insertDirectSuppression(rewriter, funcOp, consumerOp,
                                                result, ftdOps)))
          return failure();
      }
    }

    // Once that we have considered all the consumers of the results of a
    // producer, we consider the operands of the producer. Some of these
    // operands might be the arguments of the functions, and these might need
    // to be suppressed as well.

    // Do not take into account conditional branch
    if (llvm::isa<handshake::ConditionalBranchOp>(producerOp))
      continue;

    // For all the operands of the operation, take into account only the
    // start value if exists
    for (Value operand : producerOp->getOperands()) {

      // The arguments of a function do not have a defining operation
      if (operand.getDefiningOp())
        continue;

      // Skip if we are in block 0 and no multiplexer is involved
      if (operand.getParentBlock() == producerBlock &&
          !isa<handshake::MuxOp>(producerOp))
        continue;

      // Handle the suppression
      if (failed(insertDirectSuppression(rewriter, funcOp, producerOp, operand,
                                         ftdOps)))
        return failure();
    }
  }

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::addRegen(ConversionPatternRewriter &rewriter,
                                  handshake::FuncOp &funcOp,
                                  FtdStoredOperations &ftdOps) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  auto startValue = (Value)funcOp.getArguments().back();

  DenseSet<Operation *> regenMuxes;

  // For each producer/consumer relationship
  for (Block &consumerBlock : region.getBlocks()) {
    for (Operation &consumerOp : consumerBlock.getOperations()) {

      // Skip if the consumer was added by this function, to avoid loops
      if (regenMuxes.contains(&consumerOp))
        continue;

      for (Value operand : consumerOp.getOperands()) {

        // Skip if the producer was added by this function, to avoid loops
        mlir::Operation *producerOp = operand.getDefiningOp();
        if (regenMuxes.contains(producerOp))
          continue;

        // Everything related to memories should not undergo the FTD
        // transformation. Same goes for explicit phi multiplexers and INIT
        // merges.
        if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(producerOp) ||
            llvm::isa_and_nonnull<handshake::MemoryOpInterface>(consumerOp) ||
            ftdOps.explicitPhiMerges.contains(&consumerOp) ||
            ftdOps.initMergesOperations.contains(&consumerOp) ||
            ftdOps.opsToSkip.contains(producerOp) ||
            ftdOps.opsToSkip.contains(&consumerOp) ||
            llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
            llvm::isa_and_nonnull<MemRefType>(operand.getType()))
          continue;

        // Get the parent block of the operand
        Block *producerBlock = operand.getParentBlock();
        Value producerOperand = operand;

        // Function to obtain all the loops in which the consumer is but the
        // producer is not (which specifies how many times a value has to be
        // regenerated)
        auto getLoopsConsNotInProd =
            [&](Block *cons, Block *prod) -> SmallVector<CFGLoop *> {
          SmallVector<CFGLoop *> result;

          // Get all the loops in which the consumer is but the producer is
          // not, starting from the innermost
          for (CFGLoop *loop = loopInfo.getLoopFor(cons); loop;
               loop = loop->getParentLoop()) {
            if (!loop->contains(prod))
              result.push_back(loop);
          }

          // Reverse to the get the loops from outermost to innermost
          std::reverse(result.begin(), result.end());
          return result;
        };

        // Get all the loops for which we need to regenerate the
        // corresponding value
        SmallVector<CFGLoop *> loops =
            getLoopsConsNotInProd(&consumerBlock, producerBlock);
        auto cstType = rewriter.getIntegerType(1);
        auto cstAttr = IntegerAttr::get(cstType, 0);

        // For each of the loop, from the outermost to the innermost
        for (auto *it = loops.begin(); it != loops.end(); ++it) {

          // If we are in the innermost loop (thus the iterator is at its end)
          // and the consumer is a loop merge, stop
          if (std::next(it) == loops.end() &&
              ftdOps.memDepLoopMerges.contains(&consumerOp))
            break;

          // Add the merge to the network, by substituting the operand with
          // the output of the merge, and forwarding the output of the merge
          // to its inputs.
          //
          rewriter.setInsertionPointToStart((*it)->getHeader());

          // The type of the input must be channelified
          producerOperand.setType(channelifyType(producerOperand.getType()));

          // Create an INIT merge to provide the select of the multiplexer
          auto constOp = rewriter.create<handshake::ConstantOp>(
              consumerOp.getLoc(), cstAttr, startValue);
          ftdOps.initMergesOperations.insert(constOp);
          Value conditionValue = ftdOps.conditionToValue[getBlockCondition(
              (*it)->getExitingBlock())];
          SmallVector<Value> mergeOperands;
          mergeOperands.push_back(constOp.getResult());
          mergeOperands.push_back(conditionValue);
          auto initMergeOp = rewriter.create<handshake::MergeOp>(
              consumerOp.getLoc(), mergeOperands);
          ftdOps.initMergesOperations.insert(initMergeOp);

          // Create the multiplexer
          auto selectSignal = initMergeOp->getResult(0);
          selectSignal.setType(channelifyType(selectSignal.getType()));

          SmallVector<Value> muxOperands;
          muxOperands.push_back(producerOperand);
          muxOperands.push_back(producerOperand);

          auto muxOp = rewriter.create<handshake::MuxOp>(
              producerOperand.getLoc(), producerOperand.getType(), selectSignal,
              muxOperands);

          // The new producer operand is the output of the multiplxer
          producerOperand = muxOp.getResult();
          // Set the output of the mux as its input as well
          muxOp->setOperand(2, muxOp->getResult(0));
          regenMuxes.insert(muxOp);
        }
        consumerOp.replaceUsesOfWith(operand, producerOperand);
      }
    }
  }

  // Once that all the multiplexers have been added, it is necessary to modify
  // the type of the result, for it to be a channel type (that could not be
  // done before)
  auto muxes = funcOp.getBody().getOps<handshake::MuxOp>();
  for (auto mux : muxes)
    mux->getResult(0).setType(channelifyType(mux->getResult(0).getType()));

  return success();
}

LogicalResult FtdLowerFuncToHandshake::convertUndefinedValues(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp) const {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each undefined value
  auto undefinedValues = funcOp.getBody().getOps<LLVM::UndefOp>();

  for (auto undefOp : llvm::make_early_inc_range(undefinedValues)) {
    // Create an attribute of the appropriate type for the constant
    auto resType = undefOp.getRes().getType();
    TypedAttr cstAttr;
    if (isa<IndexType>(resType)) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    } else if (isa<IntegerType>(resType)) {
      cstAttr = rewriter.getIntegerAttr(resType, 0);
    } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
      cstAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    }

    // Create a constant with a default value and replace the undefined value
    rewriter.setInsertionPoint(undefOp);
    auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(),
                                                        cstAttr, startValue);
    cstOp->setDialectAttrs(undefOp->getAttrDictionary());
    undefOp.getResult().replaceAllUsesWith(cstOp.getResult());
    namer.replaceOp(cstOp, cstOp);
    rewriter.replaceOp(undefOp, cstOp.getResult());
  }

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::convertConstants(ConversionPatternRewriter &rewriter,
                                          handshake::FuncOp &funcOp) const {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each constant
  auto constants = funcOp.getBody().getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : llvm::make_early_inc_range(constants)) {

    rewriter.setInsertionPoint(cstOp);

    // Convert the constant to the handshake equivalent, using the start value
    // as control signal
    TypedAttr cstAttr = cstOp.getValue();

    if (isa<IndexType>(cstAttr.getType())) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = IntegerAttr::get(
          intType, cast<IntegerAttr>(cstAttr).getValue().trunc(32));
    }

    auto newCstOp = rewriter.create<handshake::ConstantOp>(cstOp.getLoc(),
                                                           cstAttr, startValue);

    newCstOp->setDialectAttrs(cstOp->getDialectAttrs());

    // Replace the constant and the usage of its result
    namer.replaceOp(cstOp, newCstOp);
    cstOp.getResult().replaceAllUsesWith(newCstOp.getResult());
    rewriter.replaceOp(cstOp, newCstOp->getResults());
  }
  return success();
}

LogicalResult FtdLowerFuncToHandshake::ftdVerifyAndCreateMemInterfaces(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
    MemInterfacesInfo &memInfo, FtdStoredOperations &ftdOps) const {

  /// Given an LSQ, extract the list of operations which require that same LSQ
  auto getLSQOperations =
      [&](const llvm::MapVector<unsigned, SmallVector<Operation *>> &lsqPorts)
      -> SmallVector<Operation *> {
    // Result vector holding the result
    SmallVector<Operation *> combinedOperations;

    // Iterate over the MapVector and add all Operation* to the
    // combinedOperations vector
    for (const auto &entry : lsqPorts) {
      const SmallVector<Operation *> &operations = entry.second;
      combinedOperations.insert(combinedOperations.end(), operations.begin(),
                                operations.end());
    }
    return combinedOperations;
  };

  if (memInfo.empty())
    return success();

  // Get the CFG loop information
  mlir::DominanceInfo domInfo;

  // Create a mapping between each block and all the other blocks it
  // properly dominates so that we can quickly determine whether LSQ groups
  // make sense
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : funcOp) {
    // Start with an empty set of dominated blocks for each potential
    // dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : funcOp) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Find the control value indicating the last control flow decision in the
  // function; it will be fed to memory interfaces to indicate that no more
  // group allocations will be coming
  Value ctrlEnd;
  auto returns = funcOp.getOps<func::ReturnOp>();
  assert(!returns.empty() && "no returns in function");
  if (std::distance(returns.begin(), returns.end()) == 1) {
    ctrlEnd = getBlockControl((*returns.begin())->getBlock());
  } else {
    // Merge the control signals of all blocks with a return to create a
    // control representing the final control flow decision
    SmallVector<Value> controls;
    func::ReturnOp lastRetOp;
    for (func::ReturnOp retOp : returns) {
      lastRetOp = retOp;
      controls.push_back(getBlockControl(retOp->getBlock()));
    }
    rewriter.setInsertionPointToStart(lastRetOp->getBlock());
    auto mergeOp =
        rewriter.create<handshake::MergeOp>(lastRetOp.getLoc(), controls);
    ctrlEnd = mergeOp.getResult();

    // The merge goes into an extra "end block" after all others, this will
    // be where the function end terminator will be located as well
    mergeOp->setAttr(BB_ATTR_NAME,
                     rewriter.getUI32IntegerAttr(funcOp.getBlocks().size()));
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [blockIdx, block] : llvm::enumerate(funcOp))
    ctrlVals.insert({blockIdx, getBlockControl(&block)});

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
    SmallPtrSet<Block *, 4> controlBlocks;

    FtdMemoryInterfaceBuilder memBuilder(funcOp, memref, memAccesses.memStart,
                                         ctrlEnd, ctrlVals);

    // Add MC ports to the interface builder
    for (auto &[_, mcBlockOps] : memAccesses.mcPorts) {
      for (Operation *mcOp : mcBlockOps)
        memBuilder.addMCPort(mcOp);
    }

    // Determine LSQ group validity and add ports the interface builder
    // at the same time
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      assert(!groupOps.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<Operation *>> opsPerBlock;
      for (Operation *op : groupOps)
        opsPerBlock[op->getBlock()].push_back(op);

      // Check whether there is a clear "linear dominance" relationship
      // between all blocks, and derive a port ordering for the group from
      // it
      SmallVector<Block *> order;
      if (failed(computeLinearDominance(dominations, opsPerBlock, order)))
        return failure();

      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(order.front()); !newCtrl)
        return groupOps.front()->emitError()
               << "Inconsistent LSQ group for memory interface the "
                  "operation "
                  "references. No two groups can have the same control "
                  "signal.";

      // Add all group ports in the correct order to the builder. Within
      // each block operations are naturally in program order since we
      // always use ordered maps and iterated over the operations in program
      // order to begin with
      for (Block *block : order) {
        for (Operation *lsqOp : opsPerBlock[block])
          memBuilder.addLSQPort(group, lsqOp);
      }
    }

    // Build the memory interfaces.
    // If the memory accesses require an LSQ, then the Fast Load-Store queue
    // allocation method from FPGA'23 is used. In particular, first the
    // groups allocation is performed together with the creation of the fork
    // graph. Afterwards, the FTD methodology is used to interconnect the
    // elements correctly.
    if (memAccesses.lsqPorts.size() > 0) {

      mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

      // Get all the operations associated to an LSQ
      SmallVector<Operation *> allOperations =
          getLSQOperations(memAccesses.lsqPorts);

      // Get all the dependencies among the BBs of the related operations.
      // Two memory operations are dependant if:
      // 1. They are in different BBs;
      // 2. One of them is a write operations;
      // 3. They are not mutually exclusive.
      SmallVector<ProdConsMemDep> allMemDeps;
      identifyMemoryDependencies(allOperations, allMemDeps, loopInfo);

      for (auto &dep : allMemDeps)
        dep.printDependency();

      // Get the initial start signal, which is the last argument of the
      // function
      auto startValue = (Value)funcOp.getArguments().back();

      // Stores the Groups graph required for the allocation network
      // analysis
      DenseSet<Group *> groupsGraph;
      constructGroupsGraph(allOperations, allMemDeps, groupsGraph);
      minimizeGroupsConnections(groupsGraph);

      for (auto &g : groupsGraph)
        g->printDependenices();

      // Build the memory interfaces
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;

      // As we instantiate the interfaces for the LSQ for each memory
      // operation, we need to add some forks in order for the control input
      // to be propagated. In particular, we want to keep track of the control
      // value associated to each basic block in the region
      DenseMap<Block *, Operation *> forksGraph;

      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groupsGraph, forksGraph, startValue)))
        return failure();

      // [TBD] Instead of adding things this way, introduce a custom pass
      // about the analysis of these merges. Unify them with SSA?
      if (failed(addMergeNonLoop(funcOp, rewriter, allMemDeps, groupsGraph,
                                 forksGraph, ftdOps, startValue)))
        return failure();

      if (failed(addMergeLoop(funcOp, rewriter, allMemDeps, groupsGraph,
                              forksGraph, ftdOps, startValue)))
        return failure();

      if (failed(joinInsertion(rewriter, groupsGraph, forksGraph)))
        return failure();
    } else {
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
        return failure();
    }
  }

  return success();
}

void FtdLowerFuncToHandshake::identifyMemoryDependencies(
    const SmallVector<Operation *> &operations,
    SmallVector<ProdConsMemDep> &allMemDeps,
    const mlir::CFGLoopInfo &li) const {

  // Returns true if there exist a path between `op1` and `op2`
  auto isThereAPath = [](Operation *op1, Operation *op2) -> bool {
    return !findAllPaths(op1->getBlock(), op2->getBlock()).empty();
  };

  // Returns true if two operations are both load
  auto areBothLoad = [](Operation *op1, Operation *op2) {
    return (isa<handshake::LSQLoadOp>(op1) && isa<handshake::LSQLoadOp>(op2));
  };

  // Returns true if two operations belong to the same block
  auto isSameBlock = [](Operation *op1, Operation *op2) {
    return (op1->getBlock() == op2->getBlock());
  };

  // Given all the operations which are assigned to an LSQ, loop over them
  // and skip those which are not memory operations
  for (Operation *i : operations) {

    if (!isHandhsakeLSQOperation(i))
      continue;

    // Loop over all the other operations in the LSQ. There is no dependency
    // in the following cases:
    // 1. One of them is not a memory operation;
    // 2. The two operation are in the same group, thus they are in the same
    // BB;
    // 3. They are both load operations;
    // 4. The operations are mutually exclusive (i.e. there is no path which
    // goes from i to j and vice-versa);
    for (Operation *j : operations) {

      if (!isHandhsakeLSQOperation(j) || isSameBlock(i, j) ||
          areBothLoad(i, j) || (!isThereAPath(i, j) && !isThereAPath(j, i)))
        continue;

      // Get the two blocks
      Block *bbI = i->getBlock();
      Block *bbJ = j->getBlock();

      // If the relationship was already present, then skip the pairs of
      // blocks
      auto *it = llvm::find_if(allMemDeps, [bbI, bbJ](ProdConsMemDep p) {
        return p.prodBb == bbJ && p.consBb == bbI;
      });

      if (it != allMemDeps.end())
        continue;

      // Insert a dependency only if index _j is smaller than index _i: in
      // this case i is the producer, j is the consumer. If this doesn't
      // hold, the dependency will be added when the two blocks are analyzed
      // in the opposite direction
      if (lessThanBlocks(bbJ, bbI)) {

        // and add it to the list of dependencies
        ProdConsMemDep oneMemDep(bbJ, bbI, false);
        allMemDeps.push_back(oneMemDep);

        // If the two blocks are in the same loop, then bbI is also a
        // consumer, while bbJ is a producer. This relationship is backward.
        if (isSameLoopBlocks(bbI, bbJ, li)) {
          ProdConsMemDep opp(bbI, bbJ, true);
          allMemDeps.push_back(opp);
        }
      }
    }
  }
}

LogicalResult FtdLowerFuncToHandshake::addMergeNonLoop(
    handshake::FuncOp &funcOp, OpBuilder &builder,
    SmallVector<ProdConsMemDep> &allMemDeps, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, FtdStoredOperations &ftdOps,
    Value startCtrl) const {

  // Get entry block of the function to lower
  Block *entryBlock = &funcOp.getRegion().front();

  // Stores the information related to the control dependencies ob basic
  // blocks within an handshake::funcOp object
  ControlDependenceAnalysis<dynamatic::handshake::FuncOp> cdgAnalysis(funcOp);

  // For each group within the groups graph
  for (Group *producerGroup : groups) {

    // Get the block associated to that same group
    Block *producerBlock = producerGroup->bb;

    // Compute all the forward dependencies of that block
    auto res = cdgAnalysis.getBlockForwardControlDeps(producerBlock);
    DenseSet<Block *> producerControlDeps = res.value_or(DenseSet<Block *>());

    // For each successor (which is now considered as a consumer)
    for (Group *consumerGroup : producerGroup->succs) {

      // Get its basic block
      Block *consumerBlock = consumerGroup->bb;

      // Compute all the forward dependencies of that block
      auto res = cdgAnalysis.getBlockForwardControlDeps(consumerBlock);
      DenseSet<Block *> consumerControlDeps = res.value_or(DenseSet<Block *>());

      // Remove the common forward dependencies among the two blocks
      eliminateCommonBlocks(producerControlDeps, consumerControlDeps);

      // Compute the boolean function `fProd`, that is true when the producer
      // is going to produce the token (cfr. FPGA'22, IV.C: Generating and
      // Suppressing Tokens)
      BoolExpression *fProd =
          enumeratePaths(entryBlock, producerBlock, producerControlDeps);

      // Compute the boolean function `fCons`, that is true when the consumer
      // is going to consume the token (cfr. FPGA'22, IV.C: Generating and
      // Suppressing Tokens)
      BoolExpression *fCons =
          enumeratePaths(entryBlock, consumerBlock, consumerControlDeps);

      // A token needs to be generated when the consumer consumes  but the
      // producer does not producer. Compute the corresponding function and
      // minimize it.
      BoolExpression *fGen =
          BoolExpression::boolAnd(fCons, fProd->boolNegate())->boolMinimize();

      // If the condition for the generation is not zero a `GENERATE` block
      // needs to be inserted, which is a multiplexer/mux
      if (fGen->type != experimental::boolean::ExpressionType::Zero) {

        // Find the memory dependence related to the current producer and
        // consumer
        auto *memDepIt = llvm::find_if(
            allMemDeps,
            [producerBlock, consumerBlock](const ProdConsMemDep &dep) {
              return dep.prodBb == producerBlock && dep.consBb == consumerBlock;
            });
        if (memDepIt == allMemDeps.end())
          return failure();
        ProdConsMemDep &memDep = *memDepIt;

        // The merge needs to be inserted before the consumer and its fork
        builder.setInsertionPointToStart(consumerBlock);
        Location loc = forksGraph[consumerBlock]->getLoc();

        // Adjust insertion position: if there is a block which is always
        // traversed between the producer and the consumer, the merge/generate
        // block can be put right after it.
        //
        // If the relationship between consumer and producer is backward, we
        // are interested in a block which is a successor of the producer and
        // post-dominates the consumer. If the relationship is forward, we are
        // interested in a block which dominates the consumer and
        // post-dominates the consumer.
        Block *bbNewLoc = nullptr;
        if (memDep.isBackward) {
          bbNewLoc = getPostDominantSuccessor(producerBlock, consumerBlock);
        } else {
          bbNewLoc = getPredecessorDominatingAndPostDominating(producerBlock,
                                                               consumerBlock);
        }
        if (bbNewLoc) {
          builder.setInsertionPointToStart(bbNewLoc);
          loc = bbNewLoc->getOperations().front().getLoc();
        }

        // The possible inputs of the merge are the start value and the first
        // output of the producer fork
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(startCtrl);
        mergeOperands.push_back(forksGraph[producerBlock]->getResult(0));
        auto mergeOp = builder.create<handshake::MergeOp>(loc, mergeOperands);

        // At this point, the output of the merge is the new producer, which
        // becomes an input for the consumer.
        forksGraph[consumerBlock]->replaceUsesOfWith(
            forksGraph[producerBlock]->getResult(0), mergeOp->getResult(0));
      }
    }
  }
  return success();
}

LogicalResult FtdLowerFuncToHandshake::addMergeLoop(
    handshake::FuncOp &funcOp, OpBuilder &builder,
    SmallVector<ProdConsMemDep> &allMemDeps, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, FtdStoredOperations &ftdOps,
    Value startCtrl) const {

  // Get the dominance info about the current region, in order to compute the
  // properties of loop
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each group within the groups graph
  for (Group *consGroup : groups) {
    Block *cons = consGroup->bb;

    // For each predecessor (which is now considered as a producer)
    for (Group *prodGroup : consGroup->preds) {
      Block *prod = prodGroup->bb;

      // If the consumer comes before a producer, it might mean the two basic
      // blocks are within a loop
      if (!greaterThanBlocks(prod, cons))
        continue;

      if (auto *loop = getInnermostCommonLoop(prod, cons, loopInfo); loop) {

        // A merge is inserted at the beginning of the loop, getting as
        // operands both `start` and the result of the producer as operand.
        // As many merges must be added as the number of paths from the
        // producer to the consumer (corresponding to the amount of loops
        // which involve both of them).
        Block *loopHeader = loop->getHeader();
        builder.setInsertionPointToStart(loopHeader);

        std::vector<std::vector<Operation *>> allPaths =
            findAllPaths(forksGraph[prod], forksGraph[cons]);

        // For each path
        for (std::vector<Operation *> path : allPaths) {

          SmallVector<Value> operands;
          // Get the result of the operation before the consumer (this will
          // become an input of the merge)
          Value muxOperand = path.at(path.size() - 2)->getResult(0);
          operands.push_back(startCtrl);
          operands.push_back(muxOperand);

          auto cstType = builder.getIntegerType(1);
          auto cstAttr = IntegerAttr::get(cstType, 0);
          auto constOp = builder.create<handshake::ConstantOp>(
              muxOperand.getLoc(), cstAttr, startCtrl);

          ftdOps.initMergesOperations.insert(constOp);

          Value conditionValue =
              ftdOps
                  .conditionToValue[getBlockCondition(loop->getExitingBlock())];

          SmallVector<Value> mergeOperands;
          mergeOperands.push_back(constOp.getResult());
          mergeOperands.push_back(conditionValue);

          // Create the merge
          auto initMergeOp = builder.create<handshake::MergeOp>(
              muxOperand.getLoc(), mergeOperands);

          auto selectSignal = initMergeOp->getResult(0);
          selectSignal.setType(channelifyType(selectSignal.getType()));

          ftdOps.initMergesOperations.insert(initMergeOp);

          // Add the merge and update the FTD data structures
          auto muxOp = builder.create<handshake::MuxOp>(muxOperand.getLoc(),
                                                        muxOperand.getType(),
                                                        selectSignal, operands);

          ftdOps.memDepLoopMerges.insert(muxOp);

          // The merge becomes the producer now, so connect the result of
          // the MERGE as an operand of the Consumer. Also remove the old
          // connection between the producer's LazyFork and the consumer's
          // LazyFork Connect the MERGE to the consumer's LazyFork
          forksGraph[cons]->replaceUsesOfWith(muxOperand, muxOp->getResult(0));
        }
      }
    }
  }
  return success();
}

LogicalResult
FtdLowerFuncToHandshake::addExplicitPhi(func::FuncOp funcOp,
                                        ConversionPatternRewriter &rewriter,
                                        FtdStoredOperations &ftdOps) const {

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
  // Also, a GAMMA function might have an empty data input: GAMMA(c, EMPTY,
  // V). In this case, the function is translated into a branch.
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

  if (funcOp.getBlocks().size() == 1)
    return success();

  auto gsaAnalysis = gsa::GsaAnalysis<func::FuncOp>(funcOp);

  // List of missing GSA functions
  SmallVector<MissingGsa> missingGsaList;
  // List of gammas with only one input
  DenseSet<Operation *> oneInputGammaList;
  // Maps the index of each GSA function to each real operation
  DenseMap<unsigned, Operation *> gsaList;
  ControlDependenceAnalysis<func::FuncOp> cdgAnalysis(funcOp);

  // For each block excluding the first one, which has no gsa
  for (Block &block : llvm::drop_begin(funcOp)) {

    // For each GSA function
    auto *phis = gsaAnalysis.getPhis(&block);
    for (auto &phi : *phis) {

      // Skip if it's a phi
      if (phi->gsaGateFunction == gsa::PhiGate)
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
        if (operand->type == gsa::PhiInputType) {
          operands.emplace_back(operand->phi->result);
          missingGsaList.emplace_back(
              MissingGsa(phi->index, operand->phi->index, operandIndex));
        } else if (operand->type == gsa::EmptyInputType) {
          nullOperand = operandIndex;
          operands.emplace_back(nullptr);
        } else {
          auto val = operand->v;
          val.setType(channelifyType(val.getType()));
          operands.emplace_back(val);
        }
        operandIndex++;
      }

      // The condition value is provided by the `condition` field of the phi
      rewriter.setInsertionPointAfterValue(phi->result);
      Value conditionValue = ftdOps.conditionToValue[phi->condition];

      // If the function is MU, then we create a merge and use its result as
      // condition
      if (phi->gsaGateFunction == gsa::MuGate) {
        Region &region = funcOp.getBody();
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

        ftdOps.initMergesOperations.insert(initMergeOp);

        // Replace the new condition value
        conditionValue = initMergeOp->getResult(0);
        conditionValue.setType(channelifyType(conditionValue.getType()));
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
      if (nullOperand >= 0) {
        oneInputGammaList.insert(mux);
        ftdOps.opsToSkip.insert(mux);
      }

      if (phi->isRoot)
        phi->result.replaceAllUsesWith(mux.getResult());

      gsaList.insert({phi->index, mux});
      ftdOps.explicitPhiMerges.insert(mux);

      // It might be that the condition of a block was coming from a block
      // argument. For this reason, a remapping of the block conditions is
      // necessary
      mapConditionsToValues(funcOp.getRegion(), ftdOps);
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
    ftdOps.explicitPhiMerges.erase(op);
    rewriter.eraseOp(op);
  }

  // Remove all the block arguments for all the non starting blocks
  for (Block &block : llvm::drop_begin(funcOp))
    block.eraseArguments(0, block.getArguments().size());

  // Each terminator must be replaced so that it does not provide any block
  // arguments
  for (Block &block : funcOp) {
    Operation *terminator = block.getTerminator();
    if (terminator) {
      rewriter.setInsertionPointAfter(terminator);
      if (isa<cf::CondBranchOp>(terminator)) {
        auto condBranch = dyn_cast<cf::CondBranchOp>(terminator);
        auto newCondBranch = rewriter.create<cf::CondBranchOp>(
            condBranch->getLoc(), condBranch.getCondition(),
            condBranch.getTrueDest(), condBranch.getFalseDest());
        rewriter.replaceOp(condBranch, newCondBranch);
      } else if (isa<cf::BranchOp>(terminator)) {
        auto branch = dyn_cast<cf::BranchOp>(terminator);
        auto newBranch =
            rewriter.create<cf::BranchOp>(branch->getLoc(), branch.getDest());
        rewriter.replaceOp(branch, newBranch);
      }
    }
  }

  // Since the terminators have been modified, a new remapping is necessary as
  // well
  mapConditionsToValues(funcOp.getRegion(), ftdOps);

  return success();
}
std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
