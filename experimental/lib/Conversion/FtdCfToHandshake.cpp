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
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
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
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<ControlDependenceAnalysis>(),
        getAnalysis<gsa::GSAAnalysis>(), getAnalysis<NameAnalysis>(), converter,
        ctx);

    patterns.add<ConvertCalls,
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

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

void ftd::FtdLowerFuncToHandshake::exportGsaGatesInfo(
    handshake::FuncOp funcOp) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  std::ofstream ofs;

  // We print to this file information about the GSA gates and the innermost
  // loops, if any, containing each.
  ofs.open("ftdscripting/gsaGatesInfo.txt", std::ofstream::out);
  std::string loopDescription;
  llvm::raw_string_ostream loopDescriptionStream(loopDescription);

  auto muxes = funcOp.getBody().getOps<handshake::MuxOp>();
  for (auto phi : muxes) {
    ofs << namer.getName(phi).str();
    if (llvm::isa<handshake::MergeOp>(phi->getOperand(0).getDefiningOp()))
      ofs << " (MU)\n";
    else
      ofs << " (GAMMA)\n";
    if (!loopInfo.getLoopFor(phi->getBlock()))
      ofs << "Not inside any loop\n";
    else
      loopInfo.getLoopFor(phi->getBlock())
          ->print(loopDescriptionStream, false, false, 0);
    ofs << loopDescription << "\n";
    loopDescription = "";
  }

  ofs.close();
}

// --- Helper functions ---

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, create a group graph.
static void
constructGroupsGraph(SmallVector<handshake::MemPortOpInterface> &operations,
                     SmallVector<ProdConsMemDep> &allMemDeps,
                     DenseSet<Group *> &groups) {

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : operations) {
    Block *b = op->getBlock();
    auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
    if (it == groups.end()) {
      Group *g = new Group(b);
      groups.insert(g);
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

  // Add a self dependency each time you have a group with no dependency
  for (Group *g : groups) {
    if (!g->preds.size()) {
      g->preds.insert(g);
      g->succs.insert(g);
    }
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
static void minimizeGroupsConnections(DenseSet<Group *> &groupsGraph,
                                      const BlockIndexing &bi) {

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
        if (sp->bb == bp->bb || bi.greaterIndex(sp->bb, bp->bb))
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

LogicalResult ftd::FtdLowerFuncToHandshake::addSupp(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    ControlDependenceAnalysis::BlockControlDepsMap &cdaDeps,
    const BlockIndexing &bi) const {

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
  for (Block &producerBlock : funcOp.getBlocks()) {
    for (Operation &producerOp : producerBlock.getOperations())
      producersToCover.push_back(&producerOp);
  }

  // Loop through the vector until all the elements have been analyzed
  unsigned producerIndex = 0;
  while (producerIndex < producersToCover.size()) {
    Operation *producerOp = producersToCover.at(producerIndex++);
    if (failed(addSuppToProducer(rewriter, funcOp, producerOp, bi,
                                 producersToCover, cdaDeps)))
      return failure();
  }

  return success();
}

LogicalResult
ftd::FtdLowerFuncToHandshake::addRegen(ConversionPatternRewriter &rewriter,
                                       handshake::FuncOp &funcOp) const {

  // For each producer/consumer relationship
  for (Operation &consumerOp : funcOp.getOps()) {
    if (failed(addRegenToConsumer(rewriter, funcOp, &consumerOp)))
      return failure();
  }

  // Considering each mux that was added, the inputs and output values must be
  // channellified
  for (Operation *mux : funcOp.getOps<handshake::MuxOp>()) {
    mux->getOperand(1).setType(channelifyType(mux->getOperand(1).getType()));
    mux->getOperand(2).setType(channelifyType(mux->getOperand(2).getType()));
    mux->getResult(0).setType(channelifyType(mux->getResult(0).getType()));
  }

  return success();
}

LogicalResult ftd::FtdLowerFuncToHandshake::convertUndefinedValues(
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

LogicalResult ftd::FtdLowerFuncToHandshake::convertConstants(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp) const {

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

LogicalResult ftd::FtdLowerFuncToHandshake::ftdVerifyAndCreateMemInterfaces(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
    MemInterfacesInfo &memInfo, const BlockIndexing &bi) const {

  /// Given an LSQ, extract the list of operations which require that same LSQ
  auto getLSQOperations =
      [&](const llvm::MapVector<
          unsigned, SmallVector<handshake::MemPortOpInterface>> &lsqPorts)
      -> SmallVector<handshake::MemPortOpInterface> {
    // Result vector holding the result
    SmallVector<handshake::MemPortOpInterface> combinedOperations;

    // Iterate over the MapVector and add all Operation* to the
    // combinedOperations vector
    for (const auto &entry : lsqPorts) {
      combinedOperations.insert(combinedOperations.end(), entry.second.begin(),
                                entry.second.end());
    }
    return combinedOperations;
  };

  if (memInfo.empty())
    return success();

  // Get the initial start signal, which is the last argument of the
  // function
  auto startValue = (Value)funcOp.getArguments().back();

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
      for (handshake::MemPortOpInterface portOp : mcBlockOps)
        memBuilder.addMCPort(portOp);
    }

    // Determine LSQ group validity and add ports the interface builder
    // at the same time
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      assert(!groupOps.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<handshake::MemPortOpInterface>>
          opsPerBlock;
      for (handshake::MemPortOpInterface portOp : groupOps)
        opsPerBlock[portOp->getBlock()].push_back(portOp);

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
      for (Block *block : order)
        for (handshake::MemPortOpInterface portOp : opsPerBlock[block])
          memBuilder.addLSQPort(group, portOp);
    }

    // Build the memory interfaces.
    // If the memory accesses require an LSQ, then the Fast Load-Store queue
    // allocation method from FPGA'23 is used. In particular, first the
    // groups allocation is performed together with the creation of the fork
    // graph. Afterwards, the FTD methodology that comes later in the flow is
    // used to interconnect the elements correctly.
    if (memAccesses.lsqPorts.size() > 0) {

      mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

      // Get all the operations associated to an LSQ
      SmallVector<handshake::MemPortOpInterface> allOperations =
          getLSQOperations(memAccesses.lsqPorts);

      // Get all the dependencies among the BBs of the related operations.
      // Two memory operations are dependant if:
      // 1. They are in different BBs;
      // 2. One of them is a Store operation;
      // 3. They are not mutually exclusive (according to the control flow).
      SmallVector<ProdConsMemDep> allMemDeps;
      identifyMemoryDependencies(allOperations, allMemDeps, loopInfo, bi);

      for (auto &dep : allMemDeps)
        dep.printDependency();

      // Stores the Groups graph required for the allocation network
      // analysis
      DenseSet<Group *> groupsGraph;
      constructGroupsGraph(allOperations, allMemDeps, groupsGraph);
      minimizeGroupsConnections(groupsGraph, bi);

      for (auto &g : groupsGraph)
        g->printDependenices();

      // Build the memory interfaces
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;

      // As we instantiate the interfaces for the LSQ for each memory
      // operation, we need to add one fork to model each group and propagate
      // control between dependent groups.
      DenseMap<Block *, Operation *> forksGraph;

      // Instantiate the interfaces and a lazy fork for each group
      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groupsGraph, forksGraph, startValue)))
        return failure();

      for (Group *consumerGroup : groupsGraph) {
        SmallVector<Value> differentInputs;
        Operation *consumerLF = forksGraph[consumerGroup->bb];
        for (Group *producerGroup : consumerGroup->preds) {
          Operation *producerLF = forksGraph[producerGroup->bb];

          // Every Fork takes Start as a predecessor along with any other
          // predecessor it may have in the groups graph The below function
          // running SSA will then either add a Merge or eliminate the Start
          // predecessor depending on control flow
          SmallVector<Value> forkValuesToConnect = {startValue,
                                                    producerLF->getResult(0)};
          // This function adds SSA Phis (Merges) by studying the deps between
          // forks and the blocks containing them
          auto phiNetworkOrFailure = createPhiNetwork(
              funcOp.getRegion(), rewriter, forkValuesToConnect);
          if (failed(phiNetworkOrFailure))
            return failure();

          auto &phiNetwork = *phiNetworkOrFailure;
          differentInputs.push_back(phiNetwork[consumerGroup->bb]);
        }

        if (differentInputs.size() == 0)
          differentInputs.push_back(startValue);

        consumerLF->setOperands(differentInputs);
      }

      if (failed(joinInsertion(rewriter, groupsGraph, forksGraph)))
        return failure();

    } else {
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
        return failure();
    }
  }

  // Create a backedge for the start value, to be sued during the merges to
  // muxes conversion
  BackedgeBuilder edgeBuilderStart(rewriter, funcOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());

  // For each merge that was signed with the `NEW_PHI` attribute, substitute it
  // with its GSA equivalent
  for (handshake::MergeOp merge : funcOp.getOps<handshake::MergeOp>()) {
    if (!merge->hasAttr(NEW_PHI))
      continue;
    gsa::GSAAnalysis gsa(merge, funcOp.getRegion());
    if (failed(gsa::GSAAnalysis::addGsaGates(funcOp.getRegion(), rewriter, gsa,
                                             startValueBackedge, false)))
      return failure();

    // Get rid of the merge
    rewriter.eraseOp(merge);
  }

  // Replace the backedge
  startValueBackedge.setValue(startValue);

  return success();
}

void ftd::FtdLowerFuncToHandshake::identifyMemoryDependencies(
    const SmallVector<handshake::MemPortOpInterface> &operations,
    SmallVector<ProdConsMemDep> &allMemDeps, const mlir::CFGLoopInfo &li,
    const BlockIndexing &bi) const {

  // Returns true if there exist a path between `op1` and `op2`
  auto isThereAPath = [&](Operation *op1, Operation *op2) -> bool {
    return !findAllPaths(op1->getBlock(), op2->getBlock(), bi).empty();
  };

  // Returns true if two operations are both load
  auto areBothLoad = [](Operation *op1, Operation *op2) {
    return (isa<handshake::LoadOp>(op1) && isa<handshake::LoadOp>(op2));
  };

  // Returns true if two operations belong to the same block
  auto isSameBlock = [](Operation *op1, Operation *op2) {
    return (op1->getBlock() == op2->getBlock());
  };

  // Given all the operations which are assigned to an LSQ, loop over them
  // and skip those which are not memory operations
  for (handshake::MemPortOpInterface i : operations) {

    // Loop over all the other operations in the LSQ. There is no dependency
    // in the following cases:
    // 1. One of them is not a memory operation;
    // 2. The two operations are in the same group, thus they are in the same
    // BB;
    // 3. They are both load operations;
    // 4. The operations are mutually exclusive (i.e. there is no path which
    // goes from i to j and vice-versa);
    for (handshake::MemPortOpInterface j : operations) {

      if (isSameBlock(i, j) || areBothLoad(i, j) ||
          (!isThereAPath(i, j) && !isThereAPath(j, i)))
        continue;

      // Get the two blocks
      Block *bbI = i->getBlock(), *bbJ = j->getBlock();

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
      if (bi.lessIndex(bbJ, bbI)) {

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

LogicalResult ftd::FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Get the map of control dependencies for the blocks in the function to
  // lower
  auto cdaDeps = cdAnalysis.getAllBlockDeps();

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // Add the muxes as obtained by the GSA analysis pass
  BackedgeBuilder edgeBuilderStart(rewriter, lowerFuncOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());
  if (failed(gsa::GSAAnalysis::addGsaGates(lowerFuncOp.getRegion(), rewriter,
                                           gsaAnalysis, startValueBackedge)))
    return failure();

  // Save pointers to old block
  SmallVector<Block *> blocksCfFunction;
  for (auto &bb : lowerFuncOp.getBlocks())
    blocksCfFunction.push_back(&bb);

  // First lower the parent function itself, without modifying its body
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // Save pointers from new blocks
  SmallVector<Block *> blocksHandshakeFunc;
  for (auto &bb : funcOp.getBlocks())
    blocksHandshakeFunc.push_back(&bb);

  // Remap control dependency analysis (in correspondence to the new block
  // pointers)
  for (unsigned i = 0; i < blocksCfFunction.size(); i++) {
    auto deps = cdaDeps[blocksCfFunction[i]];

    for (unsigned i = 0; i < blocksCfFunction.size(); i++) {
      if (deps.allControlDeps.contains(blocksCfFunction[i]))
        deps.allControlDeps.insert(blocksHandshakeFunc[i]);
      if (deps.forwardControlDeps.contains(blocksCfFunction[i]))
        deps.forwardControlDeps.insert(blocksHandshakeFunc[i]);
    }

    cdaDeps.erase(blocksCfFunction[i]);
    cdaDeps.insert({blocksHandshakeFunc[i], deps});
  }

  // Get the block indexing
  BlockIndexing bi(funcOp.getRegion());

  // When GSA-MU functions are translated into multiplexers, an `init merge`
  // is created to feed them. This merge requires the start value of the
  // function as one of its data inputs. However, the start value was not
  // present yet when `addGsaGates` is called, thus we need to reconnect
  // it.
  startValueBackedge.setValue((Value)funcOp.getArguments().back());

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
  if (failed(ftdVerifyAndCreateMemInterfaces(funcOp, rewriter, memInfo, bi)))
    return failure();

  // Convert the constants and undefined values from the `arith` dialect to
  // the `handshake` dialect, while also using the start value as their
  // control value
  if (failed(convertConstants(rewriter, funcOp)) ||
      failed(convertUndefinedValues(rewriter, funcOp)))
    return failure();

  if (funcOp.getBlocks().size() != 1) {
    // Add muxes for regeneration of values in loop
    if (failed(addRegen(rewriter, funcOp)))
      return failure();

    // This function prints to a file information about all GSA gates and the
    // loops containing them (if any)
    exportGsaGatesInfo(funcOp);

    // Add suppression blocks between each pair of producer and consumer
    if (failed(addSupp(rewriter, funcOp, cdaDeps, bi)))
      return failure();
  }

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  cfg::annotateCFG(funcOp, rewriter);

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass> ftd::createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}
