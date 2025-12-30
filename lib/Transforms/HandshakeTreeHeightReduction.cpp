#include "dynamatic/Transforms/HandshakeTreeHeightReduction.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

// Boilerplate: Include this for the pass option defintitions
namespace dynamatic {
// import auto-generated base class definition
// and put it under the dynamatic namespace.
#define GEN_PASS_DEF_HANDSHAKETREEHEIGHTREDUCTION
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

namespace {

struct AddTreeHeightReduction : public OpRewritePattern<handshake::AddIOp> {
  using OpRewritePattern<handshake::AddIOp>::OpRewritePattern;

  unsigned computeAddTreeHeight(Operation *op) const {
    if (auto add = dyn_cast_or_null<handshake::AddIOp>(op)) {
      return 1 + std::max(computeAddTreeHeight(add.getLhs().getDefiningOp()),
                          computeAddTreeHeight(add.getRhs().getDefiningOp()));
    }
    return 0;
  }

  LogicalResult matchAndRewrite(handshake::AddIOp op,
                                PatternRewriter &rewriter) const override {

    // Collect operands in a flat list
    SmallVector<Value> addTreeOperands;
    SmallVector<handshake::AddIOp> toBeErased;
    std::function<void(Value)> collect = [&](Value v) {
      if (auto add = v.getDefiningOp<handshake::AddIOp>()) {
        collect(add.getLhs());
        collect(add.getRhs());
        if (add != op)
          toBeErased.push_back(add);
      } else
        addTreeOperands.push_back(v);
    };
    collect(op.getResult());
    assert(addTreeOperands.size() >= 2);

    // The tree height is optimal
    if (computeAddTreeHeight(op) == llvm::Log2_64_Ceil(addTreeOperands.size()))
      return failure();

    // Build balanced tree
    std::function<Value(ArrayRef<Value>)> build =
        [&](ArrayRef<Value> vals) -> Value {
      assert(vals.size() > 0);
      if (vals.size() == 1)
        return vals[0];
      auto mid = vals.size() / 2;
      auto lhs = build(vals.take_front(mid));
      auto rhs = build(vals.drop_front(mid));
      return rewriter.create<handshake::AddIOp>(op.getLoc(), lhs, rhs);
    };

    Value newTree = build(addTreeOperands);

    rewriter.replaceOp(op, newTree);

    // Erase backwards: toBeErased adds first the leaf operations (which use
    // the results of their parent ops).
    for (auto opToRemove : llvm::reverse(toBeErased)) {
      rewriter.eraseOp(opToRemove);
    }
    return success();
  }
};

struct HandshakeTreeHeightReductionPass
    : public dynamatic::impl::HandshakeTreeHeightReductionBase<
          HandshakeTreeHeightReductionPass> {

  /// \note: Use the auto-generated construtors from tblgen
  using HandshakeTreeHeightReductionBase::HandshakeTreeHeightReductionBase;
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();

    NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
    if (!nameAnalysis.isAnalysisValid())
      return signalPassFailure();

    mlir::GreedyRewriteConfig config;

    RewritePatternSet patterns{ctx};
    patterns.add<AddTreeHeightReduction>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();

    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    namer.nameAllUnnamedOps();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
