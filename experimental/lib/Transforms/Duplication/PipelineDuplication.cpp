#include "experimental/Transforms/Duplication/DuplicationLogic.h"

// Include some other useful headers.
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PIPELINEDUPLICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {
  /*
  struct AddConstantBranch : public OpRewritePattern<arith::AddfOp> {
    using OpRewritePattern<arith::AddfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddfOp op, PatternRewriter &rewriter) const override {
      // Find the end of my pipeline with addf, to get the same constants
      auto nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
      if (!nameAttr || nameAttr.getValue() != "addf0")
        return failure();

      if (op->hasAttr("processed")) // is this necessary?
        return failure();
    
      Location loc = op.getLoc();

      // get the other constants
      auto mulfOp = op.getLhs().getDefiningOp<arith::MulfOp>();
      if (!mulfOp) return failure();

      // constants -2.0 and 15.0 that are used for the other operations
      Value cnstNegTwo = mulfOp.getRhs();
      Value cnstFifteen = op.getRhs();

      // actually create the new values
      Value cnstFive = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(rewriter.getF64Type(), 5.0));

      Value newMulf = rewriter.create<arith::MulfOp>(loc, cstFive, cstNegTwo).getResult();
      Value newAddf = rewriter.create<arith::AddfOp>(loc, newMulf, cstFifteen).getResult();
      Value newTrunc = rewriter.create<arith::TruncfOp>(
        loc, rewriter.getF32Type(), newAddf).getResult();

      
      for (auto *user : op.getResult().getUsers()) {
      if (auto truncOp = dyn_cast<arith::TruncfOp>(user)) {
        for (auto *truncUser : truncOp.getResult().getUsers()) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(truncUser)) {
            
            Value sharedIndex = storeOp.getIndices()[0]; 
            Value targetMemref = storeOp.getMemref();

            // Create the 4th branch store
            rewriter.create<memref::StoreOp>(loc, newTrunc, targetMemref, sharedIndex);
            break;
          }
        }
      }
      }
      op->setAttr("processed", rewriter.getUnitAttr());
      return success();
    }
  };
  */


  // wrapper
  struct PipelineDuplicationPass 
      : public dynamatic::experimental::impl::PipelineDuplicationBase<
      PipelineDuplicationPass> {

    using PipelineDuplicationBase::PipelineDuplicationBase;

    void runDynamaticPass() override;
  };
  
} // namespace

void HandshakeRigidificationPass::runDynamaticPass() {
  FormalPropertyTable table;
  if (failed(table.addPropertiesFromJSON(jsonPath)))
    llvm::errs() << "[WARNING] Formal property retrieval failed\n";

  for (const auto &property : table.getProperties()) {
    if (property->getTag() == FormalProperty::TAG::OPT &&
        property->getCheck() != std::nullopt && *property->getCheck()) {

      if (auto *p = dyn_cast<AbsenceOfBackpressure>(property.get())) {
        if (failed(insertReadyRemover(*p)))
          return signalPassFailure();

      } else if (auto *p = dyn_cast<ValidEquivalence>(property.get())) {
        if (failed(insertValidMerger(*p)))
          return signalPassFailure();
      }
    }
  }
}
