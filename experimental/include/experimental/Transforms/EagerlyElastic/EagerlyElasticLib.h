#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace dynamatic;
using namespace mlir;

/// Helper function to add the bbAttr and name to new operations. You can either
/// pass one operation or multiple ones
template <typename... Args>
void setupMetadata(Attribute bbAttr, NameAnalysis &namer, Args... ops) {
  // Create a braced initializer list to unpack and process each operation
  (..., [&]() {
    if (ops) {
      if (bbAttr)
        ops->setAttr("handshake.bb", bbAttr);
      namer.setName(ops);
    }
  }());
}
Value getForkTop(Value value, bool &isInverted);

bool isSourced(Value value);

bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                                   Operation *targetOp);

void performSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                             DenseSet<handshake::ConditionalBranchOp> &frontier,
                             NameAnalysis &namer, int DRewrite = 0);

void applyRewriteB(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   handshake::ConditionalBranchOp falseBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteD(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   handshake::InitOp initOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteC(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteE(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer,
                   int inverted = 0);

void applyRewriteF(handshake::ConditionalBranchOp branchOp,
                   handshake::ConditionalBranchOp topSuppLeft,
                   handshake::ConditionalBranchOp topSuppRight,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteG(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp blueBranchA,
                   handshake::ConditionalBranchOp blueBranchB,
                   handshake::ConditionalBranchOp topSuppressorA,
                   handshake::ConditionalBranchOp topSuppressorB,
                   handshake::ConditionalBranchOp topSuppressorC,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteH(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   handshake::InitOp initOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);
