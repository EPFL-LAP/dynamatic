#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace dynamatic;
using namespace mlir;

#define HANDSHAKEBB "handshake.bb"
#define SUBLOOP_INFO_ATTR "handshake.subloop_info"
#define ENTRY_OPS "entry_ops"
#define STORES "stores"
#define HEADER_BB "header_bb"

enum class BypassResult : bool { Ineligible = false, Eligible = true };

/// Helper function to add the bbAttr and name to new operations.
void setHandshakeAttrs(Attribute bbAttr, NameAnalysis &namer,
                       ArrayRef<Operation *> ops);

bool checkConditionsMatch(Value valA, Value valB, bool expectSamePolarity);

bool isSourced(Value value);

BypassResult isEligibleForBypass(handshake::ConditionalBranchOp branchOp,
                                 Operation *targetOp);

void moveSuppressorPastOp(handshake::ConditionalBranchOp branchOp,
                          Operation *targetOp,
                          DenseSet<handshake::ConditionalBranchOp> &frontier,
                          NameAnalysis &namer, int DRewrite = 0);

void applyRewriteB(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   handshake::ConditionalBranchOp falseBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteC(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteD(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp branchOp,
                   handshake::InitOp initOp,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer);

void applyRewriteE(handshake::MuxOp dataMux,
                   handshake::ConditionalBranchOp trueBranch,
                   DenseSet<handshake::ConditionalBranchOp> &frontier,
                   NameAnalysis &namer, int inverted = 0);

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
