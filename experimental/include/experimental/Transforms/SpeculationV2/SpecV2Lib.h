#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <fstream>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

LogicalResult replaceBranchesWithPassers(FuncOp &funcOp, unsigned bb);

/// Returns if the value is driven by a SourceOp
bool isSourced(Value value);

/// If op is LoadOp, excludes operands coming from MemoryControllerOp.
llvm::SmallVector<Value> getEffectiveOperands(Operation *op);

/// If op is LoadOp, excludes results going to MemoryControllerOp.
llvm::SmallVector<Value> getEffectiveResults(Operation *op);

LogicalResult movePassersDownPM(Operation *pmOp);

/// Returns if the specified PasserOp is eligible for motion past a PM unit.
bool isEligibleForPasserMotionOverPM(PasserOp passerOp, bool reason = false);

/// Move the specified passer past a PM unit.
void performPasserMotionPastPM(PasserOp passerOp,
                               DenseSet<PasserOp> &frontiers);

DenseMap<unsigned, unsigned> unifyBBs(ArrayRef<unsigned> loopBBs,
                                      FuncOp funcOp);

void recalculateMCBlocks(FuncOp funcOp);

bool tryErasePasser(PasserOp passer);

void introduceGSAMux(FuncOp &funcOp, unsigned branchBB);
