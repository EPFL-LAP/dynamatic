#include "HandshakeCopyBuffers.h"
#include "JSONImporter.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_HANDSHAKECOPYBUFFERS
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct HandshakeCopyBuffersPass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeCopyBuffersBase<HandshakeCopyBuffersPass> {
  using HandshakeCopyBuffersBase<
      HandshakeCopyBuffersPass>::HandshakeCopyBuffersBase;
  void runDynamaticPass() override;
};

static std::tuple<Value, Value> copyBufferRecursively(Value pre, Value post) {
  Operation *preUser = *pre.getUsers().begin();
  if (auto preBuffer = dyn_cast<BufferOp>(preUser)) {
    OpBuilder builder(post.getContext());
    builder.setInsertionPointAfterValue(post);
    BufferOp postBuffer = builder.create<BufferOp>(
        builder.getUnknownLoc(), post, preBuffer.getNumSlots(),
        preBuffer.getBufferType());
    inheritBBFromValue(post, postBuffer);
    post.replaceAllUsesExcept(postBuffer.getResult(), postBuffer);
    return copyBufferRecursively(preBuffer.getResult(), postBuffer.getResult());
  }
  return std::make_tuple(pre, post);
}

static LogicalResult performDFS(Operation *preOp, Operation *postOp,
                                DenseSet<Operation *> &visited, unsigned preBB,
                                unsigned postBB) {
  if (visited.contains(preOp))
    return success();
  visited.insert(preOp);

  if (isa<StoreOp>(preOp)) {
    if (isa<StoreOp>(postOp)) {
      return success();
    }
    llvm::errs() << "Mismatch in operations during DFS\n";
    preOp->dump();
    postOp->dump();
    return failure();
  }

  if (preOp->getNumResults() != postOp->getNumResults()) {
    llvm::errs() << "Mismatch in number of results between operations\n";
    preOp->dump();
    postOp->dump();
    return failure();
  }

  for (unsigned int i = 0; i < preOp->getNumResults(); ++i) {
    auto [preRes, postRes] =
        copyBufferRecursively(preOp->getResult(i), postOp->getResult(i));
    Operation *preUser = *preRes.getUsers().begin();
    Operation *postUser = *postRes.getUsers().begin();
    if (getLogicBB(preUser) != preBB) {
      if (getLogicBB(postUser) != postBB) {
        continue;
      }
      llvm::errs() << "Mismatch in BBs between operations\n";
      preUser->dump();
      postUser->dump();
      return failure();
    }
    if (failed(performDFS(preUser, postUser, visited, preBB, postBB)))
      return failure();
  }

  return success();
}

void HandshakeCopyBuffersPass::runDynamaticPass() {
  MLIRContext &context = getContext();
  OwningOpRef<ModuleOp> mod =
      parseSourceFile<ModuleOp>(preUnrollingPath, &context);
  if (!mod) {
    llvm::errs() << "Error parsing pre-unrolling file: " << preUnrollingPath
                 << "\n";
    return signalPassFailure();
  }

  FuncOp preUnrollingFunc = *mod->getOps<FuncOp>().begin();
  ModuleOp postUnrollingMod = getOperation();
  FuncOp postUnrollingFunc = *postUnrollingMod.getOps<FuncOp>().begin();
  MuxOp firstMux;
  bool firstMuxFound = false;
  for (auto mux : preUnrollingFunc.getOps<MuxOp>()) {
    if (getLogicBB(mux) == preUnrollingBB) {
      firstMux = mux;
      firstMuxFound = true;
      break;
    }
  }
  bool legacyMode = false;
  if (firstMuxFound) {
    MuxOp postUnrollingEntryMux;
    bool postUnrollingEntryMuxFound = false;
    for (auto mux : postUnrollingFunc.getOps<MuxOp>()) {
      if (getLogicBB(mux) == postUnrollingBB) {
        postUnrollingEntryMux = mux;
        postUnrollingEntryMuxFound = true;
        break;
      }
    }
    if (postUnrollingEntryMuxFound) {
      legacyMode = true;
      DenseSet<Operation *> visited;
      if (failed(performDFS(firstMux, postUnrollingEntryMux, visited,
                            preUnrollingBB, postUnrollingBB))) {
        llvm::errs() << "Error during DFS traversal for copying buffers\n";
        return signalPassFailure();
      }
    }
  }

  if (!legacyMode) {
    llvm::errs() << "Non-legacy mode\n";
    llvm::SmallVector<Operation *> preEntryPoints;
    llvm::SmallVector<Operation *> postEntryPoints;
    preUnrollingFunc.walk([&](Operation *op) {
      if (getLogicBB(op) != preUnrollingBB)
        return;
      for (OpOperand &operand : op->getOpOperands()) {
        Operation *defOp = operand.get().getDefiningOp();
        if (defOp && getLogicBB(defOp) == preUnrollingBB - 1) {
          preEntryPoints.push_back(op);
        }
      }
    });
    postUnrollingFunc.walk([&](Operation *op) {
      if (getLogicBB(op) != postUnrollingBB)
        return;
      for (OpOperand &operand : op->getOpOperands()) {
        Operation *defOp = operand.get().getDefiningOp();
        if (defOp && getLogicBB(defOp) == postUnrollingBB - 1) {
          postEntryPoints.push_back(op);
        }
      }
    });
    if (preEntryPoints.size() != postEntryPoints.size()) {
      llvm::errs() << "Mismatch in number of entry point operations\n";
      return signalPassFailure();
    }
    DenseSet<Operation *> visited;
    for (size_t i = 0; i < preEntryPoints.size(); ++i) {
      Operation *preOp = preEntryPoints[i];
      Operation *postOp = postEntryPoints[i];
      if (failed(performDFS(preOp, postOp, visited, preUnrollingBB,
                            postUnrollingBB))) {
        llvm::errs() << "Error during DFS traversal for copying buffers\n";
        return signalPassFailure();
      }
    }
  }
}
