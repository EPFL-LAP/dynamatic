//===- HandshakeRigidificationcpp - Rigidification --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-rigidification pass.
//
//===----------------------------------------------------------------------===//

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
// #include "experimental/Support/FormalProperty.h"
#include "experimental/Transforms/Rigidification/HandshakeRigidification.h"
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
using namespace dynamatic::experimental::rigidification;

namespace {

struct HandshakeRigidificationPass
    : public dynamatic::experimental::rigidification::impl::
          HandshakeRigidificationBase<HandshakeRigidificationPass> {

  HandshakeRigidificationPass(const std::string &jsonPath = "") {
    this->jsonPath = jsonPath;
  }

  void runDynamaticPass() override;

private:
  json::Array propertyTable;
};
} // namespace

void HandshakeRigidificationPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();
  llvm::json::Array propTable;

  // for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
  //   for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
  //     auto propIDs = getFormalProperties(&op);
  //     for (auto propID : propIDs) {
  //       llvm::json::Value v = findProperty(propID);
  //       auto property = FormalProperty::fromJSON();
  //       if (property.tag == FormalProperty::TAG::OPT) {
  //         if (isa<AOBProperty>(property)) {
  //           insertChannelRigidifier(op, property);
  //         } else if (isa<VEQProperty>(property)) {
  //           insertValidMerger(op, property);
  //         }
  //       }
  //     }
  //   }
  // }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::rigidification::createRigidification(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeRigidificationPass>(jsonPath);
}

// namespace {
// /// Rewrite pattern that will match on all muxes in the IR and replace each
// of
// /// them with a merge taking the same inputs (except the `select` input which
// /// merges do not have due to their undeterministic nature).
// struct HandshakeRigidification : public OpRewritePattern<Operation> {
//   using OpRewritePattern<Operation>::OpRewritePattern;

//   LogicalResult matchAndRewrite(Operation muxOp,
//                                 PatternRewriter &rewriter) const override {

//     rewriter.insert(op);
//     return success();
//   }
// };

// /// Simple driver for the pass that replaces all muxes with merges.
// struct HandshakeRigidificationPass
//     : public impl::HandshakeRigidificationBase<HandshakeRigidificationPass> {

//   void runDynamaticPass() override {
//     // Get the MLIR context for the current operation being transformed
//     MLIRContext *ctx = &getContext();
//     // Get the operation being transformed (the top level module)
//     ModuleOp mod = getOperation();

//     for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
//       auto name = funcOp->getName();
//       // llvm::errs() << funcOp << "\n\n\n\n";
//       if (failed(performRigidification(funcOp, ctx)))
//         return signalPassFailure();
//     }
//     // mod->walk([&](Operation *op) {
//     //   auto name = op->getName();
//     //   llvm::errs() << name.getIdentifier().str() << "\n";
//     //   for (auto ch : op->getResults()) {
//     //     bool is_mem = false;
//     //     for (auto &use : llvm::make_early_inc_range(ch.getUses()))
//     //       if (isa<handshake::LSQOp, handshake::MemoryControllerOp>(
//     //               use.getOwner()))
//     //         is_mem = true;
//     //     Type resType = ch.getType();
//     //     if (llvm::dyn_cast<handshake::ChannelType>(resType) && !is_mem) {
//     //       rigidifyChannel(&ch, ctx);
//     //     }
//     //   }
//     // });

//     // MLIRContext *ctx = &getContext();

//     // // Define the set of rewrite patterns we want to apply to the IR
//     // RewritePatternSet patterns(ctx);

//     // patterns.add<HandshakeRigidification<handshake::MuxOp>>(ctx);

//     // // Run a greedy pattern rewriter on the entire IR under the top-level
//     // // module
//     // // operation
//     // mlir::GreedyRewriteConfig config;
//     // if (failed(
//     //         applyPatternsAndFoldGreedily(mod, std::move(patterns),
//     //         config))) {
//     //   // If the greedy pattern rewriter fails, the pass must also fail
//     //   return signalPassFailure();
//     // }
//   };
//   static LogicalResult performRigidification(handshake::FuncOp funcOp,
//                                              MLIRContext *ctx) {
//     for (mlir::Block &block : funcOp) {
//       for (auto it = block.begin(), end = block.end(); it != end;
//            /* no increment here */) {
//         mlir::Operation &op = *it;
//         ++it; // Increment before modifying the block to avoid iterator
//         // invalidation
//         for (auto res : op.getResults()) {
//           Type opType = res.getType();
//           if (llvm::dyn_cast<handshake::ChannelType>(opType) &&
//               op.getAttrOfType<mlir::ArrayAttr>("resultNames"))
//             rigidifyChannel(res, ctx);
//         }
//       }
//     }
//     return success();
//   }
// };

// } // namespace

// /// Implementation of our pass constructor, which just returns an instance of
// /// the `HandshakeMuxToMergePass` struct.
