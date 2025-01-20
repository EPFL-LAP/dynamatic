//===- ConvertMuxToMerge.cpp - Convert Mux To Merge  -----*---- C++ -*-----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ConvertMuxToMerge.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

Operation *getOpByName(handshake::FuncOp &funcOp, const std::string &name) {
  constexpr llvm::StringLiteral nameAttr("handshake.name");
  std::string otherName;

  // For each operaiton
  for (Operation &op : funcOp.getOps()) {

    // Continue if the operation has no name attribute
    if (!op.hasAttr(nameAttr))
      continue;

    // Return the operation if it has the same name as the input
    auto opName = op.getAttrOfType<mlir::StringAttr>(nameAttr);
    std::string opNameStr = opName.str();
    if (name == opNameStr)
      return &op;
  }

  llvm::errs() << "No operation with name " << name << " was found.\n";

  return nullptr;
}

static void convertMuxToMerge(handshake::FuncOp funcOp, NameAnalysis &namer) {

  llvm::dbgs() << "Start conversion\n";
  std::ifstream file("ftdscripting/muxes.txt");
  if (!file.is_open()) {
    llvm::errs() << "Error: Could not open file ftdscripting/muxes.txt\n";
    return;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line))
    lines.push_back(line);

  file.close();

  for (auto &nameOp : lines) {
    Operation *muxOp = getOpByName(funcOp, nameOp);
    if (!muxOp) {
      llvm::errs() << "Operation " << muxOp << " does not exist\n";
      return;
    }

    muxOp->print(llvm::dbgs());
    if (!llvm::isa<handshake::MuxOp>(muxOp)) {
      llvm::errs() << "Operation to convert is not a mux\n";
      return;
    }

    auto mux = llvm::cast<handshake::MuxOp>(muxOp);
    OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointAfter(mux);

    auto newCmergeOp = builder.create<handshake::ControlMergeOp>(
        mux->getLoc(), mux.getDataOperands());
    inheritBB(mux, newCmergeOp);
    namer.setName(newCmergeOp);
    mux.getResult().replaceAllUsesWith(newCmergeOp.getResult());

    auto selectOperand = mux.getSelectOperand();
    SmallVector<OpOperand *> allUses;
    for (auto &usesOperand : selectOperand.getUses())
      allUses.push_back(&usesOperand);

    for (auto &usesOperand : allUses) {
      if (auto mux = llvm::dyn_cast<handshake::MuxOp>(usesOperand->getOwner());
          mux) {
        llvm::dbgs() << "-> ";
        mux->print(llvm::dbgs());
        llvm::dbgs() << "\n";
        usesOperand->set(newCmergeOp.getIndex());
      }
    }
  }

  llvm::dbgs() << "\n-----------------------\n";
  funcOp.print(llvm::dbgs());
  llvm::dbgs() << "\n-----------------------\n";
}

struct ConvertMuxToMergePass
    : public dynamatic::experimental::ftd::impl::ConvertMuxToMergeBase<
          ConvertMuxToMergePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    ConversionPatternRewriter rewriter(ctx);

    for (handshake::FuncOp funcOp : module.getOps<handshake::FuncOp>())
      convertMuxToMerge(funcOp, namer);
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::createConvertMuxToMerge() {
  return std::make_unique<ConvertMuxToMergePass>();
}
