//===- FtdCfToHandshakeTranslator.cpp - FTD translation to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the translation behind the --ftd-lower-cf-to-handshake pass, i.e.,
// the fast token
// delivery methodology https://ieeexplore.ieee.org/abstract/document/10035134,
// together with the straight LSQ allocation
// https://dl.acm.org/doi/abs/10.1145/3543622.3573050. On top of the base
// translation (see CfToHandshakeTranslator.cpp):
//
// 1. Before the blocks are translated, the GSA gates of all blocks are created
//    in the blocks' staging blocks, with placeholders for the block conditions
//    and for the func-level values the gates take as operands.
// 2. The blocks are translated with control-only branches and merges.
// 3. Once the body is flattened, the placeholders are resolved and the
//    regeneration and suppression circuits are added, with the func-level
//    function as the CFG. During that step, each block's condition goes
//    through a proxy (a NOT tagged with FTD_COND_VAR and the block's "bb"),
//    which identifies the block the condition belongs to, even if the
//    condition is computed in another block or shared with another block.
//
//===----------------------------------------------------------------------===//

#include "FtdCfToHandshakeTranslator.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "experimental/Support/FtdSuppression.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;

LogicalResult FtdCfToHandshakeTranslator::initTranslation() {
  Region &cfRegion = cfFunc.getBody();
  blockIndexing.emplace(cfRegion);
  cfgView.shadowFunc = cfFunc;

  // The GSA gates may need the condition of a block that is translated later
  // (e.g., a loop's exit condition), so each condition starts as a placeholder
  // in the block's staging block
  Type condType = handshake::ChannelType::get(builder.getI1Type());
  for (Block &block : cfRegion) {
    if (!isa<cf::CondBranchOp>(block.getTerminator()))
      continue;
    builder.setInsertionPointToStart(stagingBlock[&block]);
    Backedge cond = edgeBuilder.get(condType);
    condPlaceholders.emplace_back(&block, cond);
    cfgView.conditionMap[blockID[&block]] = cond;
  }
  return createGsaGates();
}

LogicalResult FtdCfToHandshakeTranslator::createGsaGates() {
  using namespace experimental::gsa;
  Region &cfRegion = cfFunc.getBody();
  if (cfRegion.hasOneBlock())
    return success();
  const ftd::BlockIndexing &bi = *blockIndexing;

  // A GAMMA gate is translated into a mux driven by a block condition and fed
  // by two operands; a MU gate is translated into a mux driven by an init. The
  // operand of a gate may be another gate that is not created yet, so these
  // operands are connected once all gates are created. Each gate has a unique
  // index.
  struct MissingGate {
    // Index of the gate whose operand is another gate
    unsigned gateIndex;
    // Index of the gate providing the operand
    unsigned operandGateIndex;
    // Index of the operand among the gate's data operands
    unsigned operandIndex;
  };
  SmallVector<MissingGate> missingGates;
  // Muxes implementing a GAMMA gate with a single input, which only exist
  // during the gate creation
  DenseSet<Operation *> oneInputGammas;
  // Operation implementing each gate (nullptr once it is erased)
  llvm::MapVector<unsigned, Operation *> gateMuxes;
  // For each gate replacing a block argument, an anchor operation using the
  // gate's mux, so that the block argument can be mapped to the mux's final
  // replacement once the muxes are simplified
  SmallVector<std::pair<Value, Operation *>> rootAnchors;

  for (Block &block : llvm::drop_begin(cfRegion)) {
    Block *staging = stagingBlock[&block];
    for (Gate *gate : gsaAnalysis.getGatesPerBlock(&block)) {
      Location loc = block.front().getLoc();
      builder.setInsertionPointToStart(staging);
      SmallVector<Value> operands;
      // Index of the empty operand of a single-input GAMMA gate, if any
      int nullOperand = -1;
      for (auto [operandIdx, operand] : llvm::enumerate(gate->operands)) {
        if (operand->isTypeGate()) {
          // The other gate's mux replaces the placeholder later on
          Gate *operandGate = operand->getGate();
          operands.push_back(edgeBuilder.get(
              ftd::channelifyType(operandGate->result.getType())));
          missingGates.push_back({/*gateIndex=*/gate->index,
                                  /*operandGateIndex=*/operandGate->index,
                                  /*operandIndex=*/(unsigned)operandIdx});
        } else if (operand->isTypeEmpty()) {
          nullOperand = operandIdx;
          operands.push_back(nullptr);
        } else {
          Value val = operand->getValue();
          Backedge placeholder =
              edgeBuilder.get(ftd::channelifyType(val.getType()));
          gateOperandPlaceholders.emplace_back(val, placeholder);
          operands.push_back(placeholder);
        }
      }

      if (gate->gsaGateFunction == PhiGate) {
        assert(operands.size() == 1 && gate->isRoot &&
               "only passthrough phis survive GSA conversion");
        auto merge = builder.create<handshake::MergeOp>(
            loc, ftd::channelifyType(gate->result.getType()), operands);
        auto anchor = builder.create<UnrealizedConversionCastOp>(
            loc, merge.getResult().getType(), merge.getResult());
        rootAnchors.emplace_back(gate->result, anchor);
        gateMuxes.insert({gate->index, merge});
        continue;
      }

      Value conditionValue;
      if (gate->gsaGateFunction == MuGate) {
        // The condition of a MU gate is the condition under which the loop
        // header is reached again from itself
        Block *loopHeader = gate->getBlock();
        conditionValue = ftd::computeLoopBackedgeCondition(
            builder, loopHeader, stagingBlock[loopHeader], bi, nullptr,
            &cfgView);
      } else if (size(gate->cofactorList) > 1) {
        // Convert the BDD expansion of the gate's condition to a circuit. The
        // registry is empty since the condition does not require distribution
        BDD *bdd = ftd::buildRoutingBDD(gate->condition, gate->cofactorList);
        ftd::SignalRegistry emptyRegistry;
        conditionValue =
            ftd::bddToCircuit(builder, bdd, stagingBlock[gate->getBlock()],
                              emptyRegistry, {}, bi, nullptr, &cfgView);
      } else {
        conditionValue = cfgView.getCondition(blockID[gate->conditionBlock]);
      }

      // A MU gate's mux is driven by an init fed by its condition
      if (gate->gsaGateFunction == MuGate) {
        auto initOp = builder.create<handshake::InitOp>(loc, conditionValue);
        initOp->setAttr(ftd::FTD_INIT_MERGE, builder.getUnitAttr());
        conditionValue = initOp->getResult(0);
        conditionValue.setType(ftd::channelifyType(conditionValue.getType()));
      }

      // A single-input GAMMA gate is first created as a mux whose two data
      // operands are the input, and erased once all gates are created
      if (nullOperand >= 0) {
        operands[0] = operands[1 - nullOperand];
        operands[1] = operands[1 - nullOperand];
      }

      auto mux = builder.create<handshake::MuxOp>(
          loc, ftd::channelifyType(gate->result.getType()), conditionValue,
          operands);
      if (nullOperand >= 0)
        oneInputGammas.insert(mux);
      if (gate->isRoot) {
        auto anchor = builder.create<UnrealizedConversionCastOp>(
            loc, mux.getResult().getType(), mux.getResult());
        rootAnchors.emplace_back(gate->result, anchor);
      }
      gateMuxes.insert({gate->index, mux});
      mux->setAttr(gate->gsaGateFunction == MuGate ? ftd::FTD_EXPLICIT_MU
                                                   : ftd::FTD_EXPLICIT_GAMMA,
                   builder.getUnitAttr());
    }
  }

  // Connect the gates whose operands are other gates
  for (MissingGate &missing : missingGates) {
    Operation *gateMux = gateMuxes[missing.gateIndex];
    Value operandMuxResult = gateMuxes[missing.operandGateIndex]->getResult(0);
    // The select operand comes first
    unsigned offset = isa<handshake::MuxOp>(gateMux) ? 1 : 0;
    gateMux->setOperand(missing.operandIndex + offset, operandMuxResult);
    // Both data operands of a single-input GAMMA gate are its input
    if (oneInputGammas.contains(gateMux))
      gateMux->setOperand(2 - missing.operandIndex, operandMuxResult);
  }

  // Bypass the muxes of single-input GAMMA gates
  auto eraseMux = [&](Operation *mux) {
    for (auto &[_, gateMux] : gateMuxes) {
      if (gateMux == mux)
        gateMux = nullptr;
    }
    mux->erase();
  };
  for (Operation *mux : oneInputGammas) {
    unsigned operandToUse =
        isa_and_present<handshake::MuxOp>(mux->getOperand(1).getDefiningOp())
            ? 1
            : 2;
    mux->getResult(0).replaceAllUsesWith(mux->getOperand(operandToUse));
    eraseMux(mux);
  }

  // Two values are equivalent if they are the same value or the placeholders
  // of the same func-level value
  DenseMap<Value, Value> placeholderToCfVal;
  for (auto &[cfVal, placeholder] : gateOperandPlaceholders)
    placeholderToCfVal[placeholder] = cfVal;
  auto areEquivalent = [&](Value lhs, Value rhs) {
    if (lhs == rhs)
      return true;
    if (!lhs || !rhs)
      return false;
    return placeholderToCfVal.lookup(lhs) &&
           placeholderToCfVal.lookup(lhs) == placeholderToCfVal.lookup(rhs);
  };

  // Simplify the mux trees until they converge: merge muxes with equivalent
  // operands, and bypass muxes whose data operands are equivalent
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto it1 = gateMuxes.begin(); it1 != gateMuxes.end(); ++it1) {
      auto mux1 = dyn_cast_if_present<handshake::MuxOp>(it1->second);
      if (!mux1 || mux1.getNumOperands() != 3)
        continue;
      for (auto it2 = std::next(it1); it2 != gateMuxes.end(); ++it2) {
        auto mux2 = dyn_cast_if_present<handshake::MuxOp>(it2->second);
        if (!mux2 || mux2.getNumOperands() != 3)
          continue;
        if (areEquivalent(mux1.getSelectOperand(), mux2.getSelectOperand()) &&
            areEquivalent(mux1.getDataOperands()[0],
                          mux2.getDataOperands()[0]) &&
            areEquivalent(mux1.getDataOperands()[1],
                          mux2.getDataOperands()[1])) {
          mux2.getResult().replaceAllUsesWith(mux1.getResult());
          eraseMux(mux2);
          changed = true;
        }
      }
    }

    for (auto &[_, op] : gateMuxes) {
      auto mux = dyn_cast_if_present<handshake::MuxOp>(op);
      if (!mux || mux.getNumOperands() != 3)
        continue;
      if (areEquivalent(mux.getDataOperands()[0], mux.getDataOperands()[1])) {
        mux.getResult().replaceAllUsesWith(mux.getDataOperands()[0]);
        eraseMux(mux);
        changed = true;
      }
    }
  }

  // Map each block argument to what replaces its gate's mux
  for (auto &[blockArg, anchor] : rootAnchors) {
    valueMap.map(blockArg, anchor->getOperand(0));
    anchor->erase();
  }
  for (Block &block : llvm::drop_begin(cfRegion)) {
    for (BlockArgument arg : block.getArguments()) {
      if (!valueMap.contains(arg))
        return emitError(arg.getLoc()) << "block argument has no GSA gate";
    }
  }
  return success();
}

LogicalResult FtdCfToHandshakeTranslator::createMerges(Block *block) {
  // The block arguments are the GSA gates' muxes, so only the control needs a
  // merge
  if (preds[block].empty())
    return block->front().emitError() << "block must have predecessors";
  createControlMerge(block);
  return success();
}

SmallVector<Value>
FtdCfToHandshakeTranslator::getForwardedValues(Operation *termOp,
                                               unsigned succIdx) {
  // The GSA gates' muxes replace the successor's block arguments
  return {};
}

Value FtdCfToHandshakeTranslator::getBranchCondition(
    cf::CondBranchOp condBrOp) {
  // The block's own branches also read the condition through the proxy
  return cfgView.getCondition(blockID[condBrOp->getBlock()]);
}

bool FtdCfToHandshakeTranslator::isCstSourcable(arith::ConstantOp cstOp) {
  // A constant cannot be triggered by a source if it feeds a memory port or
  // the condition of a branch. It does not feed the other branch operands,
  // since terminators only branch the block's control
  return llvm::none_of(cstOp->getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    // The condition is a conditional branch's first operand
    if (isa<cf::CondBranchOp>(user))
      return use.getOperandNumber() == 0;
    return isa<memref::LoadOp, memref::StoreOp>(user);
  });
}

Value FtdCfToHandshakeTranslator::getBlockControlChannel(Block *block) {
  // The FTD circuits regenerate and suppress the constants as needed
  return funcOp.getArguments().back();
}

void FtdCfToHandshakeTranslator::finalizeBlocks() {
  // Operations belong to the block whose staging block they are in, including
  // the ones that the FTD circuits created with another "bb" attribute. Memory
  // interfaces do not naturally belong to any block
  for (auto [idx, staging] : llvm::enumerate(funcOp.getBody())) {
    for (Operation &op : staging) {
      if (!isa<handshake::MemoryOpInterface>(op))
        setBB(&op, idx);
    }
  }
}

LogicalResult FtdCfToHandshakeTranslator::finalizeTranslation() {
  for (auto &[cfVal, placeholder] : gateOperandPlaceholders)
    placeholder.setValue(valueMap.lookup(cfVal));
  Type condType = handshake::ChannelType::get(builder.getI1Type());
  for (auto &[block, placeholder] : condPlaceholders) {
    auto condBrOp = cast<cf::CondBranchOp>(block->getTerminator());
    Value cond = valueMap.lookup(condBrOp.getCondition());
    if (Operation *defOp = cond.getDefiningOp())
      builder.setInsertionPointAfter(defOp);
    else
      builder.setInsertionPointToStart(funcOp.getBodyBlock());
    auto proxyOp =
        builder.create<handshake::NotIOp>(condBrOp.getLoc(), condType, cond);
    proxyOp->setAttr(ftd::FTD_COND_VAR, builder.getUnitAttr());
    setBlock(proxyOp, block);
    placeholder.setValue(proxyOp.getResult());
    cfgView.conditionMap[blockID[block]] = proxyOp.getResult();
  }

  cfg::annotateCFG(funcOp, cfFunc.getBody(), [&](Value cond) {
    return namer.getName(valueMap.lookup(cond).getDefiningOp()).str();
  });

  if (!cfFunc.getBody().hasOneBlock()) {
    ftd::addRegen(funcOp, builder, cfgView);
    ftd::addSupp(funcOp, builder, cfgView);
  }
  // Bypass the condition proxies
  ftd::finalizeCondPlaceholders(funcOp);
  return success();
}

LogicalResult
dynamatic::ftdLowerCfToHandshake(ModuleOp modOp, NameAnalysis &namer,
                                 const experimental::gsa::GSAAnalysis &gsa) {
  OpBuilder builder(modOp.getContext());
  return lowerFuncsToHandshake(modOp, [&](func::FuncOp funcOp) {
    return FtdCfToHandshakeTranslator(funcOp, builder, namer, gsa).translate();
  });
}
