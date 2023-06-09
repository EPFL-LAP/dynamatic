//===- UtilsBitsUpdate.cpp - Utils support bits optimization ----*- C++ -*-===//
//
// This file contains basic functions for type updates for --optimize-bits pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

IntegerType getNewType(Value opVal, unsigned bitswidth, bool signless) {
  IntegerType::SignednessSemantics ifSign =
      IntegerType::SignednessSemantics::Signless;
  if (!signless)
    if (auto validType = dyn_cast<IntegerType>(opVal.getType()))
      ifSign = validType.getSignedness();

  return IntegerType::get(opVal.getContext(), bitswidth, ifSign);
}

IntegerType getNewType(Value opVal, unsigned bitswidth,
                       IntegerType::SignednessSemantics ifSign) {
  return IntegerType::get(opVal.getContext(), bitswidth, ifSign);
}

// specify which value to extend
std::optional<Operation *> insertWidthMatchOp(Operation *newOp, int opInd,
                                              Type newType, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  Value opVal = newOp->getOperand(opInd);

  if (!isa<IndexType, IntegerType>(opVal.getType()))
    assert(false && "Only supported width matching for Integer/Index Type!");

  unsigned int opWidth;
  if (isa<IndexType>(opVal.getType()))
    opWidth = IndexType::kInternalStorageBitWidth;
  else
    opWidth = opVal.getType().getIntOrFloatBitWidth();

  if (isa<IntegerType, IndexType>(opVal.getType())) {
    // insert Truncation operation to match the opresult width
    if (opWidth > newType.getIntOrFloatBitWidth()) {
      builder.setInsertionPoint(newOp);
      auto truncOp = builder.create<mlir::arith::TruncIOp>(newOp->getLoc(),
                                                           newType, opVal);
      newOp->setOperand(opInd, truncOp.getResult());

      return truncOp;
    }

    // insert Extension operation to match the opresult width
    if (opWidth < newType.getIntOrFloatBitWidth()) {
      builder.setInsertionPoint(newOp);
      auto extOp =
          builder.create<mlir::arith::ExtSIOp>(newOp->getLoc(), newType, opVal);
      newOp->setOperand(opInd, extOp.getResult());

      return extOp;
    }
  }
  return {};
}

namespace dynamatic::bitwidth {

void constructForwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::operand_range vecOperands)>>
        &mapOpNameWidth) {

  mapOpNameWidth[mlir::arith::AddIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return std::min(
            CPP_MAX_WIDTH,
            std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                     vecOperands[1].getType().getIntOrFloatBitWidth()) +
                1);
      };

  mapOpNameWidth[mlir::arith::SubIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::MulIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return std::min(CPP_MAX_WIDTH,
                        vecOperands[0].getType().getIntOrFloatBitWidth() +
                            vecOperands[1].getType().getIntOrFloatBitWidth());
      };

  mapOpNameWidth[mlir::arith::DivUIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return std::min(CPP_MAX_WIDTH,
                        vecOperands[0].getType().getIntOrFloatBitWidth() + 1);
      };
  mapOpNameWidth[mlir::arith::DivSIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::DivUIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::AndIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return std::min(
            CPP_MAX_WIDTH,
            std::min(vecOperands[0].getType().getIntOrFloatBitWidth(),
                     vecOperands[1].getType().getIntOrFloatBitWidth()));
      };

  mapOpNameWidth[mlir::arith::OrIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return std::min(
            CPP_MAX_WIDTH,
            std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                     vecOperands[1].getType().getIntOrFloatBitWidth()));
      };

  mapOpNameWidth[mlir::arith::XOrIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::OrIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::ShLIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        int shift_bit = 0;
        if (auto defOp = vecOperands[1].getDefiningOp();
            isa<handshake::ConstantOp>(defOp)) {
          if (handshake::ConstantOp cstOp =
                  dyn_cast<handshake::ConstantOp>(defOp))
            if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
              shift_bit = IntAttr.getValue().getZExtValue();
          return std::min(CPP_MAX_WIDTH,
                          vecOperands[0].getType().getIntOrFloatBitWidth() +
                              shift_bit);
        }
        return CPP_MAX_WIDTH;
      };

  mapOpNameWidth[mlir::arith::ShRSIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        int shiftBit = 0;
        if (auto defOp = vecOperands[1].getDefiningOp();
            isa<handshake::ConstantOp>(defOp))
          if (handshake::ConstantOp cstOp =
                  dyn_cast<handshake::ConstantOp>(defOp))
            if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
              shiftBit = IntAttr.getValue().getZExtValue();

        return std::min(CPP_MAX_WIDTH,
                        vecOperands[0].getType().getIntOrFloatBitWidth() -
                            shiftBit);
      };

  mapOpNameWidth[mlir::arith::ShRUIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::ShRSIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::CmpIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) { return unsigned(1); };

  mapOpNameWidth[mlir::arith::ExtSIOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        return vecOperands[0].getType().getIntOrFloatBitWidth();
      };

  mapOpNameWidth[mlir::arith::ExtUIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::ExtSIOp::getOperationName()];

  mapOpNameWidth[handshake::ControlMergeOp::getOperationName()] =
      [](Operation::operand_range vecOperands) {
        unsigned ind = vecOperands.size(); // record number of operators
        unsigned indexWidth = 1;
        if (ind > 1)
          indexWidth = ceil(log2(ind));

        return indexWidth;
      };

  // mapOpNameWidth[handshake::BranchOp::getOperationName()] = 
  //   [](Operation::operand_range vecOperands) {
  //     return vecOperands[0].getType().getIntOrFloatBitWidth();
  //   };

  // mapOpNameWidth[handshake::ConditionalBranchOp::getOperationName()] = 
  //   [](Operation::operand_range vecOperands) {
  //     return vecOperands[1].getType().getIntOrFloatBitWidth();
  //   };
};

void constructBackwardFuncMap(
    DenseMap<StringRef,
             std::function<unsigned(Operation::result_range vecResults)>>
        &mapOpNameWidth) {
  mapOpNameWidth[mlir::arith::AddIOp::getOperationName()] =
      [](Operation::result_range vecResults) {
        return std::min(CPP_MAX_WIDTH,
                        vecResults[0].getType().getIntOrFloatBitWidth());
      };

  mapOpNameWidth[mlir::arith::SubIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::MulIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::AndIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::OrIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::XOrIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];
}

void constructUpdateFuncMap(
    DenseMap<StringRef, std::function<std::vector<std::vector<unsigned int>>(
                            Operation::operand_range vecOperands,
                            Operation::result_range vecResults)>>
        &mapOpNameWidth) {

  mapOpNameWidth[mlir::arith::AddIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;

        unsigned int maxOpWidth =
            std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                     vecOperands[1].getType().getIntOrFloatBitWidth());

        unsigned int width = std::min(
            vecResults[0].getType().getIntOrFloatBitWidth(), maxOpWidth + 1);

        width = std::min(CPP_MAX_WIDTH, width);
        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::SubIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::AddIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::MulIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;

        unsigned int maxOpWidth =
            vecOperands[0].getType().getIntOrFloatBitWidth() +
            vecOperands[1].getType().getIntOrFloatBitWidth();

        unsigned int width = std::min(
            vecResults[0].getType().getIntOrFloatBitWidth(), maxOpWidth);

        width = std::min(CPP_MAX_WIDTH, width);

        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::DivSIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;

        unsigned int maxOpWidth =
            vecOperands[0].getType().getIntOrFloatBitWidth();

        unsigned int width = std::min(
            vecResults[0].getType().getIntOrFloatBitWidth(), maxOpWidth + 1);

        width = std::min(CPP_MAX_WIDTH, width);

        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::DivUIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::DivSIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::DivSIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::DivSIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::ShLIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        unsigned shiftBit = 0;
        if (auto defOp = vecOperands[1].getDefiningOp();
            isa<handshake::ConstantOp>(defOp))
          if (handshake::ConstantOp cstOp =
                  dyn_cast<handshake::ConstantOp>(defOp))
            if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
              shiftBit = IntAttr.getValue().getZExtValue();

        unsigned int width = std::min(
            std::min(CPP_MAX_WIDTH,
                     vecResults[0].getType().getIntOrFloatBitWidth()),
            vecOperands[0].getType().getIntOrFloatBitWidth() + shiftBit);

        width = std::min(CPP_MAX_WIDTH, width);
        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::ShRSIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        unsigned shiftBit = 0;
        if (auto defOp = vecOperands[1].getDefiningOp();
            isa<handshake::ConstantOp>(defOp))
          if (handshake::ConstantOp cstOp =
                  dyn_cast<handshake::ConstantOp>(defOp))
            if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
              shiftBit = IntAttr.getValue().getZExtValue();

        unsigned int width = std::min(
            std::min(CPP_MAX_WIDTH,
                     vecResults[0].getType().getIntOrFloatBitWidth()),
            vecOperands[0].getType().getIntOrFloatBitWidth() - shiftBit);

        width = std::min(CPP_MAX_WIDTH, width);
        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::ShRUIOp::getOperationName()] =
      mapOpNameWidth[mlir::arith::ShRSIOp::getOperationName()];

  mapOpNameWidth[mlir::arith::CmpIOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;

        unsigned int maxOpWidth =
            std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                     vecOperands[1].getType().getIntOrFloatBitWidth());

        unsigned int width = std::min(CPP_MAX_WIDTH, maxOpWidth);

        widths.push_back({width, width}); // matched widths for operators
        widths.push_back({unsigned(1)});  // matched widths for result

        return widths;
      };

  mapOpNameWidth[handshake::MuxOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        unsigned maxOpWidth = 2;

        unsigned ind = 0; // record number of operators

        for (auto oprand : vecOperands) {
          ind++;
          if (ind == 0)
            continue; // skip the width of the index

          if (!isa<NoneType>(oprand.getType()))
            if (!isa<IndexType>(oprand.getType()) &&
                oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
              maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
        }
        unsigned indexWidth = 2;
        if (ind > 2)
          indexWidth = log2(ind - 2) + 2;

        widths.push_back(
            {indexWidth}); // the bit width for the mux index result;

        if (isa<NoneType>(vecResults[0].getType())) {
          widths.push_back({});
          return widths;
        }

        unsigned int width =
            std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                     std::min(CPP_MAX_WIDTH, maxOpWidth));
        // 1st operand is the index; rest of (ind -1) operands set to width
        std::vector<unsigned> opwidths(ind - 1, width);

        widths[0].insert(widths[0].end(), opwidths.begin(),
                         opwidths.end()); // matched widths for operators
        widths.push_back({width});        // matched widths for result

        return widths;
      };

  mapOpNameWidth[StringRef("handshake.merge")] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        unsigned maxOpWidth = 2;

        unsigned ind = 0; // record number of operators

        for (auto oprand : vecOperands) {
          ind++;
          if (!isa<NoneType>(vecOperands[0].getType()))
            if (!isa<IndexType>(oprand.getType()) &&
                oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
              maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
        }

        if (isa<NoneType>(vecOperands[0].getType())) {
          widths.push_back({});
          widths.push_back({});
          return widths;
        }

        unsigned int width =
            std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                     std::min(CPP_MAX_WIDTH, maxOpWidth));
        std::vector<unsigned> opwidths(ind, width);

        widths.push_back(opwidths); // matched widths for operators
        widths.push_back({width});  // matched widths for result

        return widths;
      };

  mapOpNameWidth[StringRef("handshake.constant")] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        // Do not set the width of the input
        widths.push_back({});
        Operation *Op = vecResults[0].getDefiningOp();
        if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(*Op))
          if (auto IntAttr = cstOp.getValueAttr().dyn_cast<mlir::IntegerAttr>())
            if (auto IntType = dyn_cast<IntegerType>(IntAttr.getType())) {
              widths.push_back({IntType.getWidth()});
              return widths;
            }

        widths.push_back({});
        return widths;
      };

  mapOpNameWidth[StringRef("handshake.control_merge")] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        unsigned maxOpWidth = 2;

        unsigned ind = 0; // record number of operators

        for (auto oprand : vecOperands) {
          ind++;
          if (!isa<NoneType>(oprand.getType()))
            if (!isa<IndexType>(oprand.getType()) &&
                oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
              maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
        }

        unsigned indexWidth = 1;
        if (ind > 1)
          indexWidth = ceil(log2(ind));

        if (isa<NoneType>(vecOperands[0].getType())) {
          widths.push_back({});
          widths.push_back({0, indexWidth});
          return widths;
        }

        unsigned int width = std::min(CPP_MAX_WIDTH, maxOpWidth);
        std::vector<unsigned> opwidths(ind, width);

        widths.push_back(opwidths);     // matched widths for operators
        widths.push_back({indexWidth}); // matched widths for result

        return widths;
      };

  mapOpNameWidth[mlir::arith::SelectOp::getOperationName()] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;

        unsigned ind = 0, maxOpWidth = 2;

        for (auto oprand : vecOperands) {
          ind++;
          if (ind == 0)
            continue; // skip the width of the index

          if (!isa<NoneType>(oprand.getType()))
            if (!isa<IndexType>(oprand.getType()) &&
                oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
              maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
        }

        widths.push_back({1}); // bool like condition
        if (isa<NoneType>(vecOperands[1].getType())) {
          widths.push_back({});
          return widths;
        }

        std::vector<unsigned> opwidths(ind - 1, maxOpWidth);

        widths[0].insert(widths[0].end(), opwidths.begin(),
                         opwidths.end()); // matched widths for operators
        widths.push_back({maxOpWidth});   // matched widths for result

        return widths;
      };

  mapOpNameWidth[StringRef("handshake.d_return")] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        widths.push_back({ADDRESS_WIDTH});
        if (!isa<NoneType>(vecResults[0].getType()))
          widths.push_back({ADDRESS_WIDTH});
        else
          widths.push_back({});
        return widths;
      };

  mapOpNameWidth[StringRef("handshake.d_load")] =
      [&](Operation::operand_range vecOperands,
          Operation::result_range vecResults) {
        std::vector<std::vector<unsigned>> widths;
        widths.push_back({CPP_MAX_WIDTH, ADDRESS_WIDTH});
        widths.push_back({CPP_MAX_WIDTH, ADDRESS_WIDTH});
        return widths;
      };

  mapOpNameWidth[StringRef("handshake.d_store")] =
      mapOpNameWidth[StringRef("handshake.d_load")];
};

static bool setPassFlag(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
    .Case<handshake::BranchOp, handshake::ConditionalBranchOp>(
        [](Operation *op) { return true; })
    .Default([&](auto) { return false; });
}

static bool setMatchFlag(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
            mlir::arith::ShLIOp, mlir::arith::ShRSIOp, mlir::arith::ShRUIOp,
            mlir::arith::DivSIOp, mlir::arith::DivUIOp, mlir::arith::CmpIOp,
            mlir::arith::ShRSIOp, handshake::MuxOp, handshake::MergeOp,
            mlir::arith::SelectOp, handshake::ConstantOp,
            handshake::DynamaticLoadOp, handshake::DynamaticStoreOp,
            handshake::ControlMergeOp, handshake::DynamaticReturnOp>(
          [](Operation *op) { return true; })
      .Default([&](auto) { return false; });
}

static bool setRevertFlag(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<mlir::arith::TruncIOp, mlir::arith::ExtSIOp, mlir::arith::ExtUIOp>(
          [](Operation *op) { return true; })
      .Default([&](auto) { return false; });
}

bool propType(Operation *op) {

  if (isa<handshake::ConditionalBranchOp>(*op)) {
    for (auto resOp : op->getResults())
      resOp.setType(op->getOperand(1).getType());
    return true;
  }

  if (isa<handshake::BranchOp>(*op)) {
    op->getResult(0).setType(op->getOperand(0).getType());
    return true;
  }
  return false;
}

void replaceWithPredecessor(Operation *op) {
  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
}

void replaceWithPredecessor(Operation *op, Type resType) {
  Operation *sucNode = op->getOperand(0).getDefiningOp();

  // find the index of result in vec_results
  for (auto Val : sucNode->getResults()) {
    if (Val == op->getOperand(0)) {
      Val.setType(resType);
      break;
    }
  }

  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
}

void revertTruncOrExt(Operation *op, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  // if width(res) == width(opr) : delte the operand;

  if (op->getResult(0).getType().getIntOrFloatBitWidth() ==
      op->getOperand(0).getType().getIntOrFloatBitWidth()) {

    replaceWithPredecessor(op);
    op->erase();
    return;
  }

  // if for extension operation width(res) < width(opr),
  // change it to truncation operation
  if (isa<mlir::arith::ExtSIOp>(*op) || isa<mlir::arith::ExtUIOp>(*op))
    if (op->getResult(0).getType().getIntOrFloatBitWidth() <
        op->getOperand(0).getType().getIntOrFloatBitWidth()) {

      // builder.setInsertionPoint(op);
      // Type newType =
      //     getNewType(op->getResult(0),
      //                op->getResult(0).getType().getIntOrFloatBitWidth(), false);
      // auto truncOp = builder.create<mlir::arith::TruncIOp>(
      //     op->getLoc(), newType, op->getOperand(0));
      // op->getResult(0).replaceAllUsesWith(truncOp.getResult());
      replaceWithPredecessor(op);

      op->erase();
      return;
    }

  // if for truncation operation width(res) > width(opr),
  // change it to extension operation
  if (isa<mlir::arith::TruncIOp>(*op))
    if (op->getResult(0).getType().getIntOrFloatBitWidth() >
        op->getOperand(0).getType().getIntOrFloatBitWidth()) {

      // builder.setInsertionPoint(op);
      // Type newType =
      //     getNewType(op->getResult(0),
      //                op->getResult(0).getType().getIntOrFloatBitWidth(), false);
      // auto truncOp = builder.create<mlir::arith::ExtSIOp>(op->getLoc(), newType,
      //                                                     op->getOperand(0));
      // op->getResult(0).replaceAllUsesWith(truncOp.getResult());
      replaceWithPredecessor(op);

      op->erase();
    }
}

void matchOpResWidth(Operation *op, MLIRContext *ctx,
                     SmallVector<Operation *> &newMatchedOps) {

  DenseMap<mlir::StringRef,
           std::function<std::vector<std::vector<unsigned int>>(
               Operation::operand_range vecOperands,
               Operation::result_range vecResults)>>
      mapOpNameWidth;

  constructUpdateFuncMap(mapOpNameWidth);

  std::vector<std::vector<unsigned int>> oprsWidth =
      mapOpNameWidth[op->getName().getStringRef()](op->getOperands(),
                                                   op->getResults());

  // make operator matched the width
  for (size_t i = 0; i < oprsWidth[0].size(); ++i) {
    if (auto Operand = op->getOperand(i);
        !isa<NoneType>(Operand.getType()) &&
        Operand.getType().getIntOrFloatBitWidth() != oprsWidth[0][i]) {
      auto insertOp = insertWidthMatchOp(
          op, i, getNewType(Operand, oprsWidth[0][i], false), ctx);
      if (insertOp.has_value())
        newMatchedOps.push_back(insertOp.value());
    }
  }
  // make result matched the width
  for (size_t i = 0; i < oprsWidth[1].size(); ++i) {
    if (auto OpRes = op->getResult(i);
        oprsWidth[1][i] != 0 &&
        OpRes.getType().getIntOrFloatBitWidth() != oprsWidth[1][i]) {
      Type newType = getNewType(OpRes, oprsWidth[1][i], false);
      op->getResult(i).setType(newType);
    }
  }
}

void validateOp(Operation *op, MLIRContext *ctx,
                SmallVector<Operation *> &newMatchedOps) {
  // the operations can be divided to three types to make it validated
  // passType: branch, conditionalbranch
  // c <= op(a,b): addi, subi, mux, etc. where both a,b,c needed to be verified
  // need to be reverted or deleted : truncIOp, extIOp
  bool pass = setPassFlag(op);
  bool match = setMatchFlag(op);
  bool revert = setRevertFlag(op);

  if (pass)
    // Validate the successor operations 
    // influenced by the bit width propagation 
    if(propType(op))
      for (auto resOp : op->getResults().getUsers())
          validateOp(resOp, ctx, newMatchedOps);

  if (match)
    matchOpResWidth(op, ctx, newMatchedOps);

  if (revert)
    revertTruncOrExt(op, ctx);
}
} // namespace dynamatic::bitwidth
