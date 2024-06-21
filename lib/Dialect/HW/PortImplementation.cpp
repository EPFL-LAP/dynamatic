//===- PortImplementation.cpp - Port-related data-structures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::hw;

PortNameGenerator::PortNameGenerator(Operation *op) {
  assert(op && "cannot generate port names for null operation");
  if (auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op))
    inferFromNamedOpInterface(namedOpInterface);
  else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
    inferFromFuncOp(funcOp);
  else
    inferDefault(op);
}

void PortNameGenerator::infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
  for (size_t idx = 0, e = op->getNumOperands(); idx < e; ++idx)
    inputs.push_back(inF(idx));
  for (size_t idx = 0, e = op->getNumResults(); idx < e; ++idx)
    outputs.push_back(outF(idx));

  // The Handshake terminator forwards its non-memory inputs to its outputs, so
  // it needs port names for them
  if (handshake::EndOp endOp = dyn_cast<handshake::EndOp>(op)) {
    handshake::FuncOp funcOp = endOp->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "end must be child of handshake function");
    size_t numResults = funcOp.getFunctionType().getNumResults();
    for (size_t idx = 0, e = numResults; idx < e; ++idx)
      outputs.push_back(endOp.getDefaultResultName(idx));
  }
}

void PortNameGenerator::inferDefault(Operation *op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
            arith::CmpFOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MulFOp, arith::MulIOp,
            arith::OrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::SubFOp, arith::SubIOp, arith::XOrIOp>([&](auto) {
        infer(
            op, [](unsigned idx) { return idx == 0 ? "lhs" : "rhs"; },
            [](unsigned idx) { return "result"; });
      })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::NegFOp, arith::TruncIOp>(
          [&](auto) {
            infer(
                op, [](unsigned idx) { return "ins"; },
                [](unsigned idx) { return "outs"; });
          })
      .Case<arith::SelectOp>([&](auto) {
        infer(
            op,
            [](unsigned idx) {
              if (idx == 0)
                return "condition";
              if (idx == 1)
                return "trueValue";
              return "falseValue";
            },
            [](unsigned idx) { return "result"; });
      })
      .Default([&](auto) {
        infer(
            op, [](unsigned idx) { return "in" + std::to_string(idx); },
            [](unsigned idx) { return "out" + std::to_string(idx); });
      });
}

void PortNameGenerator::inferFromNamedOpInterface(
    handshake::NamedIOInterface namedIO) {
  auto inF = [&](unsigned idx) { return namedIO.getOperandName(idx); };
  auto outF = [&](unsigned idx) { return namedIO.getResultName(idx); };
  infer(namedIO, inF, outF);
}

void PortNameGenerator::inferFromFuncOp(handshake::FuncOp funcOp) {
  llvm::transform(funcOp.getArgNames(), std::back_inserter(inputs),
                  [](Attribute arg) { return cast<StringAttr>(arg).str(); });
  llvm::transform(funcOp.getResNames(), std::back_inserter(outputs),
                  [](Attribute res) { return cast<StringAttr>(res).str(); });
}
