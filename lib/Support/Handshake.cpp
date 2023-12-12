//===- Handshake.cpp - Helpers for Handshake-level analysis -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements helpers for working with Handshake-level IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Handshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

bool dynamatic::hasRealUses(Value val) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    return !isa<handshake::SinkOp>(user);
  });
}

void dynamatic::eraseSinkUsers(Value val) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      user->erase();
  }
}

void dynamatic::eraseSinkUsers(Value val, PatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      rewriter.eraseOp(user);
  }
}

SmallVector<Value> dynamatic::getLSQControlPaths(handshake::LSQOp lsqOp,
                                                 Operation *ctrlOp) {
  // Accumulate all outputs of the control operation that are part of the memory
  // control network
  SmallVector<Value> controlValues;
  // List of control channels to explore, starting from the control operation's
  // results
  SmallVector<Value, 4> controlChannels;
  // Set of control operations already explored from the control operation's
  // results (to avoid looping in the dataflow graph)
  SmallPtrSet<Operation *, 4> controlOps;
  for (OpResult res : ctrlOp->getResults()) {
    // We only care for control-only channels
    if (!isa<NoneType>(res.getType()))
      continue;

    // Reset the list of control channels to explore and the list of control
    // operations that we have already visited
    controlChannels.clear();
    controlOps.clear();

    controlChannels.push_back(res);
    controlOps.insert(ctrlOp);
    do {
      Value val = controlChannels.pop_back_val();
      for (Operation *succOp : val.getUsers()) {
        // Make sure that we do not loop forever over the same control
        // operations
        if (auto [_, newOp] = controlOps.insert(succOp); !newOp)
          continue;

        if (succOp == lsqOp) {
          // We have found a control path triggering a different group
          // allocation to the LSQ, add it to our list
          controlValues.push_back(res);
          break;
        }
        llvm::TypeSwitch<Operation *, void>(succOp)
            .Case<handshake::ConditionalBranchOp, handshake::BranchOp,
                  handshake::MergeOp, handshake::MuxOp, handshake::ForkOp,
                  handshake::LazyForkOp, handshake::BufferOp>([&](auto) {
              // If the successor just propagates the control path, add
              // all its results to the list of control channels to
              // explore
              for (OpResult succRes : succOp->getResults())
                controlChannels.push_back(succRes);
            })
            .Case<handshake::ControlMergeOp>(
                [&](handshake::ControlMergeOp cmergeOp) {
                  // Only the control merge's data output forwards the input
                  controlChannels.push_back(cmergeOp.getResult());
                });
      }
    } while (!controlChannels.empty());
  }

  return controlValues;
}
