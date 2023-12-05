//===- HandshakeSetBufferingProperties.h - Set buf. props. ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-set-buffering-properties pass. For now there is
// only a single policy, but it is expected that more will be defined in the
// future. Similarly to the default "fpga20", specifying a new policy amounts to
// writing a simple function that will be called on each channel present in the
// dataflow circuit and modify, if necessary, the buffering properties
// associated with it. The logic for fetching and writing back that data to the
// IR is conveniently hidden to reduce the amount of boilerplate code and
// improve performance.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakeSetBufferingProperties.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace dynamatic;
using namespace dynamatic::buffer;

/// Makes all channels adjacent to operations of the given type inside the
/// function unbufferizable.
template <typename Op>
static void makeUnbufferizable(circt::handshake::FuncOp funcOp) {
  // Channels connected to memory interfaces are not bufferizable
  for (Op op : funcOp.getOps<Op>()) {
    for (Value oprd : op->getOperands()) {
      Channel channel(oprd, true);
      channel.props->maxOpaque = 0;
      channel.props->maxTrans = 0;
    }
    for (OpResult res : op->getResults()) {
      Channel channel(res, true);
      channel.props->maxOpaque = 0;
      channel.props->maxTrans = 0;
    }
  }
}

void dynamatic::buffer::setFPGA20Properties(circt::handshake::FuncOp funcOp) {
  // Merges with more than one input should have at least a transparent slot
  // at their output
  for (handshake::MergeOp mergeOp : funcOp.getOps<handshake::MergeOp>()) {
    if (mergeOp->getNumOperands() > 1) {
      Channel channel(mergeOp.getResult(), true);
      channel.props->minTrans = std::max(channel.props->minTrans, 1U);
    }
  }

  // Channels connected to memory interfaces are not bufferizable
  makeUnbufferizable<handshake::MemoryControllerOp>(funcOp);
  makeUnbufferizable<handshake::LSQOp>(funcOp);

  // Forked control signals going to LSQs should have an opaque buffer between
  // the fork and successors that are not the LSQ
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {
    LSQPorts ports = lsqOp.getPorts();
    ValueRange lsqInputs = lsqOp.getMemOperands();

    for (LSQGroup &group : ports.getGroups()) {
      // Control signal must come from a fork for this constraint to apply
      Value ctrlVal = lsqInputs[group->ctrlPort->getCtrlInputIndex()];
      Operation *ctrlDefOp = ctrlVal.getDefiningOp();
      auto forkOp = mlir::dyn_cast_if_present<handshake::ForkOp>(ctrlDefOp);
      if (!forkOp)
        continue;

      // Force placement of an opaque buffer slot on fork output channels
      // triggering group allocations to the same LSQ
      SmallVector<Value, 4> controlChannels;
      SmallPtrSet<Operation *, 4> controlOps;
      for (OpResult forkRes : forkOp->getResults()) {
        // Ignore the fork output that immediately goes to the LSQ
        if (ctrlVal == forkRes)
          continue;

        // Reset the list of control channels to explore and the list of control
        // operations that we have already visited
        controlChannels.clear();
        controlOps.clear();

        controlChannels.push_back(forkRes);
        controlOps.insert(forkOp);
        do {
          Value val = controlChannels.pop_back_val();
          Operation *succOp = *val.getUsers().begin();

          // Make sure to not loop forever over the same control operations
          if (auto [_, newOp] = controlOps.insert(succOp); !newOp)
            continue;

          if (succOp == lsqOp) {
            // We have found a control path triggering a different group
            // allocation to the LSQ, force placement of an opaque buffer on it
            Channel channel(forkRes, true);
            channel.props->minOpaque = std::max(channel.props->minOpaque, 1U);
            break;
          }
          llvm::TypeSwitch<Operation *, void>(succOp)
              .Case<handshake::ConditionalBranchOp, handshake::BranchOp,
                    handshake::MergeOp, handshake::MuxOp, handshake::ForkOp>(
                  [&](auto) {
                    // If the successor just propagates the control path, add
                    // all its results to the list of control channels to
                    // explore
                    for (OpResult succRes : succOp->getResults())
                      controlChannels.push_back(succRes);
                  })
              .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
                // Only follow the control path if the buffer isn't opaque,
                // since we don't need to buffer datapaths that already have a
                // buffer
                if (bufOp.getBufferType() == BufferTypeEnum::fifo)
                  controlChannels.push_back(bufOp.getResult());
              })
              .Case<handshake::ControlMergeOp>(
                  [&](handshake::ControlMergeOp cmergeOp) {
                    // Only the control merge's data output forwards the input
                    controlChannels.push_back(cmergeOp.getResult());
                  });
        } while (!controlChannels.empty());
      }
    }
  }
}

namespace {

/// Simple pass driver that runs a specific buffering properties setting policy
/// on each Handshake function in the IR.
struct HandshakeSetBufferingPropertiesPass
    : public dynamatic::buffer::impl::HandshakeSetBufferingPropertiesBase<
          HandshakeSetBufferingPropertiesPass> {

  HandshakeSetBufferingPropertiesPass(const std::string &version) {
    this->version = version;
  }

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    // Check that the provided version is valid
    if (version != "fpga20") {
      modOp->emitError() << "Unkwown version \"" << version
                         << "\", expected one of [fpga20]";
      return signalPassFailure();
    }

    // Add properties to channels inside each function
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      // Buffer placement requires that all values are used exactly once
      if (failed(verifyAllValuesHasOneUse(funcOp))) {
        funcOp.emitOpError() << "Not all values are used exactly once";
        return signalPassFailure();
      }
      setFPGA20Properties(funcOp);
    }
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::buffer::createHandshakeSetBufferingProperties(
    const std::string &version) {
  return std::make_unique<HandshakeSetBufferingPropertiesPass>(version);
}
