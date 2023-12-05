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
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace dynamatic;
using namespace dynamatic::buffer;

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

      // Force placement of an opaque slot on all fork output channels, except
      // the one going to the LSQ
      for (OpResult forkRes : forkOp->getResults()) {
        if (forkRes != ctrlVal) {
          Channel channel(forkRes, true);
          channel.props->minOpaque = std::max(channel.props->minOpaque, 1U);
        }
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
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      setFPGA20Properties(funcOp);
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::buffer::createHandshakeSetBufferingProperties(
    const std::string &version) {
  return std::make_unique<HandshakeSetBufferingPropertiesPass>(version);
}
