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

void dynamatic::buffer::setFPGA20Properties(circt::handshake::FuncOp funcOp) {
  // Merges with more than one input should have at least a transparent slot
  // at their output
  for (handshake::MergeOp mergeOp : funcOp.getOps<handshake::MergeOp>()) {
    if (mergeOp->getNumOperands() > 1) {
      Channel channel(mergeOp.getResult(), true);
      channel.props->minTrans = std::max(channel.props->minTrans, 1U);
    }
  }

  // Channels connected to MCs are not bufferizable
  for (auto mcOp : funcOp.getOps<handshake::MemoryControllerOp>()) {
    for (Value oprd : mcOp->getOperands()) {
      Channel channel(oprd, true);
      channel.props->maxOpaque = 0;
      channel.props->maxTrans = 0;
    }
    for (OpResult res : mcOp->getResults()) {
      Channel channel(res, true);
      channel.props->maxOpaque = 0;
      channel.props->maxTrans = 0;
    }
  }

  // Channels connected to LSQs are not bufferizable, except control ports which
  // should have at least one opaque buffer
  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {
    // Get control indices
    DenseSet<unsigned> controlIndices;
    LSQPorts ports = lsqOp.getPorts();
    unsigned idxOffset = lsqOp.isConnectedToMC() ? 0 : 1;
    for (LSQGroup &group : ports.getGroups())
      controlIndices.insert((group->ctrlPort->getCtrlInputIndex() + idxOffset));

    for (auto [idx, oprd] : llvm::enumerate(lsqOp->getOperands())) {
      Channel channel(oprd, true);
      if (controlIndices.contains(idx)) {
        // This is a control port input
        channel.props->minOpaque = std::max(channel.props->minOpaque, 1U);
      } else {
        // This is not a control port input
        channel.props->maxOpaque = 0;
        channel.props->maxTrans = 0;
      }
    }

    for (OpResult res : lsqOp->getResults()) {
      Channel channel(res, true);
      channel.props->maxOpaque = 0;
      channel.props->maxTrans = 0;
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
