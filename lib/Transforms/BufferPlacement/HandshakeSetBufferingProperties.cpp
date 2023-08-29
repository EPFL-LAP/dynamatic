//===- HandshakeSetBufferingProperties.h - Set buf. props. ------*- C++ -*-===//
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
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace dynamatic;
using namespace dynamatic::buffer;

void dynamatic::buffer::setFPGA20Properties(Channel &channel) {
  // Merges with more than one input should have at least a transparent slot
  // at their output
  if (isa<handshake::MergeOp>(channel.producer) &&
      channel.producer.getNumOperands() > 1)
    channel.props->minTrans = std::max(channel.props->minTrans, 1U);

  // Channels connected to memory interfaces are not bufferizable
  if ((isa<handshake::MemoryControllerOp>(channel.producer)) ||
      isa<handshake::MemoryControllerOp>(channel.consumer)) {
    channel.props->maxOpaque = 0;
    channel.props->maxTrans = 0;
  }
}

/// Calls the provided callback for all channels in the function. These are all
/// producer/consumer pairs between a function argument and an operand or an
/// operation's result and an operand.
static void callOnAllChannels(handshake::FuncOp funcOp,
                              void (*callback)(Channel &)) {
  for (BlockArgument arg : funcOp.getArguments())
    for (Operation *user : arg.getUsers()) {
      Channel channel(arg, *funcOp, *user);
      callback(channel);
    }

  for (Operation &op : funcOp.getOps())
    for (OpResult res : op.getResults())
      for (Operation *user : res.getUsers()) {
        Channel channel(res, op, *user);
        callback(channel);
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

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    // Check that the provided version is valid
    if (version != "fpga20") {
      modOp->emitError() << "Unkwown version \"" << version
                         << "\", expected one of [fpga20]";
      return signalPassFailure();
    }

    // Add properties to channels inside each function
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      callOnAllChannels(funcOp, setFPGA20Properties);
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakeSetBufferingProperties(
    const std::string &version) {
  return std::make_unique<HandshakeSetBufferingPropertiesPass>(version);
}
