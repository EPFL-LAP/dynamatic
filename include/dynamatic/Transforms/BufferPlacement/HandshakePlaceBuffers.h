//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --handshake-place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractCFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {

/// Stores information related to a channel (in the buffer placement sense)
/// i.e., an SSA value along with its producer (an operation) and its unique
/// consumer (another operation). This struct is just a way to aggregate data;
/// it performs no internal verification that the producer/comsumer are
/// associated to the value in any meaningful sense. Semantically, it is
/// expected that the consumer is one of the value's users (though it may not be
/// the only one i.e., one does not need to have a materialized IR to use this
/// struct) and that the producer is either (1) the value's defining operation
/// if it is an OpResult or (2) a handshake::FuncOp instance if it is a
/// BlockArgument. Additionally, the struct allows one to lazily access the
/// channel's buffering properties that may be stored in the IR.
struct Channel {
  /// SSA value representing the channel.
  Value value;
  /// Channel's producer.
  Operation &producer;
  /// Channel's consumer.
  Operation &consumer;
  /// Lazily-loaded channel-specific buffering properties.
  LazyChannelBufProps props;

  /// Constructs a channel from its assoicated SSA value, the value's producer,
  /// and one of its comsumers. To maximize flexibility, the constructor doesn't
  /// check in any way that the provided producer and consumer correspond to the
  /// SSA value.
  Channel(Value value, Operation &producer, Operation &consumer);

  /// The lazily-loaded channel buffering properties cannot be safely copied, so
  /// neither can the channel.
  Channel(const Channel &) = delete;

  /// The lazily-loaded channel buffering properties cannot be safely copied, so
  /// neither can the channel.
  Channel &operator=(const Channel &) = delete;

  /// Determines whether the channel represents a function's argument.
  inline bool isFunArg() const {
    return isa<circt::handshake::FuncOp>(producer);
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstCFDFC = false,
                                std::string stdLevelInfo = "",
                                std::string timefile = "",
                                double targetCP = 4.0, int timeLimit = 180,
                                bool setCustom = true);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
