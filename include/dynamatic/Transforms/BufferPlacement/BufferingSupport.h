//===- BufferingSupport.h - Support for buffer placement --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infrastructure for working around the buffer placement pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGSUPPORT_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGSUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"

namespace dynamatic {
namespace buffer {

/// Helper datatype for buffer placement. Simply aggregates all the information
/// related to the Handshake function under optimization.
struct FuncInfo {
  /// The Handshake function in which to place buffers.
  handshake::FuncOp funcOp;
  /// The list of archs in the function (i.e., transitions between basic
  /// blocks).
  SmallVector<experimental::ArchBB> archs;
  /// Maps CFDFCs of the function to a boolean indicating whether they each
  /// should be optimized.
  llvm::MapVector<CFDFC *, bool> cfdfcs;

  /// Argument-less constructor so that we can use the struct as a value type
  /// for maps.
  FuncInfo() : funcOp(nullptr){};

  /// Constructs an instance from the function it refers to. Other struct
  /// members start empty.
  FuncInfo(handshake::FuncOp funcOp) : funcOp(funcOp){};
};

/// Acts as a "smart and lazy getter" around a channel's buffering properties.
/// In situations where one may want to offer easy access to a channel's
/// buffering properties without fetching the attribute from the IR if not
/// necessary, this class enables lazily-loading the attribute on the first
/// access though the -> or * operators.
///
/// If the attribute is accessed and modified, and the object was created with
/// the updateOnDestruction flag set, the channel's buffering properties are
/// automatically updated in the IR on object destruction.
class LazyChannelBufProps {
public:
  /// Constructs an instance from the channel whose buffering properties will be
  /// managed by the object.
  inline LazyChannelBufProps(Value val, bool updateOnDestruction = false)
      : val(val), updateOnDestruction(updateOnDestruction){};

  /// Returns the underlying channel the object was created with.
  inline mlir::Value getChannel() { return val; }

  /// Forces an IR update with the currently modifed buffering properties.
  /// Returns true if any update was necessary.
  bool updateIR();

  /// Returns a reference to the channel's buffering properties through which
  /// they can be manipulated.
  handshake::ChannelBufProps &operator*();

  /// Returns a pointer to the channel's buffering properties through which
  /// they can be manipulated.
  handshake::ChannelBufProps *operator->();

  /// Since the class destructor may go update the IR, it's safer to prevent
  /// object copies.
  LazyChannelBufProps(const LazyChannelBufProps &) = delete;

  /// Since the class destructor may go update the IR, it's safer to prevent
  /// object copies.
  LazyChannelBufProps &operator=(const LazyChannelBufProps &) = delete;

  /// On destruction, and if the channel's buffering properties were modified,
  /// updates them in the IR.
  ~LazyChannelBufProps();

private:
  /// Channel that the buffering properties refer to.
  Value val;
  /// Whether to update the IR on object destruction when the properties were
  /// modifed.
  bool updateOnDestruction;
  /// Lazily-loaded buffering properties (std::nullopt by default, initialized
  /// on first read), which are given reference/pointer access to by the class's
  /// -> and * operators.
  std::optional<handshake::ChannelBufProps> props;
  /// Same as props, but is never modified from the moment it is lazily-laoaded.
  /// This allows us to avoid replacing the IR attribute on object destruction
  /// if no modification has been made.
  std::optional<handshake::ChannelBufProps> unchangedProps;

  /// Attempts to read the channel's buffering properties from the defining
  /// operation's IR attribute and sets props to them (or to a set of properties
  /// representing an unconstrained channel if none were attached to the
  /// channel). Called by the * and -> operators on first access.
  void readAttribute();

  /// If the channel's buffering properties were modified, updates them in the
  /// IR. Returns true if any update was necessary.
  bool updateIRIfNecessary();
};

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
  Operation *producer;
  /// Channel's consumer.
  Operation *consumer;
  /// Lazily-loaded channel-specific buffering properties.
  LazyChannelBufProps props;

  /// Constructs a channel from its assoicated SSA value, the value's producer,
  /// and one of its comsumers. To maximize flexibility, the constructor doesn't
  /// check in any way that the provided producer and consumer correspond to the
  /// SSA value. Use at your own risk.
  Channel(Value value, Operation *producer, Operation *consumer,
          bool updateProps = false)
      : value(value), producer(producer), consumer(consumer),
        props(value, updateProps){};

  /// Constructs a channel from its associated SSA value alone.
  Channel(Value value, bool updateProps = false);

  /// Returns a reference to the operation operand corresponding to the channel.
  OpOperand &getOperand() const;

  /// The lazily-loaded channel buffering properties cannot be safely copied, so
  /// neither can the channel.
  Channel(const Channel &) = delete;

  /// The lazily-loaded channel buffering properties cannot be safely copied, so
  /// neither can the channel.
  Channel &operator=(const Channel &) = delete;

  /// Determines whether the channel represents a function's argument.
  inline bool isFunArg() const { return isa<handshake::FuncOp>(producer); }
};

/// Holds information about what type of buffer should be placed on a specific
/// channel.
struct PlacementResult {
  /// The number of ONE_SLOT_BREAK_DV that should be placed.
  unsigned numOneSlotDV = 0;
  /// The number of ONE_SLOT_BREAK_R that should be placed.
  unsigned numOneSlotR = 0;
  /// The number of FIFO_BREAK_DV slots that should be placed.
  unsigned numFifoDV = 0;
  /// The number of FIFO_BREAK_NONE slots that should be placed.
  unsigned numFifoNone = 0;
  /// The number of ONE_SLOT_BREAK_DVR that should be placed.
  unsigned numOneSlotDVR = 0;
  /// The number of SHIFT_REG_BREAK_DV that should be placed.
  unsigned numShiftRegDV = 0;
};

/// Maps channels to buffer placement decisions.
using BufferPlacement = llvm::MapVector<Value, PlacementResult>;

/// Returns the producer of the channel (in the buffer placement sense), or
/// nullptr if the channel has no producer in this context. If idx is not
/// nullptr, it is filled with the definition index of the channel in its
/// producer.
Operation *getChannelProducer(Value channel, size_t *idx = nullptr);

/// Maps all the function's channels to their specific buffering properties,
/// adjusting for buffers within units as described by the timing models. Fails
/// if the buffering properties of a channel are unsatisfiable or become
/// unsatisfiable after adjustment.
LogicalResult mapChannelsToProperties(
    handshake::FuncOp funcOp, const TimingDatabase &timingDB,
    llvm::MapVector<Value, handshake::ChannelBufProps> &channelProps);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGSUPPORT_H