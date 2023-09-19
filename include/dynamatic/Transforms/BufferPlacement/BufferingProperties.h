//===- BufferingProperties.h - Buffer placement properties ------*- C++ -*-===//
//
// Infrastructure for specifying and manipulating buffering properties, which
// are used by the buffer placement logic to constrain the nature and
// positions of buffers in dataflow circuits.
//
// Buffering properties are fundamentally an attribute of MLIR values
// (equivalently, in buffer placement jargon, of channels). However, since it is
// impossible to attach attributes to MLIR SSA values directly, we opt to attach
// these "value attributes" to the operation that defines them instead. This
// implies that an operation may contain buffering properties for multiple
// values if it defines multiple values. We special case the handshake::FuncOp
// operation, for which it's understood that any set of buffering properties
// refer to its arguments instead of its results. Internally, operations keep
// track of which buffering properties are attached to which of the values they
// define using their index. The API defined in this file largely tries to hide
// those cumbersome index management steps and exposes functions to manage
// buffering properties for a specific operation result or function argument
// directly. None of the functions declared in this file make any attempt at
// preventing the addition of unsatisfiable buffering properties to a channel
// i.e., it is the responsibility of the callers to check the properties'
// satisifability if they care about it.
//
// Since the underlying attribute that stores buffering properties maps
// properties to values using their indices, modifying the list of
// results/function arguments after having added channel buffering properties
// will break this mapping and lead to undefined behavior.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGPROPERTIES_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGPROPERTIES_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace buffer {

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
  ChannelBufProps &operator*();

  /// Returns a pointer to the channel's buffering properties through which
  /// they can be manipulated.
  ChannelBufProps *operator->();

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
  std::optional<ChannelBufProps> props;
  /// Same as props, but is never modified from the moment it is lazily-laoaded.
  /// This allows us to avoid replacing the IR attribute on object destruction
  /// if no modification has been made.
  std::optional<ChannelBufProps> unchangedProps;

  /// Attempts to read the channel's buffering properties from the defining
  /// operation's IR attribute and sets props to them (or to a set of properties
  /// representing an unconstrained channel if none were attached to the
  /// channel). Called by the * and -> operators on first access.
  void readAttribute();

  /// If the channel's buffering properties were modified, updates them in the
  /// IR. Returns true if any update was necessary.
  bool updateIRIfNecessary();
};

/// Returns the producer of the channel (in the buffer placement sense), or
/// nullptr if the channel has no producer in this context. If idx is not
/// nullptr, it is filled with the definition index of the channel in its
/// producer.
Operation *getChannelProducer(Value channel, size_t *idx = nullptr);

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
  /// NOTE: Modifying these properties will cause the IR to be updated.
  LazyChannelBufProps props;

  /// Constructs a channel from its assoicated SSA value, the value's producer,
  /// and one of its comsumers. To maximize flexibility, the constructor doesn't
  /// check in any way that the provided producer and consumer correspond to the
  /// SSA value.
  inline Channel(Value value, Operation &producer, Operation &consumer,
                 bool updateProps = false)
      : value(value), producer(producer), consumer(consumer),
        props(value, updateProps){};

  /// Constructs a channel from its associated SSA value alone. Fails if the
  /// value's producer is invalid or if does not have at least one consumer. Use
  /// at your own risk.
  inline Channel(Value value, bool updateProps = false)
      : value(value), producer(*getChannelProducer(value)),
        consumer(**value.getUsers().begin()), props(value, updateProps){};

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

// Attribute name under which buffering properties for an operation's output
// channels are expected to be stored.
const std::string BUF_PROPS_ATTR = "bufProps";

/// Returns a mapping between the results of an operation that has channel
/// buffering properties to their properties. If an operation's result is not in
/// the returned map, it means it has no specific buffering properties. If the
/// provided operation is a handshake::FuncOp, the return value maps each
/// function argument to its buffering properties instead.
mlir::DenseMap<Value, ChannelBufProps> getAllBufProps(Operation *op);

/// Returns buffering properties associated to a channel, if any are defined.
std::optional<ChannelBufProps> getBufProps(Value channel);

/// Adds buffering properties to a channel that must either be an operation's
/// result or a Handshake function argument. Fails and does nothing if the
/// channel's producer already specified buffering properties for the channel.
LogicalResult addBufProps(Value channel, ChannelBufProps &props);

/// Adds or replaces the buffering properties of a channel that must either be
/// an operation's result or a Handshake function argument. If the channel's
/// producer already had buffering properties specified for it and the last
/// argument isn't nullptr, the latter is set to true.
LogicalResult replaceBufProps(Value channel, ChannelBufProps &props,
                              bool *replaced = nullptr);

/// Clears the buffering properties attached to an operation's results.
void clearBufProps(Operation *op);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGPROPERTIES_H