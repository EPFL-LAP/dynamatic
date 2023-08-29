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
/// access though the -> or * operators. If the attribute is accessed and
/// modified, the channel's buffering properties are automatically updated
/// on object destruction.
class LazyChannelBufProps {
public:
  /// Constructs an instance from the channel whose buffering properties will be
  /// managed by the object.
  LazyChannelBufProps(Value val);

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