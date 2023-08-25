//===- BufferingProperties.h - Buffer placement properties ------*- C++ -*-===//
//
// Infrastructure for specifying and manipulating buffering properties, which
// are used by the buffer placement logic to constrain the nature and
// positions of buffers in dataflow circuits.
//
// Buffering properties are fundamentally an attribute of MLIR values
// (equivalently, in buffer placement jargon, of channels). However, since it's
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
// i.e., it's responsibility of the caller to check the properties'
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

// Attribute name under which buffering properties for an operation's output
// channels are expected to be stored.
const std::string BUF_PROPS_ATTR = "bufProps";

/// Returns a mapping between the results of an operation that have channel
/// buffering properties to their properties. If an operation's result is not in
/// the returned map, it means it has no specific buffering properties. If the
/// provided operation is a handshake::FuncOp, the return value maps each
/// function argument to its buffering properties instead.
mlir::DenseMap<Value, ChannelBufProps> getAllBufProps(Operation *op);

/// Returns buffering properties associate to the operation's result, if any.
std::optional<ChannelBufProps> getBufProps(OpResult res);

/// Returns buffering properties associate to the function argument, if any.
std::optional<ChannelBufProps> getBufProps(BlockArgument arg);

/// Clears the buffering properties attached to an operation's results.
void clearBufProps(Operation *op);

/// Adds buffering properties to an operation's result (i.e., to an output
/// channel). Fails and does nothing if the channel's producer already specified
/// buffering properties for it.
LogicalResult addBufProps(OpResult res, ChannelBufProps &props);

/// Adds buffering properties to a function argument (i.e., to a circuit's
/// input channel). Fails if the block the argument belongs to isn't the child
/// of a Handshake function or if the function already specified buffering
/// properties for it.
LogicalResult addBufProps(BlockArgument arg, ChannelBufProps &props);

/// Adds or replaces the buffering properties of an operation's result (i.e.,
/// of an output channel). If the channel's producer already had buffering
/// properties specified for it and the last argument isn't nullptr, the
/// latter is set to true.
LogicalResult replaceBufProps(OpResult res, ChannelBufProps &props,
                              bool *replaced = nullptr);

/// Adds or replaces the buffering properties of a function argument (i.e., to a
/// circuit's input channel). Fails if the block the argument belongs to isn't
/// the child of a Handshake function. If the function already had buffering
/// properties specified for it and the last argument isn't nullptr, the latter
/// is set to true.
LogicalResult replaceBufProps(BlockArgument arg, ChannelBufProps &props,
                              bool *replaced = nullptr);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGPROPERTIES_H