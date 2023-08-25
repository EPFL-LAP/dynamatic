//===- BufferingProperties.h - Buffer placement properties ------*- C++ -*-===//
//
// Infrastructure for specifying and manipulating buffering properties, which
// are used by the buffer placement logic to constrain the nature and
// positions of buffers in elastic circuits.
//
// NOTE: The underlying attribute that stores buffering properties for an
// operation's results internally uses the index of each operation's result to
// track which properties belong to which result. Modifying the list of results
// after having added channel buffering properties will therefore break this
// mapping and lead to undefined behavior.
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
/// the map, then it has no specific buffering properties.
mlir::DenseMap<mlir::OpResult, ChannelBufProps> getOpBufProps(Operation *op);

/// Clears the buffering properties attached to an operation's results.
void clearBufProps(Operation *op);

/// Adds buffering properties to an operation's result (i.e., to an output
/// channel). Fails and does nothing if the channel's producer already specified
/// buffering properties for it.
LogicalResult addChannelBufProps(OpResult res, ChannelBufProps channelProps);

/// Adds or replaces the buffering properties of an operation's results (i.e.,
/// of an output channel). If the channel's producer did not have buffering
/// properties specified for it already, adds the properties anyway and returns
/// false, otherwise returns true.
bool replaceChannelBufProps(OpResult res, ChannelBufProps channelProps);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_BUFFERINGPROPERTIES_H