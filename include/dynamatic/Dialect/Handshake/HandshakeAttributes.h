//===- HandshakeAttributes.h - Handshake attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares backing data-structures and API to express and manipulate custom
// Handshake attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_H

#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace dynamatic {
namespace handshake {

struct TimingInfo {
  std::optional<unsigned> dataLatency;
  std::optional<unsigned> validLatency;
  std::optional<unsigned> readyLatency;

  std::optional<unsigned> getLatency(SignalType type);

  TimingInfo &setLatency(SignalType type, unsigned latency);

  mlir::ParseResult parseKey(mlir::AsmParser &odsParser, mlir::StringRef key);

private:
  static constexpr llvm::StringLiteral DATA = "D", VALID = "V", READY = "R";

  std::optional<SignalType> getCorrespondingSignalType(mlir::StringRef key);
};

bool operator==(const TimingInfo &lhs, const TimingInfo &rhs);

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const TimingInfo &timing);

/// Specifies how a handshake channel (i.e. a SSA value used once) may be
/// buffered. Backing data-structure for the ChannelBufPropsAttr attribute.
struct ChannelBufProps {
  /// Minimum number of transparent slots allowed on the channel (inclusive).
  unsigned minTrans;
  /// Maximum number of transparent slots allowed on the channel (inclusive).
  std::optional<unsigned> maxTrans;
  /// Minimum number of opaque slots allowed on the channel (inclusive).
  unsigned minOpaque;
  /// Maximum number of opaque slots allowed on the channel (inclusive).
  std::optional<unsigned> maxOpaque;
  /// Combinational delay (in ns) from the output port to the buffer's input, if
  /// a buffer is placed on the channel.
  double inDelay;
  /// Combinational delay (in ns) from the buffer's output to the input port, if
  /// a buffer is placed on the channel.
  double outDelay;
  /// Total combinational channel delay (in ns) if no buffer is placed on the
  /// channel.
  double delay;

  /// Simple constructor that takes the same parameters as the struct's members.
  /// By default, all the channel is "unconstrained" w.r.t. what kind of buffers
  /// can be placed and is assumed to have 0 delay.
  ChannelBufProps(unsigned minTrans = 0,
                  std::optional<unsigned> maxTrans = std::nullopt,
                  unsigned minOpaque = 0,
                  std::optional<unsigned> maxOpaque = std::nullopt,
                  double inDelay = 0.0, double outDelay = 0.0,
                  double delay = 0.0);

  /// Determines whether these buffering properties are satisfiable i.e.,
  /// whether it's possible to create a buffer that respects them.
  bool isSatisfiable() const;

  /// Determines whether these buffering properties forbid the placement of
  /// any buffer on the associated channel.
  bool isBufferizable() const;

  /// Computes member-wise equality.
  bool operator==(const ChannelBufProps &rhs) const;
};

static inline std::string getMaxStr(std::optional<unsigned> optMax) {
  return optMax.has_value() ? (std::to_string(optMax.value()) + "]") : "inf]";
};

/// Prints the buffering properties as two closed or semi-open intervals
/// (depending on whether maximums are defined), one for tranparent slots and
/// one for opaque slots.
template <typename Os>
Os &operator<<(Os &os, ChannelBufProps &props) {
  os << "{\n\ttransparent slots: [" << props.minTrans << ", "
     << getMaxStr(props.maxTrans) << "\n\topaque slots: [" << props.minOpaque
     << ", " << getMaxStr(props.maxOpaque) << "\n\tin/out delays: ("
     << props.inDelay << ", " << props.outDelay << ")"
     << "\n\ttotal delay: " << props.delay << "\n}\n";
  return os;
}

} // namespace handshake
} // namespace dynamatic

namespace mlir {
namespace affine {
struct DependenceComponent;
} // namespace affine
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ATTRIBUTES_H
