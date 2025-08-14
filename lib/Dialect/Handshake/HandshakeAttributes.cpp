//===- HandshakeAttributes.cpp - Handshake attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the API to express and manipulate custom Handshake attributes.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// MemInterfaceAttr
//===----------------------------------------------------------------------===//

void MemInterfaceAttr::print(AsmPrinter &odsPrinter) const {
  std::optional<unsigned> group = getLsqGroup();
  if (group)
    odsPrinter << "<LSQ: " << *group << ">";
  else
    odsPrinter << "<MC>";
}

Attribute MemInterfaceAttr::parse(AsmParser &odsParser, Type odsType) {
  MLIRContext *ctx = odsParser.getContext();
  if (odsParser.parseLess())
    return nullptr;

  // Check whether the attributes indicates a connection to an MC
  if (!odsParser.parseOptionalKeyword("MC") && !odsParser.parseGreater())
    return MemInterfaceAttr::get(ctx);

  // If this is not to an MC it must be to an LSQ (LSQ: <group>)
  unsigned group = 0;
  if (odsParser.parseOptionalKeyword("LSQ") || odsParser.parseColon() ||
      odsParser.parseInteger(group))
    return nullptr;

  if (odsParser.parseGreater())
    return nullptr;
  return MemInterfaceAttr::get(ctx, group);
}

//===----------------------------------------------------------------------===//
// ChannelBufProps(Attr)
//===----------------------------------------------------------------------===//

ChannelBufProps::ChannelBufProps(unsigned minTrans,
                                 std::optional<unsigned> maxTrans,
                                 unsigned minOpaque,
                                 std::optional<unsigned> maxOpaque,
                                 unsigned minSlots, double inDelay,
                                 double outDelay, double delay)
    : minTrans(minTrans), maxTrans(maxTrans), minOpaque(minOpaque),
      maxOpaque(maxOpaque), minSlots(minSlots), inDelay(inDelay),
      outDelay(outDelay), delay(delay){};

bool ChannelBufProps::isSatisfiable() const {
  return (!maxTrans.has_value() || *maxTrans >= minTrans) &&
         (!maxOpaque.has_value() || *maxOpaque >= minOpaque);
}

bool ChannelBufProps::isBufferizable() const {
  return !maxTrans.has_value() || *maxTrans != 0 || !maxOpaque.has_value() ||
         *maxOpaque != 0;
}

bool ChannelBufProps::operator==(const ChannelBufProps &rhs) const {
  return (this->minTrans == rhs.minTrans) && (this->maxTrans == rhs.maxTrans) &&
         (this->minOpaque == rhs.minOpaque) &&
         (this->maxOpaque == rhs.maxOpaque) &&
         (this->minSlots == rhs.minSlots) && (this->inDelay == rhs.inDelay) &&
         (this->outDelay == rhs.outDelay) && (this->delay == rhs.delay);
}

/// Parses the maximum number of slots and fills the last argument with the
/// parsed value (std::nullopt if not specified).
static ParseResult parseMaxSlots(AsmParser &odsParser,
                                 std::optional<unsigned> &maxSlots) {
  if (odsParser.parseOptionalKeyword("inf")) {
    // There is a maximum
    unsigned parsedMaxSlots;
    if (odsParser.parseInteger(parsedMaxSlots))
      return failure();
    maxSlots = parsedMaxSlots;
  } else {
    // No maximum
    maxSlots = std::nullopt;
  }

  // Close the interval
  if (odsParser.parseRSquare())
    return failure();

  return success();
}

Attribute ChannelBufPropsAttr::parse(AsmParser &odsParser, Type odsType) {
  ChannelBufProps props;

  // Parse first interval (transparent slots)
  if (odsParser.parseLSquare() || odsParser.parseInteger(props.minTrans) ||
      odsParser.parseComma() || parseMaxSlots(odsParser, props.maxTrans))
    return nullptr;

  // Parse comma separating the two intervals
  if (odsParser.parseComma())
    return nullptr;

  // Parse second interval (opaque slots)
  if (odsParser.parseLSquare() || odsParser.parseInteger(props.minOpaque) ||
      odsParser.parseComma() || parseMaxSlots(odsParser, props.maxOpaque))
    return nullptr;

  // Parse minimum number of slots
  if (odsParser.parseComma() || odsParser.parseInteger(props.minSlots))
    return nullptr;

  // Parse the delays
  if (odsParser.parseComma() || odsParser.parseFloat(props.inDelay) ||
      odsParser.parseComma() || odsParser.parseFloat(props.outDelay) ||
      odsParser.parseComma() || odsParser.parseFloat(props.delay))
    return nullptr;

  return ChannelBufPropsAttr::get(odsParser.getContext(), props);
}

void ChannelBufPropsAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "[" << getMinTrans() << "," << getMaxStr(getMaxTrans()) << ", ["
             << getMinOpaque() << "," << getMaxStr(getMaxOpaque()) << ", "
             << getMinSlots() << ", " << getInDelay().getValueAsDouble() << ", "
             << getOutDelay().getValueAsDouble() << ", "
             << getDelay().getValueAsDouble();
}

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
