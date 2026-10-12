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
// MemGroupAttr
//===----------------------------------------------------------------------===//

static constexpr StringLiteral MEM_GROUP_CTRL("ctrl"), MEM_GROUP_LD("ld"),
    MEM_GROUP_ST("st");

void MemGroupAttr::print(AsmPrinter &odsPrinter) const {
  odsPrinter << "<" << getId();
  if (getHasCtrl())
    odsPrinter << ", " << MEM_GROUP_CTRL;
  odsPrinter << ", [";
  llvm::interleaveComma(getAccesses(), odsPrinter, [&](MemAccessKind kind) {
    odsPrinter << (kind == MemAccessKind::LOAD ? MEM_GROUP_LD : MEM_GROUP_ST);
  });
  odsPrinter << "]>";
}

Attribute MemGroupAttr::parse(AsmParser &odsParser, Type odsType) {
  unsigned id = 0;
  if (odsParser.parseLess() || odsParser.parseInteger(id) ||
      odsParser.parseComma())
    return nullptr;

  // The control keyword is optional
  bool hasCtrl = false;
  if (succeeded(odsParser.parseOptionalKeyword(MEM_GROUP_CTRL))) {
    hasCtrl = true;
    if (odsParser.parseComma())
      return nullptr;
  }

  SmallVector<MemAccessKind> accesses;
  auto parseAccess = [&]() -> ParseResult {
    StringRef keyword;
    llvm::SMLoc loc = odsParser.getCurrentLocation();
    if (odsParser.parseKeyword(&keyword))
      return failure();
    if (keyword == MEM_GROUP_LD)
      accesses.push_back(MemAccessKind::LOAD);
    else if (keyword == MEM_GROUP_ST)
      accesses.push_back(MemAccessKind::STORE);
    else
      return odsParser.emitError(loc)
             << "expected '" << MEM_GROUP_LD << "' or '" << MEM_GROUP_ST
             << "', but got '" << keyword << "'";
    return success();
  };
  if (odsParser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        parseAccess) ||
      odsParser.parseGreater())
    return nullptr;
  return MemGroupAttr::get(odsParser.getContext(), id, hasCtrl, accesses);
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
      outDelay(outDelay), delay(delay) {}

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
