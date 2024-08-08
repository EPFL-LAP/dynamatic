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
// MemDependenceAttr
//===----------------------------------------------------------------------===//

/// Pretty-prints a dependence component as [lb, ub] where both lb and ub are
/// optional (in which case, no bound is printed).
static void printDependenceComponent(AsmPrinter &odsPrinter,
                                     const DependenceComponentAttr &comp) {
  std::string lb =
      comp.getLb().has_value() ? std::to_string(comp.getLb().value()) : "";
  std::string ub =
      comp.getUb().has_value() ? std::to_string(comp.getUb().value()) : "";
  odsPrinter << "[" << lb << ", " << ub << "]";
}

/// Parses one optional bound from a dependence component. Succeeds when
/// parsing succeeds (even if the value is not present, since it's optional).
static ParseResult parseDependenceComponent(AsmParser &odsParser,
                                            std::optional<int64_t> &opt) {
  // For some obscure reason, trying to directly parse an int64_t from the IR
  // results in errors when the bound is either INT64_MIN or INT64_MAX. We
  // hack around this by parsing APInt instead and then explicitly checking
  // for underflow/overflow before converting back to an int64_t
  APInt bound;
  if (auto res = odsParser.parseOptionalInteger(bound); res.has_value()) {
    if (failed(res.value()))
      return failure();
    if (bound.getBitWidth() > sizeof(int64_t) * CHAR_BIT)
      opt = bound.isNegative() ? INT64_MIN : INT64_MAX;
    else
      opt = bound.getSExtValue();
  }
  return success();
}

void MemDependenceAttr::print(AsmPrinter &odsPrinter) const {
  // Print destination memory access and loop depth
  odsPrinter << "<\"" << getDstAccess().str() << "\" (" << getLoopDepth()
             << ")";

  // Print dependence components, if present
  auto components = getComponents();
  if (!components.empty()) {
    odsPrinter << " [";
    for (auto &comp : components.drop_back(1)) {
      printDependenceComponent(odsPrinter, comp);
      odsPrinter << ", ";
    }
    printDependenceComponent(odsPrinter, components.back());
    odsPrinter << "]";
  }
  odsPrinter << ">";
}

Attribute MemDependenceAttr::parse(AsmParser &odsParser, Type odsType) {

  MLIRContext *ctx = odsParser.getContext();

  // Parse destination memory access
  std::string dstName;
  if (odsParser.parseLess() || odsParser.parseString(&dstName))
    return nullptr;
  auto dstAccess = StringAttr::get(ctx, dstName);

  // Parse loop depth
  unsigned loopDepth;
  if (odsParser.parseLParen() || odsParser.parseInteger<unsigned>(loopDepth) ||
      odsParser.parseRParen())
    return nullptr;

  // Parse dependence components if present
  SmallVector<DependenceComponentAttr> components;
  if (!odsParser.parseOptionalLSquare()) {
    auto parseDepComp = [&]() -> ParseResult {
      std::optional<int64_t> optLb = std::nullopt, optUb = std::nullopt;

      // Parse [lb, ub] where both lb and ub are optional
      if (odsParser.parseLSquare() ||
          failed(parseDependenceComponent(odsParser, optLb)) ||
          odsParser.parseComma() ||
          failed(parseDependenceComponent(odsParser, optUb)) ||
          odsParser.parseRSquare())
        return failure();

      components.push_back(DependenceComponentAttr::get(ctx, optLb, optUb));
      return success();
    };

    if (odsParser.parseCommaSeparatedList(parseDepComp) ||
        odsParser.parseRSquare())
      return nullptr;
  }

  if (odsParser.parseGreater())
    return nullptr;
  return MemDependenceAttr::get(ctx, dstAccess, loopDepth, components);
}

//===----------------------------------------------------------------------===//
// ChannelBufProps(Attr)
//===----------------------------------------------------------------------===//

ChannelBufProps::ChannelBufProps(unsigned minTrans,
                                 std::optional<unsigned> maxTrans,
                                 unsigned minOpaque,
                                 std::optional<unsigned> maxOpaque,
                                 double inDelay, double outDelay, double delay)
    : minTrans(minTrans), maxTrans(maxTrans), minOpaque(minOpaque),
      maxOpaque(maxOpaque), inDelay(inDelay), outDelay(outDelay),
      delay(delay) {};

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
         (this->maxOpaque == rhs.maxOpaque) && (this->inDelay == rhs.inDelay) &&
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
             << getInDelay().getValueAsDouble() << ", "
             << getOutDelay().getValueAsDouble() << ", "
             << getDelay().getValueAsDouble();
}

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.cpp.inc"
